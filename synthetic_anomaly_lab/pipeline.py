from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Event:
    """One synthetic transaction-like event in event-time order."""

    occurred_at: int
    amount_minor: int
    account_age_days: int
    velocity_1h: int
    is_anomaly: bool

    def features(self) -> tuple[float, float, float]:
        if self.occurred_at < 0:
            raise ValueError("occurred_at must be non-negative")
        if self.amount_minor < 0:
            raise ValueError("amount_minor must be non-negative")
        if self.account_age_days < 0:
            raise ValueError("account_age_days must be non-negative")
        if self.velocity_1h < 0:
            raise ValueError("velocity_1h must be non-negative")
        return (
            math.log1p(self.amount_minor),
            math.log1p(self.account_age_days),
            math.log1p(self.velocity_1h),
        )


@dataclass(frozen=True, slots=True)
class AnomalyModel:
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    threshold: float

    def score(self, event: Event) -> float:
        values = event.features()
        if len(values) != len(self.centers):
            raise ValueError("feature count does not match model")
        return sum(
            ((value - center) / scale) ** 2
            for value, center, scale in zip(values, self.centers, self.scales)
        ) / len(values)

    def predict(self, event: Event) -> bool:
        return self.score(event) >= self.threshold


@dataclass(frozen=True, slots=True)
class Evaluation:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True)
class BacktestFold:
    """One expanding-window evaluation with an untouched future test window."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_events: int
    test_events: int
    threshold: float
    evaluation: Evaluation
    amount_psi: float


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """A model whose threshold was selected on a past-only labeled window."""

    model: AnomalyModel
    fit_events: int
    calibration_events: int
    calibration_start: int
    calibration_end: int
    calibration_evaluation: Evaluation
    strategy: str


def generate_events(count: int, *, seed: int = 7) -> list[Event]:
    """Generate deterministic labeled data; no external or personal data is used."""
    if count < 20:
        raise ValueError("count must be at least 20")
    rng = random.Random(seed)
    events: list[Event] = []
    for index in range(count):
        anomaly = rng.random() < 0.06
        amount = max(1, round(rng.lognormvariate(7.2, 0.65)))
        velocity = max(0, round(rng.gauss(3.0, 1.5)))
        account_age = rng.randint(1, 1_800)
        if anomaly:
            amount *= rng.randint(12, 25)
            velocity += rng.randint(18, 40)
            account_age = rng.randint(0, 14)
        # A modest late-period shift makes drift reporting meaningful.
        if index >= count * 4 // 5:
            amount = round(amount * 1.25)
        events.append(
            Event(
                occurred_at=1_800_000_000 + index * 60,
                amount_minor=amount,
                account_age_days=account_age,
                velocity_1h=velocity,
                is_anomaly=anomaly,
            )
        )
    return events


def chronological_split(
    events: Sequence[Event], *, train_fraction: float = 0.7
) -> tuple[list[Event], list[Event]]:
    """Split by event time, never randomly, to prevent future-to-past leakage."""
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    if len(events) < 4:
        raise ValueError("at least four events are required")
    ordered = sorted(events, key=lambda event: event.occurred_at)
    if any(
        left.occurred_at == right.occurred_at
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("occurred_at values must be unique")
    split_at = int(len(ordered) * train_fraction)
    return ordered[:split_at], ordered[split_at:]


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of empty values")
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def fit(events: Sequence[Event], *, expected_normal_fraction: float = 0.94) -> AnomalyModel:
    """Fit center, scale and threshold using training events only."""
    if len(events) < 10:
        raise ValueError("at least ten training events are required")
    if not 0.5 <= expected_normal_fraction < 1.0:
        raise ValueError("expected_normal_fraction must be in [0.5, 1.0)")
    rows = [event.features() for event in events]
    columns = list(zip(*rows))
    centers = tuple(statistics.median(column) for column in columns)
    scales = tuple(max(statistics.pstdev(column), 1e-9) for column in columns)
    provisional = AnomalyModel(centers=centers, scales=scales, threshold=math.inf)
    threshold = _quantile(
        [provisional.score(event) for event in events], expected_normal_fraction
    )
    return AnomalyModel(centers=centers, scales=scales, threshold=threshold)


def fit_historical_threshold(
    events: Sequence[Event], *, calibration_fraction: float = 0.2
) -> ThresholdSelection:
    """Select a score threshold with labels from a past-only calibration window.

    Centers and scales are fitted on the earlier prefix. Candidate thresholds are
    scored on the later calibration suffix, with deterministic tie-breaking. If
    that suffix has no positive labels, selection falls back to the unsupervised
    training quantile rather than claiming that supervised calibration occurred.
    """
    if len(events) < 20:
        raise ValueError("at least 20 historical events are required")
    if not 0.1 <= calibration_fraction <= 0.4:
        raise ValueError("calibration_fraction must be in [0.1, 0.4]")
    ordered = sorted(events, key=lambda event: event.occurred_at)
    if any(
        left.occurred_at == right.occurred_at
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("occurred_at values must be unique")

    calibration_size = max(2, math.ceil(len(ordered) * calibration_fraction))
    fit_events = ordered[:-calibration_size]
    calibration_events = ordered[-calibration_size:]
    base_model = fit(fit_events)
    strategy = "historical_label_f1"

    if any(event.is_anomaly for event in calibration_events):
        scores = sorted({base_model.score(event) for event in calibration_events})
        candidates = [*scores, math.nextafter(scores[-1], math.inf)]
        evaluated = [
            (
                evaluate(
                    dataclasses.replace(base_model, threshold=threshold),
                    calibration_events,
                ),
                threshold,
            )
            for threshold in candidates
        ]
        calibration_evaluation, threshold = max(
            evaluated,
            key=lambda item: (
                item[0].f1,
                item[0].precision,
                item[0].recall,
                item[1],
            ),
        )
        model = dataclasses.replace(base_model, threshold=threshold)
    else:
        model = base_model
        calibration_evaluation = evaluate(model, calibration_events)
        strategy = "unsupervised_quantile_fallback_no_positive_labels"

    return ThresholdSelection(
        model=model,
        fit_events=len(fit_events),
        calibration_events=len(calibration_events),
        calibration_start=calibration_events[0].occurred_at,
        calibration_end=calibration_events[-1].occurred_at,
        calibration_evaluation=calibration_evaluation,
        strategy=strategy,
    )


def evaluate(model: AnomalyModel, events: Iterable[Event]) -> Evaluation:
    tp = fp = fn = tn = 0
    for event in events:
        predicted = model.predict(event)
        if predicted and event.is_anomaly:
            tp += 1
        elif predicted:
            fp += 1
        elif event.is_anomaly:
            fn += 1
        else:
            tn += 1
    return Evaluation(tp, fp, fn, tn)


def population_stability_index(
    baseline: Sequence[float], current: Sequence[float], *, bins: int = 10
) -> float:
    """Compare one numeric feature against baseline quantile buckets."""
    if len(baseline) < bins or not current:
        raise ValueError("baseline must fill every bin and current cannot be empty")
    if bins < 2 or bins > 100:
        raise ValueError("bins must be in [2, 100]")
    if not all(math.isfinite(value) for value in (*baseline, *current)):
        raise ValueError("values must be finite")
    boundaries = sorted(
        set(_quantile(baseline, index / bins) for index in range(1, bins))
    )
    baseline_counts = [0] * (len(boundaries) + 1)
    current_counts = [0] * (len(boundaries) + 1)
    for value in baseline:
        baseline_counts[bisect.bisect_right(boundaries, value)] += 1
    for value in current:
        current_counts[bisect.bisect_right(boundaries, value)] += 1
    epsilon = 1e-6
    psi = 0.0
    for baseline_count, current_count in zip(baseline_counts, current_counts):
        baseline_share = max(baseline_count / len(baseline), epsilon)
        current_share = max(current_count / len(current), epsilon)
        psi += (current_share - baseline_share) * math.log(
            current_share / baseline_share
        )
    return psi


def rolling_backtest(
    events: Sequence[Event], *, folds: int = 4, initial_train_fraction: float = 0.5
) -> list[BacktestFold]:
    """Run expanding-window folds; each model sees only events before its test window."""
    if not 2 <= folds <= 10:
        raise ValueError("folds must be in [2, 10]")
    if not 0.4 <= initial_train_fraction <= 0.8:
        raise ValueError("initial_train_fraction must be in [0.4, 0.8]")
    if len(events) < 40:
        raise ValueError("at least 40 events are required")

    ordered = sorted(events, key=lambda event: event.occurred_at)
    if any(
        left.occurred_at == right.occurred_at
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("occurred_at values must be unique")

    initial_train_size = int(len(ordered) * initial_train_fraction)
    remaining = len(ordered) - initial_train_size
    if remaining < folds:
        raise ValueError("not enough test events for the requested folds")
    base_test_size, extra = divmod(remaining, folds)

    results: list[BacktestFold] = []
    test_start_index = initial_train_size
    for fold_index in range(folds):
        test_size = base_test_size + (1 if fold_index < extra else 0)
        test_end_index = test_start_index + test_size
        train = ordered[:test_start_index]
        test = ordered[test_start_index:test_end_index]
        model = fit(train)
        results.append(
            BacktestFold(
                train_start=train[0].occurred_at,
                train_end=train[-1].occurred_at,
                test_start=test[0].occurred_at,
                test_end=test[-1].occurred_at,
                train_events=len(train),
                test_events=len(test),
                threshold=model.threshold,
                evaluation=evaluate(model, test),
                amount_psi=population_stability_index(
                    [event.features()[0] for event in train],
                    [event.features()[0] for event in test],
                ),
            )
        )
        test_start_index = test_end_index
    return results


def run(count: int, seed: int) -> dict[str, object]:
    events = generate_events(count, seed=seed)
    train, test = chronological_split(events)
    model = fit(train)
    metrics = evaluate(model, test)
    historical_selection = fit_historical_threshold(train)
    historical_metrics = evaluate(historical_selection.model, test)
    amount_psi = population_stability_index(
        [event.features()[0] for event in train],
        [event.features()[0] for event in test],
    )
    backtest = rolling_backtest(events)
    return {
        "data": "synthetic-only",
        "seed": seed,
        "train_events": len(train),
        "test_events": len(test),
        "threshold": round(model.threshold, 6),
        "precision": round(metrics.precision, 6),
        "recall": round(metrics.recall, 6),
        "f1": round(metrics.f1, 6),
        "amount_psi": round(amount_psi, 6),
        "threshold_comparison": [
            {
                "strategy": "unsupervised_train_quantile",
                "threshold": round(model.threshold, 6),
                "precision": round(metrics.precision, 6),
                "recall": round(metrics.recall, 6),
                "f1": round(metrics.f1, 6),
            },
            {
                "strategy": historical_selection.strategy,
                "fit_events": historical_selection.fit_events,
                "calibration_events": historical_selection.calibration_events,
                "calibration_start": historical_selection.calibration_start,
                "calibration_end": historical_selection.calibration_end,
                "threshold": round(historical_selection.model.threshold, 6),
                "precision": round(historical_metrics.precision, 6),
                "recall": round(historical_metrics.recall, 6),
                "f1": round(historical_metrics.f1, 6),
            },
        ],
        "rolling_backtest": [
            {
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "train_events": fold.train_events,
                "test_events": fold.test_events,
                "threshold": round(fold.threshold, 6),
                "precision": round(fold.evaluation.precision, 6),
                "recall": round(fold.evaluation.recall, 6),
                "f1": round(fold.evaluation.f1, 6),
                "amount_psi": round(fold.amount_psi, 6),
            }
            for fold in backtest
        ],
        "limitations": [
            "statistical baseline, not a production fraud decision system",
            "labels and events are generated, not representative of real customers",
            "scores require human and domain validation before any operational use",
            "historical-label threshold selection optimizes synthetic F1, not real-world cost",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run(args.count, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
