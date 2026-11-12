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


FEATURE_NAMES = ("log_amount_minor", "log_account_age_days", "log_velocity_1h")


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
class FeatureContribution:
    """One feature's exact non-negative contribution to an anomaly score."""

    feature: str
    transformed_value: float
    center: float
    scale: float
    standardized_deviation: float
    contribution: float


@dataclass(frozen=True, slots=True)
class ScoreExplanation:
    """An additive decomposition of this baseline's squared-distance score."""

    score: float
    threshold: float
    predicted_anomaly: bool
    contributions: tuple[FeatureContribution, ...]


@dataclass(frozen=True, slots=True)
class AnomalyModel:
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    threshold: float

    def explain(self, event: Event) -> ScoreExplanation:
        values = event.features()
        if not (
            len(values) == len(FEATURE_NAMES) == len(self.centers) == len(self.scales)
        ):
            raise ValueError("feature count does not match model")
        if any(not math.isfinite(scale) or scale <= 0 for scale in self.scales):
            raise ValueError("model scales must be positive and finite")
        contributions = tuple(
            FeatureContribution(
                feature=name,
                transformed_value=value,
                center=center,
                scale=scale,
                standardized_deviation=(value - center) / scale,
                contribution=((value - center) / scale) ** 2 / len(values),
            )
            for name, value, center, scale in zip(
                FEATURE_NAMES, values, self.centers, self.scales
            )
        )
        score = sum(item.contribution for item in contributions)
        return ScoreExplanation(
            score=score,
            threshold=self.threshold,
            predicted_anomaly=score >= self.threshold,
            contributions=tuple(
                sorted(
                    contributions,
                    key=lambda item: (-item.contribution, item.feature),
                )
            ),
        )

    def score(self, event: Event) -> float:
        return self.explain(event).score

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
class FeatureDrift:
    """PSI for one fixed transformed feature and its bucket/support context."""

    feature: str
    psi: float
    baseline_support: int
    current_support: int
    requested_bins: int
    effective_bins: int
    baseline_unique_values: int
    degenerate_baseline: bool


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
    feature_drift: tuple[FeatureDrift, ...]


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


@dataclass(frozen=True, slots=True)
class SegmentEvaluation:
    """Evaluation for one bounded account-age cohort."""

    segment: str
    support: int
    positive_labels: int
    predicted_positives: int
    limited_support: bool
    evaluation: Evaluation


@dataclass(frozen=True, slots=True)
class DriftWindow:
    """One ordered synthetic monitoring window for policy simulation."""

    start: int
    end: int
    amount_psi: float
    support: int


@dataclass(frozen=True, slots=True)
class DriftPolicyDecision:
    """A simulated candidate response; it never changes model or transaction state."""

    window_start: int
    window_end: int
    amount_psi: float
    support: int
    consecutive_critical_windows: int
    candidate_response: str
    reason: str


@dataclass(frozen=True, slots=True)
class FeatureDriftWindow:
    """One ordered window containing the fixed-schema feature diagnostics."""

    start: int
    end: int
    diagnostics: tuple[FeatureDrift, ...]


@dataclass(frozen=True, slots=True)
class FeatureDriftPolicyDecision:
    """A simulated response to supported, non-degenerate per-feature PSI."""

    window_start: int
    window_end: int
    trigger_feature: str | None
    trigger_psi: float | None
    eligible_features: tuple[str, ...]
    excluded_features: tuple[str, ...]
    consecutive_critical_windows: int
    candidate_response: str
    reason: str


POLICY_RESPONSES = (
    "monitor",
    "investigate",
    "collect_more_evidence",
    "candidate_pause_and_human_review",
)


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


def fit(
    events: Sequence[Event],
    *,
    expected_normal_fraction: float = 0.94,
    scaling: str = "standard_deviation",
) -> AnomalyModel:
    """Fit center, scale and threshold using training events only.

    ``robust_mad`` uses the median absolute deviation with the 1.4826 normal-
    consistency factor for comparability with standard deviation under a normal
    distribution. Both strategies retain the median center and fit exclusively on
    the supplied history.
    """
    if len(events) < 10:
        raise ValueError("at least ten training events are required")
    if not 0.5 <= expected_normal_fraction < 1.0:
        raise ValueError("expected_normal_fraction must be in [0.5, 1.0)")
    if scaling not in {"standard_deviation", "robust_mad"}:
        raise ValueError("scaling must be 'standard_deviation' or 'robust_mad'")
    rows = [event.features() for event in events]
    columns = list(zip(*rows))
    centers = tuple(statistics.median(column) for column in columns)
    if scaling == "standard_deviation":
        scales = tuple(max(statistics.pstdev(column), 1e-9) for column in columns)
    else:
        scales = tuple(
            max(
                1.4826 * statistics.median(abs(value - center) for value in column),
                1e-9,
            )
            for column, center in zip(columns, centers)
        )
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


def account_age_segment_diagnostics(
    model: AnomalyModel, events: Iterable[Event], *, minimum_support: int = 50
) -> list[SegmentEvaluation]:
    """Report model behavior for three fixed, mutually exclusive age cohorts.

    Cohorts are deliberately bounded and include support counts. They are descriptive
    diagnostics, not fairness evidence or a policy/authorization recommendation.
    """
    if not 1 <= minimum_support <= 10_000:
        raise ValueError("minimum_support must be in [1, 10000]")
    cohorts: dict[str, list[Event]] = {
        "new_0_30_days": [],
        "growing_31_365_days": [],
        "established_366_plus_days": [],
    }
    for event in events:
        event.features()
        if event.account_age_days <= 30:
            cohorts["new_0_30_days"].append(event)
        elif event.account_age_days <= 365:
            cohorts["growing_31_365_days"].append(event)
        else:
            cohorts["established_366_plus_days"].append(event)

    results: list[SegmentEvaluation] = []
    for segment, cohort in cohorts.items():
        evaluation = evaluate(model, cohort)
        results.append(
            SegmentEvaluation(
                segment=segment,
                support=len(cohort),
                positive_labels=sum(event.is_anomaly for event in cohort),
                predicted_positives=evaluation.true_positives
                + evaluation.false_positives,
                limited_support=len(cohort) < minimum_support,
                evaluation=evaluation,
            )
        )
    return results


def population_stability_index(
    baseline: Sequence[float], current: Sequence[float], *, bins: int = 10
) -> float:
    """Compare one numeric feature against baseline quantile buckets."""
    psi, _, _ = _population_stability_details(baseline, current, bins=bins)
    return psi


def _population_stability_details(
    baseline: Sequence[float], current: Sequence[float], *, bins: int
) -> tuple[float, int, int]:
    """Return PSI, effective bucket count, and baseline cardinality."""
    if bins < 2 or bins > 100:
        raise ValueError("bins must be in [2, 100]")
    if len(baseline) < bins or not current:
        raise ValueError("baseline must fill every bin and current cannot be empty")
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
    return psi, sum(count > 0 for count in baseline_counts), len(set(baseline))


def feature_drift_diagnostics(
    baseline: Sequence[Event], current: Sequence[Event], *, bins: int = 10
) -> tuple[FeatureDrift, ...]:
    """Measure PSI for every fixed model feature using past-only bucket boundaries.

    Feature values use the same deterministic transforms as model fitting. Results
    retain ``FEATURE_NAMES`` order so consumers never infer a schema from PSI rank.
    PSI is a distribution-change diagnostic, not a cause, safety decision, or
    feature-importance measure.
    """
    if bins < 2 or bins > 100:
        raise ValueError("bins must be in [2, 100]")
    if len(baseline) < bins or not current:
        raise ValueError("baseline must fill every bin and current cannot be empty")
    baseline_rows = [event.features() for event in baseline]
    current_rows = [event.features() for event in current]
    diagnostics: list[FeatureDrift] = []
    for index, name in enumerate(FEATURE_NAMES):
        psi, effective_bins, unique_values = _population_stability_details(
            [row[index] for row in baseline_rows],
            [row[index] for row in current_rows],
            bins=bins,
        )
        diagnostics.append(
            FeatureDrift(
                feature=name,
                psi=psi,
                baseline_support=len(baseline_rows),
                current_support=len(current_rows),
                requested_bins=bins,
                effective_bins=effective_bins,
                baseline_unique_values=unique_values,
                degenerate_baseline=effective_bins < bins,
            )
        )
    return tuple(diagnostics)


def simulate_drift_policy(
    windows: Iterable[DriftWindow],
    *,
    warning_psi: float = 0.1,
    critical_psi: float = 0.25,
    minimum_support: int = 50,
    critical_windows_required: int = 2,
) -> list[DriftPolicyDecision]:
    """Simulate bounded responses to amount PSI without taking operational action.

    A critical candidate response requires consecutive sufficiently supported critical
    windows. Warning, healthy, and under-supported windows reset that streak. The
    output is a deterministic review aid, not evidence of model safety or a control
    that should be connected to financial decisions.
    """
    if not (
        math.isfinite(warning_psi)
        and math.isfinite(critical_psi)
        and 0.0 <= warning_psi < critical_psi
    ):
        raise ValueError(
            "PSI thresholds must be finite and satisfy 0 <= warning < critical"
        )
    if not 1 <= minimum_support <= 1_000_000:
        raise ValueError("minimum_support must be in [1, 1000000]")
    if not 1 <= critical_windows_required <= 10:
        raise ValueError("critical_windows_required must be in [1, 10]")

    decisions: list[DriftPolicyDecision] = []
    critical_streak = 0
    previous_end: int | None = None
    for window in windows:
        if window.start < 0 or window.end <= window.start:
            raise ValueError("window bounds must be non-negative with end after start")
        if previous_end is not None and window.start <= previous_end:
            raise ValueError("windows must be strictly ordered and non-overlapping")
        if not math.isfinite(window.amount_psi) or window.amount_psi < 0:
            raise ValueError("amount_psi must be finite and non-negative")
        if window.support < 0:
            raise ValueError("support must be non-negative")

        if window.support < minimum_support:
            critical_streak = 0
            response = "collect_more_evidence"
            reason = "insufficient_support"
        elif window.amount_psi >= critical_psi:
            critical_streak += 1
            if critical_streak >= critical_windows_required:
                response = "candidate_pause_and_human_review"
                reason = "sustained_critical_amount_psi"
            else:
                response = "investigate"
                reason = "first_critical_amount_psi"
        elif window.amount_psi >= warning_psi:
            critical_streak = 0
            response = "investigate"
            reason = "warning_amount_psi"
        else:
            critical_streak = 0
            response = "monitor"
            reason = "below_warning_amount_psi"

        decisions.append(
            DriftPolicyDecision(
                window_start=window.start,
                window_end=window.end,
                amount_psi=window.amount_psi,
                support=window.support,
                consecutive_critical_windows=critical_streak,
                candidate_response=response,
                reason=reason,
            )
        )
        previous_end = window.end
    return decisions


def simulate_feature_drift_policy(
    windows: Iterable[FeatureDriftWindow],
    *,
    warning_psi: float = 0.1,
    critical_psi: float = 0.25,
    minimum_support: int = 50,
    minimum_effective_bins: int = 2,
    critical_windows_required: int = 2,
) -> list[FeatureDriftPolicyDecision]:
    """Simulate review candidates from all fixed model-feature diagnostics.

    A feature is eligible only when both samples meet the support gate and its
    baseline retains enough effective quantile buckets. The highest-PSI eligible feature
    deterministically drives each window. Critical escalation requires the same
    feature to remain critical across consecutive windows. This simulation never
    executes an operational action.
    """
    if not (
        math.isfinite(warning_psi)
        and math.isfinite(critical_psi)
        and 0.0 <= warning_psi < critical_psi
    ):
        raise ValueError(
            "PSI thresholds must be finite and satisfy 0 <= warning < critical"
        )
    if not 1 <= minimum_support <= 1_000_000:
        raise ValueError("minimum_support must be in [1, 1000000]")
    if not 2 <= minimum_effective_bins <= 100:
        raise ValueError("minimum_effective_bins must be in [2, 100]")
    if not 1 <= critical_windows_required <= 10:
        raise ValueError("critical_windows_required must be in [1, 10]")

    decisions: list[FeatureDriftPolicyDecision] = []
    previous_end: int | None = None
    critical_feature: str | None = None
    critical_streak = 0
    expected_features = set(FEATURE_NAMES)
    for window in windows:
        if window.start < 0 or window.end <= window.start:
            raise ValueError("window bounds must be non-negative with end after start")
        if previous_end is not None and window.start <= previous_end:
            raise ValueError("windows must be strictly ordered and non-overlapping")
        names = [diagnostic.feature for diagnostic in window.diagnostics]
        if len(names) != len(set(names)) or set(names) != expected_features:
            raise ValueError("diagnostics must contain each fixed model feature exactly once")
        if any(
            not math.isfinite(diagnostic.psi) or diagnostic.psi < 0
            for diagnostic in window.diagnostics
        ):
            raise ValueError("feature PSI must be finite and non-negative")
        if any(
            diagnostic.baseline_support < 0 or diagnostic.current_support < 0
            for diagnostic in window.diagnostics
        ):
            raise ValueError("feature support must be non-negative")

        eligible = tuple(
            diagnostic
            for diagnostic in window.diagnostics
            if diagnostic.effective_bins >= minimum_effective_bins
            and diagnostic.baseline_support >= minimum_support
            and diagnostic.current_support >= minimum_support
        )
        eligible_names = tuple(sorted(item.feature for item in eligible))
        excluded_names = tuple(sorted(expected_features - set(eligible_names)))
        if not eligible:
            trigger = None
            critical_feature = None
            critical_streak = 0
            response = "collect_more_evidence"
            reason = "no_supported_feature_with_sufficient_buckets"
        else:
            trigger = min(eligible, key=lambda item: (-item.psi, item.feature))
            if trigger.psi >= critical_psi:
                if trigger.feature == critical_feature:
                    critical_streak += 1
                else:
                    critical_feature = trigger.feature
                    critical_streak = 1
                if critical_streak >= critical_windows_required:
                    response = "candidate_pause_and_human_review"
                    reason = "sustained_critical_feature_psi"
                else:
                    response = "investigate"
                    reason = "first_critical_feature_psi"
            elif trigger.psi >= warning_psi:
                critical_feature = None
                critical_streak = 0
                response = "investigate"
                reason = "warning_feature_psi"
            else:
                critical_feature = None
                critical_streak = 0
                response = "monitor"
                reason = "below_warning_feature_psi"

        decisions.append(
            FeatureDriftPolicyDecision(
                window_start=window.start,
                window_end=window.end,
                trigger_feature=trigger.feature if trigger else None,
                trigger_psi=trigger.psi if trigger else None,
                eligible_features=eligible_names,
                excluded_features=excluded_names,
                consecutive_critical_windows=critical_streak,
                candidate_response=response,
                reason=reason,
            )
        )
        previous_end = window.end
    return decisions


def summarize_feature_drift_policy(
    decisions: Iterable[FeatureDriftPolicyDecision],
) -> dict[str, object]:
    """Summarize bounded simulated review volume without executing any response.

    Zero-count responses and fixed model features remain present so reports are
    comparable across runs. Rates use the number of monitoring windows as their
    denominator and are zero when no windows are supplied.
    """
    response_counts = {response: 0 for response in POLICY_RESPONSES}
    trigger_feature_counts = {feature: 0 for feature in FEATURE_NAMES}
    window_count = 0
    max_critical_streak = 0

    for decision in decisions:
        if decision.candidate_response not in response_counts:
            raise ValueError("candidate_response is not a supported policy response")
        if decision.trigger_feature is not None:
            if decision.trigger_feature not in trigger_feature_counts:
                raise ValueError("trigger_feature is not a fixed model feature")
            trigger_feature_counts[decision.trigger_feature] += 1
        if decision.consecutive_critical_windows < 0:
            raise ValueError("consecutive_critical_windows must be non-negative")
        response_counts[decision.candidate_response] += 1
        window_count += 1
        max_critical_streak = max(
            max_critical_streak, decision.consecutive_critical_windows
        )

    denominator = window_count or 1
    return {
        "window_count": window_count,
        "response_counts": response_counts,
        "response_rates": {
            response: count / denominator
            for response, count in response_counts.items()
        },
        "trigger_feature_counts": trigger_feature_counts,
        "trigger_feature_rates": {
            feature: count / denominator
            for feature, count in trigger_feature_counts.items()
        },
        "max_consecutive_critical_windows": max_critical_streak,
    }


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
        feature_drift = feature_drift_diagnostics(train, test)
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
                amount_psi=feature_drift[0].psi,
                feature_drift=feature_drift,
            )
        )
        test_start_index = test_end_index
    return results


def run(count: int, seed: int) -> dict[str, object]:
    events = generate_events(count, seed=seed)
    train, test = chronological_split(events)
    model = fit(train)
    metrics = evaluate(model, test)
    robust_model = fit(train, scaling="robust_mad")
    robust_metrics = evaluate(robust_model, test)
    historical_selection = fit_historical_threshold(train)
    historical_metrics = evaluate(historical_selection.model, test)
    feature_drift = feature_drift_diagnostics(train, test)
    amount_psi = feature_drift[0].psi
    backtest = rolling_backtest(events)
    drift_policy = simulate_feature_drift_policy(
        FeatureDriftWindow(
            start=fold.test_start,
            end=fold.test_end,
            diagnostics=fold.feature_drift,
        )
        for fold in backtest
    )
    drift_policy_summary = summarize_feature_drift_policy(drift_policy)
    segment_diagnostics = account_age_segment_diagnostics(model, test)
    example_event = max(test, key=model.score)
    example_explanation = model.explain(example_event)
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
        "feature_drift": [
            {
                "feature": diagnostic.feature,
                "psi": round(diagnostic.psi, 6),
                "baseline_support": diagnostic.baseline_support,
                "current_support": diagnostic.current_support,
                "requested_bins": diagnostic.requested_bins,
                "effective_bins": diagnostic.effective_bins,
                "baseline_unique_values": diagnostic.baseline_unique_values,
                "degenerate_baseline": diagnostic.degenerate_baseline,
            }
            for diagnostic in feature_drift
        ],
        "highest_score_explanation": {
            "occurred_at": example_event.occurred_at,
            "score": round(example_explanation.score, 6),
            "threshold": round(example_explanation.threshold, 6),
            "predicted_anomaly": example_explanation.predicted_anomaly,
            "feature_contributions": [
                {
                    "feature": item.feature,
                    "transformed_value": round(item.transformed_value, 6),
                    "training_center": round(item.center, 6),
                    "training_scale": round(item.scale, 6),
                    "standardized_deviation": round(
                        item.standardized_deviation, 6
                    ),
                    "score_contribution": round(item.contribution, 6),
                }
                for item in example_explanation.contributions
            ],
        },
        "scaling_comparison": [
            {
                "strategy": "standard_deviation",
                "threshold": round(model.threshold, 6),
                "precision": round(metrics.precision, 6),
                "recall": round(metrics.recall, 6),
                "f1": round(metrics.f1, 6),
            },
            {
                "strategy": "robust_mad",
                "threshold": round(robust_model.threshold, 6),
                "precision": round(robust_metrics.precision, 6),
                "recall": round(robust_metrics.recall, 6),
                "f1": round(robust_metrics.f1, 6),
            },
        ],
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
        "account_age_segment_diagnostics": [
            {
                "segment": diagnostic.segment,
                "support": diagnostic.support,
                "positive_labels": diagnostic.positive_labels,
                "predicted_positives": diagnostic.predicted_positives,
                "limited_support": diagnostic.limited_support,
                "precision": round(diagnostic.evaluation.precision, 6),
                "recall": round(diagnostic.evaluation.recall, 6),
                "f1": round(diagnostic.evaluation.f1, 6),
            }
            for diagnostic in segment_diagnostics
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
                "feature_drift": [
                    {
                        "feature": diagnostic.feature,
                        "psi": round(diagnostic.psi, 6),
                        "baseline_support": diagnostic.baseline_support,
                        "current_support": diagnostic.current_support,
                        "requested_bins": diagnostic.requested_bins,
                        "effective_bins": diagnostic.effective_bins,
                        "baseline_unique_values": diagnostic.baseline_unique_values,
                        "degenerate_baseline": diagnostic.degenerate_baseline,
                    }
                    for diagnostic in fold.feature_drift
                ],
            }
            for fold in backtest
        ],
        "drift_policy_simulation": {
            "warning_psi": 0.1,
            "critical_psi": 0.25,
            "minimum_support": 50,
            "minimum_effective_bins": 2,
            "critical_windows_required": 2,
            "executes_actions": False,
            "review_volume_summary": drift_policy_summary,
            "decisions": [
                {
                    "window_start": decision.window_start,
                    "window_end": decision.window_end,
                    "trigger_feature": decision.trigger_feature,
                    "trigger_psi": (
                        round(decision.trigger_psi, 6)
                        if decision.trigger_psi is not None
                        else None
                    ),
                    "eligible_features": list(decision.eligible_features),
                    "excluded_features": list(decision.excluded_features),
                    "consecutive_critical_windows": decision.consecutive_critical_windows,
                    "candidate_response": decision.candidate_response,
                    "reason": decision.reason,
                }
                for decision in drift_policy
            ],
        },
        "limitations": [
            "statistical baseline, not a production fraud decision system",
            "labels and events are generated, not representative of real customers",
            "scores require human and domain validation before any operational use",
            "historical-label threshold selection optimizes synthetic F1, not real-world cost",
            "account-age cohorts are descriptive diagnostics, not fairness or policy evidence",
            "scaling comparisons use synthetic labels and do not establish production robustness",
            "score contributions are mathematical decompositions, not causal explanations",
            "drift policy output is a simulation for human review and executes no actions",
            "per-feature PSI policy excludes degenerate or under-supported diagnostics and executes no actions",
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
