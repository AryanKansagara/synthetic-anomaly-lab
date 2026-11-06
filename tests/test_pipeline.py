import dataclasses
import math
import unittest

from synthetic_anomaly_lab.pipeline import (
    Event,
    chronological_split,
    evaluate,
    fit,
    fit_historical_threshold,
    generate_events,
    population_stability_index,
    rolling_backtest,
    run,
)


class PipelineTests(unittest.TestCase):
    def test_generator_is_deterministic_and_synthetic(self) -> None:
        self.assertEqual(generate_events(100, seed=11), generate_events(100, seed=11))
        self.assertNotEqual(generate_events(100, seed=11), generate_events(100, seed=12))
        self.assertEqual(run(100, 11)["data"], "synthetic-only")

    def test_chronological_split_sorts_and_has_no_overlap(self) -> None:
        events = list(reversed(generate_events(100)))
        train, test = chronological_split(events)
        self.assertEqual((len(train), len(test)), (70, 30))
        self.assertLess(train[-1].occurred_at, test[0].occurred_at)
        self.assertTrue(all(a.occurred_at < b.occurred_at for a, b in zip(train, train[1:])))

    def test_future_mutation_cannot_change_fitted_model(self) -> None:
        train, test = chronological_split(generate_events(500))
        original_model = fit(train)
        changed_test = [
            dataclasses.replace(event, amount_minor=event.amount_minor * 10_000)
            for event in test
        ]
        self.assertEqual(original_model, fit(train))
        self.assertNotEqual(
            evaluate(original_model, test), evaluate(original_model, changed_test)
        )

    def test_seeded_baseline_finds_injected_anomalies(self) -> None:
        train, test = chronological_split(generate_events(2_000, seed=7))
        metrics = evaluate(fit(train), test)
        self.assertGreaterEqual(metrics.precision, 0.70)
        self.assertGreaterEqual(metrics.recall, 0.70)
        self.assertGreaterEqual(metrics.f1, 0.70)

    def test_historical_threshold_uses_only_past_calibration_labels(self) -> None:
        train, future = chronological_split(generate_events(2_000, seed=7))
        selection = fit_historical_threshold(train)
        mutated_future = [
            dataclasses.replace(event, is_anomaly=not event.is_anomaly)
            for event in future
        ]
        self.assertEqual(selection, fit_historical_threshold(train))
        self.assertNotEqual(
            evaluate(selection.model, future),
            evaluate(selection.model, mutated_future),
        )
        self.assertLess(selection.calibration_end, future[0].occurred_at)
        self.assertEqual(selection.fit_events + selection.calibration_events, len(train))
        self.assertEqual(selection.strategy, "historical_label_f1")
        labels_removed_from_calibration = [
            dataclasses.replace(event, is_anomaly=False)
            if event.occurred_at >= selection.calibration_start
            else event
            for event in train
        ]
        self.assertEqual(
            fit_historical_threshold(labels_removed_from_calibration).strategy,
            "unsupervised_quantile_fallback_no_positive_labels",
        )

    def test_historical_threshold_falls_back_without_calibration_positives(self) -> None:
        events = [
            dataclasses.replace(event, is_anomaly=False)
            for event in generate_events(100, seed=3)
        ]
        selection = fit_historical_threshold(events)
        self.assertEqual(
            selection.strategy, "unsupervised_quantile_fallback_no_positive_labels"
        )
        self.assertEqual(selection.calibration_evaluation.true_positives, 0)

    def test_population_stability_index(self) -> None:
        baseline = [float(value) for value in range(100)]
        self.assertAlmostEqual(
            population_stability_index(baseline, list(baseline)), 0.0
        )
        shifted = [value + 100.0 for value in baseline]
        self.assertGreater(population_stability_index(baseline, shifted), 1.0)

    def test_rolling_backtest_uses_ordered_non_overlapping_future_windows(self) -> None:
        events = list(reversed(generate_events(200, seed=13)))
        folds = rolling_backtest(events)
        self.assertEqual(len(folds), 4)
        self.assertEqual([fold.train_events for fold in folds], [100, 125, 150, 175])
        self.assertEqual([fold.test_events for fold in folds], [25, 25, 25, 25])
        self.assertTrue(all(fold.train_end < fold.test_start for fold in folds))
        self.assertTrue(
            all(left.test_end < right.test_start for left, right in zip(folds, folds[1:]))
        )
        self.assertEqual(folds[-1].test_end, max(event.occurred_at for event in events))

    def test_later_future_mutation_cannot_change_earlier_backtest_fold(self) -> None:
        events = generate_events(200, seed=17)
        original = rolling_backtest(events)
        changed = [
            dataclasses.replace(event, amount_minor=event.amount_minor * 10_000)
            if index >= 150
            else event
            for index, event in enumerate(events)
        ]
        mutated = rolling_backtest(changed)
        self.assertEqual(original[0], mutated[0])
        self.assertNotEqual(original[-1], mutated[-1])

    def test_validation_rejects_unsafe_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 20"):
            generate_events(19)
        with self.assertRaisesRegex(ValueError, "train_fraction"):
            chronological_split(generate_events(20), train_fraction=1.0)
        duplicate = Event(1, 1, 1, 1, False)
        with self.assertRaisesRegex(ValueError, "unique"):
            chronological_split([duplicate] * 4)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            Event(1, -1, 1, 1, False).features()
        with self.assertRaisesRegex(ValueError, "finite"):
            population_stability_index(
                [float(value) for value in range(10)], [math.inf]
            )
        with self.assertRaisesRegex(ValueError, "at least 40"):
            rolling_backtest(generate_events(20))
        with self.assertRaisesRegex(ValueError, "folds"):
            rolling_backtest(generate_events(100), folds=1)
        with self.assertRaisesRegex(ValueError, "at least 20 historical"):
            fit_historical_threshold(generate_events(20)[:19])
        with self.assertRaisesRegex(ValueError, "calibration_fraction"):
            fit_historical_threshold(generate_events(100), calibration_fraction=0.5)

    def test_report_is_bounded_and_explains_limitations(self) -> None:
        report = run(500, 9)
        self.assertEqual(report["train_events"], 350)
        self.assertEqual(report["test_events"], 150)
        limitations = report["limitations"]
        precision = report["precision"]
        recall = report["recall"]
        self.assertIsInstance(limitations, list)
        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)
        assert isinstance(limitations, list)
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        self.assertEqual(len(limitations), 4)
        self.assertTrue(0.0 <= precision <= 1.0)
        self.assertTrue(0.0 <= recall <= 1.0)
        backtest = report["rolling_backtest"]
        self.assertIsInstance(backtest, list)
        assert isinstance(backtest, list)
        self.assertEqual(len(backtest), 4)
        self.assertEqual(sum(fold["test_events"] for fold in backtest), 250)
        self.assertTrue(
            all(fold["train_end"] < fold["test_start"] for fold in backtest)
        )
        comparison = report["threshold_comparison"]
        self.assertIsInstance(comparison, list)
        assert isinstance(comparison, list)
        self.assertEqual(len(comparison), 2)
        self.assertLess(comparison[1]["calibration_end"], backtest[-1]["test_end"])


if __name__ == "__main__":
    unittest.main()
