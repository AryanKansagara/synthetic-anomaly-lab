import dataclasses
import math
import unittest

from synthetic_anomaly_lab.pipeline import (
    Event,
    chronological_split,
    evaluate,
    fit,
    generate_events,
    population_stability_index,
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

    def test_population_stability_index(self) -> None:
        baseline = [float(value) for value in range(100)]
        self.assertAlmostEqual(
            population_stability_index(baseline, list(baseline)), 0.0
        )
        shifted = [value + 100.0 for value in baseline]
        self.assertGreater(population_stability_index(baseline, shifted), 1.0)

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
        self.assertEqual(len(limitations), 3)
        self.assertTrue(0.0 <= precision <= 1.0)
        self.assertTrue(0.0 <= recall <= 1.0)


if __name__ == "__main__":
    unittest.main()
