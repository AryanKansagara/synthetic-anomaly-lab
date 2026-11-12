import dataclasses
import math
import unittest

from synthetic_anomaly_lab.pipeline import (
    AnomalyModel,
    DriftWindow,
    Event,
    FeatureDrift,
    FeatureDriftWindow,
    account_age_segment_diagnostics,
    chronological_split,
    evaluate,
    feature_drift_diagnostics,
    fit,
    fit_historical_threshold,
    generate_events,
    population_stability_index,
    rolling_backtest,
    run,
    simulate_drift_policy,
    simulate_feature_drift_policy,
    summarize_feature_drift_policy,
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

    def test_score_explanation_is_exact_ranked_and_uses_fitted_parameters(self) -> None:
        train, future = chronological_split(generate_events(500, seed=7))
        model = fit(train)
        event = future[0]
        explanation = model.explain(event)

        self.assertAlmostEqual(
            sum(item.contribution for item in explanation.contributions),
            model.score(event),
        )
        self.assertEqual(explanation.predicted_anomaly, model.predict(event))
        self.assertEqual(
            [item.contribution for item in explanation.contributions],
            sorted(
                (item.contribution for item in explanation.contributions), reverse=True
            ),
        )
        self.assertEqual(
            {item.center for item in explanation.contributions}, set(model.centers)
        )
        self.assertEqual(
            {item.scale for item in explanation.contributions}, set(model.scales)
        )

    def test_score_explanation_rejects_invalid_model_dimensions_and_scales(self) -> None:
        event = generate_events(20)[0]
        with self.assertRaisesRegex(ValueError, "feature count"):
            AnomalyModel((0.0,), (1.0,), 1.0).explain(event)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            AnomalyModel((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), 1.0).explain(event)

    def test_robust_scaling_is_train_only_and_less_outlier_sensitive(self) -> None:
        train, future = chronological_split(generate_events(500, seed=7))
        standard = fit(train, scaling="standard_deviation")
        robust = fit(train, scaling="robust_mad")
        changed_future = [
            dataclasses.replace(event, amount_minor=event.amount_minor * 1_000_000)
            for event in future
        ]

        self.assertEqual(robust, fit(train, scaling="robust_mad"))
        self.assertEqual(robust.centers, standard.centers)
        self.assertNotEqual(robust.scales, standard.scales)
        self.assertNotEqual(evaluate(robust, future), evaluate(robust, changed_future))

        contaminated = [
            *train,
            dataclasses.replace(
                train[-1],
                occurred_at=train[-1].occurred_at + 1,
                amount_minor=10**18,
            ),
        ]
        contaminated_standard = fit(contaminated, scaling="standard_deviation")
        contaminated_robust = fit(contaminated, scaling="robust_mad")
        self.assertGreater(
            contaminated_standard.scales[0] / standard.scales[0],
            contaminated_robust.scales[0] / robust.scales[0],
        )

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

    def test_feature_drift_covers_fixed_schema_and_uses_past_boundaries(self) -> None:
        baseline, current = chronological_split(generate_events(500, seed=19))
        diagnostics = feature_drift_diagnostics(baseline, current)
        shifted_velocity = [
            dataclasses.replace(event, velocity_1h=event.velocity_1h + 1_000)
            for event in current
        ]
        shifted = feature_drift_diagnostics(baseline, shifted_velocity)

        self.assertEqual(
            [item.feature for item in diagnostics],
            ["log_amount_minor", "log_account_age_days", "log_velocity_1h"],
        )
        self.assertTrue(all(item.baseline_support == 350 for item in diagnostics))
        self.assertTrue(all(item.current_support == 150 for item in diagnostics))
        self.assertEqual(diagnostics[:2], shifted[:2])
        self.assertGreater(shifted[2].psi, diagnostics[2].psi)

    def test_feature_drift_validates_support_and_events(self) -> None:
        events = generate_events(20)
        with self.assertRaisesRegex(ValueError, "fill every bin"):
            feature_drift_diagnostics(events[:9], events[9:])
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            feature_drift_diagnostics(events[:10], [])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            feature_drift_diagnostics(events[:10], [Event(1, 1, 1, -1, False)])

    def test_feature_drift_exposes_collapsed_quantile_buckets(self) -> None:
        baseline = [Event(index, 100, 30, 2, False) for index in range(20)]
        current = [Event(100 + index, 100, 30, 2, False) for index in range(10)]

        diagnostics = feature_drift_diagnostics(baseline, current)

        self.assertTrue(all(item.psi == 0.0 for item in diagnostics))
        self.assertTrue(all(item.requested_bins == 10 for item in diagnostics))
        self.assertTrue(all(item.effective_bins == 1 for item in diagnostics))
        self.assertTrue(all(item.baseline_unique_values == 1 for item in diagnostics))
        self.assertTrue(all(item.degenerate_baseline for item in diagnostics))

    def test_segment_diagnostics_are_bounded_complete_and_support_aware(self) -> None:
        train, test = chronological_split(generate_events(2_000, seed=7))
        model = fit(train)
        overall = evaluate(model, test)
        diagnostics = account_age_segment_diagnostics(model, test)

        self.assertEqual(
            [diagnostic.segment for diagnostic in diagnostics],
            [
                "new_0_30_days",
                "growing_31_365_days",
                "established_366_plus_days",
            ],
        )
        self.assertEqual(sum(item.support for item in diagnostics), len(test))
        self.assertEqual(
            sum(item.evaluation.true_positives for item in diagnostics),
            overall.true_positives,
        )
        self.assertEqual(
            sum(item.evaluation.false_positives for item in diagnostics),
            overall.false_positives,
        )
        self.assertTrue(any(item.limited_support for item in diagnostics))
        self.assertTrue(any(not item.limited_support for item in diagnostics))

    def test_segment_diagnostics_reject_invalid_support_and_events(self) -> None:
        model = fit(generate_events(100))
        with self.assertRaisesRegex(ValueError, "minimum_support"):
            account_age_segment_diagnostics(model, [], minimum_support=0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            account_age_segment_diagnostics(model, [Event(1, 1, -1, 1, False)])

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

    def test_drift_policy_requires_supported_consecutive_critical_windows(self) -> None:
        decisions = simulate_drift_policy(
            [
                DriftWindow(1, 2, 0.05, 100),
                DriftWindow(3, 4, 0.30, 100),
                DriftWindow(5, 6, 0.40, 100),
                DriftWindow(7, 8, 0.90, 10),
                DriftWindow(9, 10, 0.15, 100),
            ]
        )
        self.assertEqual(
            [decision.candidate_response for decision in decisions],
            [
                "monitor",
                "investigate",
                "candidate_pause_and_human_review",
                "collect_more_evidence",
                "investigate",
            ],
        )
        self.assertEqual(
            [decision.consecutive_critical_windows for decision in decisions],
            [0, 1, 2, 0, 0],
        )

    def test_drift_policy_validates_configuration_and_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "warning < critical"):
            simulate_drift_policy([], warning_psi=0.25, critical_psi=0.25)
        with self.assertRaisesRegex(ValueError, "strictly ordered"):
            simulate_drift_policy(
                [DriftWindow(1, 3, 0.0, 50), DriftWindow(3, 4, 0.0, 50)]
            )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            simulate_drift_policy([DriftWindow(1, 2, math.inf, 50)])

    def test_feature_drift_policy_gates_and_tracks_the_same_trigger(self) -> None:
        def diagnostic(
            feature: str, psi: float, *, support: int = 100, degenerate: bool = False
        ) -> FeatureDrift:
            return FeatureDrift(
                feature=feature,
                psi=psi,
                baseline_support=support,
                current_support=support,
                requested_bins=10,
                effective_bins=1 if degenerate else 10,
                baseline_unique_values=20,
                degenerate_baseline=degenerate,
            )

        decisions = simulate_feature_drift_policy(
            [
                FeatureDriftWindow(
                    1,
                    2,
                    (
                        diagnostic("log_amount_minor", 0.30),
                        diagnostic("log_account_age_days", 0.05),
                        diagnostic("log_velocity_1h", 0.90, degenerate=True),
                    ),
                ),
                FeatureDriftWindow(
                    3,
                    4,
                    (
                        diagnostic("log_amount_minor", 0.40),
                        diagnostic("log_account_age_days", 0.35),
                        diagnostic("log_velocity_1h", 0.05),
                    ),
                ),
                FeatureDriftWindow(
                    5,
                    6,
                    (
                        diagnostic("log_amount_minor", 0.30, support=10),
                        diagnostic("log_account_age_days", 0.40, support=10),
                        diagnostic("log_velocity_1h", 0.50, support=10),
                    ),
                ),
            ]
        )

        self.assertEqual(
            [decision.candidate_response for decision in decisions],
            [
                "investigate",
                "candidate_pause_and_human_review",
                "collect_more_evidence",
            ],
        )
        self.assertEqual(decisions[0].trigger_feature, "log_amount_minor")
        self.assertEqual(decisions[0].excluded_features, ("log_velocity_1h",))
        self.assertEqual(decisions[1].consecutive_critical_windows, 2)
        self.assertIsNone(decisions[2].trigger_feature)

    def test_feature_drift_policy_validates_fixed_schema_and_order(self) -> None:
        diagnostic = FeatureDrift("log_amount_minor", 0.1, 100, 100, 10, 10, 100, False)
        with self.assertRaisesRegex(ValueError, "each fixed model feature"):
            simulate_feature_drift_policy([FeatureDriftWindow(1, 2, (diagnostic,))])
        with self.assertRaisesRegex(ValueError, "minimum_effective_bins"):
            simulate_feature_drift_policy([], minimum_effective_bins=1)
        complete = tuple(
            dataclasses.replace(diagnostic, feature=feature)
            for feature in (
                "log_amount_minor",
                "log_account_age_days",
                "log_velocity_1h",
            )
        )
        with self.assertRaisesRegex(ValueError, "strictly ordered"):
            simulate_feature_drift_policy(
                [FeatureDriftWindow(1, 3, complete), FeatureDriftWindow(3, 4, complete)]
            )

    def test_feature_drift_policy_summary_is_bounded_and_zero_complete(self) -> None:
        report = run(500, 9)
        policy = report["drift_policy_simulation"]
        self.assertIsInstance(policy, dict)
        assert isinstance(policy, dict)
        summary = policy["review_volume_summary"]
        self.assertIsInstance(summary, dict)
        assert isinstance(summary, dict)
        response_counts = summary["response_counts"]
        response_rates = summary["response_rates"]
        trigger_counts = summary["trigger_feature_counts"]
        self.assertIsInstance(response_counts, dict)
        self.assertIsInstance(response_rates, dict)
        self.assertIsInstance(trigger_counts, dict)
        assert isinstance(response_counts, dict)
        assert isinstance(response_rates, dict)
        assert isinstance(trigger_counts, dict)

        self.assertEqual(summary["window_count"], 4)
        self.assertEqual(
            set(response_counts),
            {
                "monitor",
                "investigate",
                "collect_more_evidence",
                "candidate_pause_and_human_review",
            },
        )
        self.assertEqual(sum(response_counts.values()), 4)
        self.assertAlmostEqual(sum(response_rates.values()), 1.0)
        self.assertEqual(
            set(trigger_counts),
            {"log_amount_minor", "log_account_age_days", "log_velocity_1h"},
        )
        self.assertLessEqual(sum(trigger_counts.values()), 4)

        empty = summarize_feature_drift_policy([])
        empty_response_counts = empty["response_counts"]
        empty_response_rates = empty["response_rates"]
        empty_trigger_counts = empty["trigger_feature_counts"]
        empty_trigger_rates = empty["trigger_feature_rates"]
        assert isinstance(empty_response_counts, dict)
        assert isinstance(empty_response_rates, dict)
        assert isinstance(empty_trigger_counts, dict)
        assert isinstance(empty_trigger_rates, dict)
        self.assertEqual(empty["window_count"], 0)
        self.assertEqual(set(empty_response_counts.values()), {0})
        self.assertEqual(set(empty_response_rates.values()), {0.0})
        self.assertEqual(set(empty_trigger_counts.values()), {0})
        self.assertEqual(set(empty_trigger_rates.values()), {0.0})

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
        with self.assertRaisesRegex(ValueError, "scaling"):
            fit(generate_events(100), scaling="future_normalizer")

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
        self.assertEqual(len(limitations), 9)
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
        diagnostics = report["account_age_segment_diagnostics"]
        self.assertIsInstance(diagnostics, list)
        assert isinstance(diagnostics, list)
        self.assertEqual(len(diagnostics), 3)
        self.assertEqual(sum(item["support"] for item in diagnostics), 150)
        self.assertTrue(all("limited_support" in item for item in diagnostics))
        scaling = report["scaling_comparison"]
        self.assertIsInstance(scaling, list)
        assert isinstance(scaling, list)
        self.assertEqual(
            [item["strategy"] for item in scaling],
            ["standard_deviation", "robust_mad"],
        )
        self.assertTrue(all(0.0 <= item["f1"] <= 1.0 for item in scaling))
        explanation = report["highest_score_explanation"]
        self.assertIsInstance(explanation, dict)
        assert isinstance(explanation, dict)
        contributions = explanation["feature_contributions"]
        self.assertIsInstance(contributions, list)
        assert isinstance(contributions, list)
        self.assertEqual(len(contributions), 3)
        self.assertEqual(
            [item["feature"] for item in contributions],
            [
                item["feature"]
                for item in sorted(
                    contributions,
                    key=lambda item: (-item["score_contribution"], item["feature"]),
                )
            ],
        )
        feature_drift = report["feature_drift"]
        self.assertIsInstance(feature_drift, list)
        assert isinstance(feature_drift, list)
        self.assertEqual(
            [item["feature"] for item in feature_drift],
            ["log_amount_minor", "log_account_age_days", "log_velocity_1h"],
        )
        self.assertTrue(
            all(item["baseline_support"] == 350 for item in feature_drift)
        )
        self.assertTrue(all(item["current_support"] == 150 for item in feature_drift))
        self.assertTrue(all(item["requested_bins"] == 10 for item in feature_drift))
        self.assertTrue(
            all(
                1 <= item["effective_bins"] <= item["requested_bins"]
                for item in feature_drift
            )
        )
        self.assertTrue(
            all(item["baseline_unique_values"] >= 1 for item in feature_drift)
        )
        self.assertTrue(all("degenerate_baseline" in item for item in feature_drift))
        self.assertTrue(all(len(fold["feature_drift"]) == 3 for fold in backtest))
        policy = report["drift_policy_simulation"]
        self.assertIsInstance(policy, dict)
        assert isinstance(policy, dict)
        self.assertFalse(policy["executes_actions"])
        self.assertIn("review_volume_summary", policy)
        decisions = policy["decisions"]
        self.assertIsInstance(decisions, list)
        assert isinstance(decisions, list)
        self.assertEqual(len(decisions), 4)
        self.assertTrue(
            all(
                "trigger_feature" in item and "eligible_features" in item
                for item in decisions
            )
        )
        self.assertTrue(
            all(item["window_start"] < item["window_end"] for item in decisions)
        )


if __name__ == "__main__":
    unittest.main()
