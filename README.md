# Synthetic Anomaly Lab

A small, reproducible ML baseline for anomaly detection on **generated transaction-like events only**. It demonstrates the parts that are easy to get wrong in an anomaly prototype: event-time splitting, train-only preprocessing and threshold selection, deterministic evaluation, and feature-drift measurement.

This independent portfolio project is not affiliated with Wealthsimple or any financial institution. It processes no real money or personal data and provides no financial, fraud, credit, or investment advice.

## What it does

- generates deterministic labeled events with amount, account-age, and one-hour velocity features;
- uses a chronological 70/30 split so future observations never train the past;
- fits log transforms, median centers, scales, and a score threshold on the training partition only;
- compares standard-deviation scaling with normalized median absolute deviation (MAD)
  scaling on the same untouched holdout, with both variants fitted on past data only;
- reports holdout precision, recall, F1, and amount-feature population stability index (PSI);
- decomposes the highest holdout score into exact, ranked per-feature contributions
  using only the fitted training centers and scales;
- reports the same metrics with support counts for three fixed account-age cohorts,
  explicitly flagging cohorts below the minimum diagnostic sample size;
- compares the training-quantile threshold with an F1-selected threshold whose
  labels come only from a chronological historical calibration window;
- runs four expanding-window temporal backtests, refitting on past-only history before
  each untouched future window and reporting per-window metrics and PSI;
- simulates a bounded amount-PSI response policy that requires two consecutive,
  sufficiently supported critical windows before proposing a pause and human review;
- provides unit coverage for reproducibility, leakage boundaries, model utility, drift, and validation;
- runs dependency-free on Python 3.11+ with no dataset download or network call.

## Run

```bash
python -m synthetic_anomaly_lab.pipeline --count 2000 --seed 7
python -W error::ResourceWarning -m unittest discover -s tests -v
```

Example output is JSON and explicitly marks the data as `synthetic-only`. Exact metrics are deterministic for a given seed.

## Design boundaries

This is an educational statistical baseline, not a production fraud decision system. Synthetic labels are intentionally obvious and are not evidence of performance on real customers. The model has no identity, authorization, case-management, feedback, fairness, probability calibration, causal explainability, or human-review workflow. The additive feature contributions exactly decompose this model's squared standardized-distance score; they are not causal reasons, counterfactuals, probabilities, SHAP values, or evidence that a transaction is fraudulent. PSI indicates distribution change; it does not explain the cause or determine whether a model remains safe. The drift policy is a deterministic simulation: it executes no pause, transaction, model, or customer action, and its generic thresholds have not been validated for real data. Account-age cohort metrics are descriptive debugging aids, not fairness evidence or a basis for authorization, credit, fraud, or customer-treatment policy; small cohorts are explicitly flagged and zero-valued rates can reflect absent labels or predictions. A production system would require representative consented data, domain-specific cost analysis, privacy/security review, monitored deployment, governance, and human escalation.

Standard-deviation scaling is sensitive to contaminated training data; normalized MAD is more resistant to extreme values but can collapse for low-variance features and therefore uses a small numerical floor. Neither strategy establishes robustness on real data, and both quantile thresholds assume a configured normal fraction. The historical-label strategy maximizes F1 on generated labels; it is not probability calibration and does not encode real investigation costs. It falls back to the unsupervised quantile when its calibration window has no positive label. Rolling folds are not independent because their training histories overlap, and good synthetic backtest metrics do not establish production performance. Later slices can compare supervised baselines without weakening the chronological boundary.
