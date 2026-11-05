# Synthetic Anomaly Lab

A small, reproducible ML baseline for anomaly detection on **generated transaction-like events only**. It demonstrates the parts that are easy to get wrong in an anomaly prototype: event-time splitting, train-only preprocessing and threshold selection, deterministic evaluation, and feature-drift measurement.

This independent portfolio project is not affiliated with Wealthsimple or any financial institution. It processes no real money or personal data and provides no financial, fraud, credit, or investment advice.

## What the first slice does

- generates deterministic labeled events with amount, account-age, and one-hour velocity features;
- uses a chronological 70/30 split so future observations never train the past;
- fits log transforms, robust centers, scales, and a score threshold on the training partition only;
- reports holdout precision, recall, F1, and amount-feature population stability index (PSI);
- provides unit coverage for reproducibility, leakage boundaries, model utility, drift, and validation;
- runs dependency-free on Python 3.11+ with no dataset download or network call.

## Run

```bash
python -m synthetic_anomaly_lab.pipeline --count 2000 --seed 7
python -W error::ResourceWarning -m unittest discover -s tests -v
```

Example output is JSON and explicitly marks the data as `synthetic-only`. Exact metrics are deterministic for a given seed.

## Design boundaries

This is an educational statistical baseline, not a production fraud decision system. Synthetic labels are intentionally obvious and are not evidence of performance on real customers. The model has no identity, authorization, case-management, feedback, fairness, calibration, explainability, or human-review workflow. PSI indicates distribution change; it does not explain the cause or determine whether a model remains safe. A production system would require representative consented data, domain-specific cost analysis, privacy/security review, monitored deployment, governance, and human escalation.

The standard-deviation scale is learned from contaminated training data, and the quantile threshold assumes a configured normal fraction. Later slices can compare robust scaling and supervised baselines, add rolling backtests, and report segment-level behavior without weakening the chronological boundary.
