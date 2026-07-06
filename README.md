# MoA Prediction ML Pipeline

Reproducible machine learning pipeline for predicting drug mechanisms of action from high-dimensional cellular response data.

This project uses the Kaggle **Mechanisms of Action (MoA) Prediction** dataset, where each sample contains compound treatment metadata, gene expression features, and cell viability features. The task is multi-label classification across 206 scored MoA targets.

## Current Status

Implemented a classical ML baseline with:

* Reusable data loading utilities
* Feature grouping for metadata, gene expression, and cell viability columns
* Multi-label target construction
* Scikit-learn preprocessing pipeline
* One-vs-Rest Logistic Regression baseline
* Mean multi-label log loss evaluation
* Per-target validation error report
* Unit tests for data and metric utilities
* GitHub Actions CI for automated test execution

## Dataset

Input features:

```text
cp_type, cp_time, cp_dose
g-0 ... g-771
c-0 ... c-99
```

Targets:

```text
206 binary MoA labels
```

Data files are not committed to the repository. They should be downloaded from Kaggle and placed under:

```text
data/raw/
```

## Project Structure

```text
.github/
  workflows/
    ci.yml

src/moa/
  config.py
  data.py
  metrics.py
  train_baseline.py

tests/
  test_data.py
  test_metrics.py

reports/
  experiment_log.md
  baseline_top206.csv

requirements.txt
requirements-dev.txt
```

## Baseline

Model:

```text
StandardScaler + OneHotEncoder
→ OneVsRestClassifier(LogisticRegression)
```

Run:

```bash
PYTHONPATH=src python -m moa.train_baseline
```

## Testing and CI

This repository includes unit tests for the core data and metric utilities.

Run the full test suite locally:

```bash
PYTHONPATH=src python -m pytest tests -q
```

Current test coverage includes:

* Feature group extraction for metadata, gene-expression, and cell-viability columns
* Multi-label target construction from the most frequent scored targets
* Sample alignment validation through `sig_id`
* Mean multi-label log loss calculation
* Per-target result table generation

GitHub Actions runs the same test suite automatically on pull requests to `main`.

The CI workflow uses small synthetic test data, so it does not require Kaggle credentials or raw competition files.

## Results

Top 20 frequent targets:

```text
Mean multi-label log loss: 0.231601
```

All 206 scored targets:

```text
Mean multi-label log loss: 0.056092
```

The full 206-target score is not directly comparable to the top-20 score because many rare targets are mostly zero.

## Next Steps

* Add a batch prediction script for `test_features.csv`
* Generate Kaggle-style submission files
* Add a dummy prior baseline for comparison
* Improve validation for rare labels and one-class splits
* Add tree-based baselines
* Add a PyTorch MLP baseline
* Add a FastAPI inference endpoint