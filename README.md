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
src/moa/
  config.py
  data.py
  metrics.py
  train_baseline.py

reports/
  experiment_log.md
  baseline_top206.csv
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

* Save trained model artifacts
* Generate Kaggle-style submission files
* Add tree-based baselines
* Add PyTorch MLP baseline
* Improve rare-label validation
* Add FastAPI inference endpoint
