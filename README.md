# Mechanism of Action Prediction

A reproducible multi-label machine learning pipeline for predicting biological mechanisms of action from high-dimensional cellular response data.

The project focuses on rigorous evaluation: strong baselines, multilabel-stratified out-of-fold validation, domain-aware prediction rules, and target-level diagnostics for highly imbalanced labels.

## Highlights

- 206-target multi-label classification using gene-expression, cell-viability, and treatment metadata
- 5-fold multilabel-stratified out-of-fold (OOF) evaluation
- Feature-free target-prevalence baseline and One-vs-Rest Logistic Regression
- Domain-aware handling of vehicle-control samples
- Reusable `src/`-layout Python package
- Unit-tested data, validation, modeling, prediction, and evaluation utilities
- GitHub Actions CI with reproducible package installation
- Experiment outputs separated from model implementation

## Current Benchmark

All results below use 5-fold OOF evaluation across the 206 scored targets.

| Model | Control rule | OOF mean log loss |
|---|---:|---:|
| Target prevalence prior | Yes | **0.020481** |
| Logistic Regression (`C=0.01`) | Yes | 0.020741 |
| Target prevalence prior | No | 0.020751 |
| Logistic Regression (`C=0.01`) | No | 0.022466 |

Lower is better.

The prevalence baseline currently remains slightly stronger than Logistic Regression.

This motivates the current analysis: determine which targets benefit from feature-based learning, which targets degrade, and whether performance differences are concentrated among rare labels.

## Problem

Each sample contains treatment metadata together with high-dimensional cellular response measurements:

```text
Treatment metadata:
cp_type, cp_time, cp_dose

Gene-expression features:
g-0 ... g-771

Cell-viability features:
c-0 ... c-99
```

The prediction target consists of:

```text
206 binary mechanism-of-action labels
```

The task is highly imbalanced, with many mechanisms occurring in only a small fraction of samples. This makes target prevalence a strong baseline and makes probability quality particularly important under log-loss evaluation.

## Evaluation Strategy

The project uses multilabel-stratified cross-validation to preserve label distributions across folds.

For each model:

1. training occurs only on the training portion of each fold
2. predictions are generated for the held-out fold
3. held-out predictions are combined into one OOF prediction matrix
4. mean multi-label log loss is computed across all samples and targets

This provides a consistent evaluation framework for comparing increasingly complex models.

Vehicle-control samples are also evaluated with a domain-aware rule that sets predicted mechanism probabilities to zero.

## Current Model

The current feature-based baseline uses:

```text
Gene expression + cell viability
            ↓
      StandardScaler

Treatment metadata
            ↓
       OneHotEncoder

            ↓
One-vs-Rest Logistic Regression
            ↓
   206 target probabilities
```

The model is intentionally simple so that changes in performance can be attributed to specific modeling and validation decisions.

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   └── moa/
│       ├── config.py
│       ├── data.py
│       ├── metrics.py
│       ├── modeling.py
│       ├── oof.py
│       ├── predict.py
│       ├── prior_baseline.py
│       ├── validation.py
│       ├── train_baseline.py
│       ├── train_logistic_comparison.py
│       ├── train_oof_comparison.py
│       └── train_prior_baseline.py
│
├── tests/
├── reports/
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Installation

Create and activate a Python virtual environment, then install the project with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

The editable installation makes the `moa` package importable directly while keeping local source changes immediately available.

## Run the OOF Benchmark

Run the current prior-vs-Logistic Regression benchmark with:

```bash
python -m moa.train_oof_comparison
```

The experiment evaluates:

- target-prevalence OOF predictions
- Logistic Regression OOF predictions
- predictions with and without the vehicle-control rule
- fold-level and overall mean log loss

Experiment results are written to the `reports/` directory.

## Testing and CI

Run the full test suite locally with:

```bash
python -m pytest -q
```

The tests cover core behavior including:

- feature and target construction
- sample alignment validation
- mean multi-label log loss
- multilabel fold assignment
- OOF prediction generation
- modeling utilities
- prediction post-processing
- prior baseline behavior

GitHub Actions installs the project in a clean environment and runs the test suite automatically for changes targeting `main`.

## Data

The project uses the public Mechanisms of Action (MoA) Prediction dataset originally released through Kaggle.

Raw data files are not committed to the repository and should be placed under:

```text
data/raw/
```

## Current Investigation

The strongest baseline currently predicts target prevalence without using biological features.

Logistic Regression closes most of that gap after regularization and control-aware post-processing, but does not yet outperform the prevalence baseline overall.

The next diagnostic step is therefore target-level rather than model-level:

```text
overall log loss
        ↓
per-target log loss
        ↓
label prevalence
        ↓
identify where Logistic Regression helps or hurts
        ↓
design the next controlled experiment
```

This analysis is intended to distinguish failures caused by rare-label behavior, calibration, model capacity, or training strategy before introducing more complex models.

## Roadmap

- Per-target OOF diagnostics and rare-label analysis
- Treatment-only training with control-aware evaluation
- Logistic Regression regularization analysis
- Nonlinear classical ML baselines
- PyTorch MLP evaluated under the same OOF framework
- Probability calibration and model ensembling
- Lightweight inference API after the modeling pipeline is stable