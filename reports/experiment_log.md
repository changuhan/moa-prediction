## V1 — Full 206 Logistic Regression Baseline

### Goal
Expand the top-20 baseline to all 206 scored MoA targets.

### Setup
- Model: OneVsRestClassifier(LogisticRegression)
- Features: cp_type, cp_time, cp_dose, g-* gene expression features, c-* cell viability features
- Numeric preprocessing: StandardScaler
- Categorical preprocessing: OneHotEncoder
- Validation split: 80/20, random_state=42
- Metric: mean multi-label log loss

### Result
Mean multi-label log loss: 0.056092

### Notes
The full 206-target score is much lower than the top-20 score because many rare targets are mostly zero, making them easier to score with low predicted probabilities. This does not mean the full model is better than the top-20 model; the evaluation target set changed.

### Warning
One target had only one class present in the training split. This is expected for rare MoA labels and should be handled more carefully in future validation design.

### Hardest targets
- glutamate_receptor_antagonist: 0.499598
- adrenergic_receptor_antagonist: 0.461712
- sodium_channel_inhibitor: 0.441173
- dopamine_receptor_antagonist: 0.439691
- acetylcholine_receptor_antagonist: 0.415334

### Next
- Save model with joblib
- Create prediction script for test_features.csv