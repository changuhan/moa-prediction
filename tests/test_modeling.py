from sklearn.compose import ColumnTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from moa.modeling import build_logistic_ovr_pipeline, build_preprocessor


def test_build_preprocessor_creates_column_transformer():
    numeric_features = ["g-0", "c-0"]
    categorical_features = ["cp_type", "cp_dose"]

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    assert isinstance(preprocessor, ColumnTransformer)
    assert [name for name, _, _ in preprocessor.transformers] == ["num", "cat"]


def test_build_logistic_ovr_pipeline_creates_pipeline():
    numeric_features = ["g-0", "c-0"]
    categorical_features = ["cp_type", "cp_dose"]

    model = build_logistic_ovr_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        class_weight="balanced",
    )

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps
    assert isinstance(model.named_steps["classifier"], OneVsRestClassifier)
    assert model.named_steps["classifier"].estimator.class_weight == "balanced"


def test_build_logistic_ovr_pipeline_allows_unbalanced_model():
    numeric_features = ["g-0", "c-0"]
    categorical_features = ["cp_type", "cp_dose"]

    model = build_logistic_ovr_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        class_weight=None,
    )

    assert model.named_steps["classifier"].estimator.class_weight is None