from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_logistic_ovr_pipeline(
    numeric_features,
    categorical_features,
    class_weight="balanced",
    max_iter=1000,
    solver="liblinear",
    n_jobs=-1,
):
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    logistic_model = LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", OneVsRestClassifier(logistic_model, n_jobs=n_jobs)),
        ]
    )

    return model