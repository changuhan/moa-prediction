import joblib
import numpy as np

from moa.config import MODEL_DIR, SUBMISSION_DIR
from moa.data import load_raw_data


def load_model_artifact(model_path):
    artifact = joblib.load(model_path)

    if isinstance(artifact, dict):
        if "model" not in artifact or "target_names" not in artifact:
            raise ValueError(
                "The loaded artifact dictionary must contain 'model' and 'target_names' keys."
            )
        return artifact["model"], artifact["target_names"]

    raise ValueError(
        "The loaded artifact is not a dictionary. Please ensure the model was saved "
        "as a dictionary containing 'model' and 'target_names'."
    )

def zero_control_predictions(features, predictions):
    if "cp_type" not in features.columns:
        raise ValueError(
            "features must contain a 'cp_type' column."
        )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if predictions.ndim != 2:
        raise ValueError(
            "predictions must be a two-dimensional array."
        )

    if len(features) != predictions.shape[0]:
        raise ValueError(
            "features and predictions must have the same number of rows."
        )

    adjusted_predictions = predictions.copy()

    control_mask = (
        features["cp_type"]
        .eq("ctl_vehicle")
        .to_numpy()
    )
    adjusted_predictions[control_mask] = 0.0

    return adjusted_predictions

def build_submission(test_features, sample_submission, predictions, target_names):
    expected_targets = [col for col in sample_submission.columns if col != "sig_id"]
    missing_targets = [target for target in target_names if target not in expected_targets]

    if missing_targets:
        raise ValueError(
            f"Predicted targets are missing from sample submission: {missing_targets}"
        )

    if predictions.shape != (len(test_features), len(target_names)):
        raise ValueError(
            "Prediction shape does not match test rows and target names: "
            f"{predictions.shape} vs ({len(test_features)}, {len(target_names)})"
        )

    submission = sample_submission.copy()
    submission["sig_id"] = test_features["sig_id"].values
    submission.loc[:, target_names] = predictions

    return submission


def main():
    _, _, test_features, sample_submission = load_raw_data()

    model_path = MODEL_DIR / "baseline_model.joblib"
    model, target_names = load_model_artifact(model_path)

    X_test = test_features.drop(columns=["sig_id"])
    predictions = model.predict_proba(X_test)

    predictions = zero_control_predictions(
    features=X_test,
    predictions=predictions,
)

    submission = build_submission(
        test_features=test_features,
        sample_submission=sample_submission,
        predictions=predictions,
        target_names=target_names,
    )

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSION_DIR / "baseline_submission.csv"
    submission.to_csv(output_path, index=False)

    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()