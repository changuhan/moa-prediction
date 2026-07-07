import joblib

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