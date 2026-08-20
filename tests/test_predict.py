import numpy as np
import pandas as pd
import pytest

from moa.predict import (
    build_submission,
    zero_control_predictions,
)


def test_build_submission_preserves_sample_submission_format():
    test_features = pd.DataFrame(
        {
            "sig_id": ["test_1", "test_2"],
            "cp_type": ["trt_cp", "ctl_vehicle"],
            "cp_time": [24, 48],
            "cp_dose": ["D1", "D2"],
            "g-0": [0.1, 0.2],
            "c-0": [0.3, 0.4],
        }
    )

    sample_submission = pd.DataFrame(
        {
            "sig_id": ["test_1", "test_2"],
            "target_a": [0.0, 0.0],
            "target_b": [0.0, 0.0],
        }
    )

    predictions = np.array(
        [
            [0.10, 0.90],
            [0.25, 0.75],
        ]
    )

    submission = build_submission(
        test_features=test_features,
        sample_submission=sample_submission,
        predictions=predictions,
        target_names=["target_a", "target_b"],
    )

    assert list(submission.columns) == ["sig_id", "target_a", "target_b"]
    assert list(submission["sig_id"]) == ["test_1", "test_2"]
    assert np.allclose(submission[["target_a", "target_b"]].values, predictions)


def test_build_submission_rejects_unknown_targets():
    test_features = pd.DataFrame({"sig_id": ["test_1"]})
    sample_submission = pd.DataFrame({"sig_id": ["test_1"], "target_a": [0.0]})
    predictions = np.array([[0.10]])

    with pytest.raises(ValueError, match="missing from sample submission"):
        build_submission(
            test_features=test_features,
            sample_submission=sample_submission,
            predictions=predictions,
            target_names=["target_b"],
        )


def test_build_submission_rejects_prediction_shape_mismatch():
    test_features = pd.DataFrame({"sig_id": ["test_1", "test_2"]})
    sample_submission = pd.DataFrame(
        {"sig_id": ["test_1", "test_2"], "target_a": [0.0, 0.0]}
    )
    predictions = np.array([[0.10]])

    with pytest.raises(ValueError, match="Prediction shape"):
        build_submission(
            test_features=test_features,
            sample_submission=sample_submission,
            predictions=predictions,
            target_names=["target_a"],
        )

def test_zero_control_predictions_sets_control_rows_to_zero():
    features = pd.DataFrame(
        {
            "cp_type": [
                "trt_cp",
                "ctl_vehicle",
                "trt_cp",
            ]
        }
    )
    predictions = np.array(
        [
            [0.10, 0.90],
            [0.25, 0.75],
            [0.30, 0.70],
        ]
    )
    original_predictions = predictions.copy()

    adjusted = zero_control_predictions(
        features=features,
        predictions=predictions,
    )

    expected = np.array(
        [
            [0.10, 0.90],
            [0.00, 0.00],
            [0.30, 0.70],
        ]
    )

    np.testing.assert_allclose(adjusted, expected)
    np.testing.assert_allclose(
        predictions,
        original_predictions,
    )


def test_zero_control_predictions_rejects_missing_cp_type():
    features = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
        }
    )
    predictions = np.array(
        [
            [0.10],
            [0.20],
        ]
    )

    with pytest.raises(
        ValueError,
        match="cp_type",
    ):
        zero_control_predictions(
            features=features,
            predictions=predictions,
        )


def test_zero_control_predictions_rejects_row_mismatch():
    features = pd.DataFrame(
        {
            "cp_type": [
                "trt_cp",
                "ctl_vehicle",
            ]
        }
    )
    predictions = np.array(
        [
            [0.10, 0.90],
        ]
    )

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        zero_control_predictions(
            features=features,
            predictions=predictions,
        )