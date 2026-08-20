import numpy as np
import pandas as pd

from moa.oof import (
    make_model_oof_predictions,
    make_prior_oof_predictions,
)


def test_prior_oof_uses_only_training_folds() -> None:
    y = pd.DataFrame(
        {
            "target_a": [1, 1, 0, 0],
            "target_b": [0, 0, 1, 1],
        }
    )
    fold_ids = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )

    predictions = make_prior_oof_predictions(
        y=y,
        fold_ids=fold_ids,
    )

    expected = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    np.testing.assert_allclose(
        predictions,
        expected,
    )

class TrainingMeanEstimator:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ):
        self.target_means_ = y.mean(axis=0).to_numpy()
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        return np.tile(
            self.target_means_,
            (len(X), 1),
        )


def test_model_oof_uses_only_training_folds() -> None:
    X = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
        }
    )
    y = pd.DataFrame(
        {
            "target_a": [1, 1, 0, 0],
            "target_b": [0, 0, 1, 1],
        }
    )
    fold_ids = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )

    predictions = make_model_oof_predictions(
        X=X,
        y=y,
        fold_ids=fold_ids,
        model_builder=TrainingMeanEstimator,
    )

    expected = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    np.testing.assert_allclose(
        predictions,
        expected,
    )
