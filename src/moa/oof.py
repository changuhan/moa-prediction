import numpy as np
import pandas as pd

from moa.prior_baseline import (
    compute_target_priors,
    make_prior_predictions,
)


def make_prior_oof_predictions(
    y: pd.DataFrame,
    fold_ids: np.ndarray,
) -> np.ndarray:
    fold_ids = np.asarray(fold_ids)

    predictions = np.empty(
        shape=(len(y), y.shape[1]),
        dtype=np.float64,
    )

    for fold_id in np.unique(fold_ids):
        train_mask = fold_ids != fold_id
        valid_mask = fold_ids == fold_id

        target_priors = compute_target_priors(
            y.loc[train_mask]
        )
        fold_predictions = make_prior_predictions(
            n_rows=int(valid_mask.sum()),
            target_priors=target_priors,
        )

        predictions[valid_mask] = fold_predictions

    return predictions

def make_model_oof_predictions(
    X: pd.DataFrame,
    y: pd.DataFrame,
    fold_ids: np.ndarray,
    model_builder,
) -> np.ndarray:
    fold_ids = np.asarray(fold_ids)

    predictions = np.empty(
        shape=(len(y), y.shape[1]),
        dtype=np.float64,
    )

    for fold_id in np.unique(fold_ids):
        train_mask = fold_ids != fold_id
        valid_mask = fold_ids == fold_id

        model = model_builder()

        model.fit(
            X.loc[train_mask],
            y.loc[train_mask],
        )
        fold_predictions = np.asarray(
            model.predict_proba(
                X.loc[valid_mask]
            ),
            dtype=np.float64,
        )

        expected_shape = (
            int(valid_mask.sum()),
            y.shape[1],
        )
        if fold_predictions.shape != expected_shape:
            raise ValueError(
                "Model predictions must have shape "
                f"{expected_shape}, but received "
                f"{fold_predictions.shape}."
            )

        predictions[valid_mask] = fold_predictions

    return predictions
