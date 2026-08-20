import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def make_multilabel_folds(
    y: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    if y.empty:
        raise ValueError("y must contain at least one sample and one target.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if len(y) < n_splits:
        raise ValueError("n_splits cannot exceed the number of samples.")

    if not y.isin([0, 1]).all().all():
        raise ValueError("y must contain only binary 0/1 labels.")

    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_ids = np.full(
        shape=len(y),
        fill_value=-1,
        dtype=np.int64,
    )

    dummy_X = np.zeros(
        shape=(len(y), 1),
        dtype=np.int8,
    )

    y_array = y.to_numpy(dtype=np.int8)

    for fold_id, (_, valid_idx) in enumerate(
        splitter.split(dummy_X, y_array)
    ):
        fold_ids[valid_idx] = fold_id

    if np.any(fold_ids == -1):
        raise RuntimeError(
            "Some samples were not assigned to a validation fold."
        )

    return fold_ids
