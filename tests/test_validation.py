import numpy as np
import pandas as pd
import pytest

from moa.validation import make_multilabel_folds


def make_balanced_targets() -> pd.DataFrame:
    patterns = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )

    targets = np.tile(patterns, (3, 1))

    return pd.DataFrame(
        targets,
        columns=["target_a", "target_b", "target_c"],
    )


def test_returns_one_valid_fold_id_per_sample() -> None:
    y = make_balanced_targets()

    fold_ids = make_multilabel_folds(
        y,
        n_splits=3,
        random_state=42,
    )

    assert isinstance(fold_ids, np.ndarray)
    assert fold_ids.shape == (len(y),)
    assert fold_ids.dtype == np.int64
    assert set(fold_ids) == {0, 1, 2}


def test_folds_have_balanced_sample_counts() -> None:
    y = make_balanced_targets()

    fold_ids = make_multilabel_folds(
        y,
        n_splits=3,
        random_state=42,
    )

    fold_sizes = np.bincount(fold_ids, minlength=3)

    assert fold_sizes.sum() == len(y)
    assert fold_sizes.max() - fold_sizes.min() <= 1


def test_label_counts_are_balanced_across_folds() -> None:
    y = make_balanced_targets()

    fold_ids = make_multilabel_folds(
        y,
        n_splits=3,
        random_state=42,
    )

    label_counts = np.stack(
        [
            y.to_numpy()[fold_ids == fold_id].sum(axis=0)
            for fold_id in range(3)
        ]
    )

    assert np.all(np.ptp(label_counts, axis=0) <= 1)


def test_same_random_state_produces_same_folds() -> None:
    y = make_balanced_targets()

    first_fold_ids = make_multilabel_folds(
        y,
        n_splits=3,
        random_state=42,
    )
    second_fold_ids = make_multilabel_folds(
        y,
        n_splits=3,
        random_state=42,
    )

    np.testing.assert_array_equal(
        first_fold_ids,
        second_fold_ids,
    )


@pytest.mark.parametrize(
    "y",
    [
        pd.DataFrame(),
        pd.DataFrame(columns=["target_a"]),
        pd.DataFrame(index=range(3)),
    ],
)
def test_rejects_empty_targets(y: pd.DataFrame) -> None:
    with pytest.raises(
        ValueError,
        match="at least one sample and one target",
    ):
        make_multilabel_folds(y)


@pytest.mark.parametrize("n_splits", [0, 1, -1])
def test_rejects_fewer_than_two_splits(n_splits: int) -> None:
    y = make_balanced_targets()

    with pytest.raises(
        ValueError,
        match="n_splits must be at least 2",
    ):
        make_multilabel_folds(
            y,
            n_splits=n_splits,
        )


def test_rejects_more_splits_than_samples() -> None:
    y = pd.DataFrame(
        {
            "target_a": [0, 1, 0],
            "target_b": [1, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="n_splits cannot exceed the number of samples",
    ):
        make_multilabel_folds(
            y,
            n_splits=4,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [-1, 2, 0.5, np.nan],
)
def test_rejects_nonbinary_targets(invalid_value: float) -> None:
    y = make_balanced_targets().astype(float)
    y.iloc[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="only binary 0/1 labels",
    ):
        make_multilabel_folds(
            y,
            n_splits=3,
        )
