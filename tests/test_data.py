import pandas as pd
import pytest

from moa.data import get_feature_groups, make_X_y


def test_get_feature_groups_separates_metadata_gene_and_cell_columns():
    train_features = pd.DataFrame(
        {
            "sig_id": ["id_1"],
            "cp_type": ["trt_cp"],
            "cp_time": [24],
            "cp_dose": ["D1"],
            "g-0": [0.10],
            "g-1": [0.20],
            "c-0": [0.30],
            "unrelated_column": [999],
        }
    )

    meta_cols, gene_cols, cell_cols = get_feature_groups(train_features)

    assert meta_cols == ["cp_type", "cp_time", "cp_dose"]
    assert gene_cols == ["g-0", "g-1"]
    assert cell_cols == ["c-0"]
    assert "unrelated_column" not in gene_cols
    assert "unrelated_column" not in cell_cols


def test_make_X_y_selects_most_frequent_targets():
    train_features = pd.DataFrame(
        {
            "sig_id": ["id_1", "id_2", "id_3", "id_4"],
            "cp_type": ["trt_cp", "trt_cp", "trt_cp", "ctl_vehicle"],
            "cp_time": [24, 48, 72, 24],
            "cp_dose": ["D1", "D2", "D1", "D2"],
            "g-0": [0.10, 0.20, 0.30, 0.40],
            "c-0": [0.50, 0.60, 0.70, 0.80],
        }
    )

    train_targets = pd.DataFrame(
        {
            "sig_id": ["id_1", "id_2", "id_3", "id_4"],
            "target_a": [1, 1, 1, 0],
            "target_b": [1, 1, 0, 0],
            "target_c": [1, 0, 0, 0],
        }
    )

    X, y, selected_targets, target_counts = make_X_y(
        train_features=train_features,
        train_targets=train_targets,
        top_n_targets=2,
    )

    assert "sig_id" not in X.columns
    assert selected_targets == ["target_a", "target_b"]
    assert list(y.columns) == ["target_a", "target_b"]
    assert X.shape == (4, 5)
    assert y.shape == (4, 2)
    assert target_counts["target_a"] == 3
    assert target_counts["target_b"] == 2
    assert target_counts["target_c"] == 1


def test_make_X_y_rejects_misaligned_sample_ids():
    train_features = pd.DataFrame(
        {
            "sig_id": ["id_1", "id_2"],
            "cp_type": ["trt_cp", "trt_cp"],
            "cp_time": [24, 48],
            "cp_dose": ["D1", "D2"],
            "g-0": [0.10, 0.20],
            "c-0": [0.30, 0.40],
        }
    )

    train_targets = pd.DataFrame(
        {
            "sig_id": ["id_1", "wrong_id"],
            "target_a": [0, 1],
            "target_b": [1, 0],
        }
    )

    with pytest.raises(ValueError, match="sig_id"):
        make_X_y(
            train_features=train_features,
            train_targets=train_targets,
            top_n_targets=2,
        )