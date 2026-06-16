import pandas as pd

from moa.config import DATA_DIR


def load_raw_data():
    train_features = pd.read_csv(DATA_DIR / "train_features.csv")
    train_targets = pd.read_csv(DATA_DIR / "train_targets_scored.csv")
    test_features = pd.read_csv(DATA_DIR / "test_features.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

    return train_features, train_targets, test_features, sample_submission


def get_feature_groups(train_features):
    meta_cols = ["cp_type", "cp_time", "cp_dose"]
    gene_cols = [col for col in train_features.columns if col.startswith("g-")]
    cell_cols = [col for col in train_features.columns if col.startswith("c-")]

    return meta_cols, gene_cols, cell_cols


def make_X_y(train_features, train_targets, top_n_targets=20):
    target_cols = [col for col in train_targets.columns if col != "sig_id"]
    target_counts = train_targets[target_cols].sum().sort_values(ascending=False)
    selected_targets = target_counts.head(top_n_targets).index.tolist()

    X = train_features.drop(columns=["sig_id"])
    y = train_targets[selected_targets]

    return X, y, selected_targets, target_counts
