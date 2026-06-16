import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def mean_multilabel_log_loss(y_true, y_pred, target_names):
    """
    Compute the mean multi-label log loss across all targets.

    Parameters:
    y_true (pd.DataFrame): DataFrame of true binary labels (shape: [n_samples, n_targets])
    y_pred (pd.DataFrame): DataFrame of predicted probabilities (shape: [n_samples, n_targets])
    target_names (list): List of target names

    Returns:
    float: Mean multi-label log loss
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    losses = []

    for i, target in enumerate(target_names):
        loss = log_loss(y_true[target], y_pred[:, i], labels=[0, 1])
        losses.append(loss)

    return float(np.mean(losses)), losses


def make_results_table(target_names, target_counts, losses):
    return pd.DataFrame(
        {
            "target": target_names,
            "positive_count": target_counts[target_names].values,
            "valid_log_loss": losses,
        }
    ).sort_values("valid_log_loss", ascending=False)
