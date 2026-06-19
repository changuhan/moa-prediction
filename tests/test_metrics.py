import numpy as np
import pandas as pd 

from moa.metrics import mean_multilabel_log_loss, make_results_table

def test_mean_multilabel_log_loss():
    y_true = pd.DataFrame(
        {
            "target_a": [1, 0, 1, 0],
            "target_b": [0, 1, 0, 1]
        }
    )

    y_pred = np.array(
        [
            [0.10, 0.90],
            [0.80, 0.20],
            [0.20, 0.20],
            [0.80, 0.80],
        ]
    )

    mean_loss, losses = mean_multilabel_log_loss(
        y_true = y_true,
        y_pred = y_pred,
        target_names = ["target_a", "target_b"],
    )

    assert isinstance(mean_loss, float)
    assert len(losses) == 2
    assert all(np.isfinite(loss) for loss in losses)
    assert np.isclose(mean_loss, np.mean(losses))

def test_better_predictions_have_lower_loss():
    y_true = pd.DataFrame(
        {
            "target_a": [1, 0, 1, 0],
            "target_b": [0, 1, 0, 1]
        }
    )

    good_predictions = np.array(
        [
            [0.95, 0.05],
            [0.05, 0.95],
            [0.90, 0.10],
            [0.10, 0.90],      
        ]
    )
    
    bad_predictions = np.array(
        [
            [0.05, 0.95],
            [0.95, 0.05],
            [0.10, 0.90],
            [0.90, 0.10],
        ]
    )

    good_loss, _ = mean_multilabel_log_loss(
        y_true=y_true,
        y_pred=good_predictions,
        target_names=["target_a", "target_b"],
    )

    bad_loss, _ = mean_multilabel_log_loss(
        y_true=y_true,
        y_pred=bad_predictions,
        target_names=["target_a", "target_b"],
    )

    assert good_loss < bad_loss 

def test_log_loss_clips_zero_and_one_predictions():
    y_true = pd.DataFrame(
        {
            "target_a": [1, 0]
        }
    )

    y_pred = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    mean_loss, losses = mean_multilabel_log_loss(
        y_true=y_true,
        y_pred=y_pred,
        target_names=["target_a"],
    )

    assert np.isfinite(mean_loss)
    assert len(losses) == 1
    assert np.isfinite(losses[0])

def test_make_results_table_sorts_by_loss_descending():
    target_names = ["target_a", "target_b"]

    target_counts = pd.Series(
        {
            "target_a": 10,
            "target_b": 5,
        }
    )

    losses = [0.20, 0.80]

    results = make_results_table(
        target_names=target_names,
        target_counts=target_counts,
        losses=losses,
    )

    assert list(results.columns) == [
        "target",
        "positive_count",
        "valid_log_loss",
    ]

    assert results.iloc[0]["target"] == "target_b"
    assert results.iloc[0]["positive_count"] == 5
    assert np.isclose(results.iloc[0]["valid_log_loss"], 0.80)

