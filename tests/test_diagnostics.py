import numpy as np 
import pandas as pd
import pytest

from moa.diagnostics import make_per_target_results

def test_make_per_target_results():
    target_names = ["common_target", "rare_target"]

    target_counts = pd.Series(
        {
            "common_target": 20,
            "rare_target": 2,
        }
    )

    prior_losses = [0.10, 0.02]
    logistic_losses = [0.08, 0.03]

    result = make_per_target_results(
        target_names=target_names,
        target_counts=target_counts,
        n_samples=100,
        prior_losses=prior_losses,
        logistic_losses=logistic_losses,
    )

    assert list(result.columns) == [
        "target",
        "positive_count",
        "prevalence",
        "prior_loss",
        "logistic_loss",
        "improvement",
        "winner",
    ]

    common_result = result.loc[result["target"] == "common_target"].iloc[0]
    rare_result = result.loc[result["target"] == "rare_target"].iloc[0]

    assert common_result["positive_count"] == 20
    assert np.isclose(common_result["prevalence"], 0.20)
    assert np.isclose(common_result["improvement"], 0.02)
    assert common_result["winner"] == "logistic"

    assert rare_result["positive_count"] == 2
    assert np.isclose(rare_result["prevalence"], 0.02)
    assert np.isclose(rare_result["improvement"], -0.01)
    assert rare_result["winner"] == "prior"



def test_make_per_target_results_rejects_nonpositive_sample_count():
    target_counts = pd.Series(
        {
            "target_a": 10,
        }
    )

    with pytest.raises(
        ValueError,
        match="n_samples must be greater than 0",
    ):
        make_per_target_results(
            target_names=["target_a"],
            target_counts=target_counts,
            n_samples=0,
            prior_losses=[0.10],
            logistic_losses=[0.08],
        )

def test_make_per_target_results_rejects_mismatched_loss_lengths():
    target_counts = pd.Series(
        {
            "target_a": 10,
            "target_b": 5,
        }
    )

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        make_per_target_results(
            target_names=["target_a", "target_b"],
            target_counts=target_counts,
            n_samples=100,
            prior_losses=[0.10, 0.05],
            logistic_losses=[0.08],
        )

def test_make_per_target_results_rejects_missing_target_counts():
    target_counts = pd.Series(
        {
            "target_a": 10,
        }
    )

    with pytest.raises(
        ValueError,
        match="Target counts missing",
    ):
        make_per_target_results(
            target_names=["target_a", "target_b"],
            target_counts=target_counts,
            n_samples=100,
            prior_losses=[0.10, 0.05],
            logistic_losses=[0.08, 0.04],
        )