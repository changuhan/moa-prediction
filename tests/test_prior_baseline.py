import numpy as np 
import pandas as pd 

from moa.prior_baseline import compute_target_priors, make_prior_predictions, make_baseline_comparison

def test_compute_target_priors_returns_training_positive_rates():
    y_train = pd.DataFrame(
        {
            "target_a": [1, 0, 1, 0],
            "target_b": [0, 0, 1, 0]                   
        }
    )

    target_priors = compute_target_priors(y_train)

    assert np.isclose(target_priors["target_a"], 0.50)
    assert np.isclose(target_priors["target_b"], 0.25)

def test_make_prior_predictions_repeat_priors_for_each_row():
    target_priors = pd.Series(
        {
            "target_a": 0.5,
            "target_b": 0.25
        }
    )

    predictions = make_prior_predictions(n_rows=3, target_priors=target_priors)

    expected = np.array(
        [
            [0.5, 0.25],
            [0.5, 0.25],
            [0.5, 0.25]
        ]
    )

    assert predictions.shape == (3, 2)
    assert np.allclose(predictions, expected)

def test_make_baseline_comparison_sorts_rows_by_mean_log_loss():
    comparison = make_baseline_comparison(
        [
            {"model": "logistic_regression", "mean_log_loss": 0.2},
            {"model": "dummy_model", "mean_log_loss": 0.30},
        ]
    )

    assert list(comparison["model"]) == ["logistic_regression", "dummy_model"]

    
