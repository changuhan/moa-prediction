import numpy as np
import pandas as pd 

def compute_target_priors(y_train):
    return y_train.mean(axis=0)

def make_prior_predictions(n_rows, target_priors):
    predictions = np.tile(target_priors.to_numpy(), (n_rows, 1))
    return predictions

def make_baseline_comparison(rows):
    return pd.DataFrame(rows).sort_values("mean_log_loss")

