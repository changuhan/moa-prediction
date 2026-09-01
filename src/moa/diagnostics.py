import numpy as np
import pandas as pd

def make_per_target_results(
    target_names,
    target_counts,
    n_samples,
    prior_losses,
    logistic_losses,
):
    """
    Build a target-level comparison table for prior and logistic OOF results.

    Expected columns:
    - target
    - positive_count
    - prevalence
    - prior_loss
    - logistic_loss
    - improvement
    - winner
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0.")

    if not (
        len(target_names)
        == len(prior_losses)
        == len(logistic_losses)
    ):
        raise ValueError(
            "All input lists must have the same length."
        )
    
    missing_targets = [
        target for target in target_names if target not in target_counts.index
    ]

    if missing_targets:
        raise ValueError(
            f"Target counts missing for targets: {missing_targets}"
        )

    positive_counts = target_counts.loc[target_names].to_numpy()

    prevalence = positive_counts / n_samples

    prior_losses = np.asarray(prior_losses, dtype=np.float64)
    logistic_losses = np.asarray(logistic_losses, dtype=np.float64)

    improvement = prior_losses - logistic_losses

    winner = np.where(
        np.isclose(improvement, 0.0),
        "tie",
        np.where(improvement > 0, "logistic", "prior"),
    )

    results_df = pd.DataFrame({
        "target": target_names,
        "positive_count": positive_counts,
        "prevalence": prevalence,
        "prior_loss": prior_losses,
        "logistic_loss": logistic_losses,
        "improvement": improvement,
        "winner": winner,
    })          

    return results_df.sort_values("improvement", ascending=False).reset_index(drop=True)
