from sklearn.model_selection import train_test_split

import pandas as pd 

from moa.config import RANDOM_STATE, REPORT_DIR, TEST_SIZE, TOP_N_TARGETS
from moa.data import load_raw_data, make_X_y
from moa.metrics import mean_multilabel_log_loss
from moa.prior_baseline import (
    compute_target_priors,
    make_prior_predictions,
    make_baseline_comparison,
)

def main():
    train_features, train_targets, _, _ = load_raw_data()

    X, y, selected_targets, _ = make_X_y(
        train_features=train_features,
        train_targets=train_targets,
        top_n_targets=TOP_N_TARGETS,
    )

    _, _, y_train, y_valid = train_test_split(
        X, 
        y, 
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    target_priors = compute_target_priors(y_train)
    y_valid_pred = make_prior_predictions(
        n_rows=len(y_valid),
        target_priors=target_priors,
    )

    mean_loss, losses = mean_multilabel_log_loss(
        y_true=y_valid,
        y_pred=y_valid_pred,
        target_names=selected_targets,
    )

    comparison_rows = [
        {
            "model": "Prior Baseline",
            "mean_log_loss": mean_loss,
            "uses_features": False,
            "description": "Predicts the mean of each target from the training set for all test samples.",
        }
    ]

    logistic_report_path = REPORT_DIR / f"baseline_top{len(selected_targets)}.csv"
    if logistic_report_path.exists():
        logistic_results = pd.read_csv(logistic_report_path)
        comparison_rows.append(
            {
                "model": "logistic_regression_ovr",
                "mean_log_loss": logistic_results["valid_log_loss"].mean(),
                "uses_features": True,
                "description": "Uses metadata, gene-expression, and cell-viability features.",
            }
        )

    comparison = make_baseline_comparison(comparison_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "baseline_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print(f"Dummy prior mean multi-label log loss: {mean_loss:.6f}")
    print(f"Saved baseline comparison to {output_path}")


if __name__ == "__main__":
    main()