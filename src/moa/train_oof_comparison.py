import argparse
from time import perf_counter

import numpy as np
import pandas as pd

from moa.config import (
    RANDOM_STATE,
    REPORT_DIR,
    TOP_N_TARGETS,
)
from moa.data import (
    get_feature_groups,
    load_raw_data,
    make_X_y,
)
from moa.diagnostics import make_per_target_results
from moa.metrics import mean_multilabel_log_loss
from moa.modeling import build_logistic_ovr_pipeline
from moa.oof import (
    make_model_oof_predictions,
    make_prior_oof_predictions,
)
from moa.predict import zero_control_predictions
from moa.validation import make_multilabel_folds


DEFAULT_N_SPLITS = 5
DEFAULT_C_VALUE = 0.01


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare prior and Logistic Regression "
            "using multilabel OOF predictions."
        )
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
    )
    parser.add_argument(
        "--top-n-targets",
        type=int,
        default=TOP_N_TARGETS,
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=DEFAULT_C_VALUE,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )
    return parser.parse_args()


def make_evaluation_rows(
    model_name,
    y,
    predictions,
    target_names,
    fold_ids,
    control_override,
    c_value,
):
    rows = []

    for fold_id in np.unique(fold_ids):
        valid_mask = fold_ids == fold_id

        fold_loss, _ = mean_multilabel_log_loss(
            y_true=y.loc[valid_mask],
            y_pred=predictions[valid_mask],
            target_names=target_names,
        )

        rows.append(
            {
                "model": model_name,
                "C": c_value,
                "control_override": control_override,
                "scope": "fold",
                "fold_id": int(fold_id),
                "n_samples": int(valid_mask.sum()),
                "mean_log_loss": fold_loss,
            }
        )

    overall_loss, _ = mean_multilabel_log_loss(
        y_true=y,
        y_pred=predictions,
        target_names=target_names,
    )

    rows.append(
        {
            "model": model_name,
            "C": c_value,
            "control_override": control_override,
            "scope": "overall",
            "fold_id": -1,
            "n_samples": len(y),
            "mean_log_loss": overall_loss,
        }
    )

    return rows


def main():
    args = parse_args()

    print("Loading MoA data...")
    train_features, train_targets, _, _ = load_raw_data()

    meta_cols, gene_cols, cell_cols = get_feature_groups(
        train_features
    )

    X, y, selected_targets, _ = make_X_y(
        train_features=train_features,
        train_targets=train_targets,
        top_n_targets=args.top_n_targets,
    )

    print(
        f"Creating {args.n_splits} multilabel folds "
        f"for {len(y)} samples and "
        f"{len(selected_targets)} targets..."
    )

    fold_ids = make_multilabel_folds(
        y=y,
        n_splits=args.n_splits,
        random_state=RANDOM_STATE,
    )

    print(f"Fold sizes: {np.bincount(fold_ids)}")

    comparison_rows = []

    print("Generating prior OOF predictions...")
    prior_predictions = make_prior_oof_predictions(
        y=y,
        fold_ids=fold_ids,
    )
    prior_controlled_predictions = zero_control_predictions(
        features=X,
        predictions=prior_predictions,
    )

    comparison_rows.extend(
        make_evaluation_rows(
            model_name="dummy_prior",
            y=y,
            predictions=prior_predictions,
            target_names=selected_targets,
            fold_ids=fold_ids,
            control_override=False,
            c_value=None,
        )
    )
    comparison_rows.extend(
        make_evaluation_rows(
            model_name="dummy_prior",
            y=y,
            predictions=prior_controlled_predictions,
            target_names=selected_targets,
            fold_ids=fold_ids,
            control_override=True,
            c_value=None,
        )
    )

    fold_counter = 0

    def model_builder():
        nonlocal fold_counter
        fold_counter += 1

        print(
            f"Building Logistic Regression model "
            f"{fold_counter}/{args.n_splits}..."
        )

        return build_logistic_ovr_pipeline(
            numeric_features=gene_cols + cell_cols,
            categorical_features=meta_cols,
            class_weight=None,
            C=args.c_value,
            n_jobs=args.n_jobs,
        )

    print(
        "Generating Logistic Regression OOF predictions "
        f"with C={args.c_value}..."
    )
    start_time = perf_counter()

    logistic_predictions = make_model_oof_predictions(
        X=X,
        y=y,
        fold_ids=fold_ids,
        model_builder=model_builder,
    )

    elapsed_seconds = perf_counter() - start_time

    logistic_controlled_predictions = zero_control_predictions(
        features=X,
        predictions=logistic_predictions,
    )

    logistic_model_name = f"logistic_c_{args.c_value:g}"

    comparison_rows.extend(
        make_evaluation_rows(
            model_name=logistic_model_name,
            y=y,
            predictions=logistic_predictions,
            target_names=selected_targets,
            fold_ids=fold_ids,
            control_override=False,
            c_value=args.c_value,
        )
    )
    comparison_rows.extend(
        make_evaluation_rows(
            model_name=logistic_model_name,
            y=y,
            predictions=logistic_controlled_predictions,
            target_names=selected_targets,
            fold_ids=fold_ids,
            control_override=True,
            c_value=args.c_value,
        )
    )

    comparison = pd.DataFrame(comparison_rows)

    overall_results = (
        comparison[comparison["scope"] == "overall"]
        .sort_values("mean_log_loss")
        .reset_index(drop=True)
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = REPORT_DIR / (
        "oof_baseline_comparison_"
        f"top{len(selected_targets)}_"
        f"folds{args.n_splits}.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    print()
    _, prior_target_losses = mean_multilabel_log_loss(
        y_true=y,
        y_pred=prior_controlled_predictions,
        target_names=selected_targets,
    )

    _, logistic_target_losses = mean_multilabel_log_loss(
        y_true=y,
        y_pred=logistic_controlled_predictions,
        target_names=selected_targets,
    )

    target_counts = y[selected_targets].sum(axis=0)

    per_target_results = make_per_target_results(
        target_names=selected_targets,
        target_counts=target_counts,
        n_samples=len(y),
        prior_losses=prior_target_losses,
        logistic_losses=logistic_target_losses,
    )

    per_target_output_path = REPORT_DIR / (
        "oof_per_target_diagnostics_"
        f"top{len(selected_targets)}_"
        f"folds{args.n_splits}.csv"
    )

    per_target_results.to_csv(
        per_target_output_path,
        index=False,
    )

    print("Overall OOF results:")
    print(overall_results.to_string(index=False))
    print()
    print(
        "Logistic Regression elapsed time: "
        f"{elapsed_seconds:.1f} seconds"
    )
    print(f"Saved results to {output_path}")
    print(f"Saved per-target diagnostics to {per_target_output_path}")


if __name__ == "__main__":
    main()