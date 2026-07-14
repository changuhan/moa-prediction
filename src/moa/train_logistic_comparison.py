from sklearn.model_selection import train_test_split

from moa.config import RANDOM_STATE, REPORT_DIR, TEST_SIZE, TOP_N_TARGETS
from moa.data import load_raw_data, get_feature_groups, make_X_y
from moa.metrics import mean_multilabel_log_loss
from moa.modeling import build_logistic_ovr_pipeline
from moa.prior_baseline import (
    compute_target_priors,
    make_baseline_comparison,
    make_prior_predictions,
)


def evaluate_model(model, X_train, X_valid, y_train, y_valid, target_names):
    model.fit(X_train, y_train)
    y_valid_pred = model.predict_proba(X_valid)

    mean_loss, _ = mean_multilabel_log_loss(
        y_true=y_valid,
        y_pred=y_valid_pred,
        target_names=target_names,
    )

    return mean_loss


def main():
    train_features, train_targets, _, _ = load_raw_data()

    meta_cols, gene_cols, cell_cols = get_feature_groups(train_features)
    X, y, selected_targets, _ = make_X_y(
        train_features=train_features,
        train_targets=train_targets,
        top_n_targets=TOP_N_TARGETS,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    numeric_features = gene_cols + cell_cols
    categorical_features = meta_cols

    comparison_rows = []

    target_priors = compute_target_priors(y_train)
    prior_predictions = make_prior_predictions(
        n_rows=len(y_valid),
        target_priors=target_priors,
    )

    prior_mean_loss, _ = mean_multilabel_log_loss(
        y_true=y_valid,
        y_pred=prior_predictions,
        target_names=selected_targets,
    )

    comparison_rows.append(
        {
            "model": "dummy_prior",
            "mean_log_loss": prior_mean_loss,
            "class_weight": "none",
            "uses_features": False,
            "description": "Predicts each target's training-set positive rate for every validation sample.",
        }
    )

    logistic_configs = [
        {
            "model": "logistic_balanced",
            "class_weight": "balanced",
            "class_weight_label": "balanced",
        },
        {
            "model": "logistic_unbalanced",
            "class_weight": None,
            "class_weight_label": "none",
        },
    ]

    for config in logistic_configs:
        print(f"Training {config['model']}...")

        model = build_logistic_ovr_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            class_weight=config["class_weight"],
        )

        mean_loss = evaluate_model(
            model=model,
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            target_names=selected_targets,
        )

        comparison_rows.append(
            {
                "model": config["model"],
                "mean_log_loss": mean_loss,
                "class_weight": config["class_weight_label"],
                "uses_features": True,
                "description": "Uses metadata, gene-expression, and cell-viability features.",
            }
        )

    comparison = make_baseline_comparison(comparison_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "logistic_class_weight_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print(comparison)
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()