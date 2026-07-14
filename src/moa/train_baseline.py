from sklearn.model_selection import train_test_split

import joblib
from moa.config import PROJECT_ROOT, RANDOM_STATE, TEST_SIZE, TOP_N_TARGETS, MODEL_DIR
from moa.data import load_raw_data, get_feature_groups, make_X_y
from moa.metrics import mean_multilabel_log_loss, make_results_table
from moa.modeling import build_logistic_ovr_pipeline

def main():
    train_features, train_targets, _, _ = load_raw_data()

    meta_cols, gene_cols, cell_cols = get_feature_groups(train_features)
    X, y, selected_targets, target_counts = make_X_y(
        train_features,
        train_targets,
        top_n_targets=TOP_N_TARGETS,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    numeric_features = gene_cols + cell_cols
    categorical_features = meta_cols

    model = build_logistic_ovr_pipeline(
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    class_weight="balanced",
    )

    print("Training the baseline model...")
    model.fit(X_train, y_train)

    print("Evaluating the baseline model...")

    y_valid_pred = model.predict_proba(X_valid)

    mean_loss, losses = mean_multilabel_log_loss(
        y_true=y_valid,
        y_pred=y_valid_pred,
        target_names=selected_targets,
    )

    results = make_results_table(
        target_names=selected_targets,
        target_counts=target_counts,
        losses=losses,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "baseline_model.joblib"
    
    artifact = {
        "model": model,
        "target_names": selected_targets,
    }

    joblib.dump(artifact, model_path)
    print(f"Saved baseline model to {model_path}")

    results_dir = PROJECT_ROOT / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"baseline_top{len(selected_targets)}.csv"
    results.to_csv(output_path, index=False)

    print(f"Mean multi-label log loss: {mean_loss:.6f}")
    print(results.head(10))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
