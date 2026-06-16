from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from moa.config import PROJECT_ROOT, RANDOM_STATE, TEST_SIZE, TOP_N_TARGETS
from moa.data import load_raw_data, get_feature_groups, make_X_y
from moa.metrics import mean_multilabel_log_loss, make_results_table


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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    baseline_model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", OneVsRestClassifier(baseline_model, n_jobs=-1)),
        ]
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

    results_dir = PROJECT_ROOT / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"baseline_top{len(selected_targets)}.csv"
    results.to_csv(output_path, index=False)

    print(f"Mean multi-label log loss: {mean_loss:.6f}")
    print(results.head(10))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
