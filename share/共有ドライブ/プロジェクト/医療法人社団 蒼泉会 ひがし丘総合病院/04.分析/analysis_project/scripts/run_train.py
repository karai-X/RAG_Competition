from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    manual_root = Path(__file__).resolve().parents[1]
    if str(manual_root) not in sys.path:
        sys.path.insert(0, str(manual_root))

    from src.common import (
        as_bool,
        infer_date_column,
        infer_target_column,
        infer_task_type,
        load_csv_auto,
        load_json,
        resolve_input_path,
        resolve_output_dir,
        save_json,
        to_rel,
    )
    from src.eda import summarize_dataframe
    from src.evaluate import build_metrics_base, evaluate_predictions
    from src.features import augment_date_features, augment_numeric_interactions, select_feature_columns
    from src.infer import run_inference
    from src.modeling import build_pipeline
    from src.preprocess import split_feature_target, train_test_holdout_split

    project_root = manual_root.parents[1] if manual_root.parent.name == "artifacts" else manual_root
    config_path = manual_root / "configs" / "project_config.json"
    rel_config = to_rel(config_path, project_root)

    if not config_path.exists():
        print(f"ERROR: missing config file: {rel_config}", file=sys.stderr)
        raise SystemExit(1)

    cfg = load_json(config_path)
    csv_path = resolve_input_path(str(cfg.get("data_csv", "data/train.csv")), manual_root, project_root)
    output_dir_config = str(cfg.get("output_dir", "artifacts/analysis_outputs"))
    output_dir = resolve_output_dir(output_dir_config, manual_root, project_root)
    random_state = int(cfg.get("random_state", 42))
    test_size = float(cfg.get("test_size", 0.2))
    model_type = str(cfg.get("model_type", "random_forest"))
    model_params = cfg.get("model_params", {})
    if not isinstance(model_params, dict):
        model_params = {}
    use_date_features = as_bool(cfg.get("use_date_features", True), True)
    use_numeric_interactions = as_bool(cfg.get("use_numeric_interactions", False), False)
    rel_csv = to_rel(csv_path, project_root)

    if not csv_path.exists():
        print(f"ERROR: input CSV not found: {rel_csv}", file=sys.stderr)
        raise SystemExit(1)

    df, encoding = load_csv_auto(csv_path)
    target_col = infer_target_column(df, cfg.get("target_column"))
    date_col = infer_date_column(df, target_col, cfg.get("date_column"))
    X_raw, y = split_feature_target(df, target_col)
    effective_date_col = date_col if use_date_features else None
    X = augment_date_features(X_raw, effective_date_col)
    X = augment_numeric_interactions(X, enabled=bool(use_numeric_interactions))
    X, feature_selection = select_feature_columns(X)
    excluded_columns = feature_selection["excluded_columns"]
    if excluded_columns:
        excluded_names = ", ".join(str(item.get("column", "")) for item in excluded_columns)
        print(f"INFO: excluded {len(excluded_columns)} feature columns before modeling: {excluded_names}")

    configured_task_type = str(cfg.get("task_type", "")).strip().lower()
    if configured_task_type in {"classification", "regression"}:
        task_type = configured_task_type
    else:
        task_type = infer_task_type(y)

    model = build_pipeline(
        X,
        task_type=task_type,
        random_state=random_state,
        model_type=model_type,
        model_params=model_params,
    )
    X_train, X_test, y_train, y_test = train_test_holdout_split(
        X,
        y,
        task_type=task_type,
        test_size=test_size,
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    pred = run_inference(model, X_test)
    y_score = None
    if task_type == "classification":
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
                    y_score = proba[:, -1]
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(X_test)
                y_score = decision if getattr(decision, "ndim", 1) == 1 else decision[:, -1]
        except Exception:
            y_score = None

    metrics = build_metrics_base(
        task_type=task_type,
        encoding=encoding,
        target_col=target_col,
        date_col=date_col,
        use_date_features=bool(use_date_features),
        use_numeric_interactions=bool(use_numeric_interactions),
        model_type=model_type,
        model_params=model_params,
        row_count=len(df),
        feature_count=X.shape[1],
        train_rows=len(X_train),
        test_rows=len(X_test),
    )
    metrics.update(evaluate_predictions(task_type, y_test, pred, y_score=y_score))
    metrics["eda_summary"] = summarize_dataframe(df, target_col)
    metrics["feature_selection"] = feature_selection
    metrics["selected_feature_count"] = int(len(feature_selection["selected_columns"]))
    metrics["excluded_feature_count"] = int(len(excluded_columns))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "metrics.json", metrics)

    preview = df.head(200).copy()
    preview.to_csv(output_dir / "train_preview.csv", index=False, encoding="utf-8-sig")

    try:
        output_dir_for_summary = output_dir.relative_to(project_root).as_posix()
    except ValueError:
        output_dir_for_summary = output_dir.as_posix()

    run_summary = {
        "target_column": target_col,
        "date_column": date_col,
        "task_type": task_type,
        "use_date_features": bool(use_date_features),
        "use_numeric_interactions": bool(use_numeric_interactions),
        "model_type": model_type,
        "model_params": model_params,
        "output_dir": output_dir_for_summary,
        "selected_feature_count": int(len(feature_selection["selected_columns"])),
        "excluded_feature_columns": excluded_columns,
        "categorical_unique_limit": int(feature_selection["categorical_unique_limit"]),
    }
    save_json(output_dir / "run_summary.json", run_summary)

    print("Training pipeline finished successfully.")
    print(f"Output directory: {output_dir_for_summary}")


if __name__ == "__main__":
    main()
