from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _metric_value(task_type: str, metrics: dict) -> float:
    if task_type == "regression":
        return float(metrics.get("rmse", float("inf")))
    primary = metrics.get("f1_macro")
    if primary is None:
        primary = metrics.get("accuracy", float("-inf"))
    return float(primary)


def _is_metric_better(task_type: str, candidate: float, current: float | None, min_delta: float) -> bool:
    if current is None:
        return True
    if task_type == "regression":
        return candidate < (current - float(min_delta))
    return candidate > (current + float(min_delta))


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
    from src.features import (
        augment_cyclical_time_features,
        augment_date_features,
        augment_numeric_interactions,
        select_feature_columns,
    )
    from src.infer import run_inference
    from src.modeling import build_pipeline
    from src.preprocess import split_feature_target, train_valid_test_split

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
    use_cyclical_time_features = as_bool(cfg.get("use_cyclical_time_features", False), False)
    split_strategy = str(cfg.get("split_strategy", "random_holdout")).strip() or "random_holdout"
    transform_target = str(cfg.get("transform_target", "none")).strip().lower() or "none"
    validation_size = float(cfg.get("validation_size", 0.1))
    early_stopping = as_bool(cfg.get("early_stopping", False), False)
    early_stopping_patience = int(cfg.get("early_stopping_patience", 2))
    early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    rel_csv = to_rel(csv_path, project_root)

    if not csv_path.exists():
        print(f"ERROR: input CSV not found: {rel_csv}", file=sys.stderr)
        raise SystemExit(1)

    df, encoding = load_csv_auto(csv_path)
    target_col = infer_target_column(df, cfg.get("target_column"))
    date_col = infer_date_column(df, target_col, cfg.get("date_column"))
    working_df = df.copy().reset_index(drop=True)
    split_ordering_basis = "random_holdout"
    if split_strategy == "time_ordered":
        split_ordering_basis = "input_order"
        if date_col and date_col in working_df.columns:
            parsed_dates = pd.to_datetime(working_df[date_col], errors="coerce")
            if parsed_dates.notna().sum() >= max(8, int(len(working_df) * 0.5)):
                working_df = (
                    working_df.assign(__sort_ts=parsed_dates)
                    .sort_values("__sort_ts", kind="stable")
                    .drop(columns=["__sort_ts"])
                    .reset_index(drop=True)
                )
                split_ordering_basis = f"date_column:{date_col}"
            else:
                print(
                    f"INFO: date column '{date_col}' could not be parsed reliably; keep input row order for time_ordered split."
                )
        else:
            print("INFO: time_ordered split requested without explicit date column; keep input row order.")

    X_raw, y = split_feature_target(working_df, target_col)
    effective_date_col = date_col if use_date_features else None
    X = augment_date_features(X_raw, effective_date_col)
    X = augment_cyclical_time_features(X, enabled=bool(use_cyclical_time_features))
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
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
        X,
        y,
        task_type=task_type,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
        split_strategy=split_strategy,
    )
    y_train_model = y_train
    y_train_eval = y_train
    y_valid_eval = y_valid
    y_test_eval = y_test
    if task_type == "regression" and transform_target == "log1p":
        y_train_numeric = pd.to_numeric(y_train, errors="coerce")
        y_valid_numeric = pd.to_numeric(y_valid, errors="coerce")
        y_test_numeric = pd.to_numeric(y_test, errors="coerce")
        if (
            y_train_numeric.notna().all()
            and y_valid_numeric.notna().all()
            and y_test_numeric.notna().all()
            and float(y_train_numeric.min()) >= 0.0
        ):
            y_train_model = np.log1p(y_train_numeric)
            y_train_eval = y_train_numeric
            y_valid_eval = y_valid_numeric
            y_test_eval = y_test_numeric
        else:
            transform_target = "none"
    best_validation_metrics: dict | None = None
    early_stopping_summary: dict | None = None
    model_key = model_type.strip().lower()
    supports_manual_early_stopping = model_key in {"gradient_boosting", "hist_gradient_boosting"}
    if early_stopping and len(X_valid) > 0 and supports_manual_early_stopping:
        iteration_key = "max_iter" if model_key == "hist_gradient_boosting" else "n_estimators"
        configured_limit = int(model_params.get(iteration_key, 300))
        base_step = max(25, configured_limit // 6)
        candidate_iterations = sorted(set([base_step, base_step * 2, base_step * 3, configured_limit]))
        best_iteration = candidate_iterations[0]
        best_validation_score: float | None = None
        best_model = None
        patience_left = early_stopping_patience
        history: list[dict[str, float | int]] = []
        for iter_count in candidate_iterations:
            tuned_params = dict(model_params)
            tuned_params[iteration_key] = int(iter_count)
            candidate_model = build_pipeline(
                X_train,
                task_type=task_type,
                random_state=random_state,
                model_type=model_type,
                model_params=tuned_params,
            )
            candidate_model.fit(X_train, y_train_model)
            valid_pred = run_inference(candidate_model, X_valid)
            if task_type == "regression" and transform_target == "log1p":
                valid_pred = np.expm1(pd.to_numeric(pd.Series(valid_pred), errors="coerce"))
            valid_metrics = evaluate_predictions(task_type, y_valid_eval, valid_pred, y_score=None)
            metric_value = _metric_value(task_type, valid_metrics)
            history.append({"iteration": int(iter_count), "metric": float(metric_value)})
            if _is_metric_better(task_type, metric_value, best_validation_score, early_stopping_min_delta):
                best_validation_score = metric_value
                best_validation_metrics = valid_metrics
                best_iteration = int(iter_count)
                best_model = candidate_model
                patience_left = early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_model is None:
            best_model = build_pipeline(
                X_train,
                task_type=task_type,
                random_state=random_state,
                model_type=model_type,
                model_params=model_params,
            )
            best_model.fit(X_train, y_train_model)
        final_params = dict(model_params)
        final_params[iteration_key] = int(best_iteration)
        model = build_pipeline(
            pd.concat([X_train, X_valid], axis=0),
            task_type=task_type,
            random_state=random_state,
            model_type=model_type,
            model_params=final_params,
        )
        y_train_final = pd.concat([pd.Series(y_train_model), pd.Series(np.log1p(pd.to_numeric(y_valid_eval, errors="coerce"))) if task_type == "regression" and transform_target == "log1p" else pd.Series(y_valid)], axis=0)
        model.fit(pd.concat([X_train, X_valid], axis=0), y_train_final)
        model_params = final_params
        early_stopping_summary = {
            "enabled": True,
            "mode": "time_ordered_validation" if split_strategy == "time_ordered" else "holdout_validation",
            "iteration_key": iteration_key,
            "best_iteration": int(best_iteration),
            "patience": int(early_stopping_patience),
            "min_delta": float(early_stopping_min_delta),
            "validation_history": history,
        }
    else:
        model.fit(X_train, y_train_model)
        if early_stopping and len(X_valid) > 0 and not supports_manual_early_stopping:
            early_stopping_summary = {
                "enabled": False,
                "reason": f"manual early stopping is not implemented for model_type={model_type}",
            }
    pred = run_inference(model, X_test)
    if task_type == "regression" and transform_target == "log1p":
        pred = np.expm1(pd.to_numeric(pd.Series(pred), errors="coerce"))
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
        use_cyclical_time_features=bool(use_cyclical_time_features),
        model_type=model_type,
        model_params=model_params,
        split_strategy=split_strategy,
        transform_target=transform_target,
        validation_size=validation_size,
        early_stopping=bool(early_stopping),
        early_stopping_patience=early_stopping_patience,
        row_count=len(df),
        feature_count=X.shape[1],
        train_rows=len(X_train),
        validation_rows=len(X_valid),
        test_rows=len(X_test),
    )
    metrics["split_ordering_basis"] = split_ordering_basis
    metrics.update(evaluate_predictions(task_type, y_test_eval, pred, y_score=y_score))
    train_pred = run_inference(model, X_train)
    if task_type == "regression" and transform_target == "log1p":
        train_pred = np.expm1(pd.to_numeric(pd.Series(train_pred), errors="coerce"))
    metrics["train_metrics"] = evaluate_predictions(task_type, y_train_eval, train_pred, y_score=None)
    if len(X_valid) > 0:
        valid_pred = run_inference(model, X_valid)
        if task_type == "regression" and transform_target == "log1p":
            valid_pred = np.expm1(pd.to_numeric(pd.Series(valid_pred), errors="coerce"))
        metrics["validation_metrics"] = evaluate_predictions(task_type, y_valid_eval, valid_pred, y_score=None)
    else:
        metrics["validation_metrics"] = {}
    if early_stopping_summary is not None:
        metrics["early_stopping"] = early_stopping_summary
    metrics["eda_summary"] = summarize_dataframe(working_df, target_col)
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
        "use_cyclical_time_features": bool(use_cyclical_time_features),
        "model_type": model_type,
        "model_params": model_params,
        "split_strategy": split_strategy,
        "split_ordering_basis": split_ordering_basis,
        "transform_target": transform_target,
        "validation_size": validation_size,
        "early_stopping": bool(early_stopping),
        "early_stopping_patience": int(early_stopping_patience),
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
