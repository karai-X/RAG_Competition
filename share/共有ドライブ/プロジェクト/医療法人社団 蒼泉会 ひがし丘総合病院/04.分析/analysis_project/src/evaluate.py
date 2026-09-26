from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def build_metrics_base(
    *,
    task_type: str,
    encoding: str,
    target_col: str,
    date_col: str | None,
    use_date_features: bool,
    use_numeric_interactions: bool,
    model_type: str,
    model_params: dict[str, Any],
    row_count: int,
    feature_count: int,
    train_rows: int,
    test_rows: int,
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "encoding": encoding,
        "target_column": target_col,
        "date_column": date_col or "",
        "use_date_features": bool(use_date_features),
        "use_numeric_interactions": bool(use_numeric_interactions),
        "model_type": model_type,
        "model_params": model_params,
        "row_count": int(row_count),
        "feature_count": int(feature_count),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
    }


def evaluate_predictions(
    task_type: str,
    y_test: pd.Series,
    pred: Any,
    y_score: pd.Series | None = None,
) -> dict[str, float]:
    # REGION: METRIC_SELECTION START
    if task_type == "regression":
        y_test_num = pd.to_numeric(y_test, errors="coerce")
        pred_num = pd.to_numeric(pd.Series(pred), errors="coerce")
        mask = y_test_num.notna() & pred_num.notna()
        yv = y_test_num[mask]
        pv = pred_num[mask]
        if len(yv) == 0:
            raise ValueError("regression evaluation had no valid numeric rows")
        return {
            "rmse": float(np.sqrt(mean_squared_error(yv, pv))),
            "mae": float(mean_absolute_error(yv, pv)),
            "r2": float(r2_score(yv, pv)),
        }
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
    }
    y_series = pd.Series(y_test).reset_index(drop=True)
    if y_score is not None:
        score_series = pd.Series(y_score).reset_index(drop=True)
        valid = y_series.notna() & score_series.notna()
        if valid.sum() >= 2:
            y_valid = y_series.loc[valid]
            s_valid = score_series.loc[valid]
            uniq = sorted(pd.unique(y_valid))
            if len(uniq) == 2:
                pos_label = uniq[-1]
                y_bin = (y_valid == pos_label).astype(int)
                if y_bin.nunique() == 2:
                    try:
                        metrics["auc_roc"] = float(roc_auc_score(y_bin, s_valid))
                    except Exception:
                        pass
                    try:
                        metrics["brier_score"] = float(brier_score_loss(y_bin, s_valid))
                    except Exception:
                        pass
                    k = max(1, int(math.ceil(len(s_valid) * 0.10)))
                    top_idx = s_valid.sort_values(ascending=False).head(k).index
                    if len(top_idx) > 0:
                        metrics["precision_at_top10pct"] = float(y_bin.loc[top_idx].mean())
    return metrics
    # REGION: METRIC_SELECTION END
