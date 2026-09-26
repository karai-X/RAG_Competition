from __future__ import annotations

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from src.common import to_float, to_int
from src.features import build_preprocessor


def build_pipeline(
    X: pd.DataFrame,
    task_type: str,
    random_state: int,
    model_type: str,
    model_params: dict,
    scale_numeric_features: bool = False,
) -> Pipeline:
    model_key = (model_type or "random_forest").strip().lower()
    preprocessor = build_preprocessor(
        X,
        sparse_output=model_key != "hist_gradient_boosting",
        scale_numeric_features=scale_numeric_features,
    )
    # REGION: MODEL_SELECTION START
    n_estimators = to_int(model_params.get("n_estimators"), 300)
    max_depth = model_params.get("max_depth")
    if max_depth in ("", "None", "null"):
        max_depth = None
    min_samples_leaf = to_int(model_params.get("min_samples_leaf"), 1)
    learning_rate = to_float(model_params.get("learning_rate"), 0.1)
    alpha = to_float(model_params.get("alpha"), 1.0)
    c_value = to_float(model_params.get("c"), 1.0)
    class_weight = model_params.get("class_weight")
    max_features = model_params.get("max_features")
    l2_regularization = to_float(model_params.get("l2_regularization"), 0.0)
    solver = str(model_params.get("solver", "lbfgs") or "lbfgs")
    penalty = str(model_params.get("penalty", "l2") or "l2")
    l1_ratio_raw = model_params.get("l1_ratio")
    l1_ratio = None if l1_ratio_raw in ("", "None", "null", None) else to_float(l1_ratio_raw, 0.5)
    if max_features in ("", "None", "null"):
        max_features = None
    # REGION: MODEL_SELECTION END

    if task_type == "regression":
        if model_key == "extra_trees":
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
            )
        elif model_key == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=to_int(model_params.get("max_depth"), 3),
                random_state=random_state,
            )
        elif model_key == "hist_gradient_boosting":
            model = HistGradientBoostingRegressor(
                learning_rate=learning_rate,
                max_depth=to_int(model_params.get("max_depth"), 6),
                max_iter=to_int(model_params.get("max_iter"), 300),
                min_samples_leaf=to_int(model_params.get("min_samples_leaf"), 20),
                l2_regularization=l2_regularization,
                random_state=random_state,
            )
        elif model_key == "linear_baseline":
            model = Ridge(alpha=alpha)
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
            )
    else:
        if model_key == "extra_trees":
            model = ExtraTreesClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                max_features=max_features,
            )
        elif model_key == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        elif model_key == "hist_gradient_boosting":
            model = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_depth=to_int(model_params.get("max_depth"), 6),
                max_iter=to_int(model_params.get("max_iter"), 300),
                min_samples_leaf=to_int(model_params.get("min_samples_leaf"), 20),
                l2_regularization=l2_regularization,
                random_state=random_state,
            )
        elif model_key == "linear_baseline":
            if penalty == "elasticnet" and solver != "saga":
                solver = "saga"
            if penalty == "l1" and solver not in {"liblinear", "saga"}:
                solver = "liblinear"
            logistic_kwargs = {
                "max_iter": to_int(model_params.get("max_iter"), 1000),
                "C": c_value,
                "class_weight": class_weight,
                "solver": solver,
            }
            if penalty.strip().lower() in {"none", "null"}:
                logistic_kwargs["penalty"] = None
            else:
                logistic_kwargs["penalty"] = penalty
            if penalty == "elasticnet":
                logistic_kwargs["l1_ratio"] = 0.5 if l1_ratio is None else l1_ratio
            model = LogisticRegression(**logistic_kwargs)
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                max_features=max_features,
            )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
