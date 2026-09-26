from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_feature_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"target column not found: {target_col}")
    y = df[target_col]
    X = df.drop(columns=[target_col]).copy()
    if X.empty:
        raise ValueError("no feature columns available after target split")
    return X, y


def train_test_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_kwargs: dict[str, object] = {"test_size": test_size, "random_state": random_state}
    if task_type == "classification":
        try:
            return train_test_split(X, y, stratify=y, **split_kwargs)
        except Exception:
            return train_test_split(X, y, **split_kwargs)
    return train_test_split(X, y, **split_kwargs)
