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
    split_strategy: str = "random_holdout",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if str(split_strategy).strip().lower() == "time_ordered":
        test_rows = max(1, int(round(len(X) * float(test_size))))
        if test_rows >= len(X):
            test_rows = max(1, len(X) - 1)
        split_at = len(X) - test_rows
        return (
            X.iloc[:split_at].copy(),
            X.iloc[split_at:].copy(),
            y.iloc[:split_at].copy(),
            y.iloc[split_at:].copy(),
        )
    split_kwargs: dict[str, object] = {"test_size": test_size, "random_state": random_state}
    if task_type == "classification":
        try:
            return train_test_split(X, y, stratify=y, **split_kwargs)
        except Exception:
            return train_test_split(X, y, **split_kwargs)
    return train_test_split(X, y, **split_kwargs)


def train_valid_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    test_size: float,
    validation_size: float,
    random_state: int,
    split_strategy: str = "random_holdout",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    strategy = str(split_strategy).strip().lower()
    validation_size = float(validation_size)
    if validation_size <= 0:
        X_train, X_test, y_train, y_test = train_test_holdout_split(
            X,
            y,
            task_type=task_type,
            test_size=test_size,
            random_state=random_state,
            split_strategy=split_strategy,
        )
        empty_X = X_train.iloc[0:0].copy()
        empty_y = y_train.iloc[0:0].copy()
        return X_train, empty_X, X_test, y_train, empty_y, y_test

    if strategy == "time_ordered":
        total_rows = len(X)
        test_rows = max(1, int(round(total_rows * float(test_size))))
        valid_rows = max(1, int(round(total_rows * validation_size)))
        if test_rows + valid_rows >= total_rows:
            overflow = test_rows + valid_rows - total_rows + 1
            valid_rows = max(1, valid_rows - overflow)
        train_end = total_rows - test_rows - valid_rows
        if train_end <= 0:
            raise ValueError("time_ordered split requires more rows for train/valid/test")
        valid_end = train_end + valid_rows
        return (
            X.iloc[:train_end].copy(),
            X.iloc[train_end:valid_end].copy(),
            X.iloc[valid_end:].copy(),
            y.iloc[:train_end].copy(),
            y.iloc[train_end:valid_end].copy(),
            y.iloc[valid_end:].copy(),
        )

    X_train_full, X_test, y_train_full, y_test = train_test_holdout_split(
        X,
        y,
        task_type=task_type,
        test_size=test_size,
        random_state=random_state,
        split_strategy="random_holdout",
    )
    adjusted_validation = validation_size / max(1e-12, 1.0 - float(test_size))
    adjusted_validation = min(max(adjusted_validation, 0.05), 0.4)
    X_train, X_valid, y_train, y_valid = train_test_holdout_split(
        X_train_full,
        y_train_full,
        task_type=task_type,
        test_size=adjusted_validation,
        random_state=random_state,
        split_strategy="random_holdout",
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test
