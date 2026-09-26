from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def is_pure_day_number_column(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    if not np.all(np.isclose(numeric, np.round(numeric))):
        return False
    return bool(((numeric >= 1) & (numeric <= 31)).all())


def augment_date_features(df: pd.DataFrame, date_col: str | None) -> pd.DataFrame:
    out = df.copy()
    if not date_col or date_col not in out.columns:
        return out
    # REGION: DATE_FEATURE_LOGIC START
    if is_pure_day_number_column(out[date_col]):
        print(f"INFO: skip date expansion for '{date_col}' (pure day-of-month numbers detected).")
        return out
    parsed = pd.to_datetime(out[date_col], errors="coerce")
    if parsed.notna().sum() == 0:
        print(f"INFO: skip date expansion for '{date_col}' (no valid datetime parse).")
        return out
    # REGION: DATE_FEATURE_LOGIC END
    out[f"{date_col}_year"] = parsed.dt.year
    out[f"{date_col}_month"] = parsed.dt.month
    out[f"{date_col}_day"] = parsed.dt.day
    out[f"{date_col}_weekday"] = parsed.dt.dayofweek
    out[f"{date_col}_is_month_start"] = parsed.dt.is_month_start.astype("float")
    out[f"{date_col}_is_month_end"] = parsed.dt.is_month_end.astype("float")
    out = out.drop(columns=[date_col])
    return out


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    # REGION: PREPROCESS_PIPELINE START
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # REGION: PREPROCESS_PIPELINE END
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
