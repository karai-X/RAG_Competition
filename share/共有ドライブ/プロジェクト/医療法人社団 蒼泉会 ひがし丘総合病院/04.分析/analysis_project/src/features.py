from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MAX_CATEGORICAL_UNIQUE = 50
IDENTIFIER_EXACT_NAMES = {'code', 'id', 'uuid'}
IDENTIFIER_SUFFIXES = ('_id', '_uuid', '_code')
IDENTIFIER_PREFIXES = ('id_', 'uuid_', 'code_')


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


def augment_numeric_interactions(
    df: pd.DataFrame,
    enabled: bool,
    max_numeric_features: int = 4,
    max_interaction_pairs: int = 6,
) -> pd.DataFrame:
    if not enabled:
        return df
    out = df.copy()
    numeric_cols = [c for c in out.select_dtypes(include=["number", "bool"]).columns.tolist() if c in out.columns]
    if len(numeric_cols) < 2:
        return out

    ranked = sorted(
        numeric_cols,
        key=lambda c: int(pd.to_numeric(out[c], errors="coerce").notna().sum()),
        reverse=True,
    )[:max_numeric_features]
    pair_count = 0
    for i, left in enumerate(ranked):
        for right in ranked[i + 1 :]:
            feature_name = f"{left}__x__{right}"
            out[feature_name] = pd.to_numeric(out[left], errors="coerce") * pd.to_numeric(out[right], errors="coerce")
            pair_count += 1
            if pair_count >= max_interaction_pairs:
                return out
    return out


def is_identifier_like_column(name: str) -> bool:
    normalized = str(name).strip().lower()
    if normalized in IDENTIFIER_EXACT_NAMES:
        return True
    return normalized.startswith(IDENTIFIER_PREFIXES) or normalized.endswith(IDENTIFIER_SUFFIXES)


def select_feature_columns(
    X: pd.DataFrame, categorical_unique_limit: int = MAX_CATEGORICAL_UNIQUE
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected_columns: list[str] = []
    excluded_columns: list[dict[str, object]] = []

    for col in X.columns:
        series = X[col]
        if is_identifier_like_column(col):
            excluded_columns.append(
                {
                    "column": col,
                    "reason": "identifier_like_name",
                }
            )
            continue

        is_categorical = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        )
        if is_categorical:
            unique_count = int(series.dropna().nunique())
            if unique_count >= categorical_unique_limit:
                excluded_columns.append(
                    {
                        "column": col,
                        "reason": "high_cardinality_categorical",
                        "unique_count": unique_count,
                        "threshold": categorical_unique_limit,
                    }
                )
                continue

        selected_columns.append(col)

    if not selected_columns:
        raise ValueError("no feature columns available after feature selection")

    return X[selected_columns].copy(), {
        "selected_columns": selected_columns,
        "excluded_columns": excluded_columns,
        "categorical_unique_limit": int(categorical_unique_limit),
    }


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
