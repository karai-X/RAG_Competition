from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MAX_CATEGORICAL_UNIQUE = 100
IDENTIFIER_EXACT_NAMES = {'id', 'uuid', 'code'}
IDENTIFIER_SUFFIXES = ('_id', '_uuid', '_code')
IDENTIFIER_PREFIXES = ('id_', 'uuid_', 'code_')
GROUP_RARE_CATEGORIES = False
RARE_CATEGORY_MIN_COUNT = 20
ADD_CATEGORICAL_FREQUENCY_FEATURES = False
CATEGORICAL_FREQUENCY_MIN_UNIQUE = 3
CATEGORICAL_FREQUENCY_MAX_UNIQUE = 200
CYCLE_LENGTH_BY_NAME = {
    "hr": 24,
    "hour": 24,
    "mnth": 12,
    "month": 12,
    "weekday": 7,
    "dayofweek": 7,
    "season": 4,
}
AGE_ORDER = ["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-69", "70-79", "80+"]
EXPERIENCE_ORDER = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30 +"]
EDUCATION_ORDER = [
    "I prefer not to answer",
    "No formal education past high school",
    "Some college/university study without earning a bachelor's degree",
    "Some college/university study without earning a bachelor’s degree",
    "Professional degree",
    "Bachelor's degree",
    "Bachelor’s degree",
    "Master's degree",
    "Master’s degree",
    "Doctoral degree",
]
AGE_MAP = {value: idx for idx, value in enumerate(AGE_ORDER)}
EXPERIENCE_MAP = {value: idx for idx, value in enumerate(EXPERIENCE_ORDER)}
EDUCATION_MAP = {value: idx for idx, value in enumerate(EDUCATION_ORDER)}


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


def augment_cyclical_time_features(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return df
    out = df.copy()
    for col in list(out.columns):
        normalized = str(col).strip().lower()
        period = CYCLE_LENGTH_BY_NAME.get(normalized)
        if period is None and "_" in normalized:
            suffix = normalized.rsplit("_", 1)[-1]
            period = CYCLE_LENGTH_BY_NAME.get(suffix)
        if period is None:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        radians = 2.0 * np.pi * numeric / float(period)
        out[f"{col}_sin"] = np.sin(radians)
        out[f"{col}_cos"] = np.cos(radians)
    return out


def augment_ordered_category_features(df: pd.DataFrame, enabled: bool) -> tuple[pd.DataFrame, list[str]]:
    if not enabled:
        return df, []
    out = df.copy()
    added: list[str] = []
    for source, feature_name, mapping in (
        ("Age", "Age_ord", AGE_MAP),
        ("Experience", "Experience_ord", EXPERIENCE_MAP),
        ("Education", "Education_ord", EDUCATION_MAP),
    ):
        if source not in out.columns:
            continue
        mapped = out[source].astype("string").map(mapping)
        if pd.Series(mapped).notna().sum() == 0:
            continue
        out[feature_name] = pd.to_numeric(mapped, errors="coerce")
        added.append(feature_name)
    if "Age_ord" in out.columns and "Experience_ord" in out.columns:
        out["Age_x_Experience"] = out["Age_ord"] * out["Experience_ord"]
        out["Age_minus_Experience"] = out["Age_ord"] - out["Experience_ord"]
        added.extend(["Age_x_Experience", "Age_minus_Experience"])
    if "Education_ord" in out.columns and "Experience_ord" in out.columns:
        out["Education_x_Experience"] = out["Education_ord"] * out["Experience_ord"]
        added.append("Education_x_Experience")
    return out, added


def is_identifier_like_column(name: str) -> bool:
    normalized = str(name).strip().lower()
    if normalized in IDENTIFIER_EXACT_NAMES:
        return True
    return normalized.startswith(IDENTIFIER_PREFIXES) or normalized.endswith(IDENTIFIER_SUFFIXES)


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            cols.append(col)
    return cols


def group_rare_categories(
    df: pd.DataFrame,
    enabled: bool = GROUP_RARE_CATEGORIES,
    min_count: int = RARE_CATEGORY_MIN_COUNT,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if not enabled:
        return df, []
    out = df.copy()
    grouped_columns: list[dict[str, object]] = []
    for col in _categorical_columns(out):
        if is_identifier_like_column(col):
            continue
        series = out[col].astype("string")
        unique_count = int(series.dropna().nunique())
        if unique_count < 2:
            continue
        counts = series.fillna("__MISSING__").value_counts(dropna=False)
        rare_values = counts[counts < int(min_count)].index.tolist()
        if not rare_values or len(rare_values) >= len(counts):
            continue
        collapsed = series.fillna("__MISSING__")
        collapsed = collapsed.where(~collapsed.isin(rare_values), "__RARE__")
        collapsed = collapsed.where(collapsed != "__MISSING__", pd.NA)
        out[col] = collapsed
        grouped_columns.append(
            {
                "column": col,
                "rare_value_count": int(len(rare_values)),
                "original_unique_count": unique_count,
                "min_count": int(min_count),
            }
        )
    return out, grouped_columns


def add_categorical_frequency_features(
    df: pd.DataFrame,
    enabled: bool = ADD_CATEGORICAL_FREQUENCY_FEATURES,
    min_unique: int = CATEGORICAL_FREQUENCY_MIN_UNIQUE,
    max_unique: int = CATEGORICAL_FREQUENCY_MAX_UNIQUE,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if not enabled:
        return df, []
    out = df.copy()
    added_columns: list[dict[str, object]] = []
    for col in _categorical_columns(out):
        if is_identifier_like_column(col):
            continue
        series = out[col].astype("string")
        unique_count = int(series.dropna().nunique())
        if unique_count < int(min_unique) or unique_count > int(max_unique):
            continue
        freq = series.fillna("__MISSING__").value_counts(dropna=False, normalize=True)
        feature_name = f"{col}__freq"
        out[feature_name] = (
            series.fillna("__MISSING__")
            .map(freq)
            .astype("float")
        )
        added_columns.append(
            {
                "source_column": col,
                "feature_column": feature_name,
                "unique_count": unique_count,
            }
        )
    return out, added_columns


def select_feature_columns(
    X: pd.DataFrame,
    categorical_unique_limit: int = MAX_CATEGORICAL_UNIQUE,
    group_rare_categories_enabled: bool = GROUP_RARE_CATEGORIES,
    rare_category_min_count: int = RARE_CATEGORY_MIN_COUNT,
    add_categorical_frequency_features_enabled: bool = ADD_CATEGORICAL_FREQUENCY_FEATURES,
    categorical_frequency_min_unique: int = CATEGORICAL_FREQUENCY_MIN_UNIQUE,
    categorical_frequency_max_unique: int = CATEGORICAL_FREQUENCY_MAX_UNIQUE,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected_columns: list[str] = []
    excluded_columns: list[dict[str, object]] = []
    grouped_rare_columns: list[dict[str, object]] = []
    added_frequency_columns: list[dict[str, object]] = []

    working = X.copy()
    working, grouped_rare_columns = group_rare_categories(
        working,
        enabled=group_rare_categories_enabled,
        min_count=rare_category_min_count,
    )
    working, added_frequency_columns = add_categorical_frequency_features(
        working,
        enabled=add_categorical_frequency_features_enabled,
        min_unique=categorical_frequency_min_unique,
        max_unique=categorical_frequency_max_unique,
    )

    for col in working.columns:
        series = working[col]
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

    return working[selected_columns].copy(), {
        "selected_columns": selected_columns,
        "excluded_columns": excluded_columns,
        "grouped_rare_columns": grouped_rare_columns,
        "added_frequency_columns": added_frequency_columns,
        "categorical_unique_limit": int(categorical_unique_limit),
        "group_rare_categories": bool(group_rare_categories_enabled),
        "rare_category_min_count": int(rare_category_min_count),
        "use_categorical_frequency_features": bool(add_categorical_frequency_features_enabled),
        "categorical_frequency_min_unique": int(categorical_frequency_min_unique),
        "categorical_frequency_max_unique": int(categorical_frequency_max_unique),
    }


def build_preprocessor(X: pd.DataFrame, sparse_output: bool = True) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    # REGION: PREPROCESS_PIPELINE START
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)),
        ]
    )
    # REGION: PREPROCESS_PIPELINE END
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
