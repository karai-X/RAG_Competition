from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_dataframe(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    # REGION: EDA_SUMMARY_RULES START
    summary: dict[str, Any] = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "target_column": target_col,
        "missing_ratio_top10": df.isna().mean().sort_values(ascending=False).head(10).to_dict(),
    }
    if target_col in df.columns:
        summary["target_dtype"] = str(df[target_col].dtype)
        summary["target_unique"] = int(df[target_col].nunique(dropna=True))
    # REGION: EDA_SUMMARY_RULES END
    return summary
