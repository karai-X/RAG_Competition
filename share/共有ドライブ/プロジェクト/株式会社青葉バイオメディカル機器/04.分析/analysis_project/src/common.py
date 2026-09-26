from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "shift_jis", "euc_jp")
TARGET_HINTS = ("target", "label", "y", "sales", "demand", "count", "revenue", "profit", "売上", "販売", "需要", "個数")
DATE_HINTS = ("date", "time", "day", "timestamp", "日付", "日時", "日")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def to_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def to_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def as_bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
    return default


def load_csv_auto(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    sep = "	" if path.suffix.lower() == ".tsv" else ","
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, sep=sep), enc
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to decode CSV with supported encodings")


def infer_target_column(df: pd.DataFrame, configured: str | None) -> str:
    if configured and configured in df.columns:
        return configured
    lower_map = {c: c.lower() for c in df.columns}
    for col, low in lower_map.items():
        if low == "y":
            return col
    for hint in TARGET_HINTS:
        for col, low in lower_map.items():
            if low == hint:
                return col
    for hint in TARGET_HINTS:
        for col, low in lower_map.items():
            if hint != "y" and hint in low:
                return col
    return df.columns[-1]


def infer_date_column(df: pd.DataFrame, target_col: str, configured: str | None) -> str | None:
    def _looks_like_date_text(value: object) -> bool:
        raw = str(value or "").strip()
        if not raw:
            return False
        normalized = raw.replace(".", "-").replace("/", "-")
        if re.fullmatch(r"\d{8}", raw):
            normalized = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
        for fmt in (
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m",
            "%m-%d-%Y",
            "%m-%d-%Y %H:%M:%S",
            "%d-%m-%Y",
            "%d-%m-%Y %H:%M:%S",
        ):
            try:
                dt.datetime.strptime(normalized, fmt)
                return True
            except ValueError:
                continue
        return False

    def _series_looks_date_like(series: pd.Series) -> bool:
        values = [str(v).strip() for v in series.dropna().astype("string").head(20).tolist() if str(v).strip()]
        if len(values) < 2:
            return False
        parsed = sum(1 for value in values if _looks_like_date_text(value))
        return parsed >= max(2, int(len(values) * 0.6))

    if configured and configured in df.columns and configured != target_col and _series_looks_date_like(df[configured]):
        return configured

    lower_map = {c: c.lower() for c in df.columns if c != target_col}
    for hint in DATE_HINTS:
        for col, low in lower_map.items():
            if hint in low and _series_looks_date_like(df[col]):
                return col
    return None


def infer_task_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        uniq = y.dropna().nunique()
        if uniq > 20:
            return "regression"
    return "classification"


def resolve_input_path(path_value: str, manual_root: Path, project_root: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    candidate_manual = manual_root / p
    if candidate_manual.exists():
        return candidate_manual
    candidate_project = project_root / p
    if candidate_project.exists():
        return candidate_project
    return candidate_manual


def resolve_output_dir(path_value: str, manual_root: Path, project_root: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    if manual_root.parent.name == "artifacts":
        return project_root / p
    return manual_root / p
