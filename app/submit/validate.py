"""回答CSVの形式検証。

`evaluation/src/validator.py` の実装に合わせてある:
  - 各行を `,` で split して要素数 > 1（空行・空欄・改行混入の検出）
  - `pd.read_csv(header=None, index_col=0)` で列数が1（カンマを含む回答の
    クォート漏れを検出）
  - index 集合が期待どおり・欠損なし
  - 各回答が設定されたトークン上限以下
"""
from __future__ import annotations

import codecs
import csv
from dataclasses import dataclass, field
from pathlib import Path

from app.answer.finalize import count_tokens
from app.config import CONFIG


@dataclass
class ValidationReport:
    path: str = ""
    n_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    max_tokens: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"検証: {self.path}",
                 f"  行数: {self.n_rows}  最大トークン: {self.max_tokens}"]
        for e in self.errors:
            lines.append(f"  NG {e}")
        for w in self.warnings:
            lines.append(f"  -- {w}")
        lines.append("  PASS" if self.ok else f"  FAIL ({len(self.errors)}件)")
        return "\n".join(lines)


def validate(path: str | Path, expected_indices: set[int] | None = None,
             max_tokens: int | None = None) -> ValidationReport:
    path = Path(path)
    max_tokens = max_tokens or CONFIG.answer_max_tokens
    r = ValidationReport(path=str(path))

    if not path.exists():
        r.errors.append("ファイルが存在しない")
        return r
    if path.suffix != ".csv":
        r.errors.append(f"拡張子が .csv でない: {path.suffix}")

    # --- 行レベルの検証 ---
    raw_lines: list[str] = []
    with codecs.open(str(path), "r", "utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            raw_lines.append(line)
            sp = line.rstrip("\r\n").split(",")
            if len(sp) <= 1 or (len(sp) == 2 and len(sp[-1]) == 0):
                r.errors.append(f"{i}行目: 区切り不正または空欄 {line[:60]!r}")
    r.n_rows = len(raw_lines)

    # --- CSVパースの検証 ---
    rows: list[list[str]] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if row:
                rows.append(row)
    if not rows:
        r.errors.append("空ファイル")
        return r

    ncols = {len(x) for x in rows}
    if ncols != {2}:
        r.errors.append(f"列数が2でない行がある: {sorted(ncols)}"
                        "（回答内のカンマがクォートされていない可能性）")

    seen: set[int] = set()
    for i, row in enumerate(rows, 1):
        if len(row) < 2:
            continue
        key, ans = row[0].strip(), row[1]
        if not key.lstrip("-").isdigit():
            r.errors.append(f"{i}行目: index が整数でない {key!r}")
            continue
        idx = int(key)
        if idx in seen:
            r.errors.append(f"index {idx} が重複")
        seen.add(idx)
        if not ans.strip():
            r.errors.append(f"index {idx}: 回答が空")
        if "\n" in ans or "\r" in ans:
            r.errors.append(f"index {idx}: 回答に改行が含まれる")
        if "\t" in ans:
            r.warnings.append(f"index {idx}: 回答にタブが含まれる")
        n = count_tokens(ans)
        r.max_tokens = max(r.max_tokens, n)
        if n > max_tokens:
            r.errors.append(f"index {idx}: {n} トークン（上限 {max_tokens}）")

    if expected_indices is not None:
        missing = sorted(expected_indices - seen)
        extra = sorted(seen - expected_indices)
        if missing:
            r.errors.append(f"index が欠けている: {missing[:20]}")
        if extra:
            r.errors.append(f"余分な index: {extra[:20]}")
        if len(rows) != len(expected_indices):
            r.errors.append(f"物理行数が {len(rows)}（期待 {len(expected_indices)}）")

    # pandasでも同じ列構造として読めることを確認する。
    try:
        import pandas as pd
        df = pd.read_csv(path, header=None, index_col=0, encoding="utf-8")
        if len(df.columns) != 1:
            r.errors.append(f"pandas 読み込みで列数が {len(df.columns)}（期待 1）")
        if df.isnull().any().any():
            r.errors.append("pandas 読み込みで欠損値あり")
    except Exception as e:                        # noqa: BLE001
        r.errors.append(f"pandas 読み込み失敗: {type(e).__name__}: {e}")

    return r
