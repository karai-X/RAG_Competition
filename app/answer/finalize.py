"""回答の最終整形（決定的処理・LLMなし）。

- 空・エラー・低confidence → 「わかりません」
- 改行・タブ・XML断片を除去してCSVに格納
- トークン上限を **tiktoken cl100k_base** で検証
"""
from __future__ import annotations

import functools
import re

from app.config import CONFIG

# ツール呼び出しXML等の混入アーティファクト
_ARTIFACT_RE = re.compile(
    r"</?(?:answer|invoke|parameter|antml[^>]*|function[^>]*|tool[^>]*)>",
    re.IGNORECASE)
# 思考ブロックの混入を除去する。
_THOUGHT_RE = re.compile(r"<\s*/?\s*thought[^>]*>?", re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


@functools.lru_cache(maxsize=1)
def _enc():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc().encode(text or "", disallowed_special=()))


def sanitize(a: str | None) -> str:
    """CSVへ安全に格納できる1行の文字列にする。"""
    if not a:
        return ""
    s = _THOUGHT_RE.sub("", _ARTIFACT_RE.sub("", str(a)))
    s = _FENCE_RE.sub("", s.strip())
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r" {2,}", " ", s)
    return s.strip().strip('"').strip("「」").strip()


def finalize(answer: str | None, confidence: float, stop: str) -> str:
    """最終回答を整形し、採用条件を満たさなければ既定の欠損回答を返す。"""
    if stop != "submitted":
        return CONFIG.missing_answer
    a = sanitize(answer)
    if not a:
        return CONFIG.missing_answer
    if confidence < CONFIG.confidence_floor:
        return CONFIG.missing_answer
    toks = _enc().encode(a, disallowed_special=())
    if len(toks) > CONFIG.answer_max_tokens:
        a = _enc().decode(toks[:CONFIG.answer_max_tokens - 8]).rstrip()
    return a or CONFIG.missing_answer
