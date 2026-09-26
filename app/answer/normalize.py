"""比較用の正規化。

`143,000円` と `143000`、全角半角、単位の有無などの表記差を吸収する。
"""
from __future__ import annotations

import re
import unicodedata

_SPACE = re.compile(r"[\s　]+")
_PUNCT = {"：": ":", "，": ",", "、": ",", "．": ".", "％": "%",
          "（": "(", "）": ")", "〜": "~", "－": "-", "―": "-", "ー": "-"}
# \b は CJK の直前で成立しない（"143,000円" のカンマが残り、金額比較が全滅する）。
_NUM_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_TRAILING_ZERO = re.compile(r"(?<=\.\d)0+\b|\.0+\b")


def normalize(text: str | None) -> str:
    """表記ゆれを吸収した比較用の文字列。"""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip()
    for a, b in _PUNCT.items():
        s = s.replace(a, b)
    s = _NUM_COMMA.sub("", s)            # 3桁カンマを除去
    s = _SPACE.sub("", s)
    return s.lower()


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def numbers(text: str | None) -> list[float]:
    """文字列中の数値。3桁カンマは除去してから拾う。"""
    if not text:
        return []
    s = unicodedata.normalize("NFKC", str(text))
    s = _NUM_COMMA.sub("", s)
    out = []
    for m in NUM_RE.finditer(s):
        try:
            out.append(float(m.group()))
        except ValueError:
            continue
    return out


def same_answer(a: str | None, b: str | None) -> bool:
    """2つの回答が実質同じか。文字列一致 → 数値集合一致 の順に見る。"""
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    # 単位や接尾語の違いだけなら数値が一致する（「5」と「5ページ」）
    xa, xb = numbers(a), numbers(b)
    if xa and xa == xb:
        # 数値以外の部分が包含関係にあるときのみ同一とみなす
        ra = NUM_RE.sub("", na)
        rb = NUM_RE.sub("", nb)
        return ra in rb or rb in ra
    return False


def elements(text: str | None) -> list[str]:
    """読点・カンマ区切りの列挙を要素へ分解する（順序は保持）。"""
    if not text:
        return []
    parts = re.split(r"[、,]", str(text))
    return [p for p in (x.strip() for x in parts) if p]


def same_elements(a: str | None, b: str | None) -> bool:
    """列挙として同じか（順序は問わないが過不足は不可）。"""
    ea = {normalize(x) for x in elements(a)}
    eb = {normalize(x) for x in elements(b)}
    return bool(ea) and ea == eb
