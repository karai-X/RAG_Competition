"""日本語トークナイザ（BM25用）。SudachiPy mode C + 英数字はそのまま小文字化。

BM25 は **識別子に効く**（`AG_ratio` `max_depth` `T04` など、
密ベクトルが取りこぼす類）。形態素で切りつつ、英数字トークンは壊さない。
"""
from __future__ import annotations

import re
import threading

_LOCK = threading.Lock()
_TOKENIZER = None

_ASCII_TOKEN = re.compile(r"[A-Za-z0-9_.\-]+")
_SKIP_POS = {"補助記号", "空白", "助詞", "助動詞"}
_MAX_SUDACHI_BYTES = 40_000       # Sudachi の入力上限対策


def _get():
    global _TOKENIZER
    if _TOKENIZER is None:
        from sudachipy import dictionary, tokenizer
        _TOKENIZER = (dictionary.Dictionary().create(),
                      tokenizer.Tokenizer.SplitMode.C)
    return _TOKENIZER


def _split_bytes(text: str, max_bytes: int) -> list[str]:
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]
    segs, buf, size = [], [], 0
    for line in text.split("\n"):
        while len(line.encode("utf-8")) > max_bytes:
            segs.append(line[:4000])
            line = line[4000:]
        b = len(line.encode("utf-8")) + 1
        if size + b > max_bytes and buf:
            segs.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += b
    if buf:
        segs.append("\n".join(buf))
    return segs


def tokenize(text: str) -> list[str]:
    """形態素 + 英数字トークン。SudachiPy はスレッドセーフでないのでロックする。"""
    if not text:
        return []
    out: list[str] = []
    with _LOCK:
        tok, mode = _get()
        for seg in _split_bytes(text, _MAX_SUDACHI_BYTES):
            try:
                morphemes = tok.tokenize(seg, mode)
            except Exception:                       # noqa: BLE001
                out.extend(m.lower() for m in _ASCII_TOKEN.findall(seg))
                continue
            for m in morphemes:
                if m.part_of_speech()[0] in _SKIP_POS:
                    continue
                surf = m.surface().strip()
                if surf:
                    out.append(surf.lower())
    # 識別子は分割されても元の形を残す（T04 / AG_ratio / max_depth）
    out.extend(t.lower() for t in _ASCII_TOKEN.findall(text) if len(t) > 1)
    return out
