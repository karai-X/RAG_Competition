"""日本語 cross-encoder によるリランク。

RRF 融合後の上位候補を、クエリとの直接比較で並べ直す。
**検索が本当に要る 39% の問い** で効く。モデルが無い環境では素通しする。
"""
from __future__ import annotations

import threading

from app.config import CONFIG

_LOCK = threading.Lock()
_MODEL: object | None = None
_FAILED = False


def _get():
    global _MODEL, _FAILED
    if _MODEL is None and not _FAILED:
        try:
            import torch
            from sentence_transformers import CrossEncoder
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _MODEL = CrossEncoder(CONFIG.rerank_model, device=dev, max_length=512)
        except Exception:                       # noqa: BLE001 - 無ければ素通し
            _FAILED = True
    return _MODEL


def available() -> bool:
    return _get() is not None


def rerank(query: str, items: list[dict], top_k: int,
           text_key: str = "text") -> list[dict]:
    model = _get()
    if model is None or not items:
        return items[:top_k]
    pairs = [(query, (it.get(text_key) or "")[:2000]) for it in items]
    with _LOCK:
        scores = model.predict(pairs, show_progress_bar=False)
    for it, s in zip(items, scores):
        it["rerank_score"] = float(s)
    return sorted(items, key=lambda x: -x["rerank_score"])[:top_k]
