"""密ベクトル検索（ruri-v3, ローカルGPU）+ 増分エンコード。

**embed_cache に sha1(chunk_text) → ベクトル を保存し、新規チャンクだけエンコードする**。
フォルダ追加のたびに全件エンコードし直すのを避けるため。
GPUの用途は埋め込みとリランカーのみ（推論には使わない）。
"""
from __future__ import annotations

import hashlib
import threading

import numpy as np

from app.config import CONFIG

_LOCK = threading.Lock()
_MODEL = None
_CACHE_FILE = "vectors.npz"


def _get_model():
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = SentenceTransformer(CONFIG.embed_model, device=dev,
                                     trust_remote_code=True)
        _MODEL.max_seq_length = 512          # 長大チャンクでのOOM防止
    return _MODEL


def _prefixes(name: str) -> tuple[str, str]:
    """ruri は「検索クエリ:」「検索文書:」の1+プレフィックス方式。"""
    n = name.lower()
    if "ruri" in n:
        return "検索クエリ: ", "検索文書: "
    if "e5" in n:
        return "query: ", "passage: "
    return "", ""


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, np.ndarray]:
    p = CONFIG.embed_cache_dir / _CACHE_FILE
    if not p.exists():
        return {}
    try:
        with np.load(p) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict[str, np.ndarray]) -> None:
    CONFIG.embed_cache_dir.mkdir(parents=True, exist_ok=True)
    p = CONFIG.embed_cache_dir / _CACHE_FILE
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, **cache)
    tmp.replace(p)


def encode_documents(texts: list[str], batch_size: int = 16,
                     progress: bool = True) -> np.ndarray:
    """キャッシュを使って必要な分だけエンコードする。"""
    _, doc_prefix = _prefixes(CONFIG.embed_model)
    cache = _load_cache()
    keys = [text_key(t) for t in texts]
    todo = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    # 同一内容の重複を1回に畳む
    uniq: dict[str, str] = {}
    for k, t in todo:
        uniq.setdefault(k, t)

    if uniq:
        model = _get_model()
        ks = list(uniq)
        vecs = model.encode([doc_prefix + uniq[k][:4000] for k in ks],
                            batch_size=batch_size, show_progress_bar=progress,
                            normalize_embeddings=True)
        for k, v in zip(ks, np.asarray(vecs, dtype=np.float32)):
            cache[k] = v
        _save_cache(cache)

    dim = len(next(iter(cache.values()))) if cache else 0
    return np.vstack([cache.get(k, np.zeros(dim, dtype=np.float32))
                      for k in keys]).astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    q_prefix, _ = _prefixes(CONFIG.embed_model)
    with _LOCK:                              # GPUエンコードはスレッドセーフでない
        v = _get_model().encode([q_prefix + query], normalize_embeddings=True)
    return np.asarray(v[0], dtype=np.float32)


def cache_size() -> int:
    return len(_load_cache())
