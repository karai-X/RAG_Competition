"""検索インデックス構築: チャンク化 → BM25 + 密ベクトル。

  python -m app.index.build [--no-embed] [--force]

BM25 は全再構築（403ファイル規模で数十秒）。
密ベクトルは embed_cache により **新規チャンクだけ** エンコードする。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time

import numpy as np

from app.config import CONFIG
from app.extract.catalog import load_docs
from app.index.chunk import make_chunks
from app.index.tokenize_ja import tokenize


def build(embed: bool = True, progress=None) -> dict:
    CONFIG.ensure_dirs()
    t0 = time.time()

    chunks: list[dict] = []
    for doc in load_docs():
        chunks.extend(make_chunks(doc, CONFIG.chunk_target_tokens,
                                  CONFIG.chunk_overlap_tokens))
    if progress:
        progress(f"chunks: {len(chunks)} ({time.time()-t0:.1f}s)")

    with open(CONFIG.index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    from rank_bm25 import BM25Okapi
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(CONFIG.index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25,
                     "chunk_ids": [c["chunk_id"] for c in chunks]}, f)
    if progress:
        progress(f"bm25 built ({time.time()-t0:.1f}s)")

    n_new = 0
    if embed:
        from app.index.dense import cache_size, encode_documents
        before = cache_size()
        emb = encode_documents([c["text"] for c in chunks])
        np.save(CONFIG.index_dir / "embeddings.npy", emb.astype(np.float16))
        n_new = cache_size() - before
        if progress:
            progress(f"embeddings: {emb.shape} 新規{n_new}件 "
                     f"({time.time()-t0:.1f}s)")

    meta = {"n_chunks": len(chunks), "n_docs": len({c["doc_id"] for c in chunks}),
            "embed_model": CONFIG.embed_model if embed else None,
            "n_new_embeddings": n_new,
            "elapsed_sec": round(time.time() - t0, 1)}
    (CONFIG.index_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()
    meta = build(embed=not args.no_embed, progress=lambda s: print("  " + s,
                                                                   flush=True))
    print(f"\n{meta}")
    print(f"index -> {CONFIG.index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
