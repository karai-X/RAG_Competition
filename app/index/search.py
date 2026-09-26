"""ハイブリッド検索: BM25 + 密ベクトル → RRF融合 → リランク。

日本語では片方だけだと必ず落ちる:
  BM25   … 識別子に強い（AG_ratio, max_depth, T04）
  ruri   … 言い換えに強い（「予測に影響が高い特徴量」→ feature importance）
RRF はスコアスケールの違いを無視できるので、両者の融合に適している。
"""
from __future__ import annotations

import json
import pickle
import threading

import numpy as np

from app.config import CONFIG

_RRF_POOL = 200


class SearchIndex:
    def __init__(self) -> None:
        with open(CONFIG.index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]
        with open(CONFIG.index_dir / "bm25.pkl", "rb") as f:
            d = pickle.load(f)
        self.bm25 = d["bm25"]
        emb = CONFIG.index_dir / "embeddings.npy"
        self.emb = np.load(emb).astype(np.float32) if emb.exists() else None
        if self.emb is not None and len(self.emb) != len(self.chunks):
            self.emb = None                     # 不整合なら密検索は無効化
        self._lock = threading.Lock()

    # ---- メタフィルタ ----
    def _mask(self, project, category, filetype, mount) -> np.ndarray:
        mask = np.ones(len(self.chunks), dtype=bool)
        if project:
            p = project.lower()
            mask &= np.array([p in (c["project"] or "").lower()
                              for c in self.chunks])
        if category:
            q = category.lower()
            mask &= np.array([q in (c["category"] or "").lower()
                              for c in self.chunks])
        if filetype:
            fts = {x.strip().lower().lstrip(".") for x in filetype.split(",")}
            mask &= np.array([c["filetype"] in fts for c in self.chunks])
        if mount:
            m = mount.lower()
            mask &= np.array([m in (c["mount"] or "").lower()
                              for c in self.chunks])
        return mask

    def search(self, query: str, k: int | None = None, project=None,
               category=None, filetype=None, mount=None,
               rerank: bool = True) -> list[dict]:
        from app.index.tokenize_ja import tokenize
        k = k or CONFIG.search_top_k
        mask = self._mask(project, category, filetype, mount)
        if not mask.any():
            return []

        with self._lock:                       # sudachi / GPU は非スレッドセーフ
            scores = np.asarray(self.bm25.get_scores(tokenize(query)))
            scores[~mask] = -np.inf
            order = np.argsort(-scores)
            bm_rank = {int(i): r for r, i in enumerate(order[:_RRF_POOL])
                       if scores[i] > 0}

            dn_rank: dict[int, int] = {}
            if self.emb is not None:
                from app.index.dense import encode_query
                sims = self.emb @ encode_query(query)
                sims[~mask] = -np.inf
                dorder = np.argsort(-sims)
                dn_rank = {int(i): r for r, i in enumerate(dorder[:_RRF_POOL])
                           if np.isfinite(sims[i])}

        rrf_k = CONFIG.rrf_k
        fused = []
        for i in set(bm_rank) | set(dn_rank):
            s = 0.0
            if i in bm_rank:
                s += 1.0 / (rrf_k + bm_rank[i])
            if i in dn_rank:
                s += 1.0 / (rrf_k + dn_rank[i])
            fused.append((s, i))
        fused.sort(reverse=True)

        n_cand = max(k, CONFIG.rerank_candidates) if rerank else k
        items = []
        for s, i in fused[:n_cand]:
            c = self.chunks[i]
            items.append({
                "relpath": c["relpath"], "page": c["page_no"],
                "label": c["label"], "project": c["project"],
                "category": c["category"], "filetype": c["filetype"],
                "rrf_score": round(s, 6), "text": c["text"],
            })

        if rerank and items:
            from app.index.rerank import rerank as do_rerank
            items = do_rerank(query, items, k)
        return items[:k]


_INDEX: SearchIndex | None = None
_INIT_LOCK = threading.Lock()


def get_index() -> SearchIndex:
    global _INDEX
    with _INIT_LOCK:
        if _INDEX is None:
            _INDEX = SearchIndex()
    return _INDEX
