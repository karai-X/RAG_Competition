"""画像バッチ実行（並列・md5キャッシュ・冪等）。

  python -m app.ocr.run [--engine codex] [--workers 3] [--limit N]

**本番ランの時間予算から切り離す**ために事前に埋め切る。
キャッシュヒットは即返り、失敗は空を返して次回リトライされる。
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.config import CONFIG
from app.ocr import cache
from app.ocr.cache import OcrResult
from app.ocr.engines import run_engine
from app.ocr.image_util import normalize
from app.ocr.targets import (OcrTarget, collect_targets, image_bytes,
                             image_key)


@dataclass
class BatchStats:
    total: int = 0
    cached: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)

    def line(self) -> str:
        return (f"total={self.total} cached={self.cached} done={self.done} "
                f"failed={self.failed} skipped={self.skipped} "
                f"{self.elapsed:.1f}s")


def process_one(t: OcrTarget, engine: str, fallback: str | None,
                force: bool = False) -> tuple[OcrTarget, OcrResult | None, bool]:
    """戻り値は (対象, 結果, キャッシュヒットか)。

    キャッシュ結果は latency_sec に **元の実行時間** が入っているので、
    それでヒット判定すると計上を誤る。明示的にフラグで返す。
    """
    raw = image_bytes(t)
    if not raw:
        return t, None, False
    img = normalize(raw, CONFIG.image_max_edge_px)
    md5 = image_key(img)

    if not force:
        hit = cache.get(md5, engine)
        if hit is not None and hit.ok:
            return t, hit, True

    res = run_engine(engine, img, md5, source=t.label)
    if not res.ok and fallback and fallback != engine:
        alt = cache.get(md5, fallback) if not force else None
        if alt is not None and alt.ok:
            return t, alt, True
        alt = run_engine(fallback, img, md5, source=t.label)
        if alt.ok:
            cache.put(alt)
            return t, alt, False
    cache.put(res)
    return t, res, False


def run_batch(engine: str | None = None, workers: int | None = None,
              limit: int | None = None, force: bool = False,
              fallback: str | None = None, targets: list[OcrTarget] | None = None,
              progress=None) -> BatchStats:
    import time
    engine = engine or CONFIG.model_ocr
    workers = workers or CONFIG.codex_concurrency
    CONFIG.ensure_dirs()
    ts = targets if targets is not None else collect_targets()
    if limit:
        ts = ts[:limit]

    st = BatchStats(total=len(ts))
    t0 = time.time()
    lock = threading.Lock()

    def work(t: OcrTarget):
        tt, res, hit = process_one(t, engine, fallback, force)
        with lock:
            if res is None:
                st.skipped += 1
            elif hit:
                st.cached += 1
            elif res.ok:
                st.done += 1
            else:
                st.failed += 1
                if res.error and len(st.errors) < 20:
                    st.errors.append(f"{tt.label}: {res.error[:120]}")
            st.elapsed = time.time() - t0
            if progress:
                progress(st, tt, res)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, ts))

    st.elapsed = time.time() - t0
    return st


def apply_to_docs(engine: str | None = None) -> dict:
    """キャッシュ済みOCR結果を抽出済み文書へ差し込む（冪等）。

    - スキャンPDFページ: そのページ本文へ追記
    - 埋め込み画像 / 画像ファイル: 該当ページへ追記
    差し込み済みマーカーで二重追記を防ぐ。
    """
    from app.extract.catalog import build_catalog
    from app.extract.models import ExtractedDoc

    engine = engine or CONFIG.model_ocr
    updated = 0
    by_doc: dict[str, list[tuple[OcrTarget, OcrResult]]] = {}
    for t in collect_targets():
        raw = image_bytes(t)
        if not raw:
            continue
        md5 = image_key(normalize(raw, CONFIG.image_max_edge_px))
        res = cache.get(md5, engine)
        if res is None or not res.ok:
            continue
        by_doc.setdefault(t.doc_id, []).append((t, res))

    for doc_id, items in by_doc.items():
        try:
            doc = ExtractedDoc.load(CONFIG.extracted_dir, doc_id)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        changed = False
        for t, res in items:
            page = None
            if t.kind == "pdf_page":
                page = next((p for p in doc.pages if p.no == t.page), None)
            elif t.kind == "asset":
                page = next((p for p in doc.pages if t.ref in p.assets), None)
            else:
                page = doc.pages[0] if doc.pages else None
            head = "\n\n" + cache.applied_head(t.kind, t.ref)
            # 判定は画像ごとに行う。ページに MARKER があるかだけを見ると、
            # 同じページの 2枚目以降がここで捨てられる。
            if page is None or (head + "\n") in page.text:
                continue
            page.text = (page.text.replace(
                "[スキャンページ: テキスト層なし。OCR結果は後段で追加]", "").rstrip()
                + head + "\n" + res.as_text())
            changed = True
        if changed:
            doc.meta["ocr_engine"] = engine
            doc.meta["ocr_applied"] = True
            doc.save(CONFIG.extracted_dir)
            updated += 1

    build_catalog()
    return {"docs_updated": updated, "results": sum(len(v) for v in by_doc.values())}
