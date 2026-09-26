"""OCR対象の列挙。

3系統ある:
  1. コーパス直置きの画像ファイル（png 等）
  2. Office/notebook の埋め込み画像（extract 層が assets/ に保存済み）
  3. スキャンPDFページ（テキスト層が空 → レンダリングして画像化）

EMF は extract 層で決定的に解析済みなので、**解析に成功したものは対象外**。
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from app.config import CONFIG
from app.corpus.walk import PathResolver
from app.extract.catalog import load_catalog

RASTER_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


@dataclass(frozen=True)
class OcrTarget:
    kind: str            # "file" | "asset" | "pdf_page"
    doc_id: str
    relpath: str         # 由来ドキュメント（表示・トレース用）
    ref: str             # file: raw_path / asset: asset_id / pdf_page: raw_path
    page: int = 0        # pdf_page のときのページ番号
    label: str = ""

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.doc_id}:{self.ref}:{self.page}"


def image_bytes(t: OcrTarget, resolver: PathResolver | None = None) -> bytes:
    """対象の画像バイト列を得る。取得できなければ空を返す。"""
    if t.kind == "file":
        try:
            return Path(t.ref).read_bytes()
        except OSError:
            return b""
    if t.kind == "asset":
        p = CONFIG.assets_dir / t.ref
        try:
            return p.read_bytes()
        except OSError:
            return b""
    if t.kind == "pdf_page":
        from app.extract.pdf import render_page_png
        try:
            return render_page_png(t.ref, t.page, CONFIG.image_max_edge_px)
        except (OSError, ValueError, RuntimeError, IndexError):
            return b""
    return b""


def collect_targets(resolver: PathResolver | None = None) -> list[OcrTarget]:
    resolver = resolver or PathResolver()
    out: list[OcrTarget] = []
    for e in load_catalog():
        rel, doc_id = e["relpath"], e["doc_id"]
        ext = "." + (e["filetype"] or "").lower()

        if ext in RASTER_EXTS:
            raw = resolver.resolve(rel)
            if raw:
                out.append(OcrTarget("file", doc_id, rel, raw, label=rel))
            continue

        for aid in e.get("assets") or []:
            a_ext = "." + aid.rsplit(".", 1)[-1].lower() if "." in aid else ""
            if a_ext not in RASTER_EXTS:
                continue        # EMF/WMF は決定的解析済み or 読めない
            if not (CONFIG.assets_dir / aid).exists():
                continue
            out.append(OcrTarget("asset", doc_id, rel, aid,
                                 label=f"{rel} :: {aid.split('/')[-1]}"))

        if e["filetype"] == "pdf":
            raw = resolver.resolve(rel)
            if not raw:
                continue
            for pno in _scanned_pages(doc_id):
                out.append(OcrTarget("pdf_page", doc_id, rel, raw, page=pno,
                                     label=f"{rel} p{pno}"))
    return out


def _scanned_pages(doc_id: str) -> list[int]:
    from app.extract.models import ExtractedDoc
    try:
        d = ExtractedDoc.load(CONFIG.extracted_dir, doc_id)
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return list(d.meta.get("scanned_pages") or [])


def md5_of(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def image_key(img_bytes: bytes) -> str:
    """画像のキャッシュキー。**画素そのもの**から作る。

    圧縮後のバイト列（PNG）はzlibの実装やバージョンで変わり得るため、
    デコード後の画素からハッシュを作る。

    画素・寸法・モードから作れば、圧縮の実装に依らない。
    画像として開けないものは、バイト列そのものにフォールバックする。
    """
    from PIL import Image, UnidentifiedImageError
    try:
        im = Image.open(io.BytesIO(img_bytes))
        im.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return md5_of(img_bytes)
    h = hashlib.md5()
    h.update(f"{im.mode}|{im.width}x{im.height}|".encode())
    h.update(im.tobytes())
    return h.hexdigest()
