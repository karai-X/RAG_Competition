"""テキスト系（md/txt/py/json/toml/yaml/…）と表形式（csv/tsv）と画像ファイル。"""
from __future__ import annotations

import csv as csvmod
import io
from pathlib import Path

from app.corpus.walk import CorpusFile
from app.extract.markup import gfm_table
from app.extract.models import ExtractedDoc, Page

MAX_TEXT_CHARS = 400_000
CSV_FULL_MAX_ROWS = 200
CSV_PREVIEW_ROWS = 25


def _new(cf: CorpusFile, filetype: str) -> ExtractedDoc:
    return ExtractedDoc(doc_id=cf.doc_id, relpath=cf.relpath, mount=cf.mount,
                        project=cf.project, category=cf.category,
                        filetype=filetype, md5=cf.md5)


def _read_text(path: str) -> tuple[str, str]:
    """(本文, 使ったエンコーディング)。BOM付きUTF-8 を最優先で試す。"""
    for enc in ("utf-8-sig", "utf-8", "cp932", "euc-jp", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.read(), enc
        except (UnicodeDecodeError, LookupError):
            continue
        except OSError:
            raise
    with open(path, "rb") as f:
        return f.read().decode("utf-8", "replace"), "utf-8(replace)"


def extract_plain(cf: CorpusFile, assets_dir: Path | None = None) -> ExtractedDoc:
    doc = _new(cf, cf.ext.lstrip("."))
    try:
        text, enc = _read_text(cf.raw_path)
    except OSError as e:
        doc.error = f"read failed: {type(e).__name__}: {e}"
        return doc
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n...[以降省略。全量は run_python で読むこと]"
        doc.meta["truncated"] = True
    lang = {"py": "python", "json": "json", "toml": "toml",
            "yaml": "yaml", "yml": "yaml", "sh": "bash"}.get(doc.filetype)
    body = f"```{lang}\n{text}\n```" if lang else text
    doc.pages = [Page(no=1, text=body)]
    doc.meta.update(encoding=enc, n_chars=len(text),
                    n_lines=text.count("\n") + 1)
    return doc


def extract_tabular(cf: CorpusFile, assets_dir: Path | None = None) -> ExtractedDoc:
    doc = _new(cf, cf.ext.lstrip("."))
    try:
        text, enc = _read_text(cf.raw_path)
    except OSError as e:
        doc.error = f"read failed: {type(e).__name__}: {e}"
        return doc

    delim = "\t" if cf.ext == ".tsv" else ","
    try:
        rows = list(csvmod.reader(io.StringIO(text), delimiter=delim))
    except csvmod.Error as e:
        doc.error = f"csv parse failed: {e}"
        return doc
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        doc.pages = [Page(no=1, text="(空ファイル)")]
        doc.meta.update(encoding=enc, n_rows=0, n_cols=0)
        return doc

    full = len(rows) <= CSV_FULL_MAX_ROWS
    shown = rows if full else rows[:CSV_PREVIEW_ROWS + 1]
    body = gfm_table(shown)
    if not full:
        body += (f"\n\n(全{len(rows)-1}行×{len(rows[0])}列のうち先頭"
                 f"{CSV_PREVIEW_ROWS}行のみ表示。"
                 "全量の集計・条件抽出は run_python でこのファイルを直接読むこと)")
    doc.pages = [Page(no=1, text=body)]
    doc.meta.update(encoding=enc, n_rows=len(rows) - 1, n_cols=len(rows[0]),
                    columns=rows[0][:60], truncated=not full)
    return doc


def extract_image(cf: CorpusFile, assets_dir: Path | None = None) -> ExtractedDoc:
    """画像ファイル本体。中身は ocr 層が md5 キャッシュ付きで読む。"""
    doc = _new(cf, cf.ext.lstrip("."))
    w = h = 0
    try:
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(cf.raw_path) as im:
                w, h = im.size
        except (UnidentifiedImageError, OSError, ValueError) as e:
            doc.warn(f"画像サイズ取得失敗: {type(e).__name__}: {e}")
    except ImportError:
        doc.warn("Pillow 未導入")
    doc.pages = [Page(no=1,
                      text=(f"[画像ファイル: {cf.relpath} ({w}x{h}) "
                            f"md5:{cf.md5[:8]} — read_image で確認可能]"),
                      needs_vision=True, assets=[cf.relpath])]
    doc.meta.update(width=w, height=h)
    return doc


def extract_unknown(cf: CorpusFile, assets_dir: Path | None = None) -> ExtractedDoc:
    doc = _new(cf, cf.ext.lstrip(".") or "bin")
    doc.pages = [Page(no=1, text=f"[未対応形式: {cf.ext} {cf.size}B] {cf.relpath}")]
    return doc
