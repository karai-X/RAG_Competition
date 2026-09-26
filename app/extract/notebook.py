"""ipynb 抽出: セル単位（ソース + 出力 + 画像）。

notebook は「分析の根拠」そのものなので、出力（stdout / 実行結果 / 図）まで拾う。
画像は **出力側も本文側も** asset 化して read_image で読めるようにする。

Markdownセルのbase64画像もasset化し、巨大な文字列が抽出本文へ流れないようにする。
`read` の上限では何も読めず、索引・grep・チャンクも汚染される。
さらに OCR の収集対象外でもあったため、**その画像だけ中身が誰にも見えなかった**。
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from pathlib import Path

from app.corpus.walk import CorpusFile
from app.extract.models import ExtractedDoc, Page

MAX_OUTPUT_CHARS = 4000

# markdown 本文への直書き `![alt](data:image/png;base64,XXXX)`
_DATA_URI = re.compile(r"data:image/(png|jpe?g|gif|webp);base64,([A-Za-z0-9+/=\s]+)")
# Jupyter 標準の添付参照 `![alt](attachment:name.png)`
_ATTACH_REF = re.compile(r"attachment:([^)\s]+)")


def _save_asset(blob: bytes, aid: str, assets_dir: Path | None) -> str:
    """asset を書き出して md5 を返す。出力画像・本文画像で共通に使う。"""
    if assets_dir is not None:
        path = assets_dir / aid
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    return hashlib.md5(blob).hexdigest()


def _text(x) -> str:
    if isinstance(x, list):
        return "".join(x)
    return x if isinstance(x, str) else str(x)


def extract_ipynb(cf: CorpusFile, assets_dir: Path | None = None) -> ExtractedDoc:
    doc = ExtractedDoc(doc_id=cf.doc_id, relpath=cf.relpath, mount=cf.mount,
                       project=cf.project, category=cf.category,
                       filetype="ipynb", md5=cf.md5)
    try:
        with open(cf.raw_path, encoding="utf-8-sig") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        doc.error = f"ipynb open failed: {type(e).__name__}: {e}"
        return doc

    n_images = 0
    for i, cell in enumerate(nb.get("cells", []), start=1):
        ctype = cell.get("cell_type", "")
        src = _text(cell.get("source", "")).rstrip()
        assets: list[str] = []

        # 本文に埋め込まれた画像を asset へ逃がし、参照だけを本文に残す。
        # 残さないと base64 がそのまま本文になり、read も grep も効かなくなる。
        def _swap(m, _cell=i):
            nonlocal n_images
            mime, b64 = m.group(1), m.group(2)
            try:
                blob = base64.b64decode(re.sub(r"\s", "", b64))
            except (binascii.Error, ValueError):
                doc.warn(f"cell{_cell}: 本文の埋め込み画像のデコード失敗")
                return "[画像: デコード失敗]"
            n_images += 1
            ext = ".png" if mime == "png" else (".jpg" if "jp" in mime else "." + mime)
            aid = f"{cf.doc_id}/cell{_cell}_src{n_images}{ext}"
            md5 = _save_asset(blob, aid, assets_dir)
            assets.append(aid)
            return f"[画像: {aid} md5:{md5[:8]} — read_image で確認可能]"

        src = _DATA_URI.sub(_swap, src)

        # Jupyter 標準の attachments も同じ扱いにする
        for name, payload in (cell.get("attachments") or {}).items():
            for mime, b64 in (payload or {}).items():
                if not mime.startswith("image/"):
                    continue
                try:
                    blob = base64.b64decode(re.sub(r"\s", "", _text(b64)))
                except (binascii.Error, ValueError):
                    doc.warn(f"cell{i}: 添付画像のデコード失敗: {name}")
                    continue
                n_images += 1
                ext = Path(name).suffix or (".png" if "png" in mime else ".jpg")
                aid = f"{cf.doc_id}/cell{i}_att{n_images}{ext}"
                md5 = _save_asset(blob, aid, assets_dir)
                assets.append(aid)
                src = src.replace(f"attachment:{name}",
                                  f"{aid} md5:{md5[:8]} — read_image で確認可能")

        parts = [f"## cell{i} ({ctype})"]
        if src:
            parts.append(f"```python\n{src}\n```" if ctype == "code" else src)
        for out in cell.get("outputs", []) or []:
            otype = out.get("output_type", "")
            if otype == "stream":
                t = _text(out.get("text", ""))[:MAX_OUTPUT_CHARS]
                if t.strip():
                    parts.append(f"[出力]\n```\n{t.rstrip()}\n```")
            elif otype in ("execute_result", "display_data"):
                data = out.get("data", {}) or {}
                if "text/plain" in data:
                    t = _text(data["text/plain"])[:MAX_OUTPUT_CHARS]
                    if t.strip():
                        parts.append(f"[実行結果]\n```\n{t.rstrip()}\n```")
                if "text/html" in data:
                    h = _text(data["text/html"])
                    if "<table" in h:
                        parts.append("[出力に表(HTML)あり]")
                for mime in ("image/png", "image/jpeg"):
                    if mime not in data:
                        continue
                    try:
                        blob = base64.b64decode(_text(data[mime]))
                    except (binascii.Error, ValueError):
                        doc.warn(f"cell{i}: 出力画像のデコード失敗")
                        continue
                    n_images += 1
                    ext = ".png" if mime.endswith("png") else ".jpg"
                    aid = f"{cf.doc_id}/cell{i}_img{n_images}{ext}"
                    if assets_dir is not None:
                        p = assets_dir / aid
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(blob)
                    assets.append(aid)
                    md5 = hashlib.md5(blob).hexdigest()
                    parts.append(f"[出力画像: {aid} md5:{md5[:8]} "
                                 "— read_image で確認可能]")
            elif otype == "error":
                tb = "\n".join(out.get("traceback", []))[:1000]
                parts.append(f"[エラー出力]\n```\n{tb}\n```")

        doc.pages.append(Page(no=i, text="\n\n".join(parts),
                              label=f"cell{i}", needs_vision=bool(assets),
                              assets=assets))

    doc.meta.update(n_cells=len(doc.pages), n_images=n_images)
    return doc
