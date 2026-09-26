"""OOXML (docx/xlsx/pptx) 内の埋め込みメディアの取り出し。

- `word|xl|ppt/media/*` を asset として保存し **md5 を付与** する
  （版比較で「テキストは同一だが画像が差し替わっている」を検出するため）
- EMF/WMF は決定的解析を試み、成功すれば **テキストとして焼き込む**
  （表・ピボットが画像で埋め込まれているケースを「該当なし」にしないため）
"""
from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from app.extract.emf import emf_to_markdown

MEDIA_PREFIXES = ("word/media/", "xl/media/", "ppt/media/", "word/embeddings/",
                  "xl/embeddings/", "ppt/embeddings/")
RASTER_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
VECTOR_EXTS = {".emf", ".wmf"}


@dataclass
class MediaItem:
    asset_id: str          # "<doc_id>/<name>"
    name: str              # zip 内のファイル名
    md5: str
    size: int
    ext: str
    emf_markdown: str = ""   # EMF解析に成功した場合の本文
    emf_error: str | None = None

    @property
    def is_vector(self) -> bool:
        return self.ext in VECTOR_EXTS

    @property
    def readable_as_image(self) -> bool:
        return self.ext in RASTER_EXTS


@dataclass
class MediaResult:
    items: list[MediaItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def asset_ids(self) -> list[str]:
        return [m.asset_id for m in self.items]


def extract_media(path: str | Path, doc_id: str,
                  assets_dir: Path | None) -> MediaResult:
    """埋め込みメディアを保存して一覧を返す。

    **例外を握り潰さない**: zip が壊れている等は warnings に残す。
    （旧実装ではここで NameError が起き、xlsx の抽出が丸ごと空になっていた）
    """
    res = MediaResult()
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        res.warnings.append(f"メディア走査不可: {type(e).__name__}: {e}")
        return res

    with z:
        names = [n for n in z.namelist()
                 if n.startswith(MEDIA_PREFIXES) and not n.endswith("/")]
        for n in sorted(names):
            base = n.rsplit("/", 1)[-1]
            try:
                data = z.read(n)
            except (KeyError, zipfile.BadZipFile, OSError) as e:
                res.warnings.append(f"{n}: 読み出し失敗 {type(e).__name__}")
                continue
            ext = ("." + base.rsplit(".", 1)[-1].lower()) if "." in base else ""
            item = MediaItem(asset_id=f"{doc_id}/{base}", name=base,
                             md5=hashlib.md5(data).hexdigest(),
                             size=len(data), ext=ext)
            if item.is_vector:
                md, content = emf_to_markdown(data, label=base)
                item.emf_markdown = md
                item.emf_error = content.error
            if assets_dir is not None:
                out = assets_dir / item.asset_id
                out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    out.write_bytes(data)
                except OSError as e:
                    res.warnings.append(f"{n}: asset保存失敗 {e}")
            res.items.append(item)
    return res


def media_block(res: MediaResult) -> str:
    """抽出テキストへ埋め込む、メディアの見出しブロック。"""
    if not res.items:
        return ""
    lines: list[str] = []
    for m in res.items:
        if m.emf_markdown:
            lines.append(m.emf_markdown)
        elif m.is_vector:
            lines.append(f"[ベクタ画像: {m.asset_id} md5:{m.md5[:8]} "
                         f"— EMF解析不可({m.emf_error}) read_image で確認可能]")
        else:
            lines.append(f"[画像: {m.asset_id} md5:{m.md5[:8]} "
                         f"— read_image で確認可能]")
    return "\n\n".join(lines)


def count_media(path: str | Path) -> int:
    """zip 内のメディア数（health.py の突き合わせ用）。"""
    try:
        with zipfile.ZipFile(path) as z:
            return sum(1 for n in z.namelist()
                       if n.startswith(MEDIA_PREFIXES) and not n.endswith("/"))
    except (zipfile.BadZipFile, OSError):
        return 0
