"""抽出ドキュメントの共通モデル。

全形式を「Markdown + 軽量アノテーション」に統一する。記法は markup.py が単一定義。

**設計上の約束**: 抽出の失敗を黙って捨てない。
例外は型を絞って捕捉し、握り潰した箇所は必ず `warnings` に残す。
`error` が立った文書と 0ページの文書は health.py が全件洗い出す。
"""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Page:
    """ページ / スライド / シート / セル 1単位。"""
    no: int                                          # 1始まり
    text: str                                        # アノテーション付きMarkdown
    label: str = ""                                  # シート名・スライド名など
    needs_vision: bool = False                       # 画像を見ないと分からない
    assets: list[str] = field(default_factory=list)  # asset_id のリスト


@dataclass
class ExtractedDoc:
    doc_id: str
    relpath: str
    mount: str = ""
    project: str | None = None
    category: str | None = None
    filetype: str = ""
    md5: str = ""
    pages: list[Page] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def n_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def assets(self) -> list[str]:
        return [a for p in self.pages for a in p.assets]

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def save(self, extracted_dir: Path) -> None:
        extracted_dir.mkdir(parents=True, exist_ok=True)
        tmp = extracted_dir / f".{self.doc_id}.tmp"
        tmp.write_text(json.dumps(dataclasses.asdict(self), ensure_ascii=False,
                                  indent=1), encoding="utf-8")
        tmp.replace(extracted_dir / f"{self.doc_id}.json")

    @classmethod
    def load(cls, extracted_dir: Path, doc_id: str) -> "ExtractedDoc":
        with open(extracted_dir / f"{doc_id}.json", encoding="utf-8") as f:
            d = json.load(f)
        d["pages"] = [Page(**p) for p in d["pages"]]
        return cls(**d)

    @classmethod
    def exists(cls, extracted_dir: Path, doc_id: str) -> bool:
        return (extracted_dir / f"{doc_id}.json").exists()


# 版数トークン（一般的な命名慣習のみ。特定ファイル名に依存しない）
_VERSION_TOKEN_RE = re.compile(
    r"(?:[_\-\s]?(?:v|ver|r|rev)\d+"
    r"|[_\-\s]?(?:final|draft|old|new|fix|latest|copy|bak)\d*"
    r"|[_\-\s]?(?:新|旧|最終|案|改)\d*)",
    re.IGNORECASE)
_HINT_SUFFIX_RE = re.compile(r"[_\-]pw-[^_\-.]+", re.IGNORECASE)


def version_group_key(relpath: str) -> str:
    """版数トークンを除去した正規化キー。同一キー = 同一文書の版違い候補。

    `old/` `旧/` 等のサブフォルダも同一グループに畳む。
    """
    parts = relpath.split("/")
    dirs = [p for p in parts[:-1]
            if p.lower() not in ("old", "旧", "archive", "_old", "backup")]
    stem, dot, ext = parts[-1].rpartition(".")
    if not dot:
        stem, ext = parts[-1], ""
    stem = _HINT_SUFFIX_RE.sub("", stem)
    stem = _VERSION_TOKEN_RE.sub("", stem).strip("_- ")
    return "/".join(dirs + [f"{stem}.{ext}" if ext else stem])
