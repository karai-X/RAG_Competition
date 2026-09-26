"""コーパス走査 — 全ファイルの台帳を作る。

走査時の注意:
- **NFC/NFD混在**: 走査は `os.walk`
  を使い、戻り値からパスを組む。表示・ID・検索用は NFC 正規化した relpath、
  open 用は生の raw_path を **二重に保持** する。
- Officeロック一時ファイル(`~$*`)・`.pyc`・`__pycache__` 等は除外。

project / category はマウント相対パスの位置から導出する。

  python -m app.corpus.walk --stats
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.corpus.mounts import Mount, load_mounts

# 機械生成物のみを除外する。**コンテンツを黙って落とさない**のが原則。
# 例: uv.lock は依存バージョンという実データなので除外しない。
EXCLUDE_FILE_PATTERNS = ("~$*", ".~lock.*", "*.pyc", "*.pyo",
                         ".DS_Store", "Thumbs.db", "desktop.ini")
EXCLUDE_DIR_NAMES = {"__pycache__", ".git", ".ipynb_checkpoints", ".venv",
                     "venv", "node_modules"}

# 抽出対象の拡張子（それ以外も台帳には載せる）
KNOWN_EXTS = {".pdf", ".docx", ".xlsx", ".pptx", ".ipynb", ".md", ".txt",
              ".py", ".json", ".toml", ".csv", ".tsv", ".png", ".jpg",
              ".jpeg", ".gif", ".bmp", ".emf", ".wmf", ".yaml", ".yml",
              ".html", ".xml", ".cfg", ".ini", ".sh", ".sql", ".r", ".rmd"}

MD5_MAX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class CorpusFile:
    raw_path: str        # open() にそのまま使える絶対パス（生のバイト列由来）
    relpath: str         # NFC正規化済み・表示/ID/検索用（mount_as 込み）
    doc_id: str          # relpath の sha1 先頭12桁（安定ID）
    ext: str             # 小文字拡張子
    size: int
    md5: str             # 増分判定用
    mount: str           # 由来マウントの mount_as（"" は既定コーパス）
    project: str | None  # 位置から導出（例: 案件フォルダ名）
    category: str | None # 位置から導出（例: 番号付きサブフォルダ名）

    @property
    def known(self) -> bool:
        return self.ext in KNOWN_EXTS

    @property
    def name(self) -> str:
        return self.relpath.rsplit("/", 1)[-1]


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _excluded(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_FILE_PATTERNS)


def _md5(path: str, size: int) -> str:
    """内容ハッシュ。巨大ファイルは先頭+末尾+サイズで代用する。"""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            if size <= MD5_MAX_BYTES:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            else:
                h.update(f.read(1 << 20))
                f.seek(-(1 << 20), os.SEEK_END)
                h.update(f.read())
                h.update(str(size).encode())
    except OSError:
        return ""
    return h.hexdigest()


def classify(rel_parts: tuple[str, ...]) -> tuple[str | None, str | None]:
    """相対パスの構造から project / category を機械的に導出する。

    `<コンテナ>/<案件>/<カテゴリ>/...` という一般構造のみを仮定し、
    固有名には依存しない。
      深さ>=4: project=parts[1], category=parts[2]   例 プロジェクト/A社/01.契約/x.docx
      深さ==3: project=parts[1]                      例 プロジェクト/A社/x.docx
      深さ==2: category=parts[0]                     例 社内管理/x.docx
    """
    if len(rel_parts) >= 4:
        return rel_parts[1], rel_parts[2]
    if len(rel_parts) == 3:
        return rel_parts[1], None
    if len(rel_parts) == 2:
        return None, rel_parts[0]
    return None, None


def walk_mount(m: Mount, with_md5: bool = True) -> list[CorpusFile]:
    root = str(m.root.resolve())
    prefix = m.mount_as.strip("/")
    out: list[CorpusFile] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIR_NAMES)
        for name in sorted(filenames):
            if _excluded(name):
                continue
            raw_path = os.path.join(dirpath, name)
            rel_raw = os.path.relpath(raw_path, root).replace(os.sep, "/")
            relpath = nfc(f"{prefix}/{rel_raw}" if prefix else rel_raw)
            try:
                size = os.path.getsize(raw_path)
            except OSError:
                continue
            parts = tuple(relpath.split("/"))
            project, category = classify(parts)
            out.append(CorpusFile(
                raw_path=raw_path,
                relpath=relpath,
                doc_id=hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:12],
                ext=os.path.splitext(name)[1].lower(),
                size=size,
                md5=_md5(raw_path, size) if with_md5 else "",
                mount=m.mount_as,
                project=project,
                category=category,
            ))
    return out


def scan_excluded(mounts: list[Mount] | None = None) -> list[tuple[str, str]]:
    """除外したファイルと理由。**silent drop を作らない**ために常に可視化する。"""
    out: list[tuple[str, str]] = []
    for m in (mounts if mounts is not None else load_mounts()):
        if not m.root.exists():
            continue
        root = str(m.root.resolve())
        for dirpath, dirnames, filenames in os.walk(root):
            skipped = [d for d in dirnames if d in EXCLUDE_DIR_NAMES]
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIR_NAMES)
            for d in skipped:
                rel = os.path.relpath(os.path.join(dirpath, d), root)
                out.append((nfc(rel.replace(os.sep, "/")) + "/", "除外ディレクトリ"))
            for name in filenames:
                if _excluded(name):
                    rel = os.path.relpath(os.path.join(dirpath, name), root)
                    pat = next(p for p in EXCLUDE_FILE_PATTERNS
                               if fnmatch.fnmatch(name, p))
                    out.append((nfc(rel.replace(os.sep, "/")), pat))
    return sorted(out)


def walk_corpus(with_md5: bool = True,
                mounts: list[Mount] | None = None) -> list[CorpusFile]:
    """全マウントを走査。relpath 衝突は先勝ち（既定コーパス優先）。"""
    seen: dict[str, CorpusFile] = {}
    for m in (mounts if mounts is not None else load_mounts()):
        if not m.root.exists():
            continue
        for cf in walk_mount(m, with_md5=with_md5):
            seen.setdefault(cf.relpath, cf)
    return sorted(seen.values(), key=lambda c: c.relpath)


class PathResolver:
    """NFC relpath → open できる生パス。`run_python` の P() と共有する。"""

    def __init__(self, files: list[CorpusFile] | None = None) -> None:
        self._files = files if files is not None else walk_corpus(with_md5=False)
        self._by_rel = {f.relpath: f.raw_path for f in self._files}
        self._by_name: dict[str, list[str]] = {}
        for f in self._files:
            self._by_name.setdefault(f.name, []).append(f.relpath)

    def resolve(self, relpath: str) -> str | None:
        key = nfc(relpath.strip().lstrip("/"))
        if key in self._by_rel:
            return self._by_rel[key]
        # 末尾一致で一意に決まる場合のみ救済する
        tail = key.rsplit("/", 1)[-1]
        cands = self._by_name.get(tail, [])
        if len(cands) == 1:
            return self._by_rel[cands[0]]
        cands2 = [r for r in self._by_rel if r.endswith("/" + key)]
        return self._by_rel[cands2[0]] if len(cands2) == 1 else None

    def candidates(self, relpath: str, limit: int = 5) -> list[str]:
        tail = nfc(relpath).rsplit("/", 1)[-1].lower()
        return [r for r in self._by_rel if tail in r.lower()][:limit]


def main() -> int:
    import collections
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--no-md5", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    files = walk_corpus(with_md5=not args.no_md5)
    print(f"total files: {len(files)}  (known-ext: {sum(f.known for f in files)})")

    if args.list:
        for f in files:
            print(f"  {f.relpath}  [{f.ext} {f.size}B {f.md5[:8]}]")

    if args.stats or not args.list:
        exts = collections.Counter(f.ext for f in files)
        print("by ext:", dict(exts.most_common(20)))
        projects = sorted({f.project for f in files if f.project})
        print(f"projects ({len(projects)}):")
        for p in projects:
            n = sum(1 for f in files if f.project == p)
            print(f"  - {p}  ({n})")
        cats = collections.Counter(
            f.category for f in files if f.category and f.project)
        print("categories:", dict(sorted(cats.items())))
        noproj = collections.Counter(
            f.category for f in files if not f.project)
        print("non-project:", dict(sorted(noproj.items(), key=lambda x: str(x[0]))))
        dup = [m for m, c in collections.Counter(
            f.md5 for f in files if f.md5).items() if c > 1]
        print(f"duplicate-content files: {len(dup)} md5 groups")
        exc = scan_excluded()
        print(f"excluded: {len(exc)}", dict(collections.Counter(r for _, r in exc)))
        for path, why in exc[:20]:
            print(f"  - {path}  [{why}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
