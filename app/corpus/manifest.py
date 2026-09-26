"""増分インジェストの差分判定。

`relpath -> md5` を保存し、次回走査時に **新規 / 変更 / 削除** を出す。
これがあるので「新しいフォルダを追加 → 変わった分だけ再抽出」ができる。

  python -m app.corpus.manifest diff
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.config import CONFIG
from app.corpus.walk import CorpusFile, walk_corpus


@dataclass
class Delta:
    added: list[CorpusFile] = field(default_factory=list)
    changed: list[CorpusFile] = field(default_factory=list)
    unchanged: list[CorpusFile] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)   # relpath
    removed_doc_ids: list[str] = field(default_factory=list)

    @property
    def todo(self) -> list[CorpusFile]:
        return self.added + self.changed

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.changed or self.removed)

    def summary(self) -> str:
        return (f"added={len(self.added)} changed={len(self.changed)} "
                f"removed={len(self.removed)} unchanged={len(self.unchanged)}")


def load_manifest(path: Path | None = None) -> dict[str, dict]:
    p = path or CONFIG.manifest_file
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(files: list[CorpusFile], path: Path | None = None) -> None:
    p = path or CONFIG.manifest_file
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for f in files:
        try:
            mtime_ns = Path(f.raw_path).stat().st_mtime_ns
        except OSError:
            mtime_ns = None
        data[f.relpath] = {
            "md5": f.md5, "size": f.size, "mtime_ns": mtime_ns,
            "doc_id": f.doc_id, "mount": f.mount,
        }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def compute_delta(files: list[CorpusFile] | None = None,
                  manifest: dict[str, dict] | None = None) -> Delta:
    files = files if files is not None else walk_corpus()
    prev = manifest if manifest is not None else load_manifest()
    d = Delta()
    seen: set[str] = set()
    for f in files:
        seen.add(f.relpath)
        old = prev.get(f.relpath)
        if old is None:
            d.added.append(f)
        elif old.get("md5") != f.md5:
            d.changed.append(f)
        else:
            d.unchanged.append(f)
    for rel, meta in prev.items():
        if rel not in seen:
            d.removed.append(rel)
            if meta.get("doc_id"):
                d.removed_doc_ids.append(meta["doc_id"])
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diff", "save", "clear"], default="diff",
                    nargs="?")
    args = ap.parse_args()

    if args.cmd == "clear":
        CONFIG.manifest_file.unlink(missing_ok=True)
        print("manifest cleared")
        return 0

    files = walk_corpus()
    d = compute_delta(files)
    print(d.summary())
    for f in d.added[:10]:
        print(f"  + {f.relpath}")
    for f in d.changed[:10]:
        print(f"  ~ {f.relpath}")
    for r in d.removed[:10]:
        print(f"  - {r}")

    if args.cmd == "save":
        save_manifest(files)
        print(f"saved -> {CONFIG.manifest_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
