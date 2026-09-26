"""抽出CLI。

  python -m app.extract.run [--full] [--workers 8]
  python -m app.extract.run --relpath "<相対パス>" [--pages 1-3]
"""
from __future__ import annotations

import argparse
import sys
import time

from app.config import CONFIG
from app.extract.models import ExtractedDoc
from app.extract.pipeline import run_extract


def _show(relpath: str, pages: str | None) -> int:
    from app.corpus.walk import walk_corpus, nfc
    target = nfc(relpath)
    cf = next((f for f in walk_corpus(with_md5=False)
               if f.relpath == target or f.relpath.endswith("/" + target)), None)
    if cf is None:
        print(f"見つかりません: {relpath}")
        return 1
    run_extract(only_relpath=cf.relpath, workers=1)
    doc = ExtractedDoc.load(CONFIG.extracted_dir, cf.doc_id)
    print(f"# {doc.relpath}\n  filetype={doc.filetype} pages={len(doc.pages)} "
          f"meta={doc.meta}")
    if doc.error:
        print(f"  ERROR: {doc.error}")
    for w in doc.warnings:
        print(f"  WARN: {w}")
    want = None
    if pages:
        want = set()
        for part in pages.split(","):
            if "-" in part:
                a, _, b = part.partition("-")
                want |= set(range(int(a or 1), int(b or len(doc.pages)) + 1))
            elif part.strip().isdigit():
                want.add(int(part))
    for p in doc.pages:
        if want and p.no not in want:
            continue
        print(f"\n--- p{p.no} {p.label} "
              f"{'[needs_vision]' if p.needs_vision else ''} ---")
        print(p.text[:6000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="manifest を無視して全件")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--relpath", default=None, help="1ファイルだけ抽出して表示")
    ap.add_argument("--pages", default=None)
    args = ap.parse_args()

    if args.relpath:
        return _show(args.relpath, args.pages)

    t0 = time.time()

    def progress(done: int, total: int, doc):
        if doc.error:
            print(f"  ERROR {doc.relpath}: {doc.error.splitlines()[0]}")
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total}  ({time.time()-t0:.1f}s)", flush=True)

    s = run_extract(full=args.full, workers=args.workers, progress=progress)
    print(f"\ntotal={s['total']} 処理={s['todo']} "
          f"(新規{s['added']} 変更{s['changed']} 削除{s['removed']}) "
          f"errors={s['errors']} warnings={s['warnings']} "
          f"{time.time()-t0:.1f}s")
    print(f"extracted -> {CONFIG.extracted_dir}")
    print("次: python -m app.extract.health")
    return 0


if __name__ == "__main__":
    sys.exit(main())
