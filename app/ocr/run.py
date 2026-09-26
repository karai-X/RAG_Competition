"""画像バッチCLI。

  python -m app.ocr.run [--engine codex|gemini] [--workers 3] [--limit N]
  python -m app.ocr.run --apply          # キャッシュ結果を抽出文書へ差し込む
  python -m app.ocr.run --stats
"""
from __future__ import annotations

import argparse
import sys

from app.config import CONFIG
from app.ocr import cache
from app.ocr.runner import apply_to_docs, run_batch
from app.ocr.targets import collect_targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None)
    ap.add_argument("--fallback", default=None,
                    help="失敗時に切り替えるエンジン（例: gemini）")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--kind", default=None,
                    help="file / asset / pdf_page で絞る")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if args.stats:
        ts = collect_targets()
        import collections
        print("対象:", len(ts), dict(collections.Counter(t.kind for t in ts)))
        print("キャッシュ:", cache.stats())
        return 0

    if args.apply:
        print(apply_to_docs(args.engine))
        return 0

    ts = collect_targets()
    if args.kind:
        ts = [t for t in ts if t.kind == args.kind]

    def progress(st, t, res):
        n = st.cached + st.done + st.failed + st.skipped
        if n % 10 == 0 or n == st.total:
            print(f"  {n}/{st.total}  {st.line()}", flush=True)

    st = run_batch(engine=args.engine, workers=args.workers, limit=args.limit,
                   force=args.force, fallback=args.fallback, targets=ts,
                   progress=progress)
    print(f"\n{st.line()}")
    for e in st.errors:
        print(f"  ERROR {e}")
    print(f"cache -> {CONFIG.ocr_cache_dir}")
    print("次: python -m app.ocr.run --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
