"""検索CLI。

  python -m app.index.cli search "契約金額 税込" [--project 青潮] [--k 5]
  python -m app.index.cli grep "==[^=]+=="
  python -m app.index.cli stats
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import sys

from app.config import CONFIG


def cmd_search(args) -> int:
    from app.index.search import get_index
    idx = get_index()
    hits = idx.search(args.query, k=args.k, project=args.project,
                      category=args.category, filetype=args.filetype,
                      rerank=not args.no_rerank)
    if not hits:
        print("(ヒットなし)")
        return 0
    for h in hits:
        score = h.get("rerank_score", h["rrf_score"])
        print(f"  {score:8.4f}  {h['relpath']} p{h['page']}"
              f"{'(' + h['label'] + ')' if h['label'] else ''}")
        print(f"      {h['text'][:180].replace(chr(10), ' ')}")
    return 0


def cmd_grep(args) -> int:
    """抽出済み全文への正規表現検索（書式アノテーションの網羅検索用）。"""
    from app.extract.catalog import load_docs
    try:
        rx = re.compile(args.pattern)
    except re.error as e:
        print(f"[エラー] 正規表現が不正: {e}")
        return 1
    n = 0
    for d in load_docs():
        if args.project and args.project.lower() not in (d.project or "").lower():
            continue
        if args.path_glob and not fnmatch.fnmatch(d.relpath, args.path_glob):
            continue
        for page in d.pages:
            for m in rx.finditer(page.text):
                ls = page.text.rfind("\n", 0, m.start()) + 1
                le = page.text.find("\n", m.end())
                line = page.text[ls:le if le > 0 else None]
                print(f"  {d.relpath} p{page.no}: {line.strip()[:200]}")
                n += 1
                if n >= args.max_hits:
                    print(f"  ...(上限{args.max_hits}件)")
                    return 0
    if not n:
        print("(マッチなし)")
    return 0


def cmd_stats(args) -> int:
    import json
    p = CONFIG.index_dir / "meta.json"
    print(json.loads(p.read_text(encoding="utf-8")) if p.exists() else "(未構築)")
    from app.index.rerank import available
    print("reranker:", "有効" if available() else "無効（素通し）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--project", default=None)
    s.add_argument("--category", default=None)
    s.add_argument("--filetype", default=None)
    s.add_argument("--no-rerank", action="store_true")
    s.set_defaults(fn=cmd_search)
    g = sub.add_parser("grep")
    g.add_argument("pattern")
    g.add_argument("--project", default=None)
    g.add_argument("--path-glob", default=None)
    g.add_argument("--max-hits", type=int, default=30)
    g.set_defaults(fn=cmd_grep)
    t = sub.add_parser("stats")
    t.set_defaults(fn=cmd_stats)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
