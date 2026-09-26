"""複数ランを統合して1つのランにする。

一部の質問だけを再実行すると、最新の回答が複数のランに散らばる。
**問ごとに最も新しい結果を採用**して
1つの完全なランに畳む。

  python -m app.run.merge --out t7 --base t4 --overlay t4fix,t4fix2,var3

`--base` の全問を土台にし、`--overlay` に同じ index があれば
**より新しい方**（ファイル更新時刻で判定）で置き換える。
どのランから採ったかは merge_source.json に残す。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from app.config import CONFIG


def merge(out_id: str, base: str, overlays: list[str]) -> dict:
    out_dir = CONFIG.logs_dir / out_id
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = CONFIG.logs_dir / base
    if not base_dir.exists():
        raise SystemExit(f"base が見つかりません: {base_dir}")

    chosen: dict[int, tuple[str, Path, float]] = {}
    for run in [base, *overlays]:
        d = CONFIG.logs_dir / run
        if not d.exists():
            print(f"  (skip 存在しない: {run})")
            continue
        for p in d.glob("q*.json"):
            try:
                idx = int(p.stem[1:])
            except ValueError:
                continue
            mtime = p.stat().st_mtime
            cur = chosen.get(idx)
            # base は土台。overlay は base より新しいときだけ勝たせる
            if cur is None or mtime > cur[2]:
                chosen[idx] = (run, p, mtime)

    src_count: dict[str, int] = {}
    for idx, (run, p, _) in sorted(chosen.items()):
        shutil.copy2(p, out_dir / f"q{idx}.json")
        # chain も引き連れる（経路を追えるように）
        cp = CONFIG.logs_dir / run / "chain" / f"q{idx}.jsonl"
        if cp.exists():
            (out_dir / "chain").mkdir(exist_ok=True)
            shutil.copy2(cp, out_dir / "chain" / f"q{idx}.jsonl")
        src_count[run] = src_count.get(run, 0) + 1

    for run in [base, *overlays]:
        for sp in (CONFIG.logs_dir / run / "chain").glob("system_*.txt"):
            (out_dir / "chain").mkdir(exist_ok=True)
            shutil.copy2(sp, out_dir / "chain" / sp.name)

    mf = CONFIG.logs_dir / base / "manifest.json"
    if mf.exists():
        m = json.loads(mf.read_text(encoding="utf-8"))
        m["run_id"] = out_id
        m["merged_from"] = {"base": base, "overlays": overlays}
        (out_dir / "manifest.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")

    (out_dir / "merge_source.json").write_text(
        json.dumps({str(i): r for i, (r, _, _) in sorted(chosen.items())},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return {"out": out_id, "問数": len(chosen), "採用元": src_count}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--overlay", default="")
    args = ap.parse_args()
    ov = [x.strip() for x in args.overlay.split(",") if x.strip()]
    print(merge(args.out, args.base, ov))
    return 0


if __name__ == "__main__":
    sys.exit(main())
