"""OCRエンジンの比較。

画像の4タイプ（表 / ガント・スケジュール図 / 座席・配置図 / 通常グラフ）を
同じ枚数だけ codex と Gemini Flash に投げ、品質・レイテンシ・失敗率を比べる。
「軽い図は Flash がループ内で読む / 難しいものだけ事前バッチで codex に回す」
の閾値を決めるのが目的。

タイプ分けは **ファイル名や案件名ではなく、由来メタデータ**（PDFスキャン /
埋め込み画像 / 直置き画像）と、抽出済みテキストの汎用的な手掛かりで行う。

  python -m app.ocr.compare [--per-type 3] [--engines codex,gemini]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

from app.config import CONFIG
from app.extract.catalog import load_catalog
from app.ocr.engines import run_engine
from app.ocr.image_util import normalize
from app.ocr.targets import (OcrTarget, collect_targets, image_bytes,
                             image_key)

# 由来テキストから画像の種類を推定する。
TYPE_HINTS = {
    "スケジュール図": r"ガント|スケジュール|工程|マイルストーン|タスクID",
    "配置図": r"座席|フロア|配置図|レイアウト|内線",
    "表・ピボット": r"ピボット|集計|抽出条件|クロス集計|個数 /|合計 /",
    "グラフ": r"ヒストグラム|相関|分布|推移|散布|棒グラフ|折れ線|heatmap|"
              r"distribution|correlation|trend",
}


def classify(t: OcrTarget, by_rel: dict[str, dict]) -> str:
    hay = f"{t.relpath} {t.label}"
    e = by_rel.get(t.relpath) or {}
    hay += " " + " ".join(e.get("flags") or []) + " " + (e.get("descriptor") or "")
    for name, pat in TYPE_HINTS.items():
        if re.search(pat, hay, re.IGNORECASE):
            return name
    return "スキャン文書" if t.kind == "pdf_page" else "その他"


def pick(per_type: int) -> dict[str, list[OcrTarget]]:
    by_rel = {e["relpath"]: e for e in load_catalog()}
    buckets: dict[str, list[OcrTarget]] = defaultdict(list)
    for t in collect_targets():
        buckets[classify(t, by_rel)].append(t)
    return {k: v[:per_type] for k, v in sorted(buckets.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--engines", default="codex,gemini")
    args = ap.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    picked = pick(args.per_type)
    print("比較対象:", {k: len(v) for k, v in picked.items()}, "\n")

    agg: dict[str, dict] = {e: {"n": 0, "ok": 0, "sec": 0.0, "chars": 0,
                                "confident": 0} for e in engines}
    for typ, ts in picked.items():
        print(f"=== {typ} ({len(ts)}枚)")
        for t in ts:
            raw = image_bytes(t)
            if not raw:
                print(f"  -- 画像取得不可: {t.label}")
                continue
            img = normalize(raw, CONFIG.image_max_edge_px)
            md5 = image_key(img)
            row = []
            for eng in engines:
                r = run_engine(eng, img, md5, source=t.label)
                a = agg[eng]
                a["n"] += 1
                a["ok"] += bool(r.ok)
                a["sec"] += r.latency_sec
                a["chars"] += len(r.markdown) + len(r.figure_description)
                a["confident"] += bool(r.confident)
                row.append(f"{eng}: {'OK' if r.ok else 'NG'} "
                           f"{r.latency_sec:5.1f}s "
                           f"{len(r.markdown)+len(r.figure_description):5d}字 "
                           f"conf={r.confident} type={r.content_type[:8]}"
                           + (f" err={r.error[:40]}" if r.error else ""))
            print(f"  {t.label[-60:]}")
            for x in row:
                print(f"      {x}")
        print()

    print("=== 集計 ===")
    for eng, a in agg.items():
        if not a["n"]:
            continue
        print(f"  {eng:8s} n={a['n']:3d} 成功率={a['ok']/a['n']*100:5.1f}% "
              f"平均{a['sec']/a['n']:6.1f}s 平均{a['chars']//a['n']:5d}字 "
              f"confident={a['confident']}/{a['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
