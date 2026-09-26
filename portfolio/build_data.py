"""ポートフォリオ用データ生成: 実行ログ + カタログ → portfolio/data/*.js

  uv run python portfolio/build_data.py [--run-id csv-20260830-072745]

出力は `window.RAG_*` へ代入する **.js**。file:// で index.html を直接開いても
読めるようにするため（fetch だと同一オリジンポリシーで弾かれる）。
実行ログ本体や資料そのものは同梱しない。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"

# 経路ラベル: 使われたツールの組み合わせから主経路を1つ決める。
# 上から順に判定し、最初に当たったものを採用する（強い手掛かりが上）。
ROUTE_RULES = [
    ("decrypt",     "復号",       "暗号化された文書を規則からパスワードを導出して開いた"),
    ("ask_image",   "図版照会",   "図の中の位置関係・並びを画像専用モデルに問い直して確かめた"),
    ("diff",        "版比較",     "版違いの2ファイルを突き合わせて差分を取った"),
    ("read_image",  "画像読取",   "グラフ・スキャン・書式を画像として実際に見た"),
    ("run_python",  "コード走査", "抽出済みテキストや表をコードで走査・集計した"),
    ("grep",        "全文検索",   "抽出済み全文への正規表現で網羅的に拾った"),
    ("search",      "意味検索",   "ハイブリッド検索で関連資料を特定した"),
    ("read",        "直読",       "ツリーから当たりを付けたファイルをそのまま読んだ"),
]
TOOL_ORDER = ["search", "grep", "read", "read_image", "ask_image",
              "run_python", "diff", "decrypt", "submit_answer"]


def clip(s, n):
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def build_questions(run_dir: Path) -> dict:
    files = sorted(run_dir.glob("q*.json"),
                   key=lambda p: int(p.stem[1:]))
    items = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        steps, tools, blocked = [], [], 0
        for t in d.get("trace", []):
            if t.get("type") == "assistant":
                calls = t.get("tool_calls") or []
                if not calls and not (t.get("text") or "").strip():
                    continue
                steps.append({
                    "k": "think",
                    "text": clip(t.get("text"), 400),
                    "calls": [c.get("name") for c in calls],
                    "sec": round(float(t.get("latency_sec") or 0), 1),
                })
            elif t.get("type") == "tool":
                name = t.get("name")
                tools.append(name)
                if t.get("blocked"):
                    blocked += 1
                steps.append({
                    "k": "tool",
                    "name": name,
                    "args": {a: clip(v, 600) for a, v in (t.get("args") or {}).items()},
                    "out": clip(t.get("result_preview"), 700),
                    "sec": round(float(t.get("elapsed_sec") or 0), 2),
                    "blocked": bool(t.get("blocked")),
                })
            elif t.get("type") == "error":
                steps.append({"k": "error", "text": clip(t.get("error"), 300)})

        used = [x for x in tools if x != "submit_answer"]
        route = next((label for tool, label, _ in ROUTE_RULES if tool in used),
                     "未探索")
        entry = next((x for x in used), None)
        items.append({
            "i": d["index"],
            "q": d["question"],
            "a": d.get("answer") or "",
            "conf": d.get("confidence"),
            "ev": d.get("evidence") or [],
            "stop": "submitted" if d.get("stop") == "submitted" else "error",
            "route": route,
            "entry": entry,
            "tools": sorted(set(used), key=lambda x: TOOL_ORDER.index(x)
                            if x in TOOL_ORDER else 99),
            "counts": {t: used.count(t) for t in sorted(set(used))},
            "blocked": blocked,
            "turns": d.get("turns") or 0,
            "sec": round(float(d.get("elapsed_sec") or 0), 1),
            "tok": (d.get("usage") or {}).get("total_tokens") or 0,
            "steps": steps,
        })
    return {"questions": items}


def build_corpus() -> dict:
    cat = [json.loads(l) for l in
           (ROOT / "artifacts/catalog/catalog.jsonl").read_text(
               encoding="utf-8").splitlines() if l.strip()]
    files = [{
        "path": e["relpath"],
        "project": e.get("project"),
        "category": e.get("category"),
        "ft": e.get("filetype"),
        "pages": e.get("n_pages"),
        "chars": e.get("n_chars"),
        "desc": e.get("descriptor") or "",
        "flags": e.get("flags") or [],
        "peers": e.get("version_peers") or [],
        "assets": len(e.get("assets") or []),
    } for e in sorted(cat, key=lambda x: x["relpath"])]
    return {"files": files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="csv-20260830-072745")
    a = ap.parse_args()
    run_dir = ROOT / "logs" / a.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    OUT.mkdir(parents=True, exist_ok=True)
    qs = build_questions(run_dir)
    qs["run"] = {k: manifest.get(k) for k in
                 ("run_id", "model", "backend", "temperature", "seed",
                  "embed_model", "rerank_model", "parallel", "max_turns",
                  "n_questions", "elapsed_sec", "models_observed")}
    for name, var, payload in (("questions", "RAG_RUN", qs),
                               ("corpus", "RAG_CORPUS", build_corpus())):
        p = OUT / f"{name}.js"
        p.write_text(
            f"window.{var}=" + json.dumps(payload, ensure_ascii=False,
                                          separators=(",", ":")) + ";\n",
            encoding="utf-8")
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
