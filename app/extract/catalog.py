"""catalog.jsonlと、検索用のファイルツリーを生成する。

版違い（v1/r2/old）も併記し、比較対象を見つけやすくする。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.config import CONFIG
from app.extract.models import ExtractedDoc, version_group_key


def _flags(doc: ExtractedDoc) -> list[str]:
    f: list[str] = []
    m = doc.meta
    if m.get("encrypted"):
        f.append("暗号化")
    if m.get("scanned_pages"):
        f.append(f"スキャン{len(m['scanned_pages'])}p")
    if m.get("n_pivots"):
        f.append(f"ピボット{m['n_pivots']}")
    if m.get("n_emf_parsed"):
        f.append(f"EMF表{m['n_emf_parsed']}")
    if m.get("n_charts"):
        f.append(f"グラフ{m['n_charts']}")
    if m.get("n_comments"):
        f.append(f"コメント{m['n_comments']}")
    if doc.assets:
        f.append(f"画像{len(doc.assets)}")
    if m.get("truncated"):
        f.append("大規模(要run_python)")
    if doc.error:
        f.append("抽出エラー")
    return f


def _descriptor(doc: ExtractedDoc) -> str:
    m, ft = doc.meta, doc.filetype
    if ft == "xlsx":
        return "シート: " + ", ".join(m.get("sheets", []))
    if ft == "pdf":
        return f"{m.get('n_pages', len(doc.pages))}ページ"
    if ft == "pptx":
        return f"{m.get('n_slides', len(doc.pages))}スライド"
    if ft == "ipynb":
        return f"{m.get('n_cells', len(doc.pages))}セル"
    if ft in ("csv", "tsv"):
        cols = ", ".join(m.get("columns", [])[:20])
        return f"{m.get('n_rows', '?')}行×{m.get('n_cols', '?')}列: {cols}"
    if ft in ("png", "jpg", "jpeg"):
        return f"{m.get('width', '?')}x{m.get('height', '?')}"
    if ft == "docx":
        bits = []
        if m.get("n_tables"):
            bits.append(f"表{m['n_tables']}")
        return " ".join(bits)
    return ""


def load_docs() -> list[ExtractedDoc]:
    out = []
    for p in sorted(CONFIG.extracted_dir.glob("*.json")):
        try:
            out.append(ExtractedDoc.load(CONFIG.extracted_dir, p.stem))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(out, key=lambda d: d.relpath)


def build_catalog(docs: list[ExtractedDoc] | None = None) -> Path:
    docs = docs if docs is not None else load_docs()
    CONFIG.catalog_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        groups[version_group_key(d.relpath)].append(d.relpath)

    entries = []
    for d in docs:
        vg = version_group_key(d.relpath)
        entries.append({
            "doc_id": d.doc_id, "relpath": d.relpath, "mount": d.mount,
            "project": d.project, "category": d.category,
            "filetype": d.filetype, "md5": d.md5,
            "n_pages": len(d.pages), "n_chars": d.n_chars,
            "version_group": vg,
            "version_peers": sorted(p for p in groups[vg] if p != d.relpath),
            "flags": _flags(d), "descriptor": _descriptor(d),
            "assets": d.assets, "error": d.error, "warnings": d.warnings,
        })

    with open(CONFIG.catalog_dir / "catalog.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # tree.txt: インデント形式（フルパス反復を避けてトークンを節約）
    lines: list[str] = []
    printed: set[str] = set()
    for e in entries:
        parts = e["relpath"].split("/")
        for depth in range(len(parts) - 1):
            d_ = "/".join(parts[:depth + 1])
            if d_ not in printed:
                printed.add(d_)
                lines.append("  " * depth + parts[depth] + "/")
        info = [e["filetype"]]
        if e["descriptor"]:
            info.append(e["descriptor"])
        info.extend(e["flags"])
        if e["version_peers"]:
            info.append("版違い: " + ", ".join(
                p.split("/")[-1] for p in e["version_peers"]))
        lines.append("  " * (len(parts) - 1)
                     + f"{parts[-1]}  [{'; '.join(info)}]")

    out = CONFIG.catalog_dir / "tree.txt"
    out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return out


def load_catalog() -> list[dict]:
    p = CONFIG.catalog_dir / "catalog.jsonl"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
