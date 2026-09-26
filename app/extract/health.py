"""抽出のヘルスチェック — **M1の受け入れ条件**。

旧実装では `hashlib` の import 漏れで `_media_assets()` が NameError を投げ、
内側の except に捕まらず pipeline の汎用 except に落ちた結果、
**メディアを含む xlsx の抽出が丸ごと空になっていた**（EMFピボットの見落としの原因）。

同じ事故を二度と起こさないために、以下を毎回洗い出す:
  1. error が立っている文書
  2. 抽出0ページ / 本文が空の文書
  3. zip 内のメディア数と asset 数の不一致
  4. warnings が出ている文書
  5. 暗号化・スキャン・EMF未解析など、後段の処理が必要なもの

  python -m app.extract.health [--strict]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from app.config import CONFIG
from app.corpus.walk import walk_corpus
from app.extract.catalog import load_docs
from app.extract.office_media import count_media
from app.extract.pipeline import IMAGE_EXTS

OOXML = {"docx", "xlsx", "pptx"}


def check() -> dict:
    docs = load_docs()
    by_rel = {f.relpath: f for f in walk_corpus(with_md5=False)}

    errors, empties, mismatch, warns = [], [], [], []
    encrypted, scanned, emf_unparsed = [], [], []

    for d in docs:
        if d.error:
            errors.append((d.relpath, d.error.splitlines()[0]))
        # 画像ファイル自体は本文が短くて当然なので対象外
        is_image_file = f".{d.filetype}" in IMAGE_EXTS
        if not d.pages or (d.n_chars < 5 and not is_image_file
                           and not d.meta.get("encrypted")):
            empties.append((d.relpath, len(d.pages), d.n_chars))
        if d.warnings:
            warns.append((d.relpath, d.warnings))
        if d.meta.get("encrypted"):
            encrypted.append(d.relpath)
        if d.meta.get("scanned_pages"):
            scanned.append((d.relpath, len(d.meta["scanned_pages"])))
        if d.filetype in OOXML:
            cf = by_rel.get(d.relpath)
            if cf is not None and not d.meta.get("encrypted"):
                zipn = count_media(cf.raw_path)
                got = int(d.meta.get("n_media", 0))
                if zipn != got:
                    mismatch.append((d.relpath, zipn, got))

    missing = [r for r in by_rel if r not in {d.relpath for d in docs}]

    return {
        "n_docs": len(docs), "n_files": len(by_rel),
        "errors": errors, "empties": empties, "mismatch": mismatch,
        "warnings": warns, "missing": missing,
        "ocr_unapplied": _ocr_unapplied(docs),
        "encrypted": encrypted, "scanned": scanned,
        "emf_unparsed": emf_unparsed,
        "by_type": Counter(d.filetype for d in docs),
    }


def _ocr_unapplied(docs) -> list[str]:
    """OCR 済みなのに抽出文書へ差し込まれていない文書。

    `extract.run` は抽出をやり直すので、その後 `ocr.run --apply` を
    忘れると **画像の読み取り結果が黙って消える**。キャッシュはあるのに
    本文に無い、という状態は下流からは見えないので、ここで可視化する。
    """
    from app.ocr import cache
    from app.ocr.image_util import normalize
    from app.ocr.targets import collect_targets, image_bytes, image_key

    if not cache.stats()["total"]:
        return []
    by_id = {d.doc_id: d for d in docs}
    text_of: dict[str, str] = {}
    need: set[str] = set()
    for t in collect_targets():
        d = by_id.get(t.doc_id)
        if d is None:
            continue
        # 画像ごとに見る。文書に MARKER が1つでもあれば済みとみなすと、
        # 同じページの 2枚目以降が落ちていても気づけない。
        body = text_of.setdefault(
            t.doc_id, "\n".join(p.text for p in d.pages))
        if "\n\n" + cache.applied_head(t.kind, t.ref) + "\n" in body:
            continue
        raw = image_bytes(t)
        if not raw:
            continue
        res = cache.get(image_key(normalize(raw, CONFIG.image_max_edge_px)),
                        CONFIG.model_ocr)
        if res is not None and res.ok:
            need.add(d.relpath)
    return sorted(need)


def report(r: dict, verbose: bool = True) -> int:
    """致命的な件数を返す（0 なら受け入れ条件クリア）。"""
    print(f"抽出済み {r['n_docs']} / 台帳 {r['n_files']}")
    fatal = 0
    for key, title, is_fatal in (
            ("missing", "未抽出のファイル", True),
            ("errors", "抽出エラー", True),
            ("empties", "抽出0ページ / 本文が空", True),
            ("mismatch", "メディア数の不一致 (zip内, asset)", True),
            ("ocr_unapplied", "OCR済みだが本文へ未差し込み "
                              "(→ python -m app.ocr.run --apply)", True),
            ("warnings", "警告", False)):
        items = r[key]
        mark = "NG" if (items and is_fatal) else ("-- " if items else "OK")
        print(f"  {mark} {title}: {len(items)}")
        if items and verbose:
            for x in items[:15]:
                print(f"       {x}")
            if len(items) > 15:
                print(f"       ... 他 {len(items)-15} 件")
        if is_fatal:
            fatal += len(items)

    print(f"  -- 暗号化: {len(r['encrypted'])} {r['encrypted']}")
    print(f"  -- スキャンPDF: {len(r['scanned'])} "
          f"(計{sum(n for _, n in r['scanned'])}ページ) → ocr 対象")
    print(f"  -- 形式内訳: {dict(r['by_type'])}")
    print(f"\n{'PASS' if fatal == 0 else f'FAIL ({fatal} 件)'}")
    return fatal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    return 1 if report(check(), verbose=not args.quiet) else 0


if __name__ == "__main__":
    sys.exit(main())
