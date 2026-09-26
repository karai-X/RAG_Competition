"""抽出オーケストレータ（**増分**）。

  python -m app.extract.run [--full] [--workers N] [--relpath PATH]

manifest の差分（新規/変更/削除）だけを処理するので、
「新しいフォルダを追加 → 変わった分だけ再抽出」が成立する。
"""
from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from app.config import CONFIG
from app.corpus.walk import CorpusFile
from app.extract.docx import extract_docx
from app.extract.models import ExtractedDoc, Page
from app.extract.notebook import extract_ipynb
from app.extract.pdf import extract_pdf
from app.extract.plain import (extract_image, extract_plain, extract_tabular,
                               extract_unknown)
from app.extract.pptx import extract_pptx
from app.extract.xlsx import extract_xlsx

TEXT_EXTS = {".md", ".txt", ".py", ".json", ".toml", ".yaml", ".yml",
             ".html", ".xml", ".cfg", ".ini", ".sh", ".sql", ".r", ".rmd",
             ".lock", ".gitignore"}
TABULAR_EXTS = {".csv", ".tsv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
              ".emf", ".wmf"}

_DISPATCH = {
    ".docx": extract_docx, ".xlsx": extract_xlsx, ".pptx": extract_pptx,
    ".pdf": extract_pdf, ".ipynb": extract_ipynb,
}


def extractor_for(ext: str):
    if ext in _DISPATCH:
        return _DISPATCH[ext]
    if ext in TABULAR_EXTS:
        return extract_tabular
    if ext in IMAGE_EXTS:
        return extract_image
    if ext in TEXT_EXTS:
        return extract_plain
    return extract_unknown


def extract_one(cf: CorpusFile, assets_dir_s: str | None) -> ExtractedDoc:
    """1ファイル抽出。想定外の例外も **必ず記録して返す**（黙って消さない）。

    不変条件: 戻り値は常に 1 ページ以上を持つ。
    0ページで返すと `read` が何も返さず、失敗が下流で見えなくなる。
    """
    assets_dir = Path(assets_dir_s) if assets_dir_s else None
    try:
        doc = extractor_for(cf.ext)(cf, assets_dir)
        if not doc.pages:
            reason = doc.error or "本文なし"
            doc.pages = [Page(no=1, text=f"[抽出結果が空: {cf.relpath}] {reason}")]
        return doc
    except Exception as e:                       # noqa: BLE001 - 記録が目的
        doc = ExtractedDoc(doc_id=cf.doc_id, relpath=cf.relpath, mount=cf.mount,
                           project=cf.project, category=cf.category,
                           filetype=cf.ext.lstrip("."), md5=cf.md5)
        doc.error = (f"{type(e).__name__}: {e}\n"
                     + "".join(traceback.format_tb(e.__traceback__)[-3:]))
        doc.pages = [Page(no=1, text=f"[抽出失敗: {cf.relpath}]")]
        return doc


def run_extract(full: bool = False, workers: int = 8,
                only_relpath: str | None = None,
                progress=None) -> dict:
    """増分抽出を実行し、サマリを返す。"""
    from app.corpus.manifest import compute_delta, save_manifest
    from app.corpus.walk import walk_corpus
    from app.extract.catalog import build_catalog
    from app.extract import pagemap

    CONFIG.ensure_dirs()
    files = walk_corpus()
    # docx は組版してはじめてページが決まる。組版（Word 本体、無ければ
    # LibreOffice）は **親プロセスでまとめて1回** 済ませ、並列ワーカーからは
    # md5 キャッシュだけを読ませる（ワーカーごとに Word を起動させないため）。
    # コーパス全体を渡す: 較正の基準になる資料は --relpath の対象外にもある。
    pagemap.prepare(files)
    if only_relpath:
        files = [f for f in files if f.relpath == only_relpath]
        full = True

    delta = compute_delta(files, manifest={} if full else None)
    todo = delta.todo
    # 抽出結果が消えているものは差分に関わらずやり直す
    have = {p.stem for p in CONFIG.extracted_dir.glob("*.json")}
    todo += [f for f in delta.unchanged if f.doc_id not in have]

    for doc_id in delta.removed_doc_ids:
        (CONFIG.extracted_dir / f"{doc_id}.json").unlink(missing_ok=True)

    summary = {"total": len(files), "todo": len(todo),
               "added": len(delta.added), "changed": len(delta.changed),
               "removed": len(delta.removed), "errors": 0, "warnings": 0}
    if not todo:
        if not only_relpath:
            # 削除だけの差分でも台帳を更新し、処理待ち表示を解消する。
            save_manifest(files)
            build_catalog()
        return summary

    done = 0
    if workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(extract_one, cf, str(CONFIG.assets_dir)): cf
                    for cf in todo}
            for fut in as_completed(futs):
                doc = fut.result()
                doc.save(CONFIG.extracted_dir)
                summary["errors"] += bool(doc.error)
                summary["warnings"] += len(doc.warnings)
                done += 1
                if progress:
                    progress(done, len(todo), doc)
    else:
        for cf in todo:
            doc = extract_one(cf, str(CONFIG.assets_dir))
            doc.save(CONFIG.extracted_dir)
            summary["errors"] += bool(doc.error)
            summary["warnings"] += len(doc.warnings)
            done += 1
            if progress:
                progress(done, len(todo), doc)

    if not only_relpath:
        save_manifest(files)
        build_catalog()
    return summary
