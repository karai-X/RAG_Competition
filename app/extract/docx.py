"""docx 抽出: 段落・表・見出し・コメントを書式アノテーション付き Markdown へ。

書式（太字・下線・ハイライト・文字色）は **それ自体が答えになる** ので焼き込む。
暗号化ファイルはフラグのみ立て、復号は実行時ツールに任せる。
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from app.corpus.walk import CorpusFile
from app.extract.encrypted import is_ole_encrypted
from app.extract.markup import (HIGHLIGHT_NAMES, StyledRun, gfm_table,
                                normalize_color, render_runs)
from app.extract.models import ExtractedDoc, Page
from app.extract.office_media import extract_media, media_block
from app.extract.ooxml_chart import extract_charts
from app.extract.pagemap import block as page_block

_W_T = re.compile(r"<w:t[^>]*>([^<]*)</w:t>")


def _styled(run) -> StyledRun:
    f = run.font
    hl = None
    if f.highlight_color is not None:
        hl = HIGHLIGHT_NAMES.get(int(f.highlight_color), "黄")
    color = None
    if f.color is not None:
        color = normalize_color(getattr(f.color, "rgb", None))
    return StyledRun(text=run.text, bold=bool(run.bold), italic=bool(run.italic),
                     strike=bool(f.strike), underline=bool(run.underline),
                     highlight=hl, color=color)


def _para_text(para) -> str:
    txt = render_runs([_styled(r) for r in para.runs])
    style = (para.style.name or "").lower() if para.style else ""
    m = re.search(r"heading (\d)", style)
    if m and txt.strip():
        return "#" * min(int(m.group(1)) + 1, 6) + " " + txt
    return txt


def _table_md(table) -> str:
    rows = []
    for row in table.rows:
        rows.append([" ".join(_para_text(p) for p in cell.paragraphs
                              if p.text.strip())
                     for cell in row.cells])
    return gfm_table(rows)


def _comments(path: str) -> tuple[list[str], list[str]]:
    """(コメント行, 警告)。コメント本文とアンカーテキストを対応づける。"""
    warns: list[str] = []
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if "word/comments.xml" not in names:
                return [], warns
            cx = z.read("word/comments.xml").decode("utf-8", "replace")
            dx = z.read("word/document.xml").decode("utf-8", "replace")
    except (OSError, zipfile.BadZipFile, KeyError) as e:
        return [], [f"コメント抽出失敗: {type(e).__name__}: {e}"]

    out = []
    for m in re.finditer(
            r"<w:comment\b([^>]*)>(.*?)</w:comment>", cx, re.DOTALL):
        attrs, body = m.group(1), m.group(2)
        cid = (re.search(r'w:id="(\d+)"', attrs) or [None, ""])[1]
        author = (re.search(r'w:author="([^"]*)"', attrs) or [None, ""])[1]
        text = "".join(_W_T.findall(body)).strip()
        if not text:
            continue
        anchor = ""
        am = re.search(
            rf'<w:commentRangeStart[^>]*w:id="{cid}"/>(.*?)'
            rf'<w:commentRangeEnd[^>]*w:id="{cid}"/>', dx, re.DOTALL)
        if am:
            anchor = "".join(_W_T.findall(am.group(1))).strip()
        entry = f"- 対象:「{anchor}」 → コメント: {text}" if anchor \
            else f"- コメント: {text}"
        if author:
            entry += f"（作成者: {author}）"
        out.append(entry)
    return out, warns


def extract_docx(cf: CorpusFile, assets_dir: Path | None = None) -> ExtractedDoc:
    doc = ExtractedDoc(doc_id=cf.doc_id, relpath=cf.relpath, mount=cf.mount,
                       project=cf.project, category=cf.category,
                       filetype="docx", md5=cf.md5)

    if is_ole_encrypted(cf.raw_path):
        doc.meta["encrypted"] = True
        doc.pages = [Page(no=1, text=(
            f"[暗号化ファイル: {cf.relpath}] パスワード保護されています。"
            "横断参照フォルダのパスワード導出規則から導出したパスワードで "
            "decrypt ツールを使用してください。"))]
        return doc

    try:
        import docx as docx_lib
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        d = docx_lib.Document(cf.raw_path)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
        doc.error = f"docx open failed: {type(e).__name__}: {e}"
        return doc

    # ページ情報は、読み取り上限で落ちにくい本文先頭へ置く。
    head_block, page = page_block(cf.raw_path, cf.md5)

    blocks: list[str] = [head_block]
    n_tables = 0
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            t = _para_text(Paragraph(child, d))
            if t.strip():
                blocks.append(t)
        elif tag == "tbl":
            md = _table_md(Table(child, d))
            if md:
                blocks.append(md)
                n_tables += 1

    comments, warns = _comments(cf.raw_path)
    for w in warns:
        doc.warn(w)
    if comments:
        blocks.append("## コメント\n" + "\n".join(comments))

    charts, cwarns = extract_charts(cf.raw_path)
    for w in cwarns:
        doc.warn(w)
    media = extract_media(cf.raw_path, cf.doc_id, assets_dir)
    for w in media.warnings:
        doc.warn(w)
    mb = media_block(media)
    if charts or mb:
        blocks.append("## 埋め込みオブジェクト\n"
                      + "\n\n".join(charts + ([mb] if mb else [])))

    doc.pages = [Page(no=1, text="\n\n".join(blocks),
                      needs_vision=any(m.readable_as_image for m in media.items),
                      assets=media.asset_ids)]
    doc.meta.update(n_tables=n_tables, n_comments=len(comments),
                    n_charts=len(charts), page_state=page.state,
                    page_source=page.source, page_note=page.note,
                    n_pages_detected=page.n_pages or 0,
                    n_media=len(media.items),
                    n_emf_parsed=sum(1 for m in media.items if m.emf_markdown))
    return doc
