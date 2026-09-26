"""OOXML の埋め込みグラフ（DrawingML chart）をキャッシュ値から復元する。

docx / xlsx / pptx のいずれも `(word|xl|ppt)/charts/chartN.xml` に
系列・カテゴリ・値が **キャッシュされて** 入っている。
グラフしか載っていない文書（本文テキストが0の docx など）があるため、
**形式によらず一律に走査する**。
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from app.extract.markup import gfm_table

CHART_PATH = re.compile(r"^(?:word|xl|ppt)/charts/chart(?:Ex)?\d*\.xml$")
_A_T = re.compile(r"<a:t>([^<]*)</a:t>")
_C_V = re.compile(r"<c:v>([^<]*)</c:v>")
_PT = re.compile(r'<c:pt idx="(\d+)"[^>]*>\s*<c:v>([^<]*)</c:v>')


def chartex_markdown(xml: str, name: str = "") -> str:
    """Excel の新形式グラフ (ChartEx) からタイトルと系列を復元する。

    ヒストグラム等は ``xl/charts/chartExN.xml`` に保存され、openpyxl も
    従来の ``c:chart`` 用抽出も認識しない。値キャッシュを持たない場合でも
    系列名と参照式は XML 内にあるため、少なくとも「どの列のグラフか」は
    失わず抽出する。
    """
    title = ""
    mt = re.search(r"<cx:title\b.*?</cx:title>", xml, re.DOTALL)
    if mt:
        title = "".join(_A_T.findall(mt.group(0))).strip()

    rows: list[list[str]] = [["系列", "グラフ種別", "データ参照"]]
    for ser in re.findall(r"<cx:series\b([^>]*)>(.*?)</cx:series>",
                          xml, re.DOTALL):
        attrs, body = ser
        layout = ""
        ml = re.search(r'\blayoutId="([^"]+)"', attrs)
        if ml:
            layout = ml.group(1)
        series_name = ""
        mtx = re.search(r"<cx:txData\b.*?</cx:txData>", body, re.DOTALL)
        if mtx:
            mv = re.search(r"<cx:v>([^<]*)</cx:v>", mtx.group(0))
            if mv:
                series_name = mv.group(1).strip()
        formula = ""
        mf = re.search(r"<cx:(?:num|str)Dim\b.*?<cx:f>([^<]*)</cx:f>",
                       xml, re.DOTALL)
        if mf:
            formula = mf.group(1).strip()
        rows.append([series_name or f"系列{len(rows)}", layout, formula])

    layouts = [row[1] for row in rows[1:] if row[1]]
    ctype = "/".join(dict.fromkeys(layouts)) or "chartEx"
    head = f"### 埋め込みグラフ ({ctype})" + (f": {title}" if title else "")
    if name:
        head += f" [{name}]"
    if len(rows) > 1:
        return head + "\n" + gfm_table(rows)
    return head + " (系列データなし)"


def _series_points(ser: str, tag: str) -> list[str]:
    m = re.search(rf"<c:{tag}>(.*?)</c:{tag}>", ser, re.DOTALL)
    if not m:
        return []
    found = _PT.findall(m.group(1))
    return [v for _, v in sorted(found, key=lambda x: int(x[0]))]


def chart_markdown(xml: str, name: str = "") -> str:
    if "drawing/2014/chartex" in xml or "<cx:chartSpace" in xml:
        return chartex_markdown(xml, name)
    m = re.search(r"<c:(\w+Chart)>", xml)
    ctype = m.group(1) if m else "chart"
    title = ""
    mt = re.search(r"<c:title>.*?</c:title>", xml, re.DOTALL)
    if mt:
        title = "".join(_A_T.findall(mt.group(0))).strip()

    # 軸タイトル（「x軸は何か」を問う質問で効く）
    axes = []
    for am in re.finditer(r"<c:(?:cat|val|date)Ax>.*?</c:(?:cat|val|date)Ax>",
                          xml, re.DOTALL):
        t = re.search(r"<c:title>.*?</c:title>", am.group(0), re.DOTALL)
        if t:
            label = "".join(_A_T.findall(t.group(0))).strip()
            if label:
                axes.append(label)

    rows: list[list[str]] = []
    has_categories = False
    for ser in re.findall(r"<c:ser>(.*?)</c:ser>", xml, re.DOTALL):
        sname = ""
        mn = re.search(r"<c:tx>.*?</c:tx>", ser, re.DOTALL)
        if mn:
            vs = _C_V.findall(mn.group(0))
            sname = vs[0] if vs else ""
        cats = _series_points(ser, "cat")
        vals = _series_points(ser, "val")
        if not rows:
            if cats:
                has_categories = True
                header = cats
            else:
                # カテゴリ値が埋め込まれていないグラフは、描画時のx軸が
                # 1,2,3… の連番になる（OOXMLの既定）。抽出でも **描画と同じ
                # ラベル** を出す。独自表記による位置のずれを避ける。
                header = [str(i + 1) for i in range(len(vals))]
            rows.append(["系列"] + header)
        rows.append([sname or f"系列{len(rows)}"] + vals)

    head = f"### 埋め込みグラフ ({ctype})" + (f": {title}" if title else "")
    if name:
        head += f" [{name}]"
    if axes:
        head += f"\n- 軸: {' / '.join(dict.fromkeys(axes))}"
    if len(rows) > 1 and not has_categories:
        head += ("\n- x軸: カテゴリ値は埋め込まれていない。"
                 "列見出しの 1,2,3… は描画上のx軸の位置（1始まり）を表す")
    if len(rows) > 1:
        return head + "\n" + gfm_table(rows)
    return head + " (キャッシュデータなし)"


def extract_charts(path: str | Path) -> tuple[list[str], list[str]]:
    """(グラフのMarkdownブロック, 警告)。"""
    blocks: list[str] = []
    warns: list[str] = []
    try:
        z = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as e:
        return blocks, [f"グラフ走査不可: {type(e).__name__}: {e}"]
    with z:
        for n in sorted(x for x in z.namelist() if CHART_PATH.match(x)):
            try:
                xml = z.read(n).decode("utf-8", "replace")
            except (KeyError, OSError) as e:
                warns.append(f"{n}: 読み出し失敗 {type(e).__name__}")
                continue
            blocks.append(chart_markdown(xml, n))
    return blocks, warns
