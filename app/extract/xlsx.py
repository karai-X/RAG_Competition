"""xlsx 抽出: シート → GFMテーブル（セル塗り色・太字・文字色・数式つき）。

このデータ群では **セルの塗り色がガント期間や状態を意味する**。
テーマ色も解決する（旧実装はコスト理由でスキップし、ガントを取りこぼしていた）。

- 小さいシート: 全セル
- 大きいシート: プレビューのみ（全量は run_python で読ませる）
- ピボットテーブル: 定義（行/列/フィルタ/値・集計方法）を明示
- 埋め込みグラフ: chart XML のキャッシュ値から表を復元
- 埋め込みメディア: EMF は決定的解析、ラスタは asset 化
"""
from __future__ import annotations

import datetime
import re
import zipfile
from pathlib import Path

from app.corpus.walk import CorpusFile
from app.extract.encrypted import is_ole_encrypted
from app.extract.markup import (color_name, fill_tag, gfm_table,
                                normalize_color)
from app.extract.models import ExtractedDoc, Page
from app.extract.office_media import extract_media, media_block
from app.extract.ooxml_chart import extract_charts

FULL_TABLE_MAX_ROWS = 200
FULL_TABLE_MAX_COLS = 60
PREVIEW_ROWS = 25
FORMULA_LOAD_MAX_BYTES = 8_000_000


# ---------------------------------------------------------------- テーマ色
def _theme_colors(path: str) -> list[str]:
    """theme1.xml の配色を取り出す（テーマ色のセル塗りを解決するため）。"""
    try:
        with zipfile.ZipFile(path) as z:
            name = next((n for n in z.namelist()
                         if n.startswith("xl/theme/") and n.endswith(".xml")), None)
            if not name:
                return []
            xml = z.read(name).decode("utf-8", "replace")
    except (OSError, zipfile.BadZipFile, KeyError):
        return []
    m = re.search(r"<a:clrScheme\b.*?</a:clrScheme>", xml, re.DOTALL)
    if not m:
        return []
    colors: list[str] = []
    for slot in re.findall(r"<a:(?:sysClr|srgbClr)\b[^>]*/>", m.group(0)):
        v = re.search(r'(?:lastClr|val)="([0-9A-Fa-f]{6})"', slot)
        colors.append(v.group(1).upper() if v else "FFFFFF")
    # OOXML のテーマ順 (lt1<->dk1, lt2<->dk2 の入れ替え) に合わせる
    if len(colors) >= 4:
        colors[0], colors[1] = colors[1], colors[0]
        colors[2], colors[3] = colors[3], colors[2]
    return colors


def _apply_tint(hex6: str, tint: float) -> str:
    """OOXML の tint（-1..1）を適用する。"""
    if not tint:
        return hex6
    out = []
    for i in (0, 2, 4):
        c = int(hex6[i:i + 2], 16)
        c = c * (1 + tint) if tint < 0 else c * (1 - tint) + 255 * tint
        out.append(f"{max(0, min(255, int(round(c)))):02X}")
    return "".join(out)


def _fill_hex(cell, theme: list[str]) -> str | None:
    f = cell.fill
    if f is None or f.patternType != "solid":
        return None
    c = f.fgColor
    if c is None:
        return None
    ctype = getattr(c, "type", None)
    if ctype == "rgb":
        return normalize_color(getattr(c, "rgb", None))
    if ctype == "theme":
        idx = getattr(c, "theme", None)
        if idx is None or idx >= len(theme):
            return None
        return _apply_tint(theme[idx], float(getattr(c, "tint", 0) or 0))
    if ctype == "indexed":
        from openpyxl.styles.colors import COLOR_INDEX
        i = getattr(c, "indexed", None)
        if i is not None and 0 <= i < len(COLOR_INDEX):
            return normalize_color(COLOR_INDEX[i])
    return None


def _font_color(cell, theme: list[str]) -> str | None:
    fc = getattr(cell.font, "color", None) if cell.font else None
    if fc is None:
        return None
    if getattr(fc, "type", None) == "theme":
        idx = getattr(fc, "theme", None)
        if idx is None or idx >= len(theme):
            return None
        return normalize_color(_apply_tint(theme[idx],
                                           float(getattr(fc, "tint", 0) or 0)))
    return normalize_color(getattr(fc, "rgb", None))


def _fmt_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, datetime.datetime):
        return v.strftime("%Y-%m-%d") if (v.hour, v.minute, v.second) == (0, 0, 0) \
            else v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, datetime.date):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _cell_text(cell, theme: list[str], fml_cell=None) -> str:
    t = _fmt_value(cell.value)
    if t and cell.font is not None:
        if cell.font.bold:
            t = f"**{t}**"
        if cell.font.underline:
            t = f"<u>{t}</u>"
        c = _font_color(cell, theme)
        if c:
            t = f"[c:{c}]{t}[/c]"
    tag = fill_tag(_fill_hex(cell, theme))
    if tag:
        t = f"{t} {tag}".strip()
    if fml_cell is not None and isinstance(fml_cell.value, str) \
            and fml_cell.value.startswith("="):
        t += f" {{fml:{fml_cell.value}}}"
    return t


def _sheet_md(ws, theme: list[str], ws_fml=None) -> tuple[str, dict]:
    from openpyxl.utils import get_column_letter
    max_row = min(ws.max_row or 0, 200_000)
    max_col = min(ws.max_column or 0, 300)
    info = {"rows": max_row, "cols": max_col, "truncated": False}
    if max_row == 0 or max_col == 0:
        return "(空シート)", info

    full = max_row <= FULL_TABLE_MAX_ROWS and max_col <= FULL_TABLE_MAX_COLS
    n_rows = max_row if full else PREVIEW_ROWS
    info["truncated"] = not full

    rows = [["行"] + [get_column_letter(c) for c in range(1, max_col + 1)]]
    for r in range(1, min(n_rows, max_row) + 1):
        row = [str(r)]
        for c in range(1, max_col + 1):
            fml = ws_fml.cell(row=r, column=c) if ws_fml is not None else None
            row.append(_cell_text(ws.cell(row=r, column=c), theme, fml))
        rows.append(row)

    md = gfm_table(rows)
    if not full:
        md += (f"\n\n(大規模シート: 全{max_row}行×{max_col}列のうち先頭{n_rows}行のみ表示。"
               "全量の集計・計算は run_python でこのファイルを直接読むこと)")
    return md, info


def _pivot_md(ws) -> tuple[str, int]:
    """ピボットテーブルの定義（行/列/フィルタ/値・集計方法）を明示する。"""
    out, n = [], 0
    for pv in getattr(ws, "_pivots", []) or []:
        n += 1
        try:
            fields = [f.name for f in pv.cache.cacheFields]
        except (AttributeError, TypeError):
            fields = []

        def fname(i):
            return fields[i] if isinstance(i, int) and 0 <= i < len(fields) \
                else f"field{i}"

        lines = [f"### ピボットテーブル定義: {getattr(pv, 'name', '?')} "
                 f"(配置: {getattr(getattr(pv, 'location', None), 'ref', '?')})"]
        for attr, label in (("rowFields", "行フィールド"),
                            ("colFields", "列フィールド"),
                            ("pageFields", "フィルタ(ページ)フィールド")):
            try:
                items = getattr(pv, attr, None) or []
                names = [fname(getattr(f, "x", getattr(f, "fld", -1)))
                         for f in items]
                if names:
                    lines.append(f"- {label}: {', '.join(names)}")
            except (AttributeError, TypeError):
                continue
        try:
            df = [f"{d.name} (元列: {fname(d.fld)}, 集計: {d.subtotal})"
                  for d in (getattr(pv, "dataFields", None) or [])]
            if df:
                lines.append(f"- 値フィールド: {'; '.join(df)}")
        except (AttributeError, TypeError):
            pass
        out.append("\n".join(lines))
    return "\n\n".join(out), n


# OOXML の比較演算子 → 日本語。**比較の主語は「セルの値」**（cellIs の定義）。
# 数値だけを渡すと、何と比べた値なのかが落ちる。
_CF_OPERATORS = {
    "lessThan": "が {0} 未満",
    "lessThanOrEqual": "が {0} 以下",
    "greaterThan": "が {0} を超える",
    "greaterThanOrEqual": "が {0} 以上",
    "equal": "が {0} と等しい",
    "notEqual": "が {0} と等しくない",
    "between": "が {0} 以上 {1} 以下",
    "notBetween": "が {0} 未満 または {1} 超",
    "containsText": "に {0} を含む",
    "notContains": "に {0} を含まない",
    "beginsWith": "が {0} で始まる",
    "endsWith": "が {0} で終わる",
}


def _dxf_fill_hex(rule, theme: list[str]) -> str | None:
    """ルールが適用する塗り色。dxf の bgColor が塗りつぶし色。"""
    dxf = getattr(rule, "dxf", None)
    fill = getattr(dxf, "fill", None) if dxf is not None else None
    if fill is None:
        return None
    for attr in ("bgColor", "fgColor"):
        c = getattr(fill, attr, None)
        if c is None:
            continue
        if getattr(c, "type", None) == "rgb":
            hx = normalize_color(getattr(c, "rgb", None))
            if hx and hx != "000000":
                return hx
        if getattr(c, "type", None) == "theme":
            idx = getattr(c, "theme", None)
            if idx is not None and idx < len(theme):
                return _apply_tint(theme[idx],
                                   float(getattr(c, "tint", 0) or 0))
    return None


def _range_edge_labels(ws, rng) -> list[str]:
    """範囲の先頭行・先頭列にあるセルの中身を並べる。

    1行目が見出しとは限らないため、位置のまま書く。

    件数は PREVIEW_ROWS（表の先頭何行を見せるかの既存の答え）に合わせる。
    """
    try:
        r0, r1 = int(rng.min_row), int(rng.max_row)
        c0, c1 = int(rng.min_col), int(rng.max_col)
    except (AttributeError, TypeError, ValueError):
        return []
    if r0 < 1 or c0 < 1:
        return []

    def _pick(cells, total):
        vals = [_fmt_value(v) for v in cells]
        vals = [v for v in vals if v]
        if not vals:
            return ""
        shown = vals[:PREVIEW_ROWS]
        tail = (f"（全{total}件のうち先頭{len(shown)}件）"
                if total > len(shown) else "")
        return " | ".join(shown) + tail

    # 範囲がシートの端から始まるときだけ出す。データ領域の内側から
    # 始まる範囲（B2:… など）の先頭行/列は値そのものなので、並べると
    # 長い数値の羅列を避ける。
    out = []
    if r0 == 1:
        top = _pick([ws.cell(row=r0, column=c).value
                     for c in range(c0, min(c1, c0 + PREVIEW_ROWS) + 1)],
                    c1 - c0 + 1)
        if top:
            out.append(f"    範囲の先頭行（シート1行目）: {top}")
    if c0 == 1:
        left = _pick([ws.cell(row=r, column=c0).value
                      for r in range(r0, min(r1, r0 + PREVIEW_ROWS) + 1)],
                     r1 - r0 + 1)
        if left:
            out.append(f"    範囲の先頭列（シートA列）: {left}")
    return out


def _conditional_formatting_md(ws, theme: list[str]) -> str:
    """条件付き書式の規則を、主語を含む形で書き出す。

    セルの塗りが規則で決まる場合、値だけを見ても塗られたことは分からない。
    規則そのものを本文に置く。`cellIs` は OOXML の定義上「セルの値」の比較
    なので、主語を明示して書く（数値だけだと何の値か伝わらない）。
    """
    cf = getattr(ws, "conditional_formatting", None)
    if cf is None:
        return ""
    lines: list[str] = []
    for rng in cf:
        for rule in getattr(rng, "rules", []) or []:
            hexc = _dxf_fill_hex(rule, theme)
            paint = (f" → 塗り {fill_tag(hexc)}" if hexc else "")
            op = getattr(rule, "operator", None)
            formula = [str(f) for f in (getattr(rule, "formula", None) or [])]
            rtype = getattr(rule, "type", "") or ""
            # 条件も書式も持たない規則は何もしない（テンプレートの残骸）。
            # 空の規則は出力しない。
            if not formula and not getattr(rule, "text", None) and not hexc:
                continue
            text = getattr(rule, "text", None)
            if rtype == "cellIs" and op in _CF_OPERATORS and formula:
                try:
                    cond = "セルの値 " + _CF_OPERATORS[op].format(*formula)
                except IndexError:
                    cond = f"セルの値 {op} {', '.join(formula)}"
            elif rtype in _CF_OPERATORS and text is not None:
                # containsText 系は比較対象が rule.text に入る。
                # formula には Excel が生成した SEARCH 式が入っており、
                # そのまま出すと条件が読めない。
                cond = "セルの値 " + _CF_OPERATORS[rtype].format(text)
            elif rtype == "expression" and formula:
                cond = f"式 {formula[0]} が真"
            else:
                cond = f"{rtype}" + (f" {op}" if op and op != rtype else "")                     + (f" {', '.join(formula)}" if formula else "")                     + (f" {text!r}" if text else "")
            lines.append(f"- 範囲 {rng.sqref}: {cond}{paint}")
            # 規則には「セルの値」としか書けないので、その範囲が
            # どのラベルに重なっているかを添える（解釈はモデルに任せる）
            for cr in getattr(rng.sqref, "ranges", []) or []:
                lines.extend(_range_edge_labels(ws, cr))
    if not lines:
        return ""
    return ("### 条件付き書式（セルの色は下の規則で決まります。"
            "値そのものには色の情報がありません）\n" + "\n".join(lines))


def _anchored_images_note(ws) -> str:
    """シートに貼られた画像を、そのシートの本文で知らせる。

    openpyxl / pandas はセルしか見ないので、図だけが貼られたシートは
    「(空シート)」と出る。実体はグラフが並んだシートで、これは抽出が
    実体と食い違っている状態。**セルに値が無いこととシートが空である
    ことは別**なので、貼られている枚数と読み取り結果の在り処を書く。

    枚数は drawing の関係から決まり、資料の内容には依存しない。
    """
    try:
        n = len(ws._images)
    except AttributeError:
        return ""
    if not n:
        return ""
    return (f"[このシートには画像が{n}枚貼られています。"
            "セルを読んでも図の中身は得られません。"
            "図の中身は末尾の「## 埋め込みグラフ・画像」の節にある"
            "読み取り結果を見てください]")


def extract_xlsx(cf: CorpusFile, assets_dir: Path | None = None) -> ExtractedDoc:
    doc = ExtractedDoc(doc_id=cf.doc_id, relpath=cf.relpath, mount=cf.mount,
                       project=cf.project, category=cf.category,
                       filetype="xlsx", md5=cf.md5)

    if is_ole_encrypted(cf.raw_path):
        doc.meta["encrypted"] = True
        doc.pages = [Page(no=1, text=(
            f"[暗号化ファイル: {cf.relpath}] パスワード保護されています。"
            "横断参照フォルダのパスワード導出規則から導出したパスワードで "
            "decrypt ツールを使用してください。"))]
        return doc

    try:
        import openpyxl
        wb = openpyxl.load_workbook(cf.raw_path, data_only=True)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as e:
        doc.error = f"xlsx open failed: {type(e).__name__}: {e}"
        return doc

    wb_fml = None
    if cf.size <= FORMULA_LOAD_MAX_BYTES:
        try:
            wb_fml = openpyxl.load_workbook(cf.raw_path, data_only=False)
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as e:
            doc.warn(f"数式ロード失敗: {type(e).__name__}: {e}")

    theme = _theme_colors(cf.raw_path)
    n_pivots = 0
    for i, ws in enumerate(wb.worksheets, start=1):
        ws_fml = (wb_fml[ws.title]
                  if wb_fml is not None and ws.title in wb_fml.sheetnames
                  else None)
        body, info = _sheet_md(ws, theme, ws_fml)
        # 非表示シートは抽出テキストが表示シートと区別できなくなる。
        # 「表示されている〜」という限定を判別できるよう見出しに出す。
        state = "" if ws.sheet_state == "visible" else f"（{ws.sheet_state}）"
        parts = [f"## シート: {ws.title}{state}"]
        # 見出しの直後に置く。冒頭だけを読まれても図の存在が伝わるように。
        note = _anchored_images_note(ws)
        if note:
            parts.append(note)
        # 表より前に置く。後ろにすると、大きなシートでは read_text の
        # 出力が切り詰められても規則が残るよう、表より前に置く。
        cfm = _conditional_formatting_md(ws, theme)
        if cfm:
            parts.append(cfm)
        parts.append(body)
        pv, n = _pivot_md(ws)
        n_pivots += n
        if pv:
            parts.append(pv)
        doc.pages.append(Page(no=i, text="\n\n".join(parts), label=ws.title))

    charts, cwarns = extract_charts(cf.raw_path)
    for w in cwarns:
        doc.warn(w)
    media = extract_media(cf.raw_path, cf.doc_id, assets_dir)
    for w in media.warnings:
        doc.warn(w)
    mb = media_block(media)
    if charts or mb:
        parts = ["## 埋め込みグラフ・画像"] + charts + ([mb] if mb else [])
        doc.pages.append(Page(
            no=len(doc.pages) + 1, text="\n\n".join(parts), label="charts_media",
            needs_vision=any(m.readable_as_image for m in media.items),
            assets=media.asset_ids))

    doc.meta.update(sheets=wb.sheetnames, n_pivots=n_pivots,
                    n_charts=len(charts), n_media=len(media.items),
                    n_emf_parsed=sum(1 for m in media.items if m.emf_markdown),
                    theme_colors=len(theme))
    wb.close()
    if wb_fml is not None:
        wb_fml.close()
    return doc
