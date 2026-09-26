"""pdf 抽出: pypdfium2 でページ単位テキスト + スキャンページ検出。

**PDF の書式（マーカー・太字・斜体・下線・色）は抽出テキストに現れない。**
書式を問われたら read_image でページを見る必要があるため、
画像を含むページには needs_vision を立てて誘導する。
テキスト層が空のページは OCR 対象（ocr 層が拾う）。
"""
from __future__ import annotations

import io
from pathlib import Path

from app.corpus.walk import CorpusFile
from app.extract.markup import color_name, normalize_color
from app.extract.models import ExtractedDoc, Page

SCANNED_TEXT_THRESHOLD = 40   # 文字数がこれ未満 + 画像あり → スキャン候補
PDFIUM_OBJ_TEXT = 1
PDFIUM_OBJ_PATH = 2
PDFIUM_OBJ_IMAGE = 3

# 白・ほぼ白の塗りはハイライトではない（紙の下地）
_WHITE_MIN = 245

# 下線は細い塗り矩形として描かれる（テキスト層・フォント名・注釈のいずれにも
# 下線情報を直接取得できないため、太さの上限は組版の一般則から取る:
# 下線の太さは字送りサイズの 1/14〜1/20 が標準（Word・LaTeX とも同水準）なので、
# 1/5 を上限にすれば余裕を持って収まり、塗りつぶしやハイライトとは区別できる。
# 位置・幅は文書から導出するのでしきい値を持たない（_underlined_chars を参照）。
_UL_MAX_THICKNESS_EM = 0.2


# 浮動小数の丸め誤差を無視するためだけの値。傾きの大小を測る値ではない
# （合成斜体の傾きは 0.2〜0.4 程度で、誤差とは桁が4つ以上違う）。
_SKEW_EPS = 1e-6


def _char_skew(tp, i) -> float:
    """文字の変換行列のせん断量。回転・拡大では 0 になる。

    PDF は italic 版フォントを持たない書体に対し、行列を傾けて斜体を作る
    ことがある（合成斜体）。この場合フォント名にも記述子の flags にも
    斜体の情報が出ないので、行列そのものを見るしかない。

    shear = (a*c + b*d) / (a^2 + b^2)
      回転  a=cosθ b=sinθ c=-sinθ d=cosθ  → 分子 0
      拡大  b=c=0                          → 分子 0
      傾き  a=1 b=0 c=k d=1                → k
    """
    import ctypes

    import pypdfium2.raw as pr
    try:
        m = pr.FS_MATRIX()
        if not pr.FPDFText_GetMatrix(tp.raw, i, ctypes.byref(m)):
            return 0.0
    except (AttributeError, OSError, ValueError, TypeError):
        return 0.0
    den = m.a * m.a + m.b * m.b
    if not den:
        return 0.0
    return (m.a * m.c + m.b * m.d) / den


def _fill_rects(page) -> list[tuple[str, tuple[float, float, float, float]]]:
    """テキストの背後にある塗り矩形。PDFのハイライトはこれで表現される。

    （注釈オブジェクトとして持つPDFもあるが、この資料群では塗り矩形だった）
    """
    import ctypes

    import pypdfium2.raw as pr
    out = []
    try:
        objs = list(page.get_objects(max_depth=3))
    except (RuntimeError, ValueError):
        return out
    for obj in objs:
        if obj.type != PDFIUM_OBJ_PATH:
            continue
        R, G, B, A = (ctypes.c_uint() for _ in range(4))
        try:
            ok = pr.FPDFPageObj_GetFillColor(obj.raw, ctypes.byref(R),
                                             ctypes.byref(G), ctypes.byref(B),
                                             ctypes.byref(A))
        except (AttributeError, OSError):
            return out
        if not ok or A.value == 0:
            continue
        if R.value >= _WHITE_MIN and G.value >= _WHITE_MIN and B.value >= _WHITE_MIN:
            continue
        l, b, r, t = (ctypes.c_float() for _ in range(4))
        if not pr.FPDFPageObj_GetBounds(obj.raw, ctypes.byref(l), ctypes.byref(b),
                                        ctypes.byref(r), ctypes.byref(t)):
            continue
        out.append((f"{R.value:02X}{G.value:02X}{B.value:02X}",
                    (l.value, b.value, r.value, t.value)))
    return out


def _underlined_chars(page, tp, n_chars: int) -> set[int]:
    """下線が引かれている文字の添字。

    **矩形→文字**の向きで判定する。文字→矩形だと、カンマや句点のように
    ボックスの下端が他の字とずれるものを取りこぼし、下線が細切れになる。

    判定はすべて**その文字自身の寸法**を基準にする:
      細いか   … 矩形の高さ < フォントサイズ × 組版上の上限
      下にあるか … 矩形の上端が文字の上端より下（取り消し線・上線を除く）
      掛かるか … 矩形が文字の中心を水平方向に含む
      近いか   … 文字の下端から1文字ぶん以内（離れた区切り線を除く）
    """
    import ctypes

    import pypdfium2.raw as pr
    rects = _fill_rects(page)
    if not rects:
        return set()
    boxes = [box for _hex, box in rects]
    out: set[int] = set()
    for i in range(n_chars):
        L, T, R, B = (ctypes.c_double() for _ in range(4))
        try:
            pr.FPDFText_GetCharBox(tp.raw, i, ctypes.byref(L), ctypes.byref(R),
                                   ctypes.byref(B), ctypes.byref(T))
            size = float(pr.FPDFText_GetFontSize(tp.raw, i))
        except (AttributeError, OSError, ValueError):
            return out
        if size <= 0:
            size = abs(T.value - B.value) or 1.0
        cx = (L.value + R.value) / 2
        c_top = max(T.value, B.value)
        c_bot = min(T.value, B.value)
        for (l, b, r, t) in boxes:
            if (t - b) >= size * _UL_MAX_THICKNESS_EM:
                continue
            if not (l <= cx <= r):
                continue
            if t >= c_top:
                continue
            if c_bot - t > size:
                continue
            out.add(i)
            break
    return out


def _styled_text(page) -> str | None:
    """文字色とハイライトを焼き込んだページテキスト。

    PDFの書式は従来 read_image でしか確認できなかったが、テキスト層があれば
    pypdfium2 から **1文字単位の塗り色** が取れる。画像から数字を目視するより
    正確なので、抽出時にマークアップとして埋め込む。
    """
    import ctypes

    import pypdfium2.raw as pr
    try:
        tp = page.get_textpage()
        n = pr.FPDFText_CountChars(tp.raw)
    except (RuntimeError, ValueError):
        return None
    if n <= 0:
        return ""

    rects = _fill_rects(page)
    underlined = _underlined_chars(page, tp, n)
    # (文字, 色, ハイライト, 太字, 斜体, 下線)
    chars: list[tuple[str, str | None, str | None, bool, bool, bool]] = []
    for i in range(n):
        ch = chr(pr.FPDFText_GetUnicode(tp.raw, i))
        R, G, B, A = (ctypes.c_uint() for _ in range(4))
        color = None
        try:
            if pr.FPDFText_GetFillColor(tp.raw, i, ctypes.byref(R), ctypes.byref(G),
                                        ctypes.byref(B), ctypes.byref(A)):
                color = normalize_color(f"{R.value:02X}{G.value:02X}{B.value:02X}")
        except (AttributeError, OSError):
            return None
        # 太字・斜体はフォントのウェイトと名前から判定する
        bold = italic = False
        try:
            if pr.FPDFText_GetFontWeight(tp.raw, i) >= 600:
                bold = True
            buf = ctypes.create_string_buffer(128)
            flags = ctypes.c_int()
            ln = pr.FPDFText_GetFontInfo(tp.raw, i, buf, 128, ctypes.byref(flags))
            if ln:
                fname = buf.raw[:ln].decode("utf-8", "replace").rstrip("\x00").lower()
                bold = bold or "bold" in fname
                italic = "italic" in fname or "oblique" in fname
        except (AttributeError, OSError, ValueError):
            pass
        # フォントが斜体を名乗らなくても、行列が傾いていれば斜体
        if not italic and abs(_char_skew(tp, i)) > _SKEW_EPS:
            italic = True
        hl = None
        if rects and not ch.isspace():
            L, T, Rt, Bt = (ctypes.c_double() for _ in range(4))
            pr.FPDFText_GetCharBox(tp.raw, i, ctypes.byref(L), ctypes.byref(Rt),
                                   ctypes.byref(Bt), ctypes.byref(T))
            cx, cy = (L.value + Rt.value) / 2, (Bt.value + T.value) / 2
            for hexv, (l, b, r, t) in rects:
                if l <= cx <= r and b <= cy <= t:
                    hl = hexv
                    break
        chars.append((ch, color, hl, bold, italic, i in underlined))

    # 行折り返しや空白でスタイルが途切れると断片だらけになるので、
    # 前後が同じスタイルなら空白もその流れに含める。
    for i, (ch, color, hl, bold, italic, ul) in enumerate(chars):
        if not ch.isspace() or ch in "\r\n":
            continue
        prev = next((chars[j] for j in range(i - 1, -1, -1)
                     if not chars[j][0].isspace()), None)
        nxt = next((chars[j] for j in range(i + 1, len(chars))
                    if not chars[j][0].isspace()), None)
        if prev and nxt and prev[1:] == nxt[1:]:
            chars[i] = (ch, prev[1], prev[2], prev[3], prev[4], prev[5])

    # 同じスタイルの連続をまとめてマークアップする
    parts: list[str] = []
    buf: list[str] = []
    cur: tuple = (None, None, False, False, False)

    def flush() -> None:
        if not buf:
            return
        text = "".join(buf)
        core = text.strip()
        if core:
            pre = text[:len(text) - len(text.lstrip())]
            post = text[len(text.rstrip()):]
            color, hl, bold, italic, ul = cur
            if italic:
                core = f"*{core}*"
            if bold:
                core = f"**{core}**"
            if ul:
                core = f"<u>{core}</u>"
            if hl:
                name = color_name(hl) or hl
                core = f"=={core}=={{{name}}}"
            if color:
                core = f"[c:{color}]{core}[/c]"
            parts.append(pre + core + post)
        else:
            parts.append(text)
        buf.clear()

    for ch, color, hl, bold, italic, ul in chars:
        if (color, hl, bold, italic, ul) != cur:
            flush()
            cur = (color, hl, bold, italic, ul)
        buf.append(ch)
    flush()
    return "".join(parts)


def extract_pdf(cf: CorpusFile, assets_dir: Path | None = None) -> ExtractedDoc:
    doc = ExtractedDoc(doc_id=cf.doc_id, relpath=cf.relpath, mount=cf.mount,
                       project=cf.project, category=cf.category,
                       filetype="pdf", md5=cf.md5)
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(cf.raw_path)
    except (OSError, ValueError, RuntimeError) as e:
        doc.error = f"pdf open failed: {type(e).__name__}: {e}"
        return doc

    scanned: list[int] = []
    n_styled = 0
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                text = page.get_textpage().get_text_range()
            except (RuntimeError, ValueError) as e:
                text = ""
                doc.warn(f"p{i+1} テキスト取得失敗: {type(e).__name__}")
            # 文字色・ハイライトを焼き込んだ版が取れればそちらを使う
            if text.strip():
                try:
                    styled = _styled_text(page)
                except Exception as e:              # noqa: BLE001
                    styled = None
                    doc.warn(f"p{i+1} 書式抽出失敗: {type(e).__name__}: {e}")
                if styled and len(styled.strip()) >= len(text.strip()) * 0.9:
                    text = styled
                    n_styled += 1
            try:
                n_images = sum(1 for obj in page.get_objects(max_depth=2)
                               if obj.type == PDFIUM_OBJ_IMAGE)
            except (RuntimeError, ValueError):
                n_images = 0

            is_scanned = (len(text.strip()) < SCANNED_TEXT_THRESHOLD
                          and n_images > 0)
            parts = [f"## ページ {i + 1}"]
            if text.strip():
                parts.append(text.strip())
            if is_scanned:
                scanned.append(i + 1)
                parts.append("[スキャンページ: テキスト層なし。OCR結果は後段で追加]")
            elif n_images:
                parts.append(f"[このページに画像/グラフ {n_images} 点 "
                             "— 書式・図の確認は read_image で行うこと]")
            doc.pages.append(Page(no=i + 1, text="\n\n".join(parts),
                                  needs_vision=is_scanned or n_images > 0))
    finally:
        pdf.close()

    doc.meta.update(n_pages=len(doc.pages), scanned_pages=scanned,
                    n_styled_pages=n_styled)
    return doc


def render_page_png(raw_path: str, page_no: int, max_edge_px: int = 1568) -> bytes:
    """PDFページをPNGへ（read_image / OCR 用）。"""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(raw_path)
    try:
        page = pdf[page_no - 1]
        w, h = page.get_size()
        bitmap = page.render(scale=max_edge_px / max(w, h))
        buf = io.BytesIO()
        bitmap.to_pil().save(buf, format="PNG")
        return buf.getvalue()
    finally:
        pdf.close()
