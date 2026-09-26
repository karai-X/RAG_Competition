"""EMF (Enhanced Metafile) の決定的解析。

Office文書では **表やピボットが EMF 画像オブジェクトとして埋め込まれている**ことがある。
その場合シートXMLにセル値が無いので「該当なし」と誤答しやすい。
実データでは docx / xlsx / pptx の3形式すべてに存在した（＝汎用処理が要る）。

旧実装はこの解析手順を *プロンプトに散文で書いて* LLMにバイナリを読ませていた。
不安定かつ高コストなので、**決定的Pythonに降ろす**。

EMFはレコード列。各レコードは `iType(u32 LE), nSize(u32 LE), payload`。
使うレコード:
  84 EMR_EXTTEXTOUTW      +36 参照点(x,y) / +44 nChars / +48 offString(レコード先頭相対) / UTF-16LE
  39 EMR_CREATEBRUSHINDIRECT  +8 ihBrush / +12 lbStyle / +16 lbColor(COLORREF=BGR)
  37 EMR_SELECTOBJECT      +8 ihObject（以後の塗りに使うブラシを選択）
  76 EMR_BITBLT            +8 rclBounds(l,t,r,b) — 塗り矩形
  43 EMR_RECTANGLE         +8 rclBox(l,t,r,b)

解析できなければ空を返す。呼び出し側は PNG ラスタライズ→画像ワーカーへフォールバックする。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from app.extract.markup import color_name, gfm_table

EMR_HEADER = 1
EMR_EOF = 14
EMR_SELECTOBJECT = 37
EMR_CREATEBRUSHINDIRECT = 39
EMR_RECTANGLE = 43
EMR_BITBLT = 76
EMR_EXTTEXTOUTA = 83
EMR_EXTTEXTOUTW = 84

# 行のまとめ判定に使う y の許容差（EMF論理単位。フォント高さ未満を想定）
_ROW_TOLERANCE_RATIO = 0.6


@dataclass
class TextItem:
    x: int
    y: int
    text: str


@dataclass
class FillRect:
    left: int
    top: int
    right: int
    bottom: int
    color: str          # RRGGBB

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass
class EmfContent:
    texts: list[TextItem] = field(default_factory=list)
    fills: list[FillRect] = field(default_factory=list)
    n_records: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.texts)


def _colorref_to_hex(v: int) -> str:
    """COLORREF は 0x00BBGGRR。RRGGBB へ並べ替える。"""
    r, g, b = v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF
    return f"{r:02X}{g:02X}{b:02X}"


def is_emf(data: bytes) -> bool:
    return (len(data) >= 44 and data[:4] == b"\x01\x00\x00\x00"
            and data[40:44] == b" EMF")


def parse_emf(data: bytes, max_records: int = 200_000) -> EmfContent:
    out = EmfContent()
    if not is_emf(data):
        out.error = "EMFシグネチャなし"
        return out

    brushes: dict[int, str] = {}       # ihBrush -> RRGGBB
    current: str | None = None
    pos, n = 0, len(data)

    while pos + 8 <= n and out.n_records < max_records:
        itype, size = struct.unpack_from("<II", data, pos)
        if size < 8 or pos + size > n:
            out.error = f"レコード境界不正 at {pos} (type={itype} size={size})"
            break
        out.n_records += 1
        rec = data[pos:pos + size]

        if itype == EMR_CREATEBRUSHINDIRECT and size >= 24:
            ih, style, color = struct.unpack_from("<III", rec, 8)
            # lbStyle 0 = BS_SOLID のみ意味を持つ
            if style == 0:
                brushes[ih] = _colorref_to_hex(color)

        elif itype == EMR_SELECTOBJECT and size >= 12:
            (ih,) = struct.unpack_from("<I", rec, 8)
            current = brushes.get(ih)          # ストックオブジェクトは None

        elif itype in (EMR_BITBLT, EMR_RECTANGLE) and size >= 24:
            l, t, r, b = struct.unpack_from("<iiii", rec, 8)
            if current and r > l and b > t:
                out.fills.append(FillRect(l, t, r, b, current))

        elif itype == EMR_EXTTEXTOUTW and size >= 52:
            x, y, nchars, off = struct.unpack_from("<iiII", rec, 36)
            end = off + nchars * 2
            if 0 < nchars < 100_000 and off >= 36 and end <= size:
                try:
                    s = rec[off:end].decode("utf-16-le", errors="replace")
                except (UnicodeDecodeError, ValueError):
                    s = ""
                s = s.replace("\x00", "").strip()
                if s:
                    out.texts.append(TextItem(x, y, s))

        elif itype == EMR_EOF:
            break

        pos += size

    if not out.texts and out.error is None:
        out.error = "テキストレコードなし（画像のみのEMF）"
    return out


def _pct(sorted_vals: list[int], q: float) -> int:
    """下位側に寄せた分位点。サンプル数が少なくても退化しないようにする。"""
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, max(0, int((len(sorted_vals) - 1) * q)))
    return sorted_vals[i]


def _cluster(values: list[int], factor: float, floor: int = 1) -> list[int]:
    """1次元の値をギャップで分割し、各クラスタの先頭値を返す。

    ギャップ分布の形で閾値を切り替える（列数やフォントサイズを仮定しない）:

    - **二峰性**（セル内の細かい間隔と、セル間の大きい間隔が混在）
      … 例: 複数行セルを持つ表。中央値 × factor で切る。
    - **均一**（密なピボット表など、行ピッチが一定）
      … 中央値で切ると全部つながってしまうので、最小ギャップ側で切る。

    判定は p90/p50 比。均一なら 1 に近く、二峰性なら大きくなる。
    """
    vs = sorted(set(values))
    if len(vs) <= 1:
        return vs
    gaps = sorted(b - a for a, b in zip(vs, vs[1:]) if b > a)
    if not gaps:
        return vs[:1]
    p50, p10, gmax = _pct(gaps, 0.5), _pct(gaps, 0.1), gaps[-1]
    # p90 ではなく max で見る（ギャップが2種類しか無い小さな表でも判定できる）
    bimodal = p50 > 0 and (gmax / p50) >= 2.0
    tol = max(floor, int(p50 * factor)) if bimodal \
        else max(floor, int(p10 * 0.9))
    heads = [vs[0]]
    for a, b in zip(vs, vs[1:]):
        if b - a > tol:
            heads.append(b)
    return heads


def _assign(value: int, heads: list[int]) -> int:
    """value が属するクラスタ index（heads は昇順）。"""
    idx = 0
    for i, h in enumerate(heads):
        if value >= h:
            idx = i
        else:
            break
    return idx


def to_grid(content: EmfContent, row_factor: float = 1.5,
            col_factor: float = 3.0) -> list[list[str]]:
    """描画座標から表のグリッドを復元する。

    行は y、列は x のギャップクラスタリングで決める。
    セル内で複数のランに分かれたテキスト（「平均約 44,000 ～ 161,000」等）は
    同じ列に落ちるので連結される。
    """
    if not content.texts:
        return []
    row_heads = _cluster([t.y for t in content.texts], row_factor)
    col_heads = _cluster([t.x for t in content.texts], col_factor)

    grid: list[list[str]] = [["" for _ in col_heads] for _ in row_heads]
    for t in sorted(content.texts, key=lambda t: (t.y, t.x)):
        r, c = _assign(t.y, row_heads), _assign(t.x, col_heads)
        cell = grid[r][c]
        grid[r][c] = f"{cell} {t.text}".strip() if cell else t.text

    keep_c = [i for i in range(len(col_heads)) if any(r[i] for r in grid)]
    out = [[r[i] for i in keep_c] for r in grid]
    return [r for r in out if any(r)]


def annotate_fills(content: EmfContent, grid_rows: list[list[TextItem]] | None = None) -> dict:
    """塗り色ごとの、その矩形内に入るテキストを返す（ハイライト位置の特定）。"""
    out: dict[str, list[str]] = {}
    for f in content.fills:
        inside = [t.text for t in content.texts if f.contains(t.x, t.y)]
        if not inside:
            continue
        name = color_name(f.color)
        key = f"#{f.color}" + (f" {name}" if name else "")
        out.setdefault(key, []).extend(inside)
    return out


def emf_to_markdown(data: bytes, label: str = "") -> tuple[str, EmfContent]:
    """EMF → アノテーション付き Markdown。失敗時は空文字列を返す。"""
    c = parse_emf(data)
    if not c.ok:
        return "", c
    grid = to_grid(c)
    parts = [f"### 埋め込みEMFオブジェクト{': ' + label if label else ''} "
             f"(テキスト{len(c.texts)}件, 塗り{len(c.fills)}件)"]
    if grid:
        parts.append(gfm_table(grid))
    else:
        parts.append(" ".join(t.text for t in c.texts))
    fills = annotate_fills(c)
    if fills:
        parts.append("塗り分け: " + "; ".join(
            f"{k} → {', '.join(dict.fromkeys(v))[:300]}" for k, v in fills.items()))
    return "\n\n".join(parts), c
