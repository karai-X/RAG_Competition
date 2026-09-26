"""書式アノテーションの単一定義。

色・太字・ハイライトなどの書式を抽出時にマークアップとして焼き込み、
検索とLLMで同じ表現として扱えるようにする。

  **太字** / *斜体* / ~~取消線~~ / <u>下線</u>
  ==ハイライト=={色名} / [c:RRGGBB]文字色[/c]
  値 {fill:#RRGGBB 色名}      セル塗り
  {shape-fill:#RRGGBB 色名}   図形塗り
  {fml:=式}                   数式
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass

# Word の highlight_color enum → 色名
HIGHLIGHT_NAMES = {
    1: "黒", 2: "青", 3: "水色", 4: "緑", 5: "紫", 6: "赤", 7: "黄",
    8: "白", 9: "濃青", 10: "濃水色", 11: "濃緑", 12: "濃紫", 13: "濃赤",
    14: "濃黄", 15: "グレー", 16: "薄グレー",
}

# 無視する文字色（黒・自動・濃灰）— 注記してもノイズにしかならない
_NEUTRAL_COLORS = {None, "000000", "0A0A0A", "1A1A1A", "212121", "333333",
                   "AUTO", ""}


@dataclass
class StyledRun:
    text: str
    bold: bool = False
    italic: bool = False
    strike: bool = False
    underline: bool = False
    highlight: str | None = None   # 色名
    color: str | None = None       # RRGGBB（黒系は None）


def normalize_color(rgb) -> str | None:
    """RGBColor / str / int → 'RRGGBB' または None（黒系・自動）。"""
    if rgb is None:
        return None
    if isinstance(rgb, int):
        s = f"{rgb:06X}"
    else:
        s = str(rgb).upper().lstrip("#")
    if len(s) == 8:          # AARRGGBB
        s = s[2:]
    if len(s) != 6 or s in _NEUTRAL_COLORS:
        return None
    if any(c not in "0123456789ABCDEF" for c in s):
        return None
    return s


def color_name(hex6: str | None) -> str:
    """HSV色相ベースの色名。淡い色も「薄オレンジ」等と判定できるようにする。

    「オレンジにハイライトされている行」のような質問は、
    厳密なRGB値ではなく **人が呼ぶ色名** で聞かれる。
    """
    if not hex6 or len(hex6) != 6:
        return ""
    try:
        r, g, b = (int(hex6[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return ""
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    deg = h * 360
    if s < 0.06:
        if v > 0.92:
            return ""                       # 白は注記しない（無意味）
        return "黒" if v < 0.25 else "グレー"
    if deg < 15 or deg >= 345:
        base = "赤"
    elif deg < 48:
        base = "茶" if (s > 0.5 and v < 0.6) else "オレンジ"
    elif deg < 68:
        base = "黄"
    elif deg < 165:
        base = "緑"
    elif deg < 210:
        base = "水色"
    elif deg < 255:
        base = "青"
    elif deg < 290:
        base = "紫"
    else:
        base = "ピンク"
    if v < 0.35:
        return "濃" + base
    if s < 0.3 and v > 0.85:
        return "薄" + base
    return base


def fill_tag(hex6: str | None, kind: str = "fill") -> str:
    """セル塗り / 図形塗りのアノテーション。白・無色は付けない。"""
    if not hex6:
        return ""
    name = color_name(hex6)
    if not name:                            # 白 = 意味を持たない
        return ""
    return f"{{{kind}:#{hex6} {name}}}"


def merge_runs(runs: list[StyledRun]) -> list[StyledRun]:
    """同一スタイルの隣接ランを結合する（Wordはランを細切れにするため）。"""
    out: list[StyledRun] = []
    for r in runs:
        if not r.text:
            continue
        key = (r.bold, r.italic, r.strike, r.underline, r.highlight, r.color)
        if out and (out[-1].bold, out[-1].italic, out[-1].strike,
                    out[-1].underline, out[-1].highlight, out[-1].color) == key:
            out[-1].text += r.text
        else:
            out.append(StyledRun(**vars(r)))
    return out


def render_runs(runs: list[StyledRun]) -> str:
    """スタイル付きラン列 → アノテーション付きテキスト。

    前後の空白は装飾の外に出す（`** 太字 **` は Markdown として壊れるため）。
    """
    parts: list[str] = []
    for r in merge_runs(runs):
        t = r.text
        core = t.strip()
        if not core:
            parts.append(t)
            continue
        pre = t[:len(t) - len(t.lstrip())]
        post = t[len(t.rstrip()):]
        if r.strike:
            core = f"~~{core}~~"
        if r.underline:
            core = f"<u>{core}</u>"
        if r.italic:
            core = f"*{core}*"
        if r.bold:
            core = f"**{core}**"
        if r.highlight:
            core = f"=={core}=={{{r.highlight}}}"
        if r.color:
            core = f"[c:{r.color}]{core}[/c]"
        parts.append(pre + core + post)
    return "".join(parts)


def gfm_table(rows: list[list[str]]) -> str:
    """行列（セル文字列）→ GFMテーブル。"""
    if not rows:
        return ""
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return ""

    def esc(c: str) -> str:
        return (c or "").replace("|", "\\|").replace("\n", " ").strip()

    lines = []
    for i, row in enumerate(rows):
        cells = [esc(c) for c in row] + [""] * (width - len(row))
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "---|" * width)
    return "\n".join(lines)


def strip_markup(text: str) -> str:
    """書式アノテーションを外した**素の本文**を返す。

    抽出テキストに含まれる書式注記を、正規表現による値抽出を妨げないよう除く。

    落とすのは**見た目の注記だけ**。数式や画像参照など中身を持つものは残す
    （`{fml:=式}` は `=式` にして式そのものは保つ）。

    **この関数はサンドボックスへ丸ごと注入されるので、自己完結させること**
    （モジュール変数を参照しない・import は関数内で行う）。
    """
    import re as _re
    if not text:
        return text
    s = text
    s = _re.sub(r"\[c:[0-9A-Fa-f]{6}\](.*?)\[/c\]", r"\1", s, flags=_re.S)
    s = _re.sub(r"\[/c\]", "", s)                       # 対応の壊れた閉じタグ
    s = _re.sub(r"==(.*?)==\{[^}]*\}", r"\1", s, flags=_re.S)
    s = _re.sub(r"\{(?:shape-)?fill:[^}]*\}", "", s)
    s = _re.sub(r"\{fml:(.*?)\}", r"\1", s, flags=_re.S)
    s = _re.sub(r"</?u>", "", s)
    s = _re.sub(r"\*\*(.+?)\*\*", r"\1", s, flags=_re.S)
    s = _re.sub(r"~~(.+?)~~", r"\1", s, flags=_re.S)
    s = _re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)", r"\1", s)
    s = _re.sub(r"[ \t]{2,}", " ", s)
    return s
