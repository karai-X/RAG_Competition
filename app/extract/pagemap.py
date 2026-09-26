"""docx のページ位置確定（インジェストの一工程）。

docx は本来「ページ」を持たない。組版してはじめて決まる。次の順で確定させる。

1. **Word 自身の改ページ記録** — Word が保存時に書き込む `w:lastRenderedPageBreak`。
   Word が実際に描画した結果そのものなので、あるならこれが最も確かな根拠。
2. **Word 本体に組版させる** — 一時コピーを Word に開かせて保存し、
   書き込まれた改ページ記録を読み戻す。1 と同じ状態を自分で作ることになる。
3. **LibreOffice で組版する** — Word が無い環境向けの縮退。
   Word と組版が一致しない場合があるため、改ページ記録を持つdocxで較正し、
   `config.pagemap_min_agreement` を満たさなければ採用しない。
4. 必要な組版器が使えない、または組版に失敗した場合は、エラーを出して前処理を止める。

**元ファイルは常に読むだけ**。組版は一時ディレクトリの複製に対して行う。
用紙サイズ(pgSz)を持たない文書は、OOXML 仕様上ページが「開いたアプリの既定用紙」で
決まってしまうため、`config.pagemap_assumed_paper` の設定を仮定して組版する。

  python -m app.extract.pagemap --validate    # 組版器の照合結果を表示
  RAG_DISABLE_WORD=1 ...                      # Word 経路を無効化（縮退の確認）
  RAG_DISABLE_SOFFICE=1 ...                   # LibreOffice 経路を無効化
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from app.config import CONFIG

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
# v2: 用紙の仮定値を Letter+35/30/30/30mm（共有資料由来）から
#     A4+25.4mm（一般値）へ変更した。組版結果が変わるので版を上げる。
CACHE_VERSION = "v2"

# 見出し行が長くなりすぎるとツール出力の上限を圧迫するので上限を設ける
MAX_MAP_LINES = 60
MAX_HEAD_CHARS = 60


# --------------------------------------------------------------------------
# Word 自身の改ページ記録
# --------------------------------------------------------------------------

def _boundaries(el) -> int:
    """el 配下にある改ページ境界の数。

    表の行が改ページをまたぐとき、Word は **その行の各セルに1つずつ**
    改ページマーカーを書く。素朴に数えるとセル数の分だけ多重に数えて
    しまうので、行の中では「各セルの k 番目のマーカーが同じ改ページ」と
    みなし、セルごとのマーカー数の最大値をその行の境界数とする。
    1つの行が複数ページにまたがる場合も正しく数えられる。
    """
    from lxml import etree
    tag = etree.QName(el).localname
    if tag == "lastRenderedPageBreak":
        return 1
    if tag == "tr":
        per_cell = [sum(_boundaries(c) for c in tc)
                    for tc in el.findall(W + "tc")]
        outside = sum(_boundaries(c) for c in el
                      if etree.QName(c).localname != "tc")
        return (max(per_cell) if per_cell else 0) + outside
    return sum(_boundaries(c) for c in el)


def paragraph_page_starts(path: str) -> list[tuple[int, str]]:
    """[(Wordのページ番号, そのページ先頭の本文)]。**段落レベルの境界のみ**。

    表の行をまたぐ改ページは、行内の各セルが横に並んで続くので
    「ページ先頭のテキスト」が一意に決まらない。位置の比較には使えないので除く。
    """
    from lxml import etree
    try:
        with zipfile.ZipFile(path) as z:
            root = etree.fromstring(z.read("word/document.xml"))
    except (OSError, zipfile.BadZipFile, KeyError, etree.XMLSyntaxError):
        return []
    body = root.find(W + "body")
    if body is None:
        return []
    page, out = 1, []
    for child in body:
        if etree.QName(child).localname != "p":
            page += _boundaries(child)
            continue
        seen, buf = False, []
        for node in child.iter():
            ln = etree.QName(node).localname
            if ln == "lastRenderedPageBreak":
                if seen and buf:
                    out.append((page, "".join(buf)))
                page += 1
                seen, buf = True, []
            elif ln == "t" and seen:
                buf.append(node.text or "")
        if seen and buf:
            out.append((page, "".join(buf)))
    return out


def _heading_style_ids(styles_xml: str) -> set[str]:
    ids = set()
    for m in re.finditer(r"<w:style\b([^>]*)>(.*?)</w:style>", styles_xml, re.S):
        sid = re.search(r'w:styleId="([^"]+)"', m.group(1))
        name = re.search(r'<w:name w:val="([^"]+)"', m.group(2))
        if sid and name and re.match(r"heading\s*\d", name.group(1), re.I):
            ids.add(sid.group(1))
    return ids


_SECTPR = re.compile(rb"<w:sectPr\b.*?</w:sectPr>|<w:sectPr\b[^>]*/>", re.S)


@dataclass
class DocOutline:
    """docx から機械的に取れる構造。"""
    headings: list[str] = field(default_factory=list)     # 文書順の見出し
    word_pages: int | None = None                         # Word の記録から
    word_heading_pages: list[int] = field(default_factory=list)
    declared_pages: int | None = None                     # docProps/app.xml
    # 文書が用紙サイズを定義しているか。定義していなければ、ページ位置は
    # 「どのアプリで開いたか」で決まる（OOXML: pgSz 省略時はアプリ既定）。
    defines_page_size: bool = False
    geometry: str = ""          # 文書が持つページ設定（PageGeometry.key 形式）
    # 段落レベルの改ページ位置 [(ページ番号, そのページ先頭の本文)]
    page_starts: list[tuple[int, str]] = field(default_factory=list)


def read_outline(path: str) -> DocOutline | None:
    """docx から見出し一覧と（あれば）Word 記録によるページ位置を取り出す。"""
    from lxml import etree
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            doc_xml = z.read("word/document.xml")
            styles = (z.read("word/styles.xml").decode("utf-8", "replace")
                      if "word/styles.xml" in names else "")
            app = (z.read("docProps/app.xml").decode("utf-8", "replace")
                   if "docProps/app.xml" in names else "")
    except (OSError, zipfile.BadZipFile, KeyError):
        return None

    out = DocOutline()
    m = re.search(r"<Pages>(\d+)</Pages>", app)
    if m:
        out.declared_pages = int(m.group(1))
    for s in _SECTPR.finditer(doc_xml):
        body = s.group(0).decode("utf-8", "replace")
        sz = re.search(r'<w:pgSz w:w="(\d+)" w:h="(\d+)"', body)
        if not sz:
            continue
        out.defines_page_size = True
        mar = dict(re.findall(r'w:(top|bottom|left|right)="(-?\d+)"',
                              re.search(r"<w:pgMar[^/]*/>", body).group(0)
                              if re.search(r"<w:pgMar[^/]*/>", body) else ""))
        out.geometry = PageGeometry(
            int(sz.group(1)), int(sz.group(2)),
            int(mar.get("top", 0)), int(mar.get("bottom", 0)),
            int(mar.get("left", 0)), int(mar.get("right", 0))).key
        break

    head_ids = _heading_style_ids(styles)
    try:
        root = etree.fromstring(doc_xml)
    except etree.XMLSyntaxError:
        return None
    body = root.find(W + "body")
    if body is None:
        return None

    has_record = b"lastRenderedPageBreak" in doc_xml
    page = 1
    first_text = ""
    for child in body:
        if etree.QName(child).localname == "p":
            pstyle = child.find(W + "pPr/" + W + "pStyle")
            sid = pstyle.get(W + "val") if pstyle is not None else None
            text = "".join(t.text or "" for t in child.iter(W + "t")).strip()
            if text and not first_text:
                first_text = text
            # 段落の本文より前にマーカーがあれば、この段落は次ページの先頭
            leading = 0
            for node in child.iter():
                ln = etree.QName(node).localname
                if ln == "lastRenderedPageBreak":
                    leading = 1
                    break
                if ln == "t" and (node.text or "").strip():
                    break
            page += leading
            if sid in head_ids and text:
                out.headings.append(text)
                out.word_heading_pages.append(page)
            page += _boundaries(child) - leading
        else:
            page += _boundaries(child)

    out.word_pages = page if has_record else None
    if not has_record:
        out.word_heading_pages = []
    # ページ先頭の本文。見出しスタイルを使っていない文書で目次の代わりにする
    out.page_starts = ([(1, first_text)] if first_text else [])
    if has_record:
        out.page_starts += paragraph_page_starts(path)
    return out


# --------------------------------------------------------------------------
# LibreOffice レンダリング
# --------------------------------------------------------------------------

_WIN_LO_SUBPATH = "LibreOffice/program/soffice.exe"
_PROFILE_DIR: str | None = None

# 用紙サイズ（twips = 1/1440 inch）。ISO/US の規格値なので固有データではない。
PAPER_TWIPS = {"a4": (11906, 16838), "letter": (12240, 15840),
               "b5": (10319, 14572), "a3": (16838, 23811),
               "legal": (12240, 20160)}
_MM_TO_TWIPS = 1440.0 / 25.4


@dataclass(frozen=True)
class PageGeometry:
    """組版に必要な最小限のページ設定。"""
    w: int
    h: int
    top: int
    bottom: int
    left: int
    right: int

    @property
    def key(self) -> str:
        return f"{self.w}x{self.h}+{self.top}.{self.bottom}.{self.left}.{self.right}"

    def sect_xml(self) -> str:
        return (f'<w:pgSz w:w="{self.w}" w:h="{self.h}"/>'
                f'<w:pgMar w:top="{self.top}" w:right="{self.right}" '
                f'w:bottom="{self.bottom}" w:left="{self.left}" '
                f'w:header="720" w:footer="720" w:gutter="0"/>')


def _key_nums(key: str) -> tuple[int, ...] | None:
    m = re.fullmatch(r"(\d+)x(\d+)\+(\d+)\.(\d+)\.(\d+)\.(\d+)", key or "")
    return tuple(int(g) for g in m.groups()) if m else None


def geometry_close(a: str, b: str, tol: int = 20) -> bool:
    """ページ設定が実質同じか。tol は twips（20 twips ≒ 0.35mm）。

    mm指定をtwipsへ丸めるとWordが書いた値とわずかにずれることがあるため、
    許容差を設ける。
    """
    na, nb = _key_nums(a), _key_nums(b)
    return bool(na and nb and all(abs(x - y) <= tol for x, y in zip(na, nb)))


def assumed_geometry() -> PageGeometry:
    """設定から「用紙未指定の文書に仮定するページ設定」を組み立てる。"""
    name = (CONFIG.pagemap_assumed_paper or "a4").strip().lower()
    w, h = PAPER_TWIPS.get(name, PAPER_TWIPS["a4"])
    try:
        mm = [float(x) for x in CONFIG.pagemap_assumed_margins_mm.split(",")]
    except ValueError:
        mm = [35.0, 30.0, 30.0, 30.0]
    while len(mm) < 4:
        mm.append(mm[-1] if mm else 30.0)
    top, bottom, left, right = (int(round(v * _MM_TO_TWIPS)) for v in mm[:4])
    return PageGeometry(w, h, top, bottom, left, right)


def _inject_geometry(doc_xml: bytes, geom: PageGeometry) -> bytes:
    """pgSz を持たない sectPr にページ設定を差し込む（**一時コピーに対してのみ**）。

    元ファイルは触らない。Word が保存時に書くのと同じ位置・同じ形の要素を足すだけ。
    """
    out = bytearray()
    pos = 0
    for m in _SECTPR.finditer(doc_xml):
        body = m.group(0)
        out += doc_xml[pos:m.start()]
        if b"<w:pgSz" in body:
            out += body
        elif body.endswith(b"/>"):                  # <w:sectPr/> 形式
            out += (b"<w:sectPr>" + geom.sect_xml().encode("utf-8")
                    + b"</w:sectPr>")
        else:
            i = body.index(b">") + 1
            out += body[:i] + geom.sect_xml().encode("utf-8") + body[i:]
        pos = m.end()
    out += doc_xml[pos:]
    return bytes(out)


def _candidate_binaries() -> list[str]:
    """soffice の候補。OS 名・WSL いずれでも同じロジックで探す。"""
    cands: list[str] = []
    explicit = os.environ.get("RAG_SOFFICE") or CONFIG.soffice_path
    if explicit:
        cands.append(explicit)
    for name in ("soffice", "libreoffice", "soffice.exe"):
        found = shutil.which(name)
        if found:
            cands.append(found)
    # Windows ネイティブ
    for var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        base = os.environ.get(var)
        if base:
            cands.append(str(Path(base) / _WIN_LO_SUBPATH))
    # WSL から Windows 側を呼ぶ経路。移行コストをかけずに Windows の
    # フォント環境で描画できる。ドライブ文字は決め打ちせず走査する。
    mnt = Path("/mnt")
    if mnt.is_dir():
        try:
            drives = sorted(d for d in mnt.iterdir() if len(d.name) == 1)
        except OSError:
            drives = []
        for d in drives:
            for pf in ("Program Files", "Program Files (x86)"):
                cands.append(str(d / pf / _WIN_LO_SUBPATH))
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def find_soffice() -> str | None:
    """使える soffice の絶対パス。無ければ None。"""
    if os.environ.get("RAG_DISABLE_SOFFICE"):
        return None
    for c in _candidate_binaries():
        try:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        except OSError:
            continue
    return None


def _is_foreign_exe(binary: str) -> bool:
    """POSIX 側から Windows の exe を呼んでいる（＝パス変換が要る）か。"""
    return binary.lower().endswith(".exe") and os.name != "nt"


def _to_native(path: str, binary: str) -> str:
    if not _is_foreign_exe(binary):
        return path
    try:
        r = subprocess.run(["wslpath", "-w", path], capture_output=True,
                           text=True, timeout=30)
        return r.stdout.strip() or path
    except (OSError, subprocess.SubprocessError):
        return path


def _profile_url(binary: str) -> str:
    """プロセス専用のユーザプロファイル。

    並列抽出で同じプロファイルを共有すると LibreOffice が起動を拒む。
    プロセスごとに1つ作り、終了時に片付ける。
    """
    global _PROFILE_DIR
    if _PROFILE_DIR is None:
        _PROFILE_DIR = tempfile.mkdtemp(prefix="ragloprof.")
        atexit.register(shutil.rmtree, _PROFILE_DIR, True)
    native = _to_native(_PROFILE_DIR, binary).replace("\\", "/")
    return "file:///" + native.replace(" ", "%20")


def _copy_with_geometry(src: str, dst: str, geom: PageGeometry | None) -> None:
    """一時コピーを作る。geom があれば word/document.xml にだけ手を入れる。"""
    if geom is None:
        shutil.copyfile(src, dst)
        return
    with zipfile.ZipFile(src) as zin, \
            zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = _inject_geometry(data, geom)
            zout.writestr(item, data)


def render_page_texts(path: str, binary: str | None = None,
                      geom: PageGeometry | None = None) -> list[str] | None:
    """docx を PDF に変換し、各ページのテキストを返す。失敗時は None。

    **元ファイルは読むだけで書き換えない**（一時ディレクトリに複製して変換する）。
    `geom` を渡すと、用紙サイズを持たない文書にそれを仮定して組版する。
    """
    binary = binary or find_soffice()
    if not binary:
        return None
    tmp = tempfile.mkdtemp(prefix="ragloconv.")
    try:
        _copy_with_geometry(path, os.path.join(tmp, "in.docx"), geom)
        native_dir = _to_native(tmp, binary)
        sep = "\\" if (_is_foreign_exe(binary) or os.name == "nt") else "/"
        cmd = [binary, f"-env:UserInstallation={_profile_url(binary)}",
               "--headless", "--norestore", "--nolockcheck", "--nodefault",
               "--nofirststartwizard", "--convert-to", "pdf",
               "--outdir", native_dir, native_dir + sep + "in.docx"]
        try:
            subprocess.run(cmd, capture_output=True,
                           timeout=CONFIG.soffice_timeout_sec)
        except (OSError, subprocess.SubprocessError):
            return None
        pdf = os.path.join(tmp, "in.pdf")
        if not os.path.exists(pdf):
            return None
        try:
            import pypdfium2
            doc = pypdfium2.PdfDocument(pdf)
            try:
                return [doc[i].get_textpage().get_text_range()
                        for i in range(len(doc))]
            finally:
                doc.close()
        except Exception:                     # noqa: BLE001 - 判定不能へ落とす
            return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# Word 本体による組版（最優先。改ページ位置が定義上そのまま得られる）
# --------------------------------------------------------------------------

_WORD_SCRIPT = Path(__file__).with_name("word_paginate.ps1")
_WORD_EXE: str | None | bool = False          # False = 未探索


def _powershell() -> str | None:
    for name in ("powershell.exe", "pwsh.exe", "powershell"):
        found = shutil.which(name)
        if found:
            return found
    for drive in _mounted_drives():
        cand = drive / "Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        if cand.is_file():
            return str(cand)
    return None


def find_word() -> str | None:
    """WINWORD.EXE のパス。無ければ None（LibreOffice へ縮退する）。

    レジストリの App Paths を唯一の情報源にする。Office のインストール先は
    バージョンや ClickToRun 構成で変わるので、パスを決め打ちしない。
    """
    global _WORD_EXE
    if os.environ.get("RAG_DISABLE_WORD"):
        return None
    if _WORD_EXE is not False:
        return _WORD_EXE                      # 探索済み（None も含む）
    _WORD_EXE = None
    explicit = os.environ.get("RAG_WINWORD")
    if explicit and os.path.exists(explicit):
        _WORD_EXE = explicit
        return _WORD_EXE
    ps = _powershell()
    if not ps:
        return None
    key = ("HKLM:" + chr(92) + "SOFTWARE" + chr(92) + "Microsoft" + chr(92)
           + "Windows" + chr(92) + "CurrentVersion" + chr(92) + "App Paths"
           + chr(92) + "winword.exe")
    query = (f"(Get-ItemProperty '{key}' "
             "-ErrorAction SilentlyContinue).'(default)'")
    try:
        r = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", query],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    win = (r.stdout or "").strip()
    if not win:
        return None
    _WORD_EXE = _from_native(win) or win
    return _WORD_EXE


def _mounted_drives() -> list[Path]:
    mnt = Path("/mnt")
    if not mnt.is_dir():
        return []
    try:
        return sorted(d for d in mnt.iterdir() if len(d.name) == 1)
    except OSError:
        return []


def _from_native(win_path: str) -> str | None:
    """Windows パス → こちら側で存在確認できるパス。"""
    if os.name == "nt":
        return win_path if os.path.exists(win_path) else None
    try:
        r = subprocess.run(["wslpath", "-u", win_path], capture_output=True,
                           text=True, timeout=30)
        p = r.stdout.strip()
        return p if p and os.path.exists(p) else None
    except (OSError, subprocess.SubprocessError):
        return None


def _win_path(path: str) -> str:
    """Windows 側プロセスへ渡すためのパス。WSL からなら UNC/ドライブ表記へ。"""
    return path if os.name == "nt" else _to_native(path, "x.exe")


_WIN_TEMP: str | None | bool = False


def _windows_tempdir() -> str | None:
    r"""Windows 側から見える一時ディレクトリの親。

    WSL の `/tmp` を渡すと `\wsl.localhost\...` という UNC パスになり、
    Word が Protected View（保護ビュー）で開いて自動化が効かなくなる。
    Windows のローカルな %TEMP% を使えばこの問題を避けられる。
    """
    global _WIN_TEMP
    if os.name == "nt":
        return None                        # tempfile の既定でよい
    if _WIN_TEMP is not False:
        return _WIN_TEMP
    _WIN_TEMP = None
    ps = _powershell()
    if ps:
        try:
            r = subprocess.run([ps, "-NoProfile", "-NonInteractive",
                                "-Command", "$env:TEMP"],
                               capture_output=True, timeout=120)
            win = r.stdout.decode("utf-8", "replace").strip()
            if win:
                _WIN_TEMP = _from_native(win)
        except (OSError, subprocess.SubprocessError):
            pass
    return _WIN_TEMP


def word_paginate(jobs: list[tuple[str, str, PageGeometry | None]],
                  timeout: int | None = None) -> dict[str, DocOutline]:
    """Word に組版させ、改ページ記録入りの複製から DocOutline を得る。

    jobs: [(id, 元ファイルのパス, 仮定するページ設定 or None)]
    **1回の Word 起動でまとめて処理する**（1件ずつ起動すると桁違いに遅い）。
    元ファイルは複製してから渡すので書き換わらない。
    """
    if not jobs or find_word() is None:
        return {}
    ps = _powershell()
    if ps is None or not _WORD_SCRIPT.exists():
        return {}
    # Word には Windows ローカルのパスを渡す（UNC だと保護ビューに入る）
    work = tempfile.mkdtemp(prefix="ragword.", dir=_windows_tempdir())
    try:
        manifest = []
        for i, (jid, src, geom) in enumerate(jobs):
            ind = os.path.join(work, f"i{i}.docx")
            outd = os.path.join(work, f"o{i}.docx")
            try:
                _copy_with_geometry(src, ind, geom)
            except (OSError, zipfile.BadZipFile):
                continue
            manifest.append({"id": jid, "in": _win_path(ind),
                             "out": _win_path(outd), "_out": outd})
        if not manifest:
            return {}
        mpath = os.path.join(work, "jobs.json")
        rpath = os.path.join(work, "result.json")
        Path(mpath).write_text(json.dumps(
            [{k: v for k, v in m.items() if not k.startswith("_")}
             for m in manifest], ensure_ascii=False), encoding="utf-8")
        cmd = [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
               "-File", _win_path(str(_WORD_SCRIPT)),
               "-Manifest", _win_path(mpath),
               "-Result", _win_path(rpath)]
        limit = timeout or (CONFIG.word_timeout_sec + 5 * len(manifest))
        try:
            subprocess.run(cmd, capture_output=True, timeout=limit)
        except (OSError, subprocess.SubprocessError):
            return {}
        # Word 自身が数えたページ数と、あればエラー内容を読む
        reported: dict[str, int] = {}
        errors: dict[str, str] = {}
        try:
            for rec in json.loads(Path(rpath).read_text(encoding="utf-8-sig")):
                if rec.get("error"):
                    errors[rec.get("id", "?")] = rec["error"]
                if isinstance(rec.get("pages"), int) and rec["pages"] > 0:
                    reported[rec.get("id", "?")] = rec["pages"]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

        out: dict[str, DocOutline] = {}
        for m in manifest:
            jid = m["id"]
            if not os.path.exists(m["_out"]):
                if jid in errors:
                    print(f"    Word 組版に失敗: {jid}: {errors[jid][:160]}")
                continue
            ol = read_outline(m["_out"])
            if ol is None:
                continue
            if ol.word_pages is None:
                # 1ページに収まる文書には改ページマーカーが1つも書かれない。
                # 「記録なし」と区別できないので、Word が数えたページ数を使う。
                n = reported.get(jid)
                if n is None:
                    continue
                ol.word_pages = n
                ol.word_heading_pages = [1] * len(ol.headings)
            out[jid] = ol
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------
# md5 キャッシュ
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def _cache_dir() -> Path:
    return CONFIG.artifacts_dir / "pagemap_cache"


def _cache_path(md5: str, geom: PageGeometry | None = None) -> Path:
    # 仮定した用紙が変われば結果も変わるので、キャッシュキーに含める
    suffix = f".{geom.key}" if geom is not None else ""
    return _cache_dir() / f"{md5}{suffix}.{CACHE_VERSION}.json"


def cached_render(path: str, md5: str, binary: str | None = None,
                  allow_render: bool = True,
                  geom: PageGeometry | None = None) -> list[str] | None:
    """正規化済みページテキスト。md5（と仮定した用紙）が同じなら再変換しない。"""
    if md5:
        p = _cache_path(md5, geom)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")).get("pages")
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
    if not allow_render:
        return None
    texts = render_page_texts(path, binary, geom)
    if texts is None:
        return None
    pages = [_norm(t) for t in texts]
    if md5:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        target = _cache_path(md5, geom)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps({"md5": md5, "pages": pages,
                                   "geometry": geom.key if geom else None},
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
    return pages


def _word_cache_path(md5: str, geom: PageGeometry | None) -> Path:
    suffix = f".{geom.key}" if geom is not None else ""
    return _cache_dir() / f"{md5}{suffix}.word.{CACHE_VERSION}.json"


def cached_word_outline(md5: str, geom: PageGeometry | None) -> DocOutline | None:
    """Word に組版させた結果（キャッシュのみ。無ければ None）。"""
    if not md5:
        return None
    p = _word_cache_path(md5, geom)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return DocOutline(headings=d.get("headings", []),
                      word_pages=d.get("word_pages"),
                      word_heading_pages=d.get("heading_pages", []),
                      defines_page_size=True, geometry=d.get("geometry", ""),
                      page_starts=[(int(a), b)
                                   for a, b in d.get("page_starts", [])])


def _save_word_outline(md5: str, geom: PageGeometry | None,
                       ol: DocOutline) -> None:
    _cache_dir().mkdir(parents=True, exist_ok=True)
    target = _word_cache_path(md5, geom)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(
        {"word_pages": ol.word_pages, "headings": ol.headings,
         "heading_pages": ol.word_heading_pages,
         "page_starts": [[n, t[:60]] for n, t in ol.page_starts],
         "geometry": geom.key if geom else ol.geometry},
        ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def _geometry_for(ol: DocOutline) -> PageGeometry | None:
    """用紙を持たない文書にだけ、設定の用紙を仮定する。"""
    return None if ol.defines_page_size else assumed_geometry()


def word_paginate_corpus(docx, verbose: bool = False) -> int:
    """未キャッシュの docx を **1回の Word 起動で** まとめて組版する。

    ここが「前処理として Word に開かせる」本体。人手の GUI 操作ではなく、
    インジェストの一工程として自動で走る。
    """
    if find_word() is None:
        return 0
    jobs, meta = [], {}
    for f in docx:
        ol = read_outline(f.raw_path)
        if ol is None:
            continue
        geom = _geometry_for(ol)
        if cached_word_outline(f.md5, geom) is not None:
            continue
        jobs.append((f.doc_id, f.raw_path, geom))
        meta[f.doc_id] = (f, geom)
    if not jobs:
        return 0
    if verbose:
        print(f"  Word で組版: {len(jobs)}件 …", flush=True)
    got = word_paginate(jobs)
    for doc_id, ol in got.items():
        f, geom = meta[doc_id]
        _save_word_outline(f.md5, geom, ol)
    if verbose:
        print(f"  Word 組版 完了: {len(got)}/{len(jobs)}件", flush=True)
    return len(got)


# --------------------------------------------------------------------------
# 較正 — Word の記録を持つ docx でレンダラの再現精度を測る
# --------------------------------------------------------------------------

@dataclass
class Calibration:
    engine: str = ""           # word | libreoffice
    renderer: str = ""
    checked: int = 0
    exact: int = 0
    headings: int = 0
    heading_hits: int = 0
    boundaries: int = 0          # 段落レベルの改ページ境界（位置を比較できるもの）
    boundary_hits: int = 0       # そのうち Word と同じ位置に入ったもの
    threshold: float = 0.0
    detail: list[dict] = field(default_factory=list)
    key: str = ""

    @property
    def page_ratio(self) -> float:
        return self.exact / self.checked if self.checked else 0.0

    @property
    def heading_ratio(self) -> float:
        # 見出しスタイルを使っていない文書しか無い場合はこの指標を課さない
        return self.heading_hits / self.headings if self.headings else 1.0

    @property
    def trustworthy(self) -> bool:
        """レンダリング結果をページ番号として採用してよいか。

        閾値 0 なら照合を課さず、レンダラがあれば採用する（照合は診断として残す）。
        閾値を上げた場合は、総ページ数と見出し位置の **両方** が基準を満たすことを
        求める。総数が合っても見出しの載るページがずれていれば、
        「何ページにあるか」の問いには答えられないため。
        """
        if self.threshold <= 0:
            return bool(self.renderer)
        return (self.checked > 0
                and self.page_ratio >= self.threshold
                and self.heading_ratio >= self.threshold)

    def summary(self) -> str:
        if not self.checked:
            return "照合対象なし"
        return (f"同じ共有ドライブ内の照合で総ページ数 {self.exact}/{self.checked}、"
                f"見出し位置 {self.heading_hits}/{self.headings}、"
                f"改ページ位置 {self.boundary_hits}/{self.boundaries} 一致")

    @property
    def boundary_ratio(self) -> float:
        return self.boundary_hits / self.boundaries if self.boundaries else 0.0


def _calib_path() -> Path:
    return _cache_dir() / f"calibration.{CACHE_VERSION}.json"


def load_calibration() -> Calibration | None:
    p = _calib_path()
    if not p.exists():
        return None
    try:
        return Calibration(**json.loads(p.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def heading_pages(headings: list[str], pages: list[str]) -> list[int | None]:
    """各見出しが何ページ目に現れるか。見出しは文書順に単調に進む。"""
    out: list[int | None] = []
    lo = 0
    for h in headings:
        nh = _norm(h)
        hit = None
        if nh:
            for i in range(lo, len(pages)):
                if nh in pages[i]:
                    hit = i + 1
                    lo = i
                    break
        out.append(hit)
    return out


def _calib_key(binary: str | None, refs) -> str:
    h = hashlib.sha1()
    h.update((binary or "none").encode("utf-8"))
    h.update(CACHE_VERSION.encode("utf-8"))
    for f, _ in sorted(refs, key=lambda t: t[0].relpath):
        h.update(f.md5.encode("utf-8"))
    return h.hexdigest()[:16]


def _reference_docs(files):
    """Word が改ページを記録している docx（＝レンダラの照合に使える資料）。"""
    refs = []
    for f in files:
        if f.ext != ".docx":
            continue
        ol = read_outline(f.raw_path)
        if ol is not None and ol.word_pages is not None:
            refs.append((f, ol))
    return refs


def run_calibration(files, threshold: float | None = None,
                    verbose: bool = False) -> Calibration:
    """Word の改ページ記録を持つ docx で、レンダラの一致率を測って保存する。

    `files` は CorpusFile 相当（raw_path / relpath / md5 / ext を持つ）。
    """
    word = find_word()
    binary = find_soffice()
    engine = "word" if word else ("libreoffice" if binary else "")
    thr = CONFIG.pagemap_min_agreement if threshold is None else threshold
    cal = Calibration(engine=engine, renderer=word or binary or "",
                      threshold=thr)
    refs = _reference_docs(files)
    cal.key = _calib_key(cal.renderer, refs)
    if not cal.renderer:
        return cal
    for f, ol in sorted(refs, key=lambda t: t[0].relpath):
        if engine == "word":
            got = cached_word_outline(f.md5, _geometry_for(ol))
            pages = None
            n = got.word_pages if got is not None else -1
            got_heads = got.word_heading_pages if got is not None else []
        else:
            pages = cached_render(f.raw_path, f.md5, binary)
            n = len(pages) if pages is not None else -1
            got_heads = heading_pages(ol.headings, pages) if pages else []
        hits = sum(1 for a, b in zip(got_heads, ol.word_heading_pages)
                   if a == b)
        # Word 由来の真値は2系統ある。改ページマーカーを畳んだ数と、
        # Word自身がdocProps/app.xmlに書いた総ページ数を優先する。
        truth = ol.declared_pages if ol.declared_pages else ol.word_pages
        # 「Word と同じ位置で改ページしたか」を段落レベルの境界で直接測る。
        # 総ページ数や見出し位置が合っていても、改ページの入る位置は別問題。
        marks = paragraph_page_starts(f.raw_path)
        bhit = 0
        if engine == "word":
            # 元の記録と、こちらの Word が入れた改ページを直接突き合わせる
            got_starts = {_norm(t)[:20]: n
                          for n, t in (got.page_starts if got else [])
                          if n >= 2}
            for pageno, text in marks:
                bhit += int(got_starts.get(_norm(text)[:20]) == pageno)
        else:
            lo_i = 0
            for pageno, text in marks:
                anchor = _norm(text)[:20]
                for i in range(lo_i, len(pages or [])):
                    if anchor and pages[i].startswith(anchor):
                        bhit += int(i + 1 == pageno)
                        lo_i = i
                        break
        cal.checked += 1
        cal.exact += int(n == truth)
        cal.headings += len(ol.headings)
        cal.heading_hits += hits
        cal.boundaries += len(marks)
        cal.boundary_hits += bhit
        cal.detail.append({"relpath": f.relpath, "word": ol.word_pages,
                           "declared": ol.declared_pages, "rendered": n,
                           "headings": len(ol.headings), "heading_hits": hits,
                           "boundaries": len(marks), "boundary_hits": bhit,
                           "geometry": ol.geometry, "exact": bool(n == truth)})
        if verbose:
            label = "Word" if engine == "word" else "LibreOffice"
            print(f"  {'OK ' if n == truth else 'NG '}"
                  f"真値={truth:>3} (マーカー{ol.word_pages:>3}/"
                  f"app.xml{str(ol.declared_pages):>4})  {label}={n:>3}  "
                  f"見出し {hits:>3}/{len(ol.headings):<3} "
                  f"改ページ位置 {bhit:>2}/{len(marks):<2} {f.relpath}")
    _cache_dir().mkdir(parents=True, exist_ok=True)
    tmp = _calib_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(cal.__dict__, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(_calib_path())
    return cal


def prepare(files, verbose: bool = False) -> Calibration | None:
    """抽出の前に一度だけ呼ぶ。全 docx のページ位置を先に確定させる。

    組版は **親プロセスでまとめて1回**行い、結果を md5 キャッシュへ置く。
    並列ワーカーは以降キャッシュを読むだけになる
    （ワーカーごとに Word / soffice を起動させないため）。
    """
    docx = [f for f in files if f.ext == ".docx"]
    if not docx:
        return None

    outlines = [(f, read_outline(f.raw_path)) for f in docx]
    need_layout = [(f, ol) for f, ol in outlines
                   if ol is not None and ol.word_pages is None]
    if not need_layout:
        return None

    word = find_word()
    soffice = find_soffice()
    if word is None and soffice is None:
        targets = "、".join(f.relpath for f, _ in need_layout[:3])
        more = f"（ほか{len(need_layout) - 3}件）" if len(need_layout) > 3 else ""
        raise RuntimeError(
            "DOCXのページ情報を確定できません。Microsoft Word または "
            f"LibreOffice が見つからないため前処理を停止します。対象: {targets}{more}")

    # ① Word 本体があれば、それに組版させる（改ページ位置が定義どおり得られる）
    if word is not None:
        word_paginate_corpus(docx, verbose=verbose)
        failed = []
        for f, ol in need_layout:
            if cached_word_outline(f.md5, _geometry_for(ol)) is None:
                failed.append(f.relpath)
        if failed:
            targets = "、".join(failed[:3])
            more = f"（ほか{len(failed) - 3}件）" if len(failed) > 3 else ""
            raise RuntimeError(
                "Microsoft Word でDOCXの組版に失敗したため前処理を停止します。"
                f"対象: {targets}{more}")

    renderer = word or soffice
    key = _calib_key(renderer, _reference_docs(docx))
    cal = load_calibration()
    if cal is None or cal.key != key:
        cal = run_calibration(docx, verbose=verbose)

    # ② Word が無い環境では LibreOffice へ縮退する
    if word is None:
        if not cal.trustworthy:
            raise RuntimeError(
                "LibreOffice のDOCX組版結果が必要な照合精度を満たさないため、"
                f"前処理を停止します（{cal.summary()}）。")
        failed = []
        for f, ol in need_layout:
            pages = cached_render(f.raw_path, f.md5, soffice,
                                  geom=_geometry_for(ol))
            if pages is None:
                failed.append(f.relpath)
        if failed:
            targets = "、".join(failed[:3])
            more = f"（ほか{len(failed) - 3}件）" if len(failed) > 3 else ""
            raise RuntimeError(
                "LibreOffice でDOCXの組版に失敗したため前処理を停止します。"
                f"対象: {targets}{more}")
    return cal


# --------------------------------------------------------------------------
# 抽出器から呼ぶ入口
# --------------------------------------------------------------------------

@dataclass
class PageInfo:
    state: str                                       # 本文先頭に出す1行
    lines: list[str] = field(default_factory=list)   # "  p3: 見出し"
    n_pages: int | None = None
    source: str = "unknown"                          # word_record|rendered|unknown
    note: str = ""                                   # 較正の内訳（人間向け・本文に出さない）


def _map_lines(pages_of: list[int | None], headings: list[str],
               n_pages: int,
               page_starts: list[tuple[int, str]] | None = None) -> list[str]:
    """ページ→そこにある見出しの目次。

    **見出しスタイルを使っていない文書**（太字段落で見出しを表す等）では
    見出しが1つも取れず目次が空になる。そのページの先頭本文で埋めて、
    「どのページに何が書かれているか」の手掛かりを必ず残す。
    実データでも 20ページと26ページの文書がこれで目次0行になっていた。
    """
    by_page: dict[int, list[str]] = {}
    for h, p in zip(headings, pages_of):
        if p is not None:
            by_page.setdefault(p, []).append(h[:MAX_HEAD_CHARS])
    for p, text in (page_starts or []):
        if p in by_page or not text.strip():
            continue
        by_page[p] = [" ".join(text.split())[:MAX_HEAD_CHARS]]
    lines: list[str] = []
    for p in sorted(by_page):
        if len(lines) >= MAX_MAP_LINES:
            lines.append(f"  …（以降省略。全{n_pages}ページ）")
            break
        lines.append(f"  p{p}: " + " / ".join(by_page[p]))
    return lines


def page_info(path: str, md5: str = "") -> PageInfo:
    """`[ページ情報]` ブロックの中身を決める。確定不能なら例外にする。"""
    try:
        ol = read_outline(path)
    except Exception as exc:                  # noqa: BLE001 - 原因を付けて停止する
        raise RuntimeError(f"DOCXのページ情報を読み取れません: {path}: {exc}") from exc
    if ol is None:
        raise RuntimeError(f"DOCXのページ情報を読み取れません: {path}")

    if ol.word_pages is not None:
        info = PageInfo(f"全{ol.word_pages}ページ"
                        "（Word が記録した改ページ位置から確定）",
                        n_pages=ol.word_pages, source="word_record")
        info.lines = _map_lines(ol.word_heading_pages, ol.headings,
                                ol.word_pages, ol.page_starts)
        return info

    # モデルに渡すのは「確定したか否か」だけでよい。理由の内訳は人間向けなので
    # doc.meta 側に残し、本文には出さない（無用な文言で文脈を汚さないため）。
    cal = load_calibration()
    calibration_note = cal.summary() if cal else "レンダラ未較正"

    # ① Word 本体に組版させた結果があればそれを使う。Word が改ページを
    #    書き込んだ状態そのものなので、記録がある文書と同じ扱いにできる。
    geom = _geometry_for(ol)
    wo = cached_word_outline(md5, geom)
    if wo is not None and wo.word_pages is not None:
        # 用紙を仮定した文書は「確定」と書かない。仮定が違えばページ数も
        # 変わるので、そのことがモデルから見えるようにする。
        if geom is None:
            head = f"全{wo.word_pages}ページ（Word で組版して確定）"
            note = "Word で組版"
        else:
            head = (f"全{wo.word_pages}ページ（この文書はページ設定を持たないため、"
                    f"{CONFIG.pagemap_assumed_paper}・余白"
                    f"{CONFIG.pagemap_assumed_margins_mm}mm を仮定して組版した結果。"
                    "用紙設定が違えばページ数も変わる）")
            note = f"Word で組版／用紙未指定のため {geom.key} を仮定"
        info = PageInfo(head, n_pages=wo.word_pages, source="word", note=note)
        info.lines = _map_lines(wo.word_heading_pages, wo.headings,
                                wo.word_pages, wo.page_starts)
        return info

    # ② 無ければ LibreOffice の組版（較正を満たす場合のみ）
    if cal is None or not cal.renderer or not cal.trustworthy:
        raise RuntimeError(
            "DOCXのページ情報を確定できないため前処理を停止します"
            f"（{calibration_note}）: {path}")

    # 用紙サイズを持たない文書は、OOXML 仕様上ページが「開いたアプリの既定用紙」で
    # 決まる。どこかで決めないと組版できないので設定の用紙を仮定する
    # （設定は特定の文書に依存しない単一の既定値。config.pagemap_assumed_paper）。
    pages = cached_render(path, md5, geom=geom)
    if pages is None:
        raise RuntimeError(
            f"LibreOffice のDOCX組版結果を読み取れないため前処理を停止します: {path}")
    how = "実際に描画して確定" if geom is None else \
        f"{CONFIG.pagemap_assumed_paper}で組版して確定"
    note = calibration_note
    if geom is not None:
        note += f"／用紙未指定のため {geom.key} を仮定"
    info = PageInfo(f"全{len(pages)}ページ（{how}）",
                    n_pages=len(pages), source="rendered", note=note)
    info.lines = _map_lines(heading_pages(ol.headings, pages), ol.headings,
                            len(pages), ol.page_starts)
    return info


def block(path: str, md5: str = "") -> tuple[str, PageInfo]:
    """本文先頭に差し込む `[ページ情報]` ブロックと、その内訳。"""
    info = page_info(path, md5)
    text = f"[ページ情報] {info.state}"
    if info.lines:
        text += "\n" + "\n".join(info.lines)
    return text, info


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    import argparse

    from app.corpus.walk import nfc, walk_corpus

    ap = argparse.ArgumentParser(description="docx ページ位置の確定と検証")
    ap.add_argument("--validate", action="store_true",
                    help="Word の改ページ記録を持つ docx で変換器を照合する")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--relpath", default=None,
                    help="1ファイルのページ情報を表示する")
    args = ap.parse_args()

    word = find_word()
    binary = word or find_soffice()
    print(f"組版器: {binary or '見つかりません'}")
    files = walk_corpus()

    if args.relpath:
        target = nfc(args.relpath)
        cf = next((f for f in files
                   if f.relpath == target or f.relpath.endswith("/" + target)),
                  None)
        if cf is None:
            print(f"見つかりません: {args.relpath}")
            return 1
        try:
            prepare([cf])
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        text, info = block(cf.raw_path, cf.md5)
        print(f"\n# {cf.relpath}  (source={info.source})")
        print(text)
        return 0

    if binary is None:
        print("ERROR: Microsoft Word / LibreOffice が無いため照合できません。",
              file=sys.stderr)
        return 1

    docx = [f for f in files if f.ext == ".docx"]
    n_ref = n_sized = n_nosize = n_bad = 0
    for f in docx:
        ol = read_outline(f.raw_path)
        if ol is None:
            n_bad += 1
        elif ol.word_pages is not None:
            n_ref += 1
        elif ol.defines_page_size:
            n_sized += 1
        else:
            n_nosize += 1
    print(f"docx {len(docx)}件の内訳:")
    print(f"  {n_ref:>3} 件 Word の改ページ記録あり  → そのまま確定できる（照合にも使う）")
    print(f"  {n_sized:>3} 件 記録なし・用紙サイズあり → そのまま組版して確定する")
    print(f"  {n_nosize:>3} 件 記録なし・用紙サイズ未定義 → "
          f"用紙を仮定して組版する（下記の設定）")
    if n_bad:
        print(f"  {n_bad:>3} 件 読めない")

    # 較正はキャッシュ済みの組版結果を読むだけなので、先に埋めておく
    word_paginate_corpus(docx, verbose=True)
    print(f"\n組版に使うもの: {'Word 本体' if find_word() else 'LibreOffice'}")
    print(f"Word の改ページ記録を持つ {n_ref}件で照合します\n")
    cal = run_calibration(files, threshold=args.threshold, verbose=True)
    print(f"\n総ページ数 完全一致: {cal.exact}/{cal.checked} "
          f"({cal.page_ratio:.0%})   採用閾値 {cal.threshold:.0%}")
    print(f"見出し位置 一致    : {cal.heading_hits}/{cal.headings} "
          f"({cal.heading_ratio:.0%})")
    print(f"改ページ位置 一致  : {cal.boundary_hits}/{cal.boundaries} "
          f"({cal.boundary_ratio:.0%})  ← Word と同じ語の直前で切れたか")
    print("判定: " + ("採用（レンダリング結果をページ番号に使う）"
                      if cal.trustworthy else
                      "不採用（前処理を停止する）"))

    # 用紙を持たない文書に何を仮定し、その結果どうなったかを必ず見せる
    geom = assumed_geometry()
    print(f"\n用紙未指定の文書に仮定するページ設定: "
          f"{CONFIG.pagemap_assumed_paper} "
          f"({geom.w}x{geom.h} twips) 余白 "
          f"{CONFIG.pagemap_assumed_margins_mm} mm  → {geom.key}")

    # 仮定した設定と **同じページ設定を持つ** 資料に絞った一致率が、
    # 仮定して組版した文書の精度の一番近い推定値になる
    same = [d for d in cal.detail
            if geometry_close(d.get("geometry", ""), geom.key)]
    if same:
        ex = sum(1 for d in same if d.get("exact"))
        hh = sum(d["heading_hits"] for d in same)
        ht = sum(d["headings"] for d in same)
        bh = sum(d.get("boundary_hits", 0) for d in same)
        bt = sum(d.get("boundaries", 0) for d in same)
        print(f"  うち同じページ設定を持つ資料 {len(same)}件での一致率:")
        print(f"    総ページ数   {ex}/{len(same)}")
        print(f"    見出しのページ {hh}/{ht}"
              + (f" ({hh/ht*100:.0f}%)" if ht else "")
              + "   ← 「この見出しは何ページ」に効く指標")
        print(f"    改ページ位置 {bh}/{bt}"
              + (f" ({bh/bt*100:.0f}%)" if bt else "")
              + "   ← Word と同じ位置で切れたか")
        print("  → 仮定して組版した文書のページ番号も、この程度の確からしさと見るべき")
    else:
        print("  （仮定した設定と同じページ設定を持つ資料が無く、精度を推定できない）")
    if not cal.trustworthy:
        return 0
    rows = [(f, read_outline(f.raw_path)) for f in docx]
    rows = [(f, o) for f, o in rows
            if o is not None and o.word_pages is None and not o.defines_page_size]
    print(f"仮定して組版した結果（{len(rows)}件）:")
    for f, ol in sorted(rows, key=lambda t: t[0].relpath):
        wo = cached_word_outline(f.md5, geom)
        if wo is not None and wo.word_pages is not None:
            n = wo.word_pages
            mapped = sum(1 for x in wo.word_heading_pages if x)
        else:
            pages = cached_render(f.raw_path, f.md5, binary, geom=geom)
            n = len(pages) if pages is not None else -1
            mapped = sum(1 for x in heading_pages(ol.headings, pages or []) if x)
        print(f"  {n:>3}ページ  見出し {mapped:>3}/{len(ol.headings):<3} "
              f"{f.relpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
