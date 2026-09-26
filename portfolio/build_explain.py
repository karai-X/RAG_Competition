"""設問ごとの解説データ生成: portfolio/explain/q*.json → materials/ の画像 + data/explain.js

  uv run python portfolio/build_explain.py [--only 0,5] [--force]

explain/qNN.json の書式:
  {
    "ask": "何を問うているか",
    "materials": [
      {"file": "プロジェクト/…/提案書.pptx", "kind": "slide", "index": 6,
       "caption": "…", "crop": [x0, y0, x1, y1]}      # crop は 0-1 の割合（任意）
    ],
    "layout": "compare" | "stack",                    # 資料を横に並べるか縦に積むか
    "reasoning": ["判断の流れ 1", "…"]
  }

kind ごとの書き出し方（Windows の Office を COM で使う。元ファイルは読むだけ）:
  slide  pptx のスライド   PowerPoint で PNG に書き出す（index はスライド番号）
  page   pdf / docx のページ  docx は Word で PDF にしてから pypdfium2 で描画（index はページ番号）
  sheet  xlsx のシート     Excel で sheet（シート名）と range（任意、例 "A1:H30"）を PDF にしてから描画
  image  画像ファイル、または artifacts/assets の埋め込み画像（asset に "doc_id/image1.png" を書く）
  text   画像にしない。text に書いた抜粋を、ページ上で等幅の文字として見せる（csv / md / py / ipynb 向け）

暗号化された資料は password に、エージェントが復号に使ったパスワードを書く。
画像は materials/qNN/ に WebP で置く。既にあれば書き出し直さない（--force で作り直す）。
Office の操作は同時に1つだけ行う（並行して動かしても、materials/.render.lock で順番待ちする）。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORPUS = ROOT / "share" / "共有ドライブ"
ASSETS = ROOT / "artifacts" / "assets"
SRC = HERE / "explain"
OUT_IMG = HERE / "materials"
OUT_JS = HERE / "data" / "explain.js"
MAX_EDGE = 1600


def _ps(script: str) -> None:
    """PowerShell で COM を操作する。失敗したら例外にする。"""
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())


def _q(p: Path) -> str:
    return "'" + str(p).replace("'", "''") + "'"


@contextlib.contextmanager
def _office_lock():
    """Office の COM 操作を1プロセスずつにする。Quit() が別プロセスの作業まで閉じるのを防ぐ。"""
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    lock = OUT_IMG / ".render.lock"
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 900:   # 落ちたプロセスの残骸
                    lock.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            time.sleep(1)
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def _corpus_path(rel: str) -> Path:
    """資料の実パス。フォルダ名の濁点がディスク上で分解形（NFD）のこともあるので、
    NFC で比べながら1段ずつたどる。"""
    import unicodedata
    direct = CORPUS / rel
    if direct.exists():
        return direct
    cur = CORPUS
    for part in unicodedata.normalize("NFC", rel).split("/"):
        hit = next((c for c in cur.iterdir()
                    if unicodedata.normalize("NFC", c.name) == part), None)
        if hit is None:
            raise FileNotFoundError(f"資料が見つからない: {rel}")
        cur = hit
    return cur


def _render_pdf_page(pdf: Path, page: int) -> "Image.Image":
    import pypdfium2 as pdfium
    # ファイルを開いたままにすると、一時フォルダを消すときに Windows で失敗する
    doc = pdfium.PdfDocument(pdf.read_bytes())
    pg = doc[page - 1]
    scale = MAX_EDGE / max(pg.get_size())
    return pg.render(scale=scale * 1.5).to_pil()


def render(m: dict, dst: Path) -> None:
    from PIL import Image
    kind = m["kind"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if kind == "slide":
            src, png = _corpus_path(m["file"]), tmp / "s.png"
            open_path = Path(f"{src}::{m['password']}::") if m.get("password") else src
            _ps(f"""$ErrorActionPreference='Stop'
$pp = New-Object -ComObject PowerPoint.Application
try {{
  $pr = $pp.Presentations.Open({_q(open_path)}, -1, 0, 0)
  $pr.Slides.Item({int(m['index'])}).Export({_q(png)}, 'PNG', {MAX_EDGE})
  $pr.Close()
}} finally {{ $pp.Quit() }}""")
            img = Image.open(png)
        elif kind == "page":
            src = _corpus_path(m["file"])
            if src.suffix.lower() == ".docx":
                pdf = tmp / "d.pdf"
                _ps(f"""$ErrorActionPreference='Stop'
$w = New-Object -ComObject Word.Application
try {{
  $d = $w.Documents.Open({_q(src)}, $false, $true, $false, {_q(Path(m.get('password', '')))})
  $d.ExportAsFixedFormat({_q(pdf)}, 17)
  $d.Close(0)
}} finally {{ $w.Quit() }}""")
                src = pdf
            img = _render_pdf_page(src, int(m["index"]))
        elif kind == "sheet":
            src, pdf = _corpus_path(m["file"]), tmp / "x.pdf"
            rng = m.get("range")
            area = f"$ws.PageSetup.PrintArea = '{rng}'" if rng else ""
            _ps(f"""$ErrorActionPreference='Stop'
$x = New-Object -ComObject Excel.Application
$x.DisplayAlerts = $false
try {{
  $wb = $x.Workbooks.Open({_q(src)}, 0, $true, 5, {_q(Path(m.get('password', '')))})
  $ws = $wb.Worksheets.Item('{m['sheet']}')
  {area}
  $ws.PageSetup.Zoom = $false
  $ws.PageSetup.FitToPagesWide = 1
  $ws.PageSetup.FitToPagesTall = 1
  $ws.ExportAsFixedFormat(0, {_q(pdf)})
  $wb.Close($false)
}} finally {{ $x.Quit() }}""")
            img = _render_pdf_page(pdf, 1)
        elif kind == "image":
            path = ASSETS / m["asset"] if m.get("asset") else _corpus_path(m["file"])
            img = Image.open(path)
        else:
            raise ValueError(f"未知の kind: {kind}")

        # 透明な画像（ノートブックの出力など）は、そのまま RGB にすると黒くなるので白で塗る
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.getchannel("A"))
            img = bg
        img = img.convert("RGB")
        if m.get("crop"):
            x0, y0, x1, y1 = m["crop"]
            w, h = img.size
            img = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        img.thumbnail((MAX_EDGE, MAX_EDGE))
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "WEBP", quality=85, method=6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="設問番号をカンマ区切りで")
    ap.add_argument("--force", action="store_true", help="既存の画像も書き出し直す")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",")} if a.only else None

    with _office_lock():
        return _build(only, a.force)


def _build(only: set[int] | None, force: bool) -> int:
    data: dict[str, dict] = {}
    for f in sorted(SRC.glob("q*.json"), key=lambda p: int(p.stem[1:])):
        i = int(f.stem[1:])
        entry = json.loads(f.read_text(encoding="utf-8"))
        for k, m in enumerate(entry.get("materials", []), 1):
            if m["kind"] == "text":
                continue
            rel = f"materials/q{i:02d}/{k}.webp"
            dst = HERE / rel
            if (only is None or i in only) and (force or not dst.exists()):
                print(f"q{i:02d} #{k} {m['kind']} {m.get('file') or m.get('asset')}")
                render(m, dst)
            m.pop("password", None)                  # パスワードはページに載せない
            if m.get("file"):
                import unicodedata
                m["file"] = unicodedata.normalize("NFC", m["file"])
            m["src"] = rel
        data[str(i)] = entry

    OUT_JS.write_text("window.RAG_EXPLAIN=" + json.dumps(data, ensure_ascii=False,
                      separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"{OUT_JS.relative_to(ROOT)}  {len(data)}問")
    return 0


if __name__ == "__main__":
    sys.exit(main())
