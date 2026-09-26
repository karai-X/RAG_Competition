"""run_python サンドボックス。

- **APIキー等の環境変数は渡さない**（漏洩防止）
- コーパスは読み取り専用で参照。`P()` / `projects()` / `files()` を prelude で注入
- **横断集計型の問い**（全案件を数える系）を1回のコード実行で解けるようにするのが要点。
  10案件を逐次 read するとターン数を使い切るため。
- タイムアウト・出力上限あり
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import CONFIG

MAX_STDOUT_CHARS = 12_000

# 切り詰めるときに末尾へ回す割合。集計結果が末尾にある場合も保持する。
TAIL_SHARE = 0.35

# 書式アノテーションの断片が「値の位置」に単独で立っている行を見つける。
# 抽出テキストには [c:RRGGBB]…[/c] や ** が焼き込まれているため、素朴な
# 正規表現が本文ではなく注記を捕捉した場合に検出する。
_MARKUP_TOKEN = (r"(?:\[/c\]|\[c:[0-9A-Fa-f]{6}\]|\*\*|==|~~|</?u>"
                 r"|\{fill:[^}]*\}|\{shape-fill:[^}]*\}|\{fml:[^}]*\})")
_BROKEN_CAPTURE = re.compile(r"[:：=,、|\t]\s*" + _MARKUP_TOKEN + r"(?:\s|[|,、])*$")
_BROKEN_MIN_LINES = 3

# サンドボックスに入っていないモジュールを import したときの案内。
# 「やりたいこと」から代替手段へ橋渡しする。特定の設問には依存しない。
_MODULE_ALTERNATIVES = {
    "cv2": "画像処理は PIL・numpy・scipy が使えます。"
           "図そのものの読み取りは read_image / ask_image を使ってください。",
    "skimage": "画像処理は PIL・numpy・scipy が使えます。"
               "図そのものの読み取りは read_image / ask_image を使ってください。",
    "pypdf": "PDF は自分でパースせず read_text(相対パス) を使ってください。"
             "書式まで焼き込んだ抽出済みテキストが返ります。",
    "PyPDF2": "PDF は自分でパースせず read_text(相対パス) を使ってください。",
    "fitz": "PDF は自分でパースせず read_text(相対パス) を使ってください。",
    "pdfplumber": "PDF は自分でパースせず read_text(相対パス) を使ってください。",
    "matplotlib": "描画はできません。値は print() で出してください。",
    "bs4": "HTML/XML は lxml が使えます。",
    "requests": "外部への通信はできません。"
                "資料は files() / read_text() で読めます。",
    "urllib3": "外部への通信はできません。",
}
# サンドボックスで利用できるモジュール。
_AVAILABLE = ("numpy, pandas, PIL(Pillow), scipy, openpyxl, "
              "python-docx, python-pptx, pypdfium2, scikit-learn, lxml, msoffcrypto")
_MISSING_MODULE = re.compile(r"ModuleNotFoundError: No module named ['\"]([\w.]+)['\"]")

# raw 文字列にしている理由: この中身はそのままサンドボックスへ渡すコードなので、
# 正規表現のバックスラッシュを Python の文字列エスケープに食わせてはならない。
PRELUDE = r'''import os, sys, unicodedata, json, re, glob, hashlib
CORPUS_ROOTS = json.loads(os.environ["CORPUS_ROOTS"])   # [[mount_as, abs_path], ...]

def _nfc(s):
    return unicodedata.normalize("NFC", s)

_PATH_MAP = None

def _build_map():
    m = {}
    for mount_as, root in CORPUS_ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git", ".ipynb_checkpoints")]
            for name in filenames:
                if name.startswith("~$") or name.endswith((".pyc", ".pyo")):
                    continue
                p = os.path.join(dirpath, name)
                rel = os.path.relpath(p, root).replace(os.sep, "/")
                if mount_as:
                    rel = mount_as.strip("/") + "/" + rel
                m.setdefault(_nfc(rel), p)
    return m

def _map():
    global _PATH_MAP
    if _PATH_MAP is None:
        _PATH_MAP = _build_map()
    return _PATH_MAP

def _rev_map():
    """実パス -> 相対パス。注記で相対パスを見せるために使う。"""
    return {v: k for k, v in _map().items()}

class _RevMap(dict):
    """遅延構築。_map() は最初のアクセスまで走査しない。"""
    def get(self, k, d=None):
        if not self:
            self.update(_rev_map())
        return dict.get(self, k, d)

_REV_MAP = _RevMap()

def P(relpath):
    """NFC正規化済み相対パス -> 実際に開けるパス。日本語パスのNFC/NFD差を吸収する。"""
    m = _map()
    key = _nfc(relpath).lstrip("/")
    if key in m:
        return m[key]
    tail = key.rsplit("/", 1)[-1]
    cands = [k for k in m if k.endswith("/" + tail)]
    if len(cands) == 1:
        return m[cands[0]]
    raise FileNotFoundError(f"{relpath} が見つかりません。近い候補: {sorted(cands)[:5]}")

import builtins as _bi
import io as _io
_real_open = _bi.open

# 生で開いたコーパスのファイル。終了時に「抽出済みもある」と知らせる。
_RAW_OPENED = set()
# extracted()/read_text()/tables() が OCR 由来の文字列を返した文書。
# 呼び出し側が数値だけを print しても、親プロセス側で元画像確認を要求できるよう
# 終了時に監査マーカーを出す。
_OCR_TEXT_USED = set()

def _open(file, *a, **kw):
    """相対パスをそのまま open できるようにする。

    LLM は P() を忘れて相対パスを直接開こうとしがちで、その往復でターンを浪費する。
    pandas / openpyxl / zipfile も最終的にこの open を通るのでまとめて救われる。

    os.path.exists で判定すると（下でそれ自身も差し替えるため）順序に依存する。
    「素直に開いてみて、駄目なら解決して開き直す」ことで順序非依存にする。
    """
    if not isinstance(file, str):
        return _real_open(file, *a, **kw)
    try:
        f = _real_open(file, *a, **kw)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        file = P(file)
        f = _real_open(file, *a, **kw)
    _RAW_OPENED.add(file)
    return f

# builtins.open と io.open は別の属性。zipfile は io.open を使うため両方差し替える。
_bi.open = _open
_io.open = _open

# python-docx / openpyxl は開く前に os.path.isfile を確認するため、
# open だけ差し替えても «Package not found» で落ちる。os.path 側も解決する。
def _wrap_ospath(fn):
    def inner(path, *a, **kw):
        if isinstance(path, str) and not fn(path, *a, **kw):
            try:
                return fn(P(path), *a, **kw)
            except (FileNotFoundError, OSError):
                return fn(path, *a, **kw)
        return fn(path, *a, **kw)
    return inner

for _name in ("exists", "isfile", "getsize"):
    setattr(os.path, _name, _wrap_ospath(getattr(os.path, _name)))

def read_csv(rel, **kw):
    import pandas as pd
    return pd.read_csv(P(rel), **kw)

def read_excel(rel, **kw):
    import pandas as pd
    return pd.read_excel(P(rel), **kw)

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

_BINARY_EXTS = (".docx", ".xlsx", ".pptx", ".pdf", ".ipynb", ".png", ".jpg",
                ".jpeg", ".emf", ".wmf")
_EXTRACTED_DIR = os.environ.get("EXTRACTED_DIR", "")

def extracted(rel):
    """パイプラインが抽出済みのテキスト（書式アノテーション付き）を返す。

    Office/PDF を自分でパースし直す必要はない。塗り色・太字・ハイライト・
    数式・ピボット定義・EMF表・OCR結果まで、すべてここに入っている。
    """
    import hashlib, json
    key = _nfc(rel).lstrip("/")
    if key not in _map():
        key = P(rel)  # 例外で候補を出す
    doc_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    path = os.path.join(_EXTRACTED_DIR, doc_id + ".json")
    if not _EXTRACTED_DIR or not os.path.isfile(path):
        raise FileNotFoundError(f"{rel} の抽出結果が見つかりません")
    with _real_open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = (chr(10) * 2).join(
        p["text"] for p in d.get("pages", []) if p.get("text"))
    if "[画像読み取り]" in out:
        _OCR_TEXT_USED.add(_nfc(rel).lstrip("/"))
    return out

_ASSETS_DIR = os.environ.get("ASSETS_DIR", "")

def read_text(rel, encoding="utf-8-sig", plain=False):
    """テキストファイルはそのまま、Office/PDF は抽出済みテキストを返す。

    plain=True で書式アノテーション（文字色・太字・塗り・ハイライト等）を外した
    素の本文を返す。**氏名や値を正規表現で拾うときは plain=True を使う**
    （書式注記がパターンに割り込んで空振りする）。
    書式そのものが答えになる問いでは plain=False（既定）のままにする。
    """
    if _nfc(rel).lower().endswith(_BINARY_EXTS):
        out = extracted(rel)
    else:
        with _real_open(P(rel), encoding=encoding) as f:
            out = f.read()
    return strip_markup(out) if plain else out

def _extracted_path(rel):
    """相対パス -> 抽出結果の json。無ければ None。

    doc_id は **相対パス** の sha1（extracted() と同じ規則）。
    実パスで引くと必ず外れる。
    """
    if not _EXTRACTED_DIR or not rel:
        return None
    doc_id = hashlib.sha1(_nfc(rel).lstrip("/").encode("utf-8")).hexdigest()[:12]
    p = os.path.join(_EXTRACTED_DIR, doc_id + ".json")
    return p if os.path.isfile(p) else None

def _raw_open_hint():
    """抽出済みの文書を生で読んでいたら、そのことを出力の最後に添える。

    生アクセスを禁じない（条件付き書式など、生でしか取れない情報がある）。
    ただし埋め込み画像を符号化したまま抱えると量が扱えなくなり、
    画像に住所が付かないので read_image へ渡せない。**それを知らせる**。
    """
    rows = []
    for real in sorted(_RAW_OPENED):
        if not _nfc(real).lower().endswith(_BINARY_EXTS):
            continue
        rel = _REV_MAP.get(real)
        ex = _extracted_path(rel)
        if not ex:
            continue
        try:
            raw_n = os.path.getsize(real)
            with _real_open(ex, encoding="utf-8") as f:
                d = json.load(f)
            ext_n = sum(len(p.get("text") or "") for p in d.get("pages", []))
        except (OSError, ValueError):
            continue
        rows.append((rel, raw_n, ext_n))
    if not rows:
        return ""
    out = ["", "[注記] 抽出済みの文書を生ファイルとして読んでいます。"
           "read_text(相対パス) なら抽出済みテキストが返ります"
           "（埋め込み画像は取り出され、[画像: <id> …] の行として現れます。"
           "その id を read_image / ask_image に渡せば中身を見られます）。"]
    for rel, raw_n, ext_n in rows[:5]:
        out.append(f"  {rel}: 生 {raw_n:,} バイト / 抽出済み {ext_n:,} 文字")
    return chr(10).join(out)

def _ocr_text_usage_hint():
    """OCR本文を利用した事実を、最終出力から消えない監査行として返す。"""
    if not _OCR_TEXT_USED:
        return ""
    out = ["", "[内部監査: OCR画像由来テキスト使用] "
           "抽出値を回答根拠にする前に、read_image または ask_image で"
           "元画像を確認してください。"]
    for rel in sorted(_OCR_TEXT_USED)[:5]:
        out.append(f"  {rel}")
    return chr(10).join(out)

import atexit as _atexit
_atexit.register(lambda: (lambda h: print(h) if h else None)(_raw_open_hint()))
_atexit.register(lambda: (lambda h: print(h) if h else None)(_ocr_text_usage_hint()))

_SEPROW = re.compile(r"^\|[\s:|-]+\|$")

def _cells(row, plain):
    parts = [c.strip() for c in row.strip().strip("|").split("|")]
    return [strip_markup(c).strip() if plain else c for c in parts]

def _as_table(block, plain):
    if not any(_SEPROW.match(b) for b in block):
        return None
    body = [b for b in block if not _SEPROW.match(b)]
    if len(body) < 1:
        return None
    header = _cells(body[0], plain)
    rows = [_cells(b, plain) for b in body[1:]]
    return {"header": header, "rows": rows, "n_rows": len(rows)}

def tables(rel, plain=True):
    """抽出テキスト中の表を**構造として**取り出す。

    戻り値: [{"header": [...], "rows": [[...], ...], "n_rows": n}, ...]

    plain=True（既定）は書式アノテーションを外した値を返す。セルの塗り色や
    太字そのものが答えになる問いでは plain=False を渡すと注記が残る。

    複数文書の表をまとめて走査できる。
    """
    out, block = [], []
    for ln in read_text(rel, plain=False).splitlines():
        s = ln.strip()
        if len(s) > 1 and s.startswith("|") and s.endswith("|"):
            block.append(s)
            continue
        if len(block) >= 2:
            t = _as_table(block, plain)
            if t:
                out.append(t)
        block = []
    if len(block) >= 2:
        t = _as_table(block, plain)
        if t:
            out.append(t)
    return out

def all_files():
    """全ファイルの相対パス一覧（NFC）。"""
    return sorted(_map())

def projects():
    """案件フォルダ名の一覧。パス構造の位置から機械的に導出する。"""
    out = set()
    for rel in _map():
        parts = rel.split("/")
        if len(parts) >= 4:
            out.add(parts[1])
    return sorted(out)

def files(project=None, pattern=None, category=None):
    """条件に合う相対パス一覧。**横断集計はこれで全案件を1回で走査する**。

    project  … 案件名の部分一致
    category … サブフォルダ名の部分一致（例 "契約", "03.データ"）
    pattern  … ファイル名の glob（例 "*.xlsx", "*.csv", "*報告*"）

    戻り値は相対パス。そのまま open / pd.read_csv に渡してよい（自動解決される）。
    """
    out = []
    for rel in sorted(_map()):
        parts = rel.split("/")
        proj = parts[1] if len(parts) >= 4 else None
        cat = parts[2] if len(parts) >= 4 else (parts[0] if len(parts) == 2 else None)
        if project and (proj is None or project not in proj):
            continue
        if category and (cat is None or category not in cat):
            continue
        if pattern and not glob.fnmatch.fnmatch(parts[-1], pattern):
            continue
        out.append(rel)
    return out

'''


def corpus_roots() -> list[tuple[str, str]]:
    from app.corpus.mounts import load_mounts
    roots = []
    for m in load_mounts():
        if m.root.exists():
            roots.append((m.mount_as, str(m.root.resolve())))
    return roots


def _corpus_snapshot() -> dict[str, tuple[int, int]]:
    """共有ドライブの現在の状態（パス → (mtime_ns, サイズ)）。

    `run_python` の前後で比較し、**サンドボックスが入力データを書き換えていないか**
    を検査するために使う。走査は数百ファイルなので数十ミリ秒で終わる。
    """
    snap: dict[str, tuple[int, int]] = {}
    for _, root in corpus_roots():
        for dirpath, _dirs, names in os.walk(root):
            for n in names:
                fp = os.path.join(dirpath, n)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                snap[fp] = (st.st_mtime_ns, st.st_size)
    return snap


def _revert_corpus_writes(before: dict[str, tuple[int, int]]) -> str:
    """サンドボックスが共有ドライブへ書いたものを取り消し、警告文を返す。

    入力資料を変更しないよう、新規作成を消して変更を検出する。
    既存ファイルの内容は元に戻せないため警告する。
    """
    after = _corpus_snapshot()
    created = sorted(set(after) - set(before))
    changed = sorted(f for f in set(after) & set(before) if after[f] != before[f])
    for f in created:
        try:
            os.remove(f)
        except OSError:
            pass
    if not created and not changed:
        return ""
    msg = ["", "[警告] 共有ドライブは読み取り専用です。書き込みは取り消されました。"]
    if created:
        msg.append(f"  作成を取り消したファイル {len(created)}件: "
                   + ", ".join(os.path.basename(f) for f in created[:5])
                   + (" …" if len(created) > 5 else ""))
    if changed:
        msg.append(f"  **内容が変更されたファイル {len(changed)}件**（元に戻せません）: "
                   + ", ".join(os.path.basename(f) for f in changed[:5]))
    msg.append("  中間ファイルが要るときはカレントディレクトリ（一時領域）に保存してください。")
    return chr(10).join(msg)


def run_python(code: str, timeout: int | None = None) -> str:
    import json
    timeout = timeout or CONFIG.sandbox_timeout_sec
    CONFIG.logs_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CONFIG.logs_dir) as tmp:
        script = Path(tmp) / "script.py"
        script.write_text(PRELUDE + code, encoding="utf-8")
        # 画像比較モジュールを複製して import 可能にする。
        # app パッケージを import させると app.config が .env を読み直して
        # APIキーがサンドボックスへ入るので、**必要な1ファイルだけ**を渡す。
        import shutil as _sh
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": tmp,
            "CORPUS_ROOTS": json.dumps(corpus_roots(), ensure_ascii=False),
            "EXTRACTED_DIR": str(CONFIG.extracted_dir),
            "ASSETS_DIR": str(CONFIG.assets_dir),
            "PYTHONIOENCODING": "utf-8",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "4",
        }
        before = _corpus_snapshot()
        try:
            # encoding は明示する。子は PYTHONIOENCODING=utf-8 で書くが、
            # 既定（ロケール）で復号すると cp932 環境で日本語出力が全て消える。
            proc = subprocess.run([sys.executable, str(script)],
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=timeout, cwd=tmp, env=env)
        except subprocess.TimeoutExpired:
            return (f"[エラー] 実行が{timeout}秒でタイムアウトしました。"
                    "処理を軽くするか、対象を絞ってください。"
                    + _revert_corpus_writes(before))
        out = (proc.stdout or "") + _revert_corpus_writes(before)
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            out += "\n[stderr]\n" + "\n".join(err[-20:])
        if not out.strip():
            out = "(出力なし — print() で結果を出力してください)"
        return _postprocess(out)


def _truncate(out: str) -> str:
    """上限を超えた出力を、**末尾を残して**切り詰める。

    集計コードは合計・件数を最後に print する。先頭だけ残す実装ではその結論が
    消え、モデルが会話に流れた断片を目で数える誤りに繋がっていた。
    """
    if len(out) <= MAX_STDOUT_CHARS:
        return out
    tail = int(MAX_STDOUT_CHARS * TAIL_SHARE)
    head = MAX_STDOUT_CHARS - tail
    omitted = len(out) - MAX_STDOUT_CHARS
    return (out[:head]
            + f"\n...[中略: 全{len(out)}文字のうち中央{omitted}文字を省略。"
              "全件を print せず、件数と集計結果だけを print してください]...\n"
            + out[-tail:])


def _markup_hint(out: str) -> str:
    """書式アノテーションを値として捕捉している兆候があればヒントを返す。

    「道具はあるのに使われない」への対処。read_text(plain=True) は存在するが、
    正規表現が空振りしたことにモデル自身が気づけない。**気づけるのは出力を見た
    瞬間だけ**なので、出力そのものに添える。
    """
    n = sum(1 for ln in out.splitlines() if _BROKEN_CAPTURE.search(ln))
    if n < _BROKEN_MIN_LINES:
        return ""
    return (f"\n\n[注意] 出力{n}行で、値の位置に書式アノテーションの断片"
            "（[c:RRGGBB] / [/c] / ** など）がそのまま入っています。"
            "正規表現が本文ではなく注記を捕捉している可能性が高いです。"
            "read_text(相対パス, plain=True) を使うと注記が外れた素の本文が返ります。"
            "**この壊れた結果を数えたり、代わりに会話の断片を目で数えたり"
            "しないでください。**")


def _module_hint(out: str) -> str:
    """入っていないモジュールを import したなら、代わりの手段を示す。

    利用できる代替モジュールや既存APIを案内する。
    """
    m = _MISSING_MODULE.search(out)
    if not m:
        return ""
    mod = m.group(1).split(".")[0]
    alt = _MODULE_ALTERNATIVES.get(mod)
    head = f"\n\n[注意] {mod} はこの環境に入っていません。"
    if alt:
        return head + alt + f"\n使えるのは: {_AVAILABLE}"
    return head + f"使えるのは: {_AVAILABLE}\n" + \
        "Office/PDF の中身は read_text(相対パス)、表は tables(相対パス) で取れます。"


def _postprocess(out: str) -> str:
    return _truncate(out) + _markup_hint(out) + _module_hint(out)
