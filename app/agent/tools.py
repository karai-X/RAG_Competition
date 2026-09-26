"""エージェントのツール定義と実装。

ツールは「決定的抽出層を見せるだけ」。LLMの仕事は
「どのファイルのどこを見るか」の選択・ツールの組み立て・最終整形に限る。

出力は決定的・上限つき（プロンプトキャッシュを壊す揮発値を含めない）。
"""
from __future__ import annotations

import base64
import os
import difflib
import fnmatch
import re
import threading
import unicodedata
from pathlib import Path

from app.config import CONFIG
from app.extract.catalog import load_catalog
from app.extract.models import ExtractedDoc
from app.index.chunk import n_tokens

TOOL_DEFS = [
    {
        "name": "search",
        "description": "共有ドライブ全体のハイブリッド全文検索（BM25+ベクトル+リランク）。関連しそうな資料を特定する。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "検索キーワード（日本語可）"},
                "k": {"type": "integer", "description": "件数（既定8、最大20）"},
                "project": {"type": "string", "description": "案件名の一部で絞り込み"},
                "category": {"type": "string", "description": "サブフォルダ種別で絞り込み（例: 提案, 契約, 計画, データ, 分析, 会議, 報告書）"},
                "filetype": {"type": "string", "description": "拡張子で絞り込み（カンマ区切り可: pdf,docx,xlsx,pptx,ipynb,py,md,csv）"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "grep",
        "description": "抽出済み全文への正規表現検索。識別子・数値・書式アノテーション（==ハイライト=={色} や {fill:#RRGGBB 色名} など）の網羅検索に使う。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "正規表現（Python re）"},
                "project": {"type": "string", "description": "案件名の一部で絞り込み"},
                "path_glob": {"type": "string", "description": "相対パスのglob（例: */契約*/*.docx）"},
                "max_hits": {"type": "integer", "description": "最大件数（既定30）"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read",
        "description": "ファイルの抽出済み内容（書式アノテーション付きMarkdown）を読む。ページ/スライド/シート範囲を指定できる。",
        "input_schema": {
            "type": "object",
            "properties": {
                "relpath": {"type": "string", "description": "ツリーに記載の相対パス"},
                "pages": {"type": "string", "description": "ページ範囲（例: '1-5', '3', '10-'）。省略時は先頭から上限まで"},
            },
            "required": ["relpath"],
        },
    },
    {
        "name": "read_image",
        "description": "画像を実際に見る（PNGファイル / PDFページ / 文書内の埋め込み画像）。グラフ読解・スキャンページ・PDFの書式確認に使う。細かい文字が読めないときは region で部分拡大して読み直す。",
        "input_schema": {
            "type": "object",
            "properties": {
                "relpath": {"type": "string", "description": "画像ファイルまたはPDF等の相対パス"},
                "page": {"type": "integer", "description": "PDFのページ番号（1始まり）"},
                "asset_id": {"type": "string", "description": "抽出テキスト中の [画像: xxx] のID"},
                "region": {"type": "string", "description": "拡大領域を元画像に対する割合で 'x0,y0,x1,y1'（0-1）。例 右上4分の1='0.5,0,1,0.5'"},
            },
            "required": [],
        },
    },
    {
        "name": "ask_image",
        "description": (
            "画像について**具体的に質問**し、画像を読む専用のモデルに答えさせる。"
            "図の中の位置関係・並び・向き・矢印の指す先など、"
            "**自分で見て判断しにくいもの**をここで確かめる。"
            "相手は質問した画像しか見ていないので、質問文に必要な文脈を書くこと"
            "。位置・方向・包含・隣接・接続・矢印などの空間関係を問う場合は、"
            "ツール側が指定範囲の候補を該当・非該当にかかわらず全件列挙し、"
            "候補ごとの関係を分類して返す。質問では基準対象と確認範囲を明確にする。"
            "画像中の空間関係を問う設問では、read_imageの自動読取だけで提出せず、"
            "必ずこのツールで関係を検証する。"
            "回答には確信度が付く。**確信度が低い場合は断定しない。**"),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string",
                             "description": "画像に対する具体的な質問"},
                "relpath": {"type": "string", "description": "資料の相対パス"},
                "asset_id": {"type": "string", "description": "抽出済み画像のID"},
                "page": {"type": "integer", "description": "PDFのページ番号（1始まり）"},
                "region": {"type": "string",
                           "description": "'x0,y0,x1,y1'（0-1の割合）で一部だけを渡す"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "run_python",
        "description": "Pythonコードを実行して結果(stdout)を得る。pandas/numpy/openpyxl 利用可。集計・計算・条件抽出・全案件横断の走査は必ずこれで行う。\n使えるヘルパ:\n  projects() / files(project=, category=, pattern=) … 対象の列挙\n  P('相対パス') … 実パスに解決（open や pandas にそのまま渡せる）\n  read_text(rel) … Office/PDF の抽出済みテキスト（書式注記つき）\n  read_text(rel, plain=True) … 書式注記を外した素の本文。**氏名や値を正規表現で拾うときはこちら**\n  tables(rel) … 文書中の表を [{'header': [...], 'rows': [[...]]}] で返す。**表を数える・突き合わせるときはこれを使い、会話に流れた行を手写ししない**\n  read_csv(rel) / read_excel(rel) … pandas の短縮版",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "実行するPythonコード。結果は print() で出力する"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "diff",
        "description": "2ファイルの抽出テキストの差分（unified diff）。版数違い（old/v1/r2）の比較に使う。画像はmd5で同一性を判定できる。",
        "input_schema": {
            "type": "object",
            "properties": {
                "relpath_a": {"type": "string", "description": "旧版の相対パス"},
                "relpath_b": {"type": "string", "description": "新版の相対パス"},
            },
            "required": ["relpath_a", "relpath_b"],
        },
    },
    {
        "name": "decrypt",
        "description": "パスワード保護されたOfficeファイルを復号して内容を抽出する。パスワードは横断参照フォルダの導出規則から自分で導出して候補を渡すこと。",
        "input_schema": {
            "type": "object",
            "properties": {
                "relpath": {"type": "string", "description": "暗号化ファイルの相対パス"},
                "passwords": {"type": "array", "items": {"type": "string"},
                              "description": "試行するパスワード候補（順に試す）"},
            },
            "required": ["relpath", "passwords"],
        },
    },
    {
        "name": "submit_answer",
        "description": "最終回答を提出する（必ず最後に呼ぶ）。answer は要求された値のみを簡潔に。",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "最終回答（値のみ、日本語）"},
                "confidence": {"type": "number", "description": "確信度 0-1。根拠の確かさを正直に"},
                "evidence": {"type": "array", "items": {"type": "string"},
                             "description": "根拠にした相対パス（ページ付き可）"},
            },
            "required": ["answer", "confidence"],
        },
    },
]

IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"}


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


class ToolBox:
    """ツール実装。索引・抽出文書はプロセス内で共有する。

    search / grep は SudachiPy と GPU エンコードが非スレッドセーフなのでロックで直列化する
    （どちらも高速なのでボトルネックにならない）。
    """

    def __init__(self) -> None:
        from app.corpus.walk import PathResolver
        self.catalog = load_catalog()
        self.by_relpath = {e["relpath"]: e for e in self.catalog}
        self.by_doc_id = {e["doc_id"]: e for e in self.catalog}
        self.resolver = PathResolver()
        self._docs: dict[str, ExtractedDoc] = {}
        # 元の質問。read_image が画像読み取りモデルへ添える。
        # 並列実行中に別の質問で上書きされないようスレッドローカルにする。
        self._question = threading.local()
        self._lock = threading.Lock()
        self._index = None

    @property
    def question(self) -> str:
        return getattr(self._question, "value", "")

    @question.setter
    def question(self, value: str) -> None:
        self._question.value = value or ""
        # ToolBox は全設問で共有されるため、画像確認状態も質問と同じ
        # thread-local に置く。新しい設問へ前問の確認済み状態を持ち越さない。
        self._question.ocr_image_pending = False
        self._question.ocr_image_context = ""
        self._question.ocr_image_relpaths = set()

    def _track_ocr_output(self, output, context: str = "",
                          relpaths: list[str] | None = None):
        """OCR由来の本文をモデルへ渡したら、元画像確認を必須にする。"""
        if isinstance(output, str) and (
                "[画像読み取り]" in output
                or "[内部監査: OCR画像由来テキスト使用]" in output):
            paths = list(relpaths or [])
            if "[内部監査: OCR画像由来テキスト使用]" in output:
                # sandbox は監査行の後に、OCR本文へ触れた資料を字下げして出す。
                tail = output.split(
                    "[内部監査: OCR画像由来テキスト使用]", 1)[-1]
                paths.extend(line.strip() for line in tail.splitlines()[1:]
                             if line.startswith("  ")
                             and line.strip() in self.by_relpath)
            self._require_original_image(context, paths)
        return output

    def _require_original_image(self, context: str = "",
                                relpaths: list[str] | None = None) -> None:
        self._question.ocr_image_pending = True
        self._question.ocr_image_context = str(context or "")[:240]
        required = set(getattr(self._question, "ocr_image_relpaths", set()))
        required.update(_nfc(p.strip().lstrip("/")) for p in (relpaths or []) if p)
        self._question.ocr_image_relpaths = required

    @staticmethod
    def _search_chunk_uses_ocr(page_text: str, chunk_text: str) -> bool:
        """検索チャンクが、同じページのOCR差し込み位置より後ろか判定する。

        長いOCRページは複数チャンクへ分割され、2個目以降には
        ``[画像読み取り]`` マーカー自体が残らない。そのチャンクから数値を
        見つけた場合も元画像確認を要求するため、本文内の位置で判定する。
        """
        marker = page_text.find("[画像読み取り]")
        if marker < 0:
            return False
        body = chunk_text.split("\n", 1)[-1].strip()
        if not body:
            return False
        # 改行やチャンク境界のわずかな差を避け、最初の非空行を照合する。
        probe = next((line.strip() for line in body.splitlines()
                      if line.strip()), "")
        if not probe:
            return False
        pos = page_text.find(probe[:160], marker)
        return pos >= marker

    def _verification_relpath(self, relpath: str | None = None,
                              asset_id: str | None = None) -> str:
        """read_image の指定から、確認した資料の相対パスを解決する。"""
        if relpath:
            e = self._entry(relpath)
            return e["relpath"] if e else _nfc(relpath.strip().lstrip("/"))
        if asset_id:
            doc_id = asset_id.replace("\\", "/").split("/", 1)[0]
            e = self.by_doc_id.get(doc_id) or self._entry(asset_id)
            return e["relpath"] if e else ""
        return ""

    def _mark_original_image_checked(self, relpath: str | None = None,
                                     asset_id: str | None = None) -> None:
        """OCRと同じ資料の元画像を取得できた時だけ、確認待ちを解除する。"""
        required = set(getattr(self._question, "ocr_image_relpaths", set()))
        actual = self._verification_relpath(relpath, asset_id)
        if required and actual not in required:
            return
        self._question.ocr_image_pending = False
        self._question.ocr_image_context = ""
        self._question.ocr_image_relpaths = set()

    def image_verification_requirement(self) -> str:
        """未確認のOCR画像があれば、提出を止めるための指示を返す。"""
        if not getattr(self._question, "ocr_image_pending", False):
            return ""
        context = getattr(self._question, "ocr_image_context", "")
        where = f"（検出元: {context}）" if context else ""
        required = sorted(getattr(self._question, "ocr_image_relpaths", set()))
        same_doc = (" 候補資料: " + " / ".join(required[:3]) + "。")
        if not required:
            same_doc = ""
        return (
            "[提出保留] 回答候補の調査でOCR画像の読み取り結果を使用しています"
            f"{where}。OCRテキストだけでは提出できません。{same_doc}"
            "根拠に使う同じ資料の元画像を "
            "read_image または ask_image で確認してから、submit_answer を"
            "やり直してください。細部が読めない場合は region で拡大してください。"
        )

    # ---------------------------------------------------------- helpers
    def _entry(self, relpath: str) -> dict | None:
        key = _nfc((relpath or "").strip().lstrip("/"))
        e = self.by_relpath.get(key)
        if e is not None:
            return e
        tail = key.rsplit("/", 1)[-1]
        cands = [x for x in self.catalog
                 if x["relpath"].endswith("/" + tail) or x["relpath"] == tail]
        if len(cands) == 1:
            return cands[0]
        cands = [x for x in self.catalog if key in x["relpath"]]
        return cands[0] if len(cands) == 1 else None

    def _doc(self, relpath: str) -> ExtractedDoc | None:
        e = self._entry(relpath)
        if e is None:
            return None
        if e["doc_id"] not in self._docs:
            try:
                self._docs[e["doc_id"]] = ExtractedDoc.load(
                    CONFIG.extracted_dir, e["doc_id"])
            except (OSError, ValueError, KeyError, TypeError):
                return None
        return self._docs[e["doc_id"]]

    def _not_found(self, relpath: str) -> str:
        cands = self.resolver.candidates(relpath)
        return (f"[エラー] {relpath} が見つかりません。"
                + (f" 候補: {cands}" if cands else " ツリーの相対パスを確認してください。"))

    # ---------------------------------------------------------- tools
    def search(self, query: str, k: int = 8, project: str | None = None,
               category: str | None = None, filetype: str | None = None) -> str:
        from app.index.search import get_index
        with self._lock:
            if self._index is None:
                self._index = get_index()
            hits = self._index.search(query, k=min(int(k or 8), 20),
                                      project=project, category=category,
                                      filetype=filetype)
        if not hits:
            return "(ヒットなし — 言い換えるか grep を試してください)"
        lines = []
        ocr_hits: list[str] = []
        for h in hits:
            loc = f"p{h['page']}" + (f"({h['label']})" if h["label"] else "")
            snippet = h["text"].replace("\n", " ")[:260]
            lines.append(f"- {h['relpath']} {loc}\n  {snippet}")
            doc = self._doc(h["relpath"])
            page = (next((p for p in doc.pages if p.no == h["page"]), None)
                    if doc else None)
            if page and self._search_chunk_uses_ocr(page.text, h["text"]):
                ocr_hits.append(f"{h['relpath']} p{h['page']}")
        result = self._track_ocr_output("\n".join(lines), f"search: {query}")
        if ocr_hits:
            self._require_original_image("search: " + " | ".join(ocr_hits[:3]),
                                         [h.split(" p", 1)[0] for h in ocr_hits])
        return result

    def grep(self, pattern: str, project: str | None = None,
             path_glob: str | None = None, max_hits: int = 30) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"[エラー] 正規表現が不正: {e}"
        limit = min(int(max_hits or 30), 100)
        out: list[str] = []
        for e in self.catalog:
            if project and project.lower() not in (e["project"] or "").lower():
                continue
            if path_glob and not fnmatch.fnmatch(e["relpath"], path_glob):
                continue
            doc = self._doc(e["relpath"])
            if doc is None:
                continue
            for page in doc.pages:
                for m in rx.finditer(page.text):
                    ls = page.text.rfind("\n", 0, m.start()) + 1
                    le = page.text.find("\n", m.end())
                    line = page.text[ls:le if le > 0 else None]
                    out.append(f"- {e['relpath']} p{page.no}: {line.strip()[:220]}")
                    marker = page.text.find("[画像読み取り]")
                    if 0 <= marker <= m.start():
                        self._require_original_image(
                            f"grep: {e['relpath']} p{page.no}", [e["relpath"]])
                    if len(out) >= limit:
                        out.append(f"...(上限{limit}件に到達。pattern を絞ってください)")
                        return self._track_ocr_output(
                            "\n".join(out), f"grep: {pattern}")
        result = "\n".join(out) if out else "(マッチなし)"
        return self._track_ocr_output(result, f"grep: {pattern}")

    def read(self, relpath: str, pages: str | None = None) -> str:
        doc = self._doc(relpath)
        if doc is None:
            return self._not_found(relpath)
        sel = _parse_pages(pages, len(doc.pages))
        budget = CONFIG.tool_result_max_tokens
        parts, used, shown = [], 0, []
        for p in doc.pages:
            if p.no not in sel:
                continue
            t = n_tokens(p.text)
            if used + t > budget and parts:
                break
            text = p.text
            if t > budget:
                # 単一ページが予算を超える場合も必ず切り詰める。
                # ここを素通りさせると巨大シート1枚でコンテキストが溢れる。
                shown_text = _clip_tokens(text, budget)
                # 切り落とした後半に何があるかを示す。黙って落とすと、
                # 末尾のEMF表・グラフ・コメントの存在を示す。
                cut = text[len(shown_text):]
                marks = _text_marks(cut)
                text = shown_text + (
                    f"\n[このページは大きいため先頭のみ表示（全{t}トークン）。"
                    "続きは run_python で read_text(相対パス) を使うと全文が読める]")
                if marks:
                    text += f"\n[未表示部分にあるもの: {marks}]"
            parts.append(text)
            used += min(t, budget)
            shown.append(p.no)
        if not parts:
            return f"(指定ページが空です。総ページ数: {len(doc.pages)})"
        header = (f"[{doc.relpath} | 全{len(doc.pages)}ページ中 "
                  f"p{shown[0]}-p{shown[-1]} を表示]")
        remain = [p for p in doc.pages if p.no in sel and p.no not in shown]
        footer = ""
        if remain:
            # 上限で落ちたページに **何があるか** を目次として示す。
            # 末尾にある埋め込みグラフ・EMF表・コメントの存在を示す。
            lines = []
            for p in remain[:40]:
                marks = _page_marks(p)
                lines.append(f"  p{p.no}"
                             + (f"({p.label})" if p.label else "")
                             + (f" — {marks}" if marks else ""))
            footer = ("\n[未表示のページ（pages 指定で読めます）]\n"
                      + "\n".join(lines))
            if len(remain) > 40:
                footer += f"\n  …他 {len(remain)-40} ページ"
        result = header + "\n" + "\n\n".join(parts) + footer
        return self._track_ocr_output(result, f"read: {doc.relpath}",
                                      [doc.relpath])

    def read_image(self, relpath: str | None = None, page: int | None = None,
                   asset_id: str | None = None,
                   region: str | None = None) -> dict | str:
        from app.ocr.image_util import crop_region, normalize
        img: bytes | None = None
        note = ""

        if asset_id:
            p = CONFIG.assets_dir / asset_id
            if p.exists():
                img, note = p.read_bytes(), asset_id
            else:
                e = self._entry(asset_id)
                if e is None:
                    return f"[エラー] asset {asset_id} が見つかりません"
                return self.read_image(relpath=e["relpath"], region=region)
        elif relpath:
            e = self._entry(relpath)
            if e is None:
                return self._not_found(relpath)
            raw = self.resolver.resolve(e["relpath"])
            if raw is None:
                return self._not_found(relpath)
            if e["filetype"] == "pdf":
                if not page:
                    return "[エラー] PDFは page 番号（1始まり）を指定してください"
                from app.extract.pdf import render_page_png
                edge = CONFIG.image_max_edge_px * (3 if region else 1)
                try:
                    img = render_page_png(raw, int(page), edge)
                except (OSError, ValueError, RuntimeError, IndexError) as ex:
                    return f"[エラー] PDFレンダリング失敗: {type(ex).__name__}: {ex}"
                note = f"{e['relpath']} p{page}"
            elif e["filetype"] in IMAGE_EXTS:
                img, note = Path(raw).read_bytes(), e["relpath"]
            else:
                doc = self._doc(e["relpath"])
                assets = doc.assets if doc else []
                return (f"[このファイルの画像は asset_id で指定してください: {assets[:20]}]"
                        if assets else "[このファイルに抽出済み画像はありません]")
        else:
            return "[エラー] relpath か asset_id を指定してください"

        if region:
            img, err = crop_region(img, region)
            if err:
                return err
            note += f" region={region}"
        hint = ""
        if not region:
            # 色領域の自動添付はしない。しきい値を特定の画像に合わせ込んでおり、
            # 新しいデータで機能する保証がないため（run_python の
            # 特定の画像に合わせ込んだしきい値に依存するため。
            hint = self._image_reading(img)
        img = normalize(img, CONFIG.image_max_edge_px, fmt="JPEG")
        # 画像ブロックを返せるところまで到達したので、メインモデルは元画像を
        # 実際に確認できる。質問付き別モデル読みの成否だけには依存させない。
        self._mark_original_image_checked(relpath=relpath, asset_id=asset_id)
        return {"blocks": [
            {"type": "image", "media_type": "image/jpeg",
             "data": base64.b64encode(img).decode()},
            {"type": "text", "text": f"[画像: {note}]" + hint},
        ]}

    def _image_bytes(self, relpath=None, page=None, asset_id=None, region=None):
        """read_image と同じ解決規則で画像バイト列を得る。(bytes, note, err)"""
        # ask_image は、専用モデルが実際に画像を読めた時点で確認済みにする。
        # 内部で read_image を再利用しても、その前に解除しないよう状態を退避する。
        was_pending = getattr(self._question, "ocr_image_pending", False)
        saved_context = getattr(self._question, "ocr_image_context", "")
        saved_relpaths = set(getattr(self._question, "ocr_image_relpaths", set()))
        r = self.read_image(relpath=relpath, page=page, asset_id=asset_id,
                            region=region)
        if was_pending:
            self._question.ocr_image_pending = True
            self._question.ocr_image_context = saved_context
            self._question.ocr_image_relpaths = saved_relpaths
        if isinstance(r, str):
            return None, "", r
        img = b""
        note = ""
        for b in r.get("blocks", []):
            if b.get("type") == "image":
                img = base64.b64decode(b["data"])
            elif b.get("type") == "text":
                note = b.get("text", "")
        if not img:
            return None, "", "[エラー] 画像を取得できませんでした"
        return img, note, None


    def _image_reading(self, img: bytes) -> str:
        """同じ画像と元の質問を、画像読み取り用のモデルにも読ませる。

        別のモデルの読みを併記し、質問は追加指示を付けずそのまま渡す。
        全体を読むときだけ呼ぶ。1回およそ300秒かかるので回数を抑える。
        """
        if not self.question:
            return ""
        try:
            from app.ocr.ask import ask, is_spatial_relation_question
            r = ask(img, self.question)
        except Exception:                  # noqa: BLE001 - 補助情報なので落とさない
            return ""
        if r.error or not r.answer.strip():
            # 読み取り失敗を呼び出し元へ返す。
            return (f"\n\n[別のモデルによる読み取り] 失敗: "
                    f"{r.error or '空の回答'}（{r.latency_sec}秒）")
        out = [f"\n\n[別のモデルによる読み取り] {r.engine}",
               f"  {r.answer.strip()}"]
        if r.evidence.strip():
            out.append(f"  根拠: {r.evidence.strip()}")
        out.append("  ※これは参考です。自分でも画像を確認し、"
                   "食い違う場合はどちらが図と整合するかで判断してください。"
                   if r.confident else
                   "  ※読み取り側も確信を持てていません。断定しないでください。")
        if is_spatial_relation_question(self.question):
            out.append(
                "  [空間関係の検証が必要] 自動読取は候補と関係を構造化検証していないため、"
                "提出前に ask_image を呼び、基準対象と確認範囲を明記して検証してください。")
        return "\n".join(out)

    def ask_image(self, question: str, relpath: str | None = None,
                  page: int | None = None, asset_id: str | None = None,
                  region: str | None = None) -> str:
        """画像に問いを添えて、画像を読む専用のモデルに答えさせる。

        事前バッチOCRは一律のプロンプトで読むので、問いに固有の読み取り
        （位置関係・向き・矢印の指す先）には届かないため、問いごとに読む。

        位置・方向・包含・隣接・接続・矢印などの空間関係質問は、候補を先に
        全件列挙する共通の構造化モードへ送る。元の設問が空間関係なら、モデルが
        組み立てた文で網羅指定や関係の範囲が縮まらないよう、スレッドローカルの
        元質問を正規形として使う。固有の設問・資料・回答では分岐しない。

        結果は 画像md5 + 正規化した質問 + モード別プロンプト版でキャッシュする。
        """
        img, _note, err = self._image_bytes(relpath=relpath, page=page,
                                            asset_id=asset_id, region=region)
        if err:
            return err
        from app.ocr.ask import (ask as _ask, ask_spatial,
                                 is_spatial_relation_question)
        original = self.question.strip()
        # 元質問が空間関係なら常にそれを使う。これにより、メインモデルが
        # `ask_image` の質問を単数形へ縮めたり言い換えたりしても意味を保存し、
        # ランごとの質問文揺れによるキャッシュミスも防ぐ。
        if original and is_spatial_relation_question(original):
            result = ask_spatial(img, original)
        elif is_spatial_relation_question(question):
            result = ask_spatial(img, question)
        else:
            result = _ask(img, question)
        if not getattr(result, "error", None):
            self._mark_original_image_checked(relpath=relpath, asset_id=asset_id)
        return result.as_text()

    def run_python(self, code: str) -> str:
        from app.agent.sandbox import run_python as _run
        return self._track_ocr_output(_run(code), "run_python")

    def diff(self, relpath_a: str, relpath_b: str) -> str:
        da, db = self._doc(relpath_a), self._doc(relpath_b)
        if da is None:
            return self._not_found(relpath_a)
        if db is None:
            return self._not_found(relpath_b)

        def norm(text: str) -> str:
            # asset_id の doc_id 接頭辞は文書ごとに違うのでノイズになる。
            # md5 は残す（テキスト同一でも画像が差し替わっている場合の判定に使う）。
            return re.sub(r"([\[ ])[0-9a-f]{12}/", r"\1DOC/", text)

        lines = list(difflib.unified_diff(
            norm(da.full_text).splitlines(), norm(db.full_text).splitlines(),
            fromfile=da.relpath, tofile=db.relpath, lineterm="", n=1))
        if not lines:
            return "(差分なし — 抽出テキストは同一です)"
        text = "\n".join(lines)
        budget = CONFIG.tool_result_max_tokens
        if n_tokens(text) > budget:
            core = [ln for ln in lines if ln[:1] in "+-@"]
            text = "\n".join(core)
            if n_tokens(text) > budget:
                text = text[:budget * 3] + "\n...[切り詰め]"
        return text + "".join(self._image_changes(lines, da, db))

    @staticmethod
    def _doc_text(doc) -> str:
        """抽出文書の本文（ページ等に分かれている形にも対応）。"""
        t = getattr(doc, "text", "") or ""
        if t:
            return t
        for k in ("pages", "sheets", "slides"):
            v = getattr(doc, k, None)
            if isinstance(v, list):
                return "\n".join(str(getattr(x, "text", x)) for x in v)
        return ""

    def _image_changes(self, lines: list[str], da, db) -> list[str]:
        """版比較で画像が差し替わっていたら、2枚を画像読み取りモデルに見せる。

        版による寸法や列幅の違いを避けるため画素差分は使わず、2枚を直接比較する。
        """
        pat = re.compile(r"\[画像: (\S+) md5:(\w+)")   # 行頭とは限らない
        old = {m.group(1).split("/")[-1]: (m.group(1), m.group(2))
               for m in pat.finditer(self._doc_text(da))}
        new = {m.group(1).split("/")[-1]: (m.group(1), m.group(2))
               for m in pat.finditer(self._doc_text(db))}
        out: list[str] = []
        for name in sorted(set(old) & set(new)):
            if old[name][1] == new[name][1]:
                continue
            pa = CONFIG.assets_dir / old[name][0]
            pb = CONFIG.assets_dir / new[name][0]
            if not (pa.exists() and pb.exists()):
                continue
            out.append(f"\n\n■ {name} が差し替わっています "
                       f"(A md5:{old[name][1]} → B md5:{new[name][1]})")
            try:
                from app.ocr.ask import compare as _cmp
                out.append(_cmp(pa.read_bytes(), pb.read_bytes()).as_text())
            except Exception as e:            # noqa: BLE001 - 補助情報
                out.append(f"[画像の比較] 失敗: {type(e).__name__}")
        return out

    def decrypt(self, relpath: str, passwords: list[str]) -> str:
        e = self._entry(relpath)
        if e is None:
            return self._not_found(relpath)
        raw = self.resolver.resolve(e["relpath"])
        if raw is None:
            return self._not_found(relpath)

        from app.extract.encrypted import try_decrypt
        data, pw = try_decrypt(raw, list(passwords or []))
        if data is None:
            return (f"[失敗] {len(passwords or [])}個の候補すべてで復号できませんでした。"
                    "導出規則を読み直して候補を作り直してください。")

        import tempfile
        from app.corpus.walk import CorpusFile
        from app.extract.pipeline import extract_one
        suffix = "." + e["filetype"]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(data)
            tmp_path = tf.name
        cf = CorpusFile(raw_path=tmp_path, relpath=e["relpath"],
                        doc_id=e["doc_id"], ext=suffix, size=len(data),
                        md5=e.get("md5", ""), mount=e.get("mount", ""),
                        project=e["project"], category=e["category"])
        doc = extract_one(cf, str(CONFIG.assets_dir))
        doc.meta["decrypted"] = True
        doc.save(CONFIG.extracted_dir)
        self._docs[e["doc_id"]] = doc
        Path(tmp_path).unlink(missing_ok=True)
        head = doc.full_text[:2500]
        return (f"[成功] パスワードが一致しました。以後 read('{e['relpath']}') で"
                f"全文を読めます。冒頭:\n{head}")

    # ---------------------------------------------------------- dispatch
    def call(self, name: str, args: dict) -> str | dict:
        fn = getattr(self, name, None)
        if fn is None or name == "call":
            return f"[エラー] 未知のツール: {name}"
        try:
            return fn(**args)
        except TypeError as e:
            return f"[エラー] 引数が不正です: {e}"
        except Exception as e:                    # noqa: BLE001 - 落とさず返す
            return f"[エラー] {type(e).__name__}: {e}"


_PAGE_MARKERS = (
    ("埋め込みEMFオブジェクト", "EMF埋め込み表あり"),
    ("### 埋め込みグラフ", "埋め込みグラフあり"),
    ("[画像読み取り]", "画像の読み取り結果あり"),
    ("## コメント", "コメントあり"),
    ("[ページ情報]", "ページ情報あり"),
    ("### ピボットテーブル定義", "ピボット定義あり"),
    ("[画像:", "画像あり"),
)


def _text_marks(text: str) -> str:
    """テキスト断片に含まれる注目要素の目印。"""
    return " / ".join(dict.fromkeys(
        label for key, label in _PAGE_MARKERS if key in text))


def _page_marks(page) -> str:
    """未表示ページに何が入っているかの目印。"""
    found = [label for key, label in _PAGE_MARKERS if key in page.text]
    head = page.text.strip().split("\n", 1)[0][:40]
    if head.startswith("##"):
        found.insert(0, head.lstrip("# ").strip())
    return " / ".join(dict.fromkeys(found))


def _clip_tokens(text: str, budget: int) -> str:
    """トークン数で切り詰める（文字数だと日本語で大きくぶれる）。"""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text, disallowed_special=())
    if len(toks) <= budget:
        return text
    return enc.decode(toks[:budget])


def _parse_pages(pages: str | None, n: int) -> set[int]:
    if not pages:
        return set(range(1, n + 1))
    out: set[int] = set()
    for part in str(pages).split(","):
        part = part.strip()
        if "-" in part:
            a, _, b = part.partition("-")
            lo = int(a) if a.strip().isdigit() else 1
            hi = int(b) if b.strip().isdigit() else n
            out |= set(range(max(1, lo), min(n, hi) + 1))
        elif part.isdigit():
            out.add(int(part))
    return out or set(range(1, n + 1))
