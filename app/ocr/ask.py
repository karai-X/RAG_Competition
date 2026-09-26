"""元の質問を添えて画像を読み取る、事前バッチOCRとは別の経路。

空間関係を問う場合は候補を列挙して関係を分類する専用スキーマを使う。
キャッシュキーには画像md5、質問文、エンジン、プロンプト版を含める。
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import CONFIG
from app.ocr.targets import image_key

# 画像を読むモデルへの指示。**特定の資料・設問には一切触れない汎用の読み方**だけを書く。
# 読み方の指示は書かない（上記の理由）。出力形式の指定だけを添える。
ASK_PROMPT = "添付の画像を見て、次の質問に答えてください。"

ASK_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "質問への回答"},
        "evidence": {"type": "string", "description": "画像のどこを見て判断したか"},
        "confident": {"type": "boolean", "description": "確信が持てるか"},
    },
    "required": ["answer", "confident"],
}


# 空間関係モードは `ask_image`（モデルが明示的に質問する経路）だけで使う。
# 特定の図・氏名・設問形式には依存せず、画像中の候補を関係で選ぶ問い全般を扱う。
SPATIAL_PROMPT_VERSION = "v4"
SPATIAL_PROMPT = """添付画像について、次の空間関係の質問を検証してください。
最終回答を先に決めず、必ず次の順序で調べてください。

1. 質問が指定する図・区画・グループ・範囲を特定する。図が名前付きの複数グループに分かれ、質問が一つの基準対象について尋ねている場合は、質問が図全体を明示しない限り、その対象を含む最小の名前付きグループを範囲とする。指定範囲外の候補は混ぜない。
2. 図中のラベルと対象を正しく対応付ける。ラベルから引出線・矢印が伸びている場合は、線をラベル端から終点まで連続して追跡し、終点が指す対象だけをそのラベルに対応付ける。ラベル枠の色、職種の色、近くにある対象の色や単なる近接は対応付けの根拠にしない。線が交差・遮蔽して終点を確定できなければ unknown とする。
3. 対応付けを終えた対象のうち、質問の答えになり得る候補を、該当しないものも含めて先に全件列挙する。各候補の binding に、ラベルからどの線を追い、終点が何を指したかを書く。
4. 基準対象の位置と向きを画像から特定する。人物・物体から見た方向なら、画面の左右ではなく基準対象の向きを使う。着席者では、対応付けた椅子からその人が使うキーボードまたはモニターへ向かう方向を優先し、島の中心や他人の位置だけから向きを決めない。
5. 各候補を基準対象から見た8方向
   front_left / front / front_right / left / right / back_left / back / back_right
   のいずれかに分類する。方向を使わない関係では unknown とし、relations に位置・包含・隣接・対面・接続・矢印などの関係を書く。
6. 各候補が質問条件に該当するかを一件ずつ判定する。

「右側」の問いでは、各候補を right / front_right / back_right / 非該当のどれかとして独立に確認し、最初の3区分をすべて記録してください。
「左側」の問いでは、各候補を left / front_left / back_left / 非該当のどれかとして独立に確認し、最初の3区分をすべて記録してください。
「右隣」「左隣」「向かい」「対面」のように関係が限定されている場合は、その限定を保ってください。
画像から確定できない候補や関係は推測せず unknown としてください。"""

_SECTORS = (
    "front_left", "front", "front_right", "left", "right",
    "back_left", "back", "back_right", "same", "unknown",
)

SPATIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "scope": {
            "type": "string",
            "description": "質問条件に従って調べた画像内の範囲・区画・グループ",
        },
        "reference": {
            "type": "string",
            "description": "位置関係の基準となる人物・物体・領域",
        },
        "facing": {
            "type": "string",
            "description": "基準対象の向きと、その判断根拠。向きが不要・不明なら空文字列",
        },
        "candidates": {
            "type": "array",
            "description": "指定範囲内で答えになり得る候補の全件インベントリ",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "候補を一意に識別できる画像内の名称・ラベル",
                    },
                    "binding": {
                        "type": "string",
                        "description": "ラベルから引出線・矢印の終点までを追って対象を対応付けた根拠。線がなければ対応付けに使った画像上の根拠",
                    },
                    "sector": {
                        "type": "string",
                        "enum": list(_SECTORS),
                        "description": "基準対象から見た8方向。適用不能・不明なら unknown",
                    },
                    "relations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "隣接・対面・包含・接続・矢印など画像から確認した関係",
                    },
                    "matches_question": {
                        "type": "boolean",
                        "description": "この候補が質問の関係条件に該当するか",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "位置・向き・線・境界など、判定に使った画像上の根拠",
                    },
                },
                "required": ["label", "binding", "sector", "relations",
                             "matches_question", "evidence"],
            },
        },
        "exhaustive": {
            "type": "boolean",
            "description": "指定範囲の候補を該当・非該当にかかわらず全件確認できたか",
        },
        "confident": {
            "type": "boolean",
            "description": "候補の対応関係と分類を画像から確信を持って判定できたか",
        },
    },
    "required": ["scope", "reference", "facing", "candidates",
                 "exhaustive", "confident"],
}


# `ask_image` は画像に対する質問だけを受け取るため、ここでは空間的な関係を
# 表す語に限定して判定する。「上」「下」単独は「以上」「以下」等に誤発火するので
# 使わない。英語は単語境界を付ける。
_SPATIAL_RELATION_RE = re.compile(
    r"から見て|視点|位置関係|配置関係|どこに位置|"
    r"右側|左側|右手側|左手側|右方向|左方向|右隣|左隣|"
    r"右斜め|左斜め|向かい|対面|正面|前方|後方|背後|"
    r"上側|下側|上方|下方|内側|外側|内部|外部|"
    r"隣接|隣り|接して|交差|接続|包含|囲ま|"
    r"矢印.{0,12}(?:指|先)|同じ(?:行|列|区画|グループ)|"
    r"並び|何番目|最も近|近接|"
    r"\b(?:right|left|above|below|opposite|adjacent|inside|outside|nearest)\b",
    re.IGNORECASE,
)

_RIGHT_NEIGHBOR_RE = re.compile(r"右隣|右となり|right\s+(?:next|neighbor)", re.I)
_LEFT_NEIGHBOR_RE = re.compile(r"左隣|左となり|left\s+(?:next|neighbor)", re.I)
_RIGHT_AREA_RE = re.compile(r"右側|右手側|右方向|\bright(?:-hand)?\s+side\b", re.I)
_LEFT_AREA_RE = re.compile(r"左側|左手側|左方向|\bleft(?:-hand)?\s+side\b", re.I)


def is_spatial_relation_question(question: str) -> bool:
    """画像候補を空間的な関係で選ぶ質問か。固有語や回答件数は見ない。"""
    text = unicodedata.normalize("NFKC", str(question or "")).strip()
    return bool(text and _SPATIAL_RELATION_RE.search(text))


def spatial_relation_kind(question: str) -> str:
    """コードで集合化できる方向領域だけを識別する。その他も空間モードで読む。"""
    text = unicodedata.normalize("NFKC", str(question or ""))
    if _RIGHT_NEIGHBOR_RE.search(text):
        return "right_neighbor"
    if _LEFT_NEIGHBOR_RE.search(text):
        return "left_neighbor"
    if _RIGHT_AREA_RE.search(text):
        return "right_area"
    if _LEFT_AREA_RE.search(text):
        return "left_area"
    return "other"


_SECTOR_ALIASES = {
    "right_front": "front_right",
    "right-back": "back_right",
    "right_back": "back_right",
    "left_front": "front_left",
    "left-back": "back_left",
    "left_back": "back_left",
    "右": "right",
    "真右": "right",
    "右前": "front_right",
    "右斜め前": "front_right",
    "右後": "back_right",
    "右斜め後ろ": "back_right",
    "左": "left",
    "真左": "left",
    "左前": "front_left",
    "左斜め前": "front_left",
    "左後": "back_left",
    "左斜め後ろ": "back_left",
    "前": "front",
    "正面": "front",
    "後": "back",
    "後ろ": "back",
}


def _sector(value) -> str:
    s = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    s = s.replace(" ", "_")
    s = _SECTOR_ALIASES.get(s, s)
    return s if s in _SECTORS else "unknown"


def _bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "はい"}
    return bool(value)


@dataclass
class SpatialAskResult:
    md5: str = ""
    question: str = ""
    engine: str = ""
    prompt_version: str = SPATIAL_PROMPT_VERSION
    scope: str = ""
    reference: str = ""
    facing: str = ""
    candidates: list[dict] = field(default_factory=list)
    exhaustive: bool = False
    confident: bool = False
    error: str | None = None
    latency_sec: float = 0.0

    def matched_labels(self) -> tuple[list[str], str]:
        """方向領域は8方向から決定し、その他は候補ごとの判定を使う。"""
        kind = spatial_relation_kind(self.question)
        wanted = {
            "right_area": {"front_right", "right", "back_right"},
            "left_area": {"front_left", "left", "back_left"},
        }.get(kind)
        rows = [c for c in self.candidates if isinstance(c, dict)]
        known_sectors = any(_sector(c.get("sector")) != "unknown" for c in rows)

        if wanted is not None and known_sectors:
            selected = [c for c in rows if _sector(c.get("sector")) in wanted]
            source = "8方向分類から機械的に選択"
        else:
            selected = [c for c in rows if _bool(c.get("matches_question"))]
            source = "候補ごとの関係判定から選択"

        labels: list[str] = []
        for c in selected:
            label = str(c.get("label") or "").strip()
            if label and label not in labels:
                labels.append(label)
        return labels, source

    def as_text(self) -> str:
        if self.error:
            return f"[画像への空間関係質問] 読み取れませんでした: {self.error}"

        labels, source = self.matched_labels()
        out = [f"[画像への空間関係質問] {self.question}"]
        out.append(
            f"  検証済み選択集合（{source}）: "
            + ("、".join(labels) if labels else "該当候補なし")
        )
        out.append(
            "  ※上の集合は下記の候補全件から関係演算で選択した結果です。"
            "候補を列挙していない自動読取の単発回答より優先してください。"
        )
        if self.scope.strip():
            out.append(f"  確認範囲: {self.scope.strip()}")
        if self.reference.strip():
            out.append(f"  基準対象: {self.reference.strip()}")
        if self.facing.strip():
            out.append(f"  基準の向き: {self.facing.strip()}")
        out.append("  候補全件:")
        for c in self.candidates:
            if not isinstance(c, dict):
                continue
            label = str(c.get("label") or "名称不明").strip()
            binding = str(c.get("binding") or "").strip()
            sector = _sector(c.get("sector"))
            rels = ", ".join(str(x) for x in (c.get("relations") or []) if str(x))
            mark = "該当" if _bool(c.get("matches_question")) else "非該当"
            detail = f"sector={sector}; {mark}"
            if binding:
                detail += f"; 対応付け={binding}"
            if rels:
                detail += f"; relations={rels}"
            evidence = str(c.get("evidence") or "").strip()
            if evidence:
                detail += f"; 根拠={evidence}"
            out.append(f"    - {label}: {detail}")
        if not self.exhaustive:
            out.append("  **網羅確認: 未完了。候補の拾い漏れがあり得ます。**")
        if not self.confident:
            out.append("  **確信度: 低い。この結果だけで断定せず、画像と候補全件を再確認してください。**")
        return "\n".join(out)


@dataclass
class AskResult:
    md5: str = ""
    question: str = ""
    engine: str = ""
    answer: str = ""
    evidence: str = ""
    confident: bool = False
    error: str | None = None
    latency_sec: float = 0.0

    def as_text(self) -> str:
        if self.error:
            return f"[画像への質問] 読み取れませんでした: {self.error}"
        head = f"[画像への質問] {self.question}"
        body = [head, f"  回答: {self.answer.strip()}"]
        if self.evidence.strip():
            body.append(f"  根拠: {self.evidence.strip()}")
        if not self.confident:
            body.append("  **確信度: 低い。この結果だけで断定せず、"
                        "read_image で該当箇所を確認するか「わかりません」と答えてください。**")
        return "\n".join(body)


def _key(md5: str, question: str, engine: str) -> str:
    q = hashlib.md5(question.strip().encode("utf-8")).hexdigest()[:12]
    return f"{md5}.{q}.{engine}"


def _path(md5: str, question: str, engine: str) -> Path:
    return CONFIG.ocr_cache_dir / f"ask.{_key(md5, question, engine)}.json"


def ask(img: bytes, question: str, engine: str | None = None,
        timeout: int | None = None, use_cache: bool = True) -> AskResult:
    """画像 + 質問 → 構造化された回答。例外は投げない。"""
    engine = engine or CONFIG.model_image
    md5 = image_key(img) if img else ""
    res = AskResult(md5=md5, question=question.strip(), engine=engine)
    if not img:
        res.error = "画像データが空"
        return res
    if not question.strip():
        res.error = "質問が空"
        return res

    p = _path(md5, question, engine)
    if use_cache and p.exists():
        try:
            return AskResult(**json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    t0 = time.time()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": ASK_PROMPT + "\n\n" + question.strip()},
        {"type": "image", "media_type": "image/png",
         "data": base64.b64encode(img).decode()},
    ]}]
    try:
        from app.ocr.engines import backend_for
        be = backend_for(engine)
    except Exception as e:                        # noqa: BLE001
        res.error = f"backend生成失敗: {type(e).__name__}: {e}"
        return res

    # 試行は1回だけ（retries は「試行回数」であって追加回数ではない）。
    # 画像読み取りは1回が数百秒かかり、再試行は時間を倍にするだけで成功率は上がらない。
    reply = be.chat_with_retry(messages, schema=ASK_SCHEMA, max_tokens=2000,
                               timeout=timeout or CONFIG.claude_cli_timeout_sec,
                               retries=1)
    res.latency_sec = round(time.time() - t0, 2)
    if not reply.ok or not isinstance(reply.data, dict):
        res.error = reply.error or "構造化出力が得られなかった"
        return res

    d = reply.data
    res.answer = str(d.get("answer") or "")
    res.evidence = str(d.get("evidence") or "")
    res.confident = bool(d.get("confident"))
    if use_cache:
        try:
            CONFIG.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(res), ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(p)
        except OSError:
            pass
    return res


def _spatial_path(md5: str, question: str, engine: str) -> Path:
    q = hashlib.md5(question.strip().encode("utf-8")).hexdigest()[:12]
    return CONFIG.ocr_cache_dir / (
        f"spatial.{SPATIAL_PROMPT_VERSION}.{md5}.{q}.{engine}.json")


def ask_spatial(img: bytes, question: str, engine: str | None = None,
                timeout: int | None = None,
                use_cache: bool = True) -> SpatialAskResult:
    """画像の空間関係を、候補インベントリを先に作ってから判定する。

    `read_image` に自動で付く直接質問では使わず、`ask_image` から明示的に
    問われた空間関係だけをこの経路へ送る。固有の画像・質問・回答には依存しない。
    """
    engine = engine or CONFIG.model_image
    md5 = image_key(img) if img else ""
    res = SpatialAskResult(md5=md5, question=question.strip(), engine=engine)
    if not img:
        res.error = "画像データが空"
        return res
    if not question.strip():
        res.error = "質問が空"
        return res

    p = _spatial_path(md5, question, engine)
    if use_cache and p.exists():
        try:
            return SpatialAskResult(**json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    t0 = time.time()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": SPATIAL_PROMPT + "\n\n質問:\n" + question.strip()},
        {"type": "image", "media_type": "image/png",
         "data": base64.b64encode(img).decode()},
    ]}]
    try:
        from app.ocr.engines import backend_for
        be = backend_for(engine)
    except Exception as e:                        # noqa: BLE001
        res.error = f"backend生成失敗: {type(e).__name__}: {e}"
        return res

    reply = be.chat_with_retry(
        messages, schema=SPATIAL_SCHEMA, max_tokens=3000,
        timeout=timeout or CONFIG.claude_cli_timeout_sec, retries=1)
    res.latency_sec = round(time.time() - t0, 2)
    if not reply.ok or not isinstance(reply.data, dict):
        res.error = reply.error or "構造化出力が得られなかった"
        return res

    d = reply.data
    res.scope = str(d.get("scope") or "")
    res.reference = str(d.get("reference") or "")
    res.facing = str(d.get("facing") or "")
    raw_candidates = d.get("candidates")
    res.candidates = ([dict(x) for x in raw_candidates if isinstance(x, dict)]
                      if isinstance(raw_candidates, list) else [])
    res.exhaustive = _bool(d.get("exhaustive"))
    res.confident = _bool(d.get("confident"))
    if not res.candidates:
        res.error = "候補インベントリが得られなかった"
        return res

    if use_cache:
        try:
            CONFIG.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(res), ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(p)
        except OSError:
            pass
    return res


COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "changed": {"type": "boolean", "description": "内容として違いがあるか"},
        "summary": {"type": "string",
                    "description": "何が変わったか。変わっていない部分も分かれば書く"},
        "confident": {"type": "boolean", "description": "確信が持てるか"},
    },
    "required": ["changed", "summary", "confident"],
}


@dataclass
class CompareResult:
    md5_a: str = ""
    md5_b: str = ""
    engine: str = ""
    changed: bool = False
    summary: str = ""
    confident: bool = False
    error: str | None = None
    latency_sec: float = 0.0

    def as_text(self) -> str:
        if self.error:
            return f"[画像の比較] 読み取れませんでした: {self.error}"
        head = "[画像の比較] " + ("内容が変わっています" if self.changed
                                  else "内容の違いは見当たりません")
        out = [head]
        if self.summary.strip():
            out.append(f"  {self.summary.strip()}")
        if not self.confident:
            out.append("  **確信度: 低い。この結果だけで断定しないでください。**")
        return "\n".join(out)


def compare(img_a: bytes, img_b: bytes, engine: str | None = None,
            timeout: int | None = None, use_cache: bool = True) -> CompareResult:
    """2枚の画像を渡し、内容として何が変わったかを答えさせる。

    版による寸法や列幅の違いを避けるため画素差分は使わず、2枚を直接比較する。
    """
    engine = engine or CONFIG.model_image
    ma = image_key(img_a) if img_a else ""
    mb = image_key(img_b) if img_b else ""
    res = CompareResult(md5_a=ma, md5_b=mb, engine=engine)
    if not img_a or not img_b:
        res.error = "画像データが空"
        return res

    p = CONFIG.ocr_cache_dir / f"cmp.{ma}.{mb}.{engine}.json"
    if use_cache and p.exists():
        try:
            return CompareResult(**json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    t0 = time.time()
    messages = [{"role": "user", "content": [
        {"type": "text",
         "text": "1枚目（変更前）と2枚目（変更後）を見比べて、"
                 "内容として変わっている点を答えてください。"},
        {"type": "image", "media_type": "image/png",
         "data": base64.b64encode(img_a).decode()},
        {"type": "image", "media_type": "image/png",
         "data": base64.b64encode(img_b).decode()},
    ]}]
    try:
        from app.ocr.engines import backend_for
        be = backend_for(engine)
    except Exception as e:                        # noqa: BLE001
        res.error = f"backend生成失敗: {type(e).__name__}: {e}"
        return res

    reply = be.chat_with_retry(messages, schema=COMPARE_SCHEMA, max_tokens=2000,
                               timeout=timeout or CONFIG.claude_cli_timeout_sec,
                               retries=1)
    res.latency_sec = round(time.time() - t0, 2)
    if not reply.ok or not isinstance(reply.data, dict):
        res.error = reply.error or "構造化出力が得られなかった"
        return res
    d = reply.data
    res.changed = bool(d.get("changed"))
    res.summary = str(d.get("summary") or "")
    res.confident = bool(d.get("confident"))
    if use_cache:
        try:
            CONFIG.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(res), ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(p)
        except OSError:
            pass
    return res
