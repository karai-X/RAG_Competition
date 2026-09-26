"""複数候補の一致判定と裁定。

- 一致 → 採用（確信度は高い方）
- 片方だけ棄権 → 回答のある方を採用
- 不一致 → 裁定ノードが根拠と照合し、支持できなければ「わかりません」

**比較は必ず normalize してから**行う。`143,000円` と `143000` を
別物と判定しないよう、候補を比較前に正規化する。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.answer.normalize import same_answer, same_elements
from app.config import CONFIG

ADJ_PROMPT = """次の質問に対して、独立に作成された2つの候補回答があります。

# 質問
{question}

# 候補A
{ans_a}

# 候補B
{ans_b}

# 根拠（両方の調査で得られた抜粋）
{evidence}

根拠に照らして、どちらが正しいかを判定してください。
- 一方が正しければ、その回答を **そのままの形式で** answer に入れる
- 両方に誤りがあれば、根拠に基づく正しい回答を answer に入れる
- 根拠から確信が持てなければ answer を「わかりません」にする

回答形式の規範（値のみ・単位・列挙の読点区切り）は候補ではなく質問文の指示に従います。"""

ADJ_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "最終回答（値のみ、日本語）"},
        "chose": {"type": "string", "description": "A / B / 新規 / 不明 のいずれか"},
        "reason": {"type": "string", "description": "判定理由を1文で"},
    },
    "required": ["answer", "chose", "reason"],
}


@dataclass
class Adjudication:
    answer: str
    mode: str                 # agree / a_only / b_only / judged / both_missing
    detail: str = ""


def agree(a: str, b: str) -> bool:
    return same_answer(a, b) or same_elements(a, b)



def evidence_text(rec: dict, limit: int = 4000) -> str:
    """トレースから根拠の抜粋を組み立てる（構造化断片のみ。生データは載せない）。

    元は verify.py にあったが、verify パスが未使用のまま残っていたので
    こちらへ移した。裁定は「どちらの回答が根拠と整合するか」を見るので、
    根拠の抜粋そのものは引き続き必要。
    """
    parts: list[str] = []
    for ev in (rec.get("evidence") or [])[:10]:
        parts.append(f"- 根拠ファイル: {ev}")
    for t in (rec.get("trace") or []):
        if t.get("type") != "tool" or t.get("name") == "submit_answer":
            continue
        prev = (t.get("result_preview") or "").strip()
        if prev:
            parts.append(f"- [{t['name']}] {prev[:600]}")
    text = "\n".join(parts)
    return text[:limit] if text else "(根拠の抜粋なし)"

def adjudicate(question: str, rec_a: dict, rec_b: dict,
               backend_name: str | None = None) -> Adjudication:
    a = (rec_a.get("answer") or "").strip()
    b = (rec_b.get("answer") or "").strip()
    miss = CONFIG.missing_answer

    if a == miss and b == miss:
        return Adjudication(miss, "both_missing")
    if agree(a, b):
        return Adjudication(a if len(a) >= len(b) else b, "agree")
    if a == miss:
        return Adjudication(b, "b_only")
    if b == miss:
        return Adjudication(a, "a_only")

    from app.agent.backend import get_backend
    ev = (evidence_text(rec_a, 2500) + "\n" + evidence_text(rec_b, 2500))[:5000]
    be = get_backend(backend_name or CONFIG.backend)
    reply = be.chat_with_retry(
        [{"role": "user", "content": ADJ_PROMPT.format(
            question=question, ans_a=a, ans_b=b, evidence=ev)}],
        schema=ADJ_SCHEMA, max_tokens=1500, retries=2)
    if not reply.ok or not isinstance(reply.data, dict):
        # 根拠に基づいて裁定できなければ欠損回答にする。
        return Adjudication(miss, "judged", f"裁定失敗: {reply.error}")
    ans = str(reply.data.get("answer") or "").strip() or miss
    return Adjudication(ans, "judged",
                        f"chose={reply.data.get('chose')} "
                        f"reason={reply.data.get('reason')}")
