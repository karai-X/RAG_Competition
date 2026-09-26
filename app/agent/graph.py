"""LangGraph の agent ↔ tools ループ。

- 状態遷移・ループ・条件分岐は LangGraph に任せる
- checkpointer（SQLite, thread_id="<run_id>:<index>"）で中断復帰
- 予算（1問デッドライン・最大ターン）は自分で持ち、超過時は abstain
- バックエンドは LLMBackend 抽象越しに扱う
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import re

from app.agent.backend import LLMBackend, Reply, ToolCall
from app.agent.tools import TOOL_DEFS, ToolBox
from app.config import CONFIG

TERMINAL_TOOL = "submit_answer"

# ツールを呼ばずにテキストだけ返したときは、一度だけ提出を促す。
NUDGE = ("回答が確定しているなら submit_answer ツールで提出してください。"
         "まだ調査が必要ならツールを使って続けてください。"
         "根拠が得られず確信が持てない場合は answer を「わかりません」として "
         "submit_answer を呼んでください。")
TEXT_ANSWER_CONFIDENCE = 0.5

# テキスト救済では、思考途中の文を除外して短い回答だけを拾う。
_THOUGHT_MARKER = re.compile(
    # タグ記号が切り落とされた断片にも一致させる。
    r"(?:<\s*/?\s*)?thought\b|いや待|待って|かもしれ|だろうか|してみよう|"
    r"試してみ|もう一度|確認しよう|考えてみ|整理する|次に調べ|えっと|うーん",
    re.IGNORECASE)
# 長すぎる自由文は回答として救済しない。
MAX_RESCUE_CHARS = 400

# 履歴が上限を超えたら、古いツール結果を要約に置き換える。
HISTORY_TOKEN_BUDGET = 250_000
KEEP_RECENT_MESSAGES = 8


class AgentState(TypedDict, total=False):
    messages: list[dict]
    # 未宣言のキーは LangGraph の状態に載らない。載らないと tools_node が
    # 呼び出しを受け取れず、submitted を立てられないまま agent へ戻り続ける。
    pending: list[tuple]
    nudged: bool
    image_nudged: bool
    # checkpointer から再開しても、ToolBox の thread-local 初期化で
    # OCR元画像の未確認状態を失わないよう状態にも複製する。
    image_verification: str
    final_text: str
    # 上限切れの検知に使う。**宣言しないと状態に載らない**（上のコメント参照）
    final_stop_reason: str
    turns: int
    answer: str | None
    confidence: float
    evidence: list[str]
    stop: str
    trace: list[dict]
    usage: dict
    deadline: float
    # 実際に応答を返したモデル（重複なし・出現順）。
    # **宣言しないと状態に載らない**（上のコメント参照）。
    models: list[str]


@dataclass
class AgentResult:
    answer: str | None = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    stop: str = "unknown"
    turns: int = 0
    model: str = ""
    # API 応答が名乗ったモデルの一覧。2つ以上なら切り替えが起きている。
    models_used: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def submitted(self) -> bool:
        return self.stop == "submitted"


def _merge_usage(acc: dict, add: dict) -> dict:
    for k, v in (add or {}).items():
        if isinstance(v, (int, float)):
            acc[k] = acc.get(k, 0) + v
    return acc


def _tool_result_blocks(tc: ToolCall, result: Any) -> list[dict]:
    """ツール結果を content-block へ。画像はそのまま画像ブロックで返す。"""
    if isinstance(result, dict) and "blocks" in result:
        blocks = [{"type": "tool_result", "tool_call_id": tc.id, "name": tc.name,
                   "content": "（画像を添付します）"}]
        return blocks + list(result["blocks"])
    return [{"type": "tool_result", "tool_call_id": tc.id, "name": tc.name,
             "content": str(result)}]


def build_graph(backend: LLMBackend, toolbox: ToolBox,
                max_turns: int | None = None, chain=None):
    """agent ↔ tools の StateGraph を組む。

    chain に ChainLogger を渡すと、**探索経路を無切り詰めで記録**する
    （trace は表示用に切り詰めてあるので、事後検証にはこちらを使う）。
    """
    from langgraph.graph import END, START, StateGraph

    max_turns = max_turns or CONFIG.max_turns

    def live_image_verification_requirement() -> str:
        """ToolBox が対応していれば、OCR元画像の未確認状態を問い合わせる。"""
        check = getattr(toolbox, "image_verification_requirement", None)
        if not callable(check):
            return ""
        try:
            return str(check() or "")
        except Exception:  # noqa: BLE001 - ゲート照会失敗で既存ツールを壊さない
            return ""

    def image_verification_requirement(state: AgentState) -> str:
        """ライブ状態を優先し、再開時はチェックポイントの状態へ退避する。"""
        return (live_image_verification_requirement()
                or str(state.get("image_verification") or ""))

    def image_tool_succeeded(name: str, result: Any) -> bool:
        if isinstance(result, dict):
            return any(isinstance(b, dict) and b.get("type") == "image"
                       for b in result.get("blocks", []))
        text = str(result or "")
        return (name == "ask_image" and bool(text.strip())
                and "[エラー]" not in text
                and "読み取れませんでした" not in text)

    def agent_node(state: AgentState) -> dict:
        if time.time() > state.get("deadline", float("inf")):
            return {"stop": "deadline"}
        if state["turns"] >= max_turns:
            return {"stop": "max_turns"}

        turn = state["turns"] + 1
        messages = prune_history(state["messages"])
        if chain:
            chain.llm_request(messages, turn)
        reply: Reply = backend.chat_with_retry(
            messages, tools=TOOL_DEFS,
            max_tokens=CONFIG.max_tokens_per_turn)
        if chain:
            chain.llm_response(turn, reply.text, reply.tool_calls, reply.usage,
                               reply.latency_sec, reply.model, reply.error)

        trace = list(state.get("trace", []))
        usage = _merge_usage(dict(state.get("usage", {})), reply.usage)
        models = list(state.get("models", []))
        if reply.model and reply.model not in models:
            models.append(reply.model)

        if not reply.ok:
            trace.append({"type": "error", "error": reply.error})
            return {"stop": f"error:{(reply.error or '')[:120]}",
                    "trace": trace, "usage": usage, "models": models}

        content: list[dict] = []
        if reply.text:
            content.append({"type": "text", "text": reply.text})
        for tc in reply.tool_calls:
            content.append({"type": "tool_use", "id": tc.id, "name": tc.name,
                            "input": tc.args, "signature": tc.signature})
        msgs = messages + [{"role": "assistant", "content": content}] \
            if content else messages

        trace.append({"type": "assistant", "text": reply.text[:800],
                      "tool_calls": [{"name": t.name, "args": _short(t.args)}
                                     for t in reply.tool_calls],
                      "latency_sec": reply.latency_sec})

        if not reply.tool_calls:
            image_requirement = image_verification_requirement(state)
            if image_requirement:
                # submit_answer を使わない短文救済でも画像確認を迂回させない。
                if state.get("image_nudged"):
                    return {"messages": msgs, "turns": state["turns"] + 1,
                            "stop": "image_verification_required",
                            "trace": trace, "usage": usage, "models": models,
                            "final_text": reply.text or "",
                            "final_stop_reason": reply.stop_reason or ""}
                return {"messages": msgs + [{"role": "user",
                                              "content": image_requirement}],
                        "turns": state["turns"] + 1, "image_nudged": True,
                        "trace": trace, "usage": usage, "models": models,
                        "final_text": reply.text or "",
                        "final_stop_reason": reply.stop_reason or ""}
            if not state.get("nudged"):
                # 1度だけ促す（構造化された confidence / evidence が欲しい）
                return {"messages": msgs + [{"role": "user", "content": NUDGE}],
                        "turns": state["turns"] + 1, "nudged": True,
                        "trace": trace, "usage": usage, "models": models,
                        "final_text": reply.text,
                        "final_stop_reason": reply.stop_reason or ""}
            # 促しても呼ばないなら、最後のテキストを回答として拾う
            return {"messages": msgs, "turns": state["turns"] + 1,
                    "stop": "text_answer", "trace": trace, "usage": usage,
                    "models": models,
                    "final_text": reply.text or state.get("final_text", ""),
                    # 上限で切られた出力は内容によらず不完全。救済の対象から外す
                    "final_stop_reason": reply.stop_reason or ""}

        return {"messages": msgs, "turns": state["turns"] + 1,
                "trace": trace, "usage": usage, "models": models,
                "pending": [(t.id, t.name, t.args) for t in reply.tool_calls]}

    def tools_node(state: AgentState) -> dict:
        pending = state.get("pending") or []
        blocks: list[dict] = []
        trace = list(state.get("trace", []))
        out: dict = {}
        image_requirement = str(state.get("image_verification") or "")

        for tid, name, args in pending:
            if chain:
                chain.tool_call(name, dict(args))
            if name == TERMINAL_TOOL:
                required = (live_image_verification_requirement()
                            or image_requirement)
                if required:
                    # tool_use に対応する tool_result を返し、画像確認後に再提出
                    # させる。stop を立てないので agent ループは継続する。
                    result = required
                    if chain:
                        chain.tool_result(name, result, 0.0)
                    tc = ToolCall(id=tid, name=name, args=args)
                    blocks.extend(_tool_result_blocks(tc, result))
                    trace.append({"type": "tool", "name": name,
                                  "args": _short(args),
                                  "result_preview": result[:400],
                                  "blocked": True})
                    continue
                out.update(answer=str(args.get("answer", "")),
                           confidence=float(args.get("confidence", 0) or 0),
                           evidence=[str(x) for x in (args.get("evidence") or [])],
                           stop="submitted")
                trace.append({"type": "tool", "name": name, "args": _short(args)})
                continue
            t0 = time.time()
            result = toolbox.call(name, dict(args))
            if chain:
                chain.tool_result(name, result, round(time.time() - t0, 2))
            tc = ToolCall(id=tid, name=name, args=args)
            blocks.extend(_tool_result_blocks(tc, result))
            preview = (result.get("blocks", [{}])[-1].get("text", "画像")
                       if isinstance(result, dict) else str(result))
            trace.append({"type": "tool", "name": name, "args": _short(args),
                          "result_preview": preview[:400],
                          "elapsed_sec": round(time.time() - t0, 2)})

            live_required = live_image_verification_requirement()
            if live_required:
                image_requirement = live_required
            elif (name in {"read_image", "ask_image"}
                  and image_tool_succeeded(name, result)):
                # 成功時だけ ToolBox が確認待ちを解除する。失敗時や別資料の画像
                # ならライブ側に要件が残るため、チェックポイント側も維持される。
                image_requirement = ""
                out["image_nudged"] = False

        out["trace"] = trace
        out["pending"] = []
        out["image_verification"] = image_requirement
        if blocks:
            out["messages"] = state["messages"] + [{"role": "user",
                                                    "content": blocks}]
        return out

    def route_after_agent(state: AgentState) -> str:
        if state.get("stop"):
            return END
        return "tools" if state.get("pending") else "agent"

    def route_after_tools(state: AgentState) -> str:
        return END if state.get("stop") else "agent"

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent,
                            {"tools": "tools", "agent": "agent", END: END})
    g.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})
    return g


def _estimate_tokens(messages: list[dict]) -> int:
    """おおまかなトークン見積り（画像は base64 長から換算）。"""
    total = 0
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            total += len(c) // 2
            continue
        for b in c or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "image":
                total += len(b.get("data") or "") // 3
            else:
                total += len(str(b.get("text") or b.get("content") or "")) // 2
    return total


def prune_history(messages: list[dict],
                  budget: int = HISTORY_TOKEN_BUDGET,
                  keep_recent: int = KEEP_RECENT_MESSAGES) -> list[dict]:
    """system と最初の質問、直近 keep_recent 件は残し、古いツール結果を要約に置く。

    LangGraph に任せる範囲ではないので自分で持つ（画像は特に重い）。
    """
    if _estimate_tokens(messages) <= budget or len(messages) <= keep_recent + 2:
        return messages
    head, tail = messages[:2], messages[-keep_recent:]
    middle = messages[2:len(messages) - keep_recent]
    pruned: list[dict] = []
    for m in middle:
        c = m["content"]
        if isinstance(c, str):
            pruned.append(m)
            continue
        blocks = []
        for b in c or []:
            if not isinstance(b, dict):
                blocks.append(b)
            elif b.get("type") == "image":
                blocks.append({"type": "text",
                               "text": "[古い画像は履歴から省略。必要なら "
                                       "read_image で取り直すこと]"})
            elif b.get("type") == "tool_result":
                body = str(b.get("content") or "")
                blocks.append({**b, "content": body[:600] +
                               (f"…[古いツール結果を省略 全{len(body)}文字]"
                                if len(body) > 600 else "")})
            else:
                blocks.append(b)
        pruned.append({**m, "content": blocks})
    return head + pruned + tail


def _short(args: dict, limit: int = 300) -> dict:
    out = {}
    for k, v in (args or {}).items():
        s = v if isinstance(v, (int, float, bool)) else str(v)
        if isinstance(s, str) and len(s) > limit:
            s = s[:limit] + "…"
        out[k] = s
    return out


def run_question(question: str, backend: LLMBackend, toolbox: ToolBox,
                 system_prompt: str, deadline: float | None = None,
                 max_turns: int | None = None,
                 thread_id: str | None = None,
                 checkpointer=None, chain=None) -> AgentResult:
    """1問を解く。checkpointer を渡すと中断復帰、chain を渡すと経路を記録する。"""
    t0 = time.time()
    # read_image が「元の質問」を画像読み取りモデルへ添えられるようにする
    toolbox.question = question
    if chain:
        chain.question(getattr(chain, "index", -1), question,
                       getattr(backend, "model", backend.name),
                       system_prompt, max_turns or CONFIG.max_turns)
    graph = build_graph(backend, toolbox, max_turns, chain=chain).compile(
        checkpointer=checkpointer)

    init: AgentState = {
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": question}],
        "turns": 0, "answer": None, "confidence": 0.0, "evidence": [],
        "stop": "", "trace": [], "usage": {}, "models": [],
        "image_verification": "",
        "deadline": deadline or (t0 + CONFIG.question_timeout_sec),
    }
    config = {"recursion_limit": (max_turns or CONFIG.max_turns) * 2 + 10}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    final = graph.invoke(init, config=config)
    stop = final.get("stop") or "incomplete"
    answer, conf = final.get("answer"), final.get("confidence", 0.0)
    if stop == "text_answer" and not answer:
        # submit_answer を呼ばなかった場合の救済。confidence は控えめに置き、
        # verify パスの発火条件（<0.85）に自動的に載るようにする。
        text = (final.get("final_text") or "").strip()
        truncated = "MAX_TOKENS" in str(final.get("final_stop_reason") or "")
        if not text:
            answer, conf, stop = None, 0.0, "no_tool_call"
        elif truncated:
            # 出力がトークン上限で切られている。文の途中で終わっているので、
            # 内容にかかわらず、不完全な出力は回答として採用しない。
            answer, conf, stop = None, 0.0, "truncated_text"
        elif _THOUGHT_MARKER.search(text) or len(text) > MAX_RESCUE_CHARS:
            # 思考途中のテキストは回答として採用しない。
            answer, conf, stop = None, 0.0, "thinking_text"
        else:
            answer, conf, stop = text, TEXT_ANSWER_CONFIDENCE, "submitted"
    if chain:
        chain.answer(answer, conf, final.get("evidence", []), stop,
                     final.get("turns", 0))
    models = list(final.get("models", []))
    return AgentResult(
        answer=answer, confidence=conf,
        evidence=final.get("evidence", []), stop=stop,
        turns=final.get("turns", 0),
        # 設定値ではなく、実際に応答したモデルを残す
        model=models[0] if models else getattr(backend, "model", backend.name),
        models_used=models,
        usage=final.get("usage", {}), trace=final.get("trace", []),
        elapsed_sec=round(time.time() - t0, 1))


def make_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3
    CONFIG.rag_home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CONFIG.checkpoints_db), check_same_thread=False)
    return SqliteSaver(conn)
