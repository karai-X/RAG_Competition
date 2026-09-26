"""LLMチェーン（探索経路）の完全記録。

`trace` は表示用に切り詰めてあるので、**後から挙動を検証する**にはこちらを使う。
1問1ファイルの JSONL に、発生順で以下を書く:

  question       質問と設定（モデル・上限・system プロンプトのハッシュと全文）
  llm_request    何メッセージ送ったか（新規分の全文つき）
  llm_response   応答テキスト・tool_calls・usage・レイテンシ
  tool_call      ツール名と **切り詰めない引数**
  tool_result    ツールの戻り（既定 40k 文字まで）
  answer         最終回答・確信度・根拠・停止理由

  python -m app.run.chain --run-id v1 --index 3          # 経路を読む
  python -m app.run.chain --run-id v1 --summary          # 全問の経路を1行ずつ
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import CONFIG

MAX_FIELD_CHARS = 40_000


def _default(o):
    """JSON化できない値の逃がし先（thought_signature 等の bytes を含む）。"""
    if isinstance(o, (bytes, bytearray)):
        return f"<bytes {len(o)}B>"
    return repr(o)[:200]


def _clip(v, limit: int = MAX_FIELD_CHARS):
    if isinstance(v, str) and len(v) > limit:
        return v[:limit] + f"…[切り詰め 全{len(v)}文字]"
    if isinstance(v, dict):
        return {k: _clip(x, limit) for k, x in v.items()}
    if isinstance(v, list):
        return [_clip(x, limit) for x in v]
    return v


def _strip_images(content):
    """画像の base64 は経路の可読性を壊すので要約に置き換える。"""
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "image":
            out.append({"type": "image",
                        "bytes": len(b.get("data") or "") * 3 // 4,
                        "media_type": b.get("media_type")})
        else:
            out.append(b)
    return out


@dataclass
class ChainLogger:
    """1問ぶんの経路を JSONL へ追記する。スレッドセーフ。"""
    path: Path
    seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _sent: int = 0                    # 送信済みメッセージ数（差分だけ記録する）

    @classmethod
    def for_question(cls, run_id: str, index: int) -> "ChainLogger":
        d = CONFIG.logs_dir / run_id / "chain"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"q{index}.jsonl"
        p.write_text("", encoding="utf-8")     # ラン再実行時は上書き
        return cls(path=p)

    def _write(self, event_type: str, **fields) -> None:
        with self._lock:
            self.seq += 1
            rec = {"seq": self.seq, "t": round(time.time(), 3),
                   "type": event_type}
            rec.update(_clip(fields))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False,
                                   default=_default) + "\n")

    # ---- イベント ----
    def question(self, index: int, question: str, model: str,
                 system_prompt: str, max_turns: int) -> None:
        # system プロンプトは全問共通なので **ラン単位で1回だけ** 保存し、
        # 各問はハッシュで参照する（1問あたり120KB → 数KB）
        sha = hashlib.sha1(system_prompt.encode("utf-8")).hexdigest()[:12]
        shared = self.path.parent / f"system_{sha}.txt"
        if not shared.exists():
            shared.write_text(system_prompt, encoding="utf-8")
        self._write("question", index=index, question=question, model=model,
                    max_turns=max_turns, system_prompt_sha1=sha,
                    system_prompt_chars=len(system_prompt),
                    system_prompt_file=shared.name)

    def llm_request(self, messages: list[dict], turn: int) -> None:
        new = messages[self._sent:]
        self._sent = len(messages)
        self._write("llm_request", turn=turn, n_messages=len(messages),
                    new_messages=[
                        {"role": m["role"],
                         "content": ("(system プロンプト — system_*.txt を参照)"
                                     if m["role"] == "system"
                                     else _strip_images(m["content"]))}
                        for m in new])

    def llm_response(self, turn: int, text: str, tool_calls: list,
                     usage: dict, latency_sec: float, model: str,
                     error: str | None = None) -> None:
        self._write("llm_response", turn=turn, text=text, model=model,
                    tool_calls=[{"id": t.id, "name": t.name, "args": t.args}
                                for t in tool_calls],
                    usage=usage, latency_sec=latency_sec, error=error)

    def tool_call(self, name: str, args: dict) -> None:
        self._write("tool_call", name=name, args=args)

    def tool_result(self, name: str, result, elapsed_sec: float) -> None:
        if isinstance(result, dict) and "blocks" in result:
            payload = {"result_kind": "image",
                       "note": next((b.get("text") for b in result["blocks"]
                                     if b.get("type") == "text"), "")}
        else:
            payload = {"result_kind": "text", "content": str(result)}
        self._write("tool_result", name=name, elapsed_sec=elapsed_sec, **payload)

    def answer(self, answer, confidence: float, evidence: list,
               stop: str, turns: int) -> None:
        self._write("answer", answer=answer, confidence=confidence,
                    evidence=evidence, stop=stop, turns=turns)


# ------------------------------------------------------------------ 読み出し
def load_chain(run_id: str, index: int) -> list[dict]:
    p = CONFIG.logs_dir / run_id / "chain" / f"q{index}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def render(events: list[dict], width: int = 160,
           show_system: bool = False, chain_dir: Path | None = None) -> str:
    """経路を人が読める形に整形する。"""
    lines: list[str] = []
    if show_system and chain_dir:
        f = next((e.get("system_prompt_file") for e in events
                  if e["type"] == "question"), None)
        if f and (chain_dir / f).exists():
            lines.append((chain_dir / f).read_text(encoding="utf-8"))
            lines.append("=" * 60)
    for e in events:
        k = e["type"]
        if k == "question":
            lines.append(f"■ q{e['index']}  model={e['model']} "
                         f"max_turns={e['max_turns']}")
            lines.append(f"  質問: {e['question']}")
            lines.append(f"  system: {e['system_prompt_chars']}字 "
                         f"sha1={e['system_prompt_sha1']} "
                         f"({e.get('system_prompt_file', '')})")
        elif k == "llm_request":
            for m in e["new_messages"]:
                body = m["content"]
                if isinstance(body, list):
                    body = " / ".join(
                        (b.get("text") or b.get("content") or b.get("name") or
                         f"<{b.get('type')}>")[:width] if isinstance(b, dict)
                        else str(b)[:width] for b in body)
                lines.append(f"  → [{m['role']}] {str(body)[:width]}")
        elif k == "llm_response":
            calls = ", ".join(f"{c['name']}(" +
                              ", ".join(f"{k2}={str(v)[:60]!r}"
                                        for k2, v in c["args"].items()) + ")"
                              for c in e["tool_calls"])
            head = (f"  ← turn{e['turn']} {e['latency_sec']}s "
                    f"[{e.get('model', '')}]")
            if e.get("error"):
                lines.append(head + f"  ERROR {e['error'][:width]}")
            else:
                if e["text"].strip():
                    lines.append(head + f"  text: {e['text'][:width]}")
                if calls:
                    lines.append(head + f"  call: {calls[:width * 2]}")
        elif k == "tool_result":
            body = e.get("content") or e.get("note") or ""
            lines.append(f"     ⤷ {e['name']} {e['elapsed_sec']}s: "
                         f"{str(body)[:width]}")
        elif k == "answer":
            lines.append(f"■ 回答: {e['answer']}  "
                         f"(conf={e['confidence']} stop={e['stop']} "
                         f"turns={e['turns']})")
            lines.append(f"  根拠: {e['evidence']}")
    return "\n".join(lines)


def summarize(run_id: str) -> list[dict]:
    """全問の経路を1行に要約する（どのツールを何回使ったか）。"""
    d = CONFIG.logs_dir / run_id / "chain"
    out = []
    if not d.exists():
        return out
    for p in sorted(d.glob("q*.jsonl"), key=lambda x: int(x.stem[1:])):
        ev = load_chain(run_id, int(p.stem[1:]))
        calls: list[str] = []
        turns = 0
        secs = 0.0
        ans = None
        for e in ev:
            if e["type"] == "llm_response":
                turns = max(turns, e.get("turn", 0))
                secs += e.get("latency_sec", 0) or 0
            elif e["type"] == "tool_call":
                calls.append(e["name"])
            elif e["type"] == "answer":
                ans = e
        out.append({"index": int(p.stem[1:]),
                    "path": " → ".join(calls) or "(ツール未使用)",
                    "turns": turns, "llm_sec": round(secs, 1),
                    "answer": (ans or {}).get("answer"),
                    "stop": (ans or {}).get("stop")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--show-system", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.summary or args.index is None:
        rows = summarize(args.run_id)
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=1))
            return 0
        for r in rows:
            print(f"q{r['index']:<3} turns={r['turns']:<3} "
                  f"{r['llm_sec']:>6.1f}s  {r['stop']:<12} {r['path']}")
            print(f"      → {str(r['answer'])[:110]}")
        return 0

    ev = load_chain(args.run_id, args.index)
    if not ev:
        print(f"経路の記録がありません: {args.run_id} q{args.index}")
        return 1
    if args.json:
        print(json.dumps(ev, ensure_ascii=False, indent=1))
        return 0
    print(render(ev, show_system=args.show_system,
                 chain_dir=CONFIG.logs_dir / args.run_id / "chain"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
