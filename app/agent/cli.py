"""単発質問CLI（デバッグ用）。

  python -m app.agent.cli --question "..." [--trace] [--backend gemini]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from app.agent.backend import get_backend
from app.agent.graph import run_question
from app.agent.prompts import build_system_prompt
from app.agent.tools import ToolBox
from app.config import CONFIG


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--chain-run", default=None,
                    help="経路を logs/<run>/chain へ記録する")
    args = ap.parse_args()

    be = get_backend(args.backend, **({"model": args.model} if args.model else {}))
    chain = None
    if args.chain_run:
        from app.run.chain import ChainLogger
        chain = ChainLogger.for_question(args.chain_run, 0)
        chain.index = 0
    res = run_question(args.question, be, ToolBox(), build_system_prompt(),
                       chain=chain,
                       deadline=time.time() + (args.timeout
                                               or CONFIG.question_timeout_sec),
                       max_turns=args.max_turns)
    if args.json:
        print(json.dumps(res.__dict__, ensure_ascii=False, indent=1,
                         default=str))
        return 0

    print(f"\n=== 回答 ===\n{res.answer}")
    print(f"\nstop={res.stop} conf={res.confidence} turns={res.turns} "
          f"{res.elapsed_sec}s model={res.model}")
    print(f"usage={res.usage}")
    print(f"evidence={res.evidence}")
    if args.trace:
        print("\n=== トレース ===")
        for i, t in enumerate(res.trace, 1):
            if t["type"] == "assistant":
                calls = ", ".join(f"{c['name']}({_kv(c['args'])})"
                                  for c in t.get("tool_calls", []))
                print(f"  [{i}] assistant {t.get('latency_sec')}s -> {calls}")
                if t.get("text"):
                    print(f"       {t['text'][:200]}")
            elif t["type"] == "tool":
                print(f"  [{i}] tool {t['name']} {t.get('elapsed_sec', '')}s")
                print(f"       args={_kv(t.get('args', {}))}")
                if t.get("result_preview"):
                    print(f"       -> {t['result_preview'][:250]}")
            else:
                print(f"  [{i}] {t}")
    return 0


def _kv(d: dict, limit: int = 120) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in (d or {}).items())
    return s[:limit] + ("…" if len(s) > limit else "")


if __name__ == "__main__":
    sys.exit(main())
