"""M0 受け入れテスト: LLMBackend の各実装が往復するか。

  python -m app.agent.selftest [--backends gemini,codex]

外部トレース送信とgroundingが無効であることも確認する。
"""
from __future__ import annotations

import argparse
import os
import sys

from app.agent.backend import BackendError, get_backend
from app.config import CONFIG

PING = [{"role": "user",
         "content": "「疎通確認」とだけ日本語で返してください。他は何も書かないこと。"}]

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "回答本文"},
        "confidence": {"type": "number", "description": "0-1の確信度"},
    },
    "required": ["answer", "confidence"],
}
SCHEMA_MSG = [{"role": "user",
               "content": "1+1 の答えを answer に、確信度を confidence に入れて返してください。"}]


def check_env() -> bool:
    ok = True
    print("== 環境チェック ==")
    for var in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
        val = os.environ.get(var, "")
        good = val.lower() in ("", "false", "0")
        print(f"  {'OK ' if good else 'NG '} {var}={val!r} (外部トレース送信は禁止)")
        ok &= good
    has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    print(f"  {'OK ' if has_key else 'NG '} GEMINI_API_KEY {'set' if has_key else 'MISSING'}")
    ok &= has_key
    print(f"  -- corpus: {CONFIG.corpus_dir} exists={CONFIG.corpus_dir.exists()}")
    print(f"  -- budget: {CONFIG.answer_budget_hours}h / parallel={CONFIG.parallel_workers}")
    return ok


def probe(name: str) -> bool:
    print(f"\n== backend: {name} ==")
    try:
        be = get_backend(name)
    except BackendError as e:
        print(f"  NG  生成失敗: {e}")
        return False

    r = be.chat_with_retry(PING, max_tokens=64)
    print(f"  text  : {'OK ' if r.ok else 'NG '} {r.latency_sec}s "
          f"{r.text.strip()[:60]!r}{'  err=' + str(r.error) if r.error else ''}")
    if not r.ok:
        return False

    r2 = be.chat_with_retry(SCHEMA_MSG, schema=SCHEMA, max_tokens=512)
    got = isinstance(r2.data, dict) and "answer" in r2.data
    print(f"  schema: {'OK ' if got else 'NG '} {r2.latency_sec}s data={r2.data}"
          f"{'  err=' + str(r2.error) if r2.error else ''}")
    if r.usage:
        print(f"  usage : {r.usage}")
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="gemini,codex")
    args = ap.parse_args()

    ok = check_env()
    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        ok &= probe(name)
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
