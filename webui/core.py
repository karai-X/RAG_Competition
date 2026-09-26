"""Web UI の中身（Streamlit に依存しない部分）。

- 前提チェック: 質問を実行できるか、何が足りないか
- LiveRunner:  1問をバックグラウンドで解く。画面を移動しても実行は続く
- 経路の読み出し: ChainLogger の JSONL（logs/<run>/chain/q*.jsonl）を画面用のステップへ
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.config import CONFIG, REPO_DIR

LIVE_PREFIX = "live-"
BUNDLED_RUN_JS = REPO_DIR / "portfolio" / "data" / "questions.js"
RESULT_PREVIEW_CHARS = 8000
MAX_QUESTION_CHARS = 1000


# ---------------------------------------------------------------- 前提チェック

def missing_prerequisites() -> list[dict]:
    """質問の実行に足りないものを名指しで返す。空なら実行できる。

    前処理（artifacts/）は配布前に済ませておく前提なので、ここでは有無だけを見る。
    """
    checks = [
        (CONFIG.index_dir / "chunks.jsonl", "検索索引（前処理済みの artifacts/）"),
        (CONFIG.catalog_dir / "catalog.jsonl", "抽出カタログ（前処理済みの artifacts/）"),
        (CONFIG.corpus_dir, "共有ドライブの資料（share/共有ドライブ/）"),
    ]
    missing = [{"what": label, "fix": f"{p.relative_to(REPO_DIR)} を配置してください"}
               for p, label in checks if not p.exists()]
    if CONFIG.backend == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        missing.append({"what": "Gemini API キー", "fix": "設定画面で入力してください"})
    missing += image_engine_problems()
    return missing


def image_engine_problems() -> list[dict]:
    """画像読み取りエンジンの設定で、実行前に分かる不足。"""
    engine = CONFIG.model_image
    if engine == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            return [{"what": "Gemini API キー（画像読み取り）",
                     "fix": "設定画面で入力してください"}]
        return []
    cli, auth, key = {
        "claude_cli": ("claude", CONFIG.claude_cli_auth, "ANTHROPIC_API_KEY"),
        "codex": ("codex", CONFIG.codex_auth, "CODEX_API_KEY"),
    }.get(engine, (None, None, None))
    if cli is None:
        return [{"what": f"画像読み取りエンジン「{engine}」", "fix": "設定画面で選び直してください"}]
    problems = []
    if not shutil.which(cli):
        problems.append({"what": f"{cli} CLI",
                         "fix": f"{cli} をインストールして PATH を通すか、"
                                "設定画面で画像読み取りエンジンを Gemini に切り替えてください"})
    if auth == "api" and not (os.environ.get(key)
                              or (cli == "codex" and os.environ.get("OPENAI_API_KEY"))):
        problems.append({"what": f"{cli} 用の API キー", "fix": "設定画面で入力してください"})
    return problems


# ---------------------------------------------------------------- ライブ実行

@dataclass
class LiveRun:
    run_id: str
    question: str
    chain_path: Path
    started: float = field(default_factory=time.time)
    phase: str = "準備中"
    finished: float | None = None
    result: dict | None = None
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.finished is not None


class LiveRunner:
    """1問ずつ実行する。ToolBox は SudachiPy と GPU エンコードを直列化しているので、
    同時に走らせても速くならない。

    ToolBox と systemプロンプトは1度だけ組んで使い回す（カタログ・索引・ツリーの読み込みに
    数十秒かかる）。LLM バックエンドは設定画面の変更を拾うため実行ごとに作る。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self._engine: dict | None = None
        self.current: LiveRun | None = None

    @property
    def busy(self) -> bool:
        return self.current is not None and not self.current.done

    def start(self, question: str) -> LiveRun:
        with self._lock:
            if self.busy:
                raise RuntimeError("別の質問を実行中です。終了を待ってください。")
            run_id = LIVE_PREFIX + datetime.now().strftime("%Y%m%d-%H%M%S")
            run = LiveRun(run_id=run_id, question=question,
                          chain_path=CONFIG.logs_dir / run_id / "chain" / "q0.jsonl")
            self.current = run
        threading.Thread(target=self._work, args=(run,), daemon=True).start()
        return run

    def _get_engine(self, run: LiveRun) -> dict:
        with self._engine_lock:
            if self._engine is None:
                run.phase = "カタログと検索索引を読み込み中（初回のみ）"
                from app.agent.prompts import build_system_prompt
                from app.agent.tools import ToolBox
                self._engine = {"toolbox": ToolBox(),
                                "system_prompt": build_system_prompt()}
            return self._engine

    def _work(self, run: LiveRun) -> None:
        """どこで落ちても finished を立てる。立てないと画面が「実行中」のまま止まる。"""
        try:
            from app.agent.backend import get_backend
            from app.agent.graph import run_question
            from app.run.chain import ChainLogger

            eng = self._get_engine(run)
            backend = get_backend(None)
            chain = ChainLogger.for_question(run.run_id, 0)
            chain.index = 0
            run.phase = "探索中"
            res = run_question(run.question, backend, eng["toolbox"],
                               eng["system_prompt"], chain=chain,
                               deadline=time.time() + CONFIG.question_timeout_sec)
            run.result = {
                "answer": res.answer or "", "confidence": res.confidence,
                "evidence": list(res.evidence or []), "stop": res.stop,
                "turns": res.turns, "sec": res.elapsed_sec, "model": res.model,
                "tok": (res.usage or {}).get("total_tokens") or 0,
            }
        except Exception as exc:                       # noqa: BLE001
            run.error = f"{type(exc).__name__}: {exc}"
        finally:
            run.finished = time.time()


# ---------------------------------------------------------------- 経路の読み出し

def read_jsonl(path: Path) -> list[dict]:
    events = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return events
    for line in text.splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue                       # 書き込み途中の行
    return events


def _peek(path: Path) -> tuple[dict | None, dict | None]:
    """先頭行（question）と末尾行（answer）だけを読む。経路本体は大きいので一覧では読まない。"""
    try:
        with open(path, "rb") as f:
            first = f.readline()
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().splitlines()
    except OSError:
        return None, None

    def load(raw: bytes) -> dict | None:
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    head = load(first) if first.strip() else None
    last = load(tail[-1]) if tail else None
    return head, (last if last and last.get("type") == "answer" else None)


def steps_from_events(events: list[dict]) -> list[dict]:
    """ChainLogger のイベント列 → 画面の経路ステップ。

    ステップの形は portfolio/build_data.py と同じ（think / tool / error）。
    tool_call と tool_result は発生順に対応づける。
    """
    steps: list[dict] = []
    pending: list[dict] = []
    for ev in events:
        kind = ev.get("type")
        if kind == "llm_response":
            if ev.get("error"):
                steps.append({"k": "error", "text": ev["error"]})
                continue
            calls = [c.get("name") for c in (ev.get("tool_calls") or [])]
            text = (ev.get("text") or "").strip()
            if calls or text:
                steps.append({"k": "think", "text": text, "calls": calls,
                              "turn": ev.get("turn"),
                              "sec": round(ev.get("latency_sec") or 0, 1)})
        elif kind == "tool_call":
            step = {"k": "tool", "name": ev.get("name"), "args": ev.get("args") or {},
                    "out": "", "sec": 0, "blocked": False, "pending": True}
            steps.append(step)
            pending.append(step)
        elif kind == "tool_result":
            step = next((s for s in pending if s["name"] == ev.get("name")), None)
            if step is None:
                continue
            pending.remove(step)
            out = (ev.get("content") if ev.get("result_kind") == "text"
                   else ev.get("note") or "（画像を受け取りました）")
            step.update(out=str(out or ""), sec=round(ev.get("elapsed_sec") or 0, 2),
                        pending=False, image=ev.get("result_kind") == "image",
                        # 提出ゲートに止められた場合だけ submit_answer に結果が返る
                        blocked=(step["name"] == "submit_answer"
                                 and str(out or "").startswith("[提出保留]")))
        elif kind == "answer":
            # 受理された submit_answer には tool_result が記録されない
            for step in pending:
                step["pending"] = False
            pending.clear()
    return steps


def record_from_chain(path: Path, run_id: str) -> dict:
    """1問ぶんの完全な記録（経路つき）。"""
    events = read_jsonl(path)
    head = next((e for e in events if e.get("type") == "question"), {})
    ans = next((e for e in reversed(events) if e.get("type") == "answer"), None)
    tok = sum((e.get("usage") or {}).get("total_tokens") or 0
              for e in events if e.get("type") == "llm_response")
    return {
        "run_id": run_id, "i": head.get("index", _index_of(path)),
        "q": head.get("question", ""), "model": head.get("model", ""),
        "t": events[0]["t"] if events else None,
        "sec": round(events[-1]["t"] - events[0]["t"], 1) if len(events) > 1 else 0,
        "tok": tok, "steps": steps_from_events(events), **_answer_fields(ans),
    }


def _answer_fields(ans: dict | None) -> dict:
    if ans is None:
        return {"a": None, "conf": None, "ev": [], "stop": "未完了", "turns": None}
    return {"a": ans.get("answer"), "conf": ans.get("confidence"),
            "ev": ans.get("evidence") or [], "stop": ans.get("stop") or "",
            "turns": ans.get("turns")}


def _index_of(path: Path) -> int:
    m = re.match(r"q(\d+)$", path.stem)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------- 履歴

@dataclass
class RunGroup:
    key: str
    label: str
    kind: str                      # live | batch | bundled
    runs: list[str] = field(default_factory=list)


def live_group() -> RunGroup:
    """Web UI から投げた質問（logs/live-*/chain/q0.jsonl）。"""
    logs = CONFIG.logs_dir
    runs = sorted((d.name for d in logs.glob(LIVE_PREFIX + "*")
                   if (d / "chain" / "q0.jsonl").exists()
                   and (d / "chain" / "q0.jsonl").stat().st_size > 0),
                  reverse=True) if logs.exists() else []
    return RunGroup("live", "Web UI からの質問", "live", runs)


def benchmark_group() -> RunGroup | None:
    """100問の一括実行。portfolio/data/questions.js と同じ実行に固定する。

    logs/ に完全な経路があればそれを、無ければ同梱の要約版を読む。
    """
    run_id = _bundled_run_id()
    if not run_id:
        return None
    if (CONFIG.logs_dir / run_id / "chain").is_dir():
        return RunGroup(run_id, run_id, "batch", [run_id])
    return RunGroup(run_id, run_id, "bundled")


def list_questions(group: RunGroup) -> list[dict]:
    """一覧用の要約（経路本体は読まない）。"""
    if group.kind == "bundled":
        return [{k: q.get(k) for k in ("i", "q", "a", "stop", "turns", "sec", "conf")}
                | {"run_id": _bundled_run_id(), "t": None}
                for q in _load_bundled()["questions"]]

    rows = []
    for run_id in group.runs:
        chain_dir = CONFIG.logs_dir / run_id / "chain"
        for p in sorted(chain_dir.glob("q*.jsonl"), key=_index_of):
            head, ans = _peek(p)
            if head is None:
                continue
            rows.append({"run_id": run_id, "i": head.get("index", _index_of(p)),
                         "q": head.get("question", ""), "t": head.get("t"),
                         **_answer_fields(ans)})
    return rows


def load_question(group: RunGroup, run_id: str, index: int) -> dict | None:
    if group.kind == "bundled":
        q = next((q for q in _load_bundled()["questions"] if q["i"] == index), None)
        return None if q is None else {**q, "run_id": run_id, "model": "", "t": None}
    path = CONFIG.logs_dir / run_id / "chain" / f"q{index}.jsonl"
    return record_from_chain(path, run_id) if path.exists() else None


_bundled_cache: dict | None = None


def _load_bundled() -> dict:
    """portfolio/data/questions.js（`window.RAG_RUN={...};`）。logs/ を配布しない場合の記録。"""
    global _bundled_cache
    if _bundled_cache is None:
        try:
            text = BUNDLED_RUN_JS.read_text(encoding="utf-8")
            _bundled_cache = json.loads(text[text.index("=") + 1:].rstrip().rstrip(";"))
        except (OSError, ValueError):
            _bundled_cache = {"questions": [], "run": {}}
    return _bundled_cache


def _bundled_run_id() -> str | None:
    return (_load_bundled().get("run") or {}).get("run_id")


# Streamlit は再実行のたびにスクリプトを読み直すが、import したモジュールは残る。
# 実行中の質問を画面の移動・再描画をまたいで保つため、ここに1つだけ置く。
RUNNER = LiveRunner()
