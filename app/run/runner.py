"""バッチ実行: 質問CSV → 各問エージェント実行 → predictions.csv。

  python -m app.run.runner --questions <csv> --run-id <id> [--parallel 8]

- 1問1トレースJSONへ永続化（既存はスキップ = **再開可能**）
- 全体デッドライン（既定12h）と1問デッドライン（既定600s）の二段管理
- 進捗を progress.json に書き、CLI再実行時にも状態を確認できるようにする
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from app.agent.backend import get_backend
from app.agent.graph import run_question
from app.agent.prompts import build_system_prompt
from app.agent.tools import ToolBox
from app.answer.finalize import finalize, sanitize
from app.run.chain import ChainLogger
from app.run.csv_io import load_question_rows
from app.config import CONFIG

# 答えに到達せず終わった問をやり直す回数。棄権の判断は覆さない
# （is_unfinished を参照）。0 で無効。
RETRY_UNFINISHED = 1


def load_questions(path: Path) -> list[dict]:
    """A=index、B=question、C=answerを既定として質問CSVを読む。"""
    return load_question_rows(path)


class RunLock:
    """同一 run_id の二重起動を防ぐ。

    複数プロセスが同じ結果ファイルを同時更新しないようロックする。
    12時間ランでは致命的なので明示的に排他する。
    """

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "run.lock"

    def acquire(self, force: bool = False) -> None:
        import os
        if self.path.exists() and not force:
            try:
                pid = int(self.path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = -1
            if pid > 0 and pid != os.getpid():
                alive = True
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    alive = False             # 死んでいるので奪ってよい
                except PermissionError:
                    alive = True              # 存在するがシグナルできないだけ
                if alive:
                    raise RuntimeError(
                        f"同じ run_id が pid={pid} で実行中です。"
                        f"終了を待つか --force を指定してください ({self.path})")
        self.path.write_text(str(os.getpid()), encoding="utf-8")

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


def _write_progress(run_dir: Path, state: dict) -> None:
    tmp = run_dir / ".progress.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(run_dir / "progress.json")


# 答えを出さずに終わった状態。**モデルの判断ではなく中断**なので、
# やり直しても棄権の判断を覆さない。
_UNFINISHED = ("max_turns", "deadline", "global_deadline")


def is_unfinished(stop: str | None) -> bool:
    """答えに到達せず終わったか。やり直してよいのはこの場合だけ。

    submit_answer で「わかりません」と答えた場合は stop='submitted' なので
    ここには入らない。根拠不足という判断を覆さないため。
    """
    s = str(stop or "")
    return s in _UNFINISHED or s.startswith("error")


def run_batch(questions_csv: Path, run_id: str, parallel: int | None = None,
              indices: set[int] | None = None, limit: int | None = None,
              backend_name: str | None = None, model: str | None = None,
              max_turns: int | None = None, budget_hours: float | None = None,
              progress=None, force: bool = False) -> dict:
    CONFIG.ensure_dirs()
    run_dir = CONFIG.logs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_lock = RunLock(run_dir)      # 進捗用の threading.Lock と別物
    run_lock.acquire(force=force)

    all_q = load_questions(questions_csv)
    todo = [q for q in all_q
            if (indices is None or q["index"] in indices)]
    if limit:
        todo = todo[:limit]

    backend = get_backend(backend_name, **({"model": model} if model else {}))
    toolbox = ToolBox()
    system_prompt = build_system_prompt()

    t0 = time.time()
    budget = (budget_hours or CONFIG.answer_budget_hours) * 3600
    global_deadline = t0 + budget
    lock = threading.Lock()
    observed_models: set[str] = set()
    state = {"run_id": run_id, "total": len(todo), "done": 0, "cached": 0,
             "submitted": 0, "abstained": 0, "errors": 0,
             "started_at": t0, "budget_sec": budget, "elapsed_sec": 0.0,
             "current": []}

    def one(q: dict) -> dict:
        qfile = run_dir / f"q{q['index']}.json"
        if qfile.exists():
            try:
                rec = json.loads(qfile.read_text(encoding="utf-8"))
                if not str(rec.get("stop", "")).startswith("error"):
                    with lock:
                        state["cached"] += 1
                    return rec
            except (OSError, json.JSONDecodeError):
                pass

        with lock:
            state["current"].append(q["index"])
        started = time.time()
        remaining = global_deadline - started
        if remaining <= 5:
            rec = {"index": q["index"], "question": q["question"],
                   "answer": CONFIG.missing_answer, "confidence": 0.0,
                   "stop": "global_deadline", "turns": 0, "evidence": [],
                   "trace": [], "usage": {}, "elapsed_sec": 0.0}
        else:
            attempts: list[str] = []
            for attempt in range(1 + RETRY_UNFINISHED):
                left = global_deadline - time.time()
                if attempt and left <= CONFIG.question_timeout_sec * 0.5:
                    break                      # やり直す時間が無い
                deadline = time.time() + min(CONFIG.question_timeout_sec, left)
                chain = ChainLogger.for_question(run_id, q["index"])
                chain.index = q["index"]
                res = run_question(
                    q["question"], backend, toolbox, system_prompt,
                    deadline=deadline, max_turns=max_turns,
                    # 試行ごとに別の thread_id。同じだと checkpointer を
                    # 使う構成で、行き詰まった状態から再開してしまう。
                    thread_id=f"{run_id}:{q['index']}:{attempt}", chain=chain)
                attempts.append(str(res.stop))
                if not is_unfinished(res.stop):
                    break
            rec = {"index": q["index"], "question": q["question"],
                   "gt": q.get("gt"),
                   "answer_raw": res.answer,
                   "answer": finalize(res.answer, res.confidence, res.stop),
                   "confidence": res.confidence, "evidence": res.evidence,
                   "stop": res.stop, "turns": res.turns,
                   # 実際に応答したモデル（設定値の書き写しではない）
                   "model": res.model, "models_used": res.models_used,
                   "usage": res.usage,
                   "elapsed_sec": round(time.time() - started, 1),
                   "trace": res.trace}
            if len(attempts) > 1:
                # 何回目で答えに到達したかを残す（効果を後から測るため）
                rec["attempts"] = attempts
        qfile.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        with lock:
            observed_models.update(rec.get("models_used") or [])
            state["current"] = [i for i in state["current"] if i != q["index"]]
            state["done"] += 1
            if rec["answer"] == CONFIG.missing_answer:
                state["abstained"] += 1
            else:
                state["submitted"] += 1
            if str(rec.get("stop", "")).startswith("error"):
                state["errors"] += 1
            state["elapsed_sec"] = round(time.time() - t0, 1)
            _write_progress(run_dir, state)
            if progress:
                progress(state, rec)
        return rec

    workers = parallel or CONFIG.parallel_workers
    try:
        if workers > 1 and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(one, todo))
        else:
            for q in todo:
                one(q)
    finally:
        run_lock.release()

    write_predictions(run_dir, all_q)
    manifest = {
        "run_id": run_id, "questions_csv": str(questions_csv),
        "n_questions": len(todo), "backend": backend.name,
        "model": model or getattr(backend, "model", ""),
        "temperature": CONFIG.generation_temperature,
        "seed": CONFIG.generation_seed,
        # 混雑時は兄弟モデルへ切り替わる。何が実際に答えたかを残す。
        "model_fallback_enabled": True,
        "retry_unfinished": RETRY_UNFINISHED,
        "models_observed": sorted(observed_models),
        "embed_model": CONFIG.embed_model, "rerank_model": CONFIG.rerank_model,
        "parallel": workers, "max_turns": max_turns or CONFIG.max_turns,
        "budget_hours": budget_hours or CONFIG.answer_budget_hours,
        "elapsed_sec": round(time.time() - t0, 1),
        "stats": {k: v for k, v in state.items() if k != "current"},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    state["elapsed_sec"] = round(time.time() - t0, 1)
    state["finished"] = True
    _write_progress(run_dir, state)
    return manifest


def write_predictions(run_dir: Path, all_q: list[dict],
                      out: Path | None = None) -> Path:
    """**全index網羅・欠けは「わかりません」で埋める**（空欄は形式違反）。"""
    out = out or (run_dir / "predictions.csv")
    import csv

    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for q in all_q:
            qfile = run_dir / f"q{q['index']}.json"
            ans = CONFIG.missing_answer
            if qfile.exists():
                try:
                    rec = json.loads(qfile.read_text(encoding="utf-8"))
                    ans = sanitize(rec.get("answer") or "") or CONFIG.missing_answer
                except (OSError, json.JSONDecodeError):
                    pass
            w.writerow([q["index"], ans])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(CONFIG.questions_csv))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--indices", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--force", action="store_true",
                    help="実行中ロックを無視して起動する")
    ap.add_argument("--no-retry", action="store_true",
                    help="答えに到達せず終わった問のやり直しをしない")
    args = ap.parse_args()

    idx = ({int(x) for x in args.indices.split(",")} if args.indices else None)

    def progress(state, rec):
        gt = f"  (正解: {str(rec.get('gt'))[:40]})" if rec.get("gt") else ""
        print(f"[{state['done']}/{state['total']}] q{rec['index']} "
              f"{rec.get('stop')} conf={rec.get('confidence')} "
              f"{rec.get('elapsed_sec')}s\n    -> {rec['answer'][:90]}{gt}",
              flush=True)

    if args.no_retry:
        globals()["RETRY_UNFINISHED"] = 0
    m = run_batch(Path(args.questions), args.run_id, parallel=args.parallel,
                  indices=idx, limit=args.limit, backend_name=args.backend,
                  model=args.model, max_turns=args.max_turns,
                  budget_hours=args.budget_hours, progress=progress,
                  force=args.force)
    print(f"\n{json.dumps(m['stats'], ensure_ascii=False)}")
    print(f"predictions: {CONFIG.logs_dir / args.run_id / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
