"""Agentic RAG のCLI操作。

使用例:
  python rag_cli.py preprocess
  python rag_cli.py preprocess --skip-processed
  python rag_cli.py answer-csv share/質問回答/questions_test.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from app.answer.finalize import sanitize
from app.config import CONFIG
from app.run.csv_io import load_question_rows, write_answers_to_c_column
from submission.pipeline import run_pipeline
from submission.state import create_job, update_job
from submission.storage import preprocessing_status, safe_component


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _preprocess(args: argparse.Namespace) -> int:
    status = preprocessing_status()
    if args.skip_processed and not status["has_pending"]:
        print("未処理のファイル／フォルダはありません。前処理を省略しました。")
        return 0

    job_id = f"cli-ingest-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    create_job(job_id, "ingest", folder_name="CLI一括前処理",
               source="share/共有ドライブ")
    request = {
        "job_id": job_id,
        # 既定は全件。--skip-processed のときだけ増分処理にする。
        "full": not args.skip_processed,
        "do_embed": not args.no_embed,
        "ocr_engine": CONFIG.model_ocr,
        "extract_workers": args.extract_workers,
        "ocr_workers": args.ocr_workers,
    }
    mode = "未処理のみ" if args.skip_processed else "全件"
    print(f"共有ドライブの前処理を開始します（{mode}）。job_id={job_id}")
    try:
        result = run_pipeline(request)
    except Exception as exc:  # noqa: BLE001 - CLI境界で表示・状態保存する
        update_job(job_id, status="failed", stage="failed",
                   message=f"前処理に失敗しました: {type(exc).__name__}: {exc}",
                   error=f"{type(exc).__name__}: {exc}")
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        return 1

    extract = result.get("extract") or {}
    ocr = result.get("ocr") or {}
    index = result.get("index") or {}
    print("前処理が完了しました。")
    print(json.dumps({
        "files": extract.get("total", 0),
        "processed": extract.get("todo", 0),
        "added": extract.get("added", 0),
        "changed": extract.get("changed", 0),
        "removed": extract.get("removed", 0),
        "ocr_done": ocr.get("done", 0),
        "ocr_cached": ocr.get("cached", 0),
        "documents": index.get("n_docs", 0),
        "chunks": index.get("n_chunks", 0),
    }, ensure_ascii=False, indent=2))
    return 0


def _status(_args: argparse.Namespace) -> int:
    status = preprocessing_status()
    rows = status["rows"]
    if rows:
        for row in rows:
            print(f"{row['status']:<8} {row['path']} "
                  f"(対象 {row['files']}件、未処理 {row['pending']}件)")
    else:
        print("共有ドライブに前処理対象がありません。")
    if status["removed"]:
        print(f"削除を索引へ反映していないファイル: {status['removed']}件")
    if not status["index_ready"]:
        print("検索索引: 未作成")
    print("最終正常完了: " + str(status["last_completed_at"] or "なし"))
    print("状態: " + ("前処理が必要" if status["has_pending"]
                      else "前処理済み"))
    return 0


def _answers_from_run(run_dir: Path, questions: list[dict],
                      only_existing: bool = False) -> dict[int, str]:
    answers: dict[int, str] = {}
    for question in questions:
        index = question["index"]
        record = run_dir / f"q{index}.json"
        if only_existing and not record.is_file():
            continue
        answer = CONFIG.missing_answer
        try:
            payload = json.loads(record.read_text(encoding="utf-8"))
            answer = sanitize(payload.get("answer") or "") or CONFIG.missing_answer
        except (OSError, json.JSONDecodeError, TypeError):
            if only_existing:
                continue
        answers[index] = answer
    return answers


def _answer_csv(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file():
        print(f"ERROR: CSVファイルが見つかりません: {csv_path}", file=sys.stderr)
        return 2
    try:
        questions = load_question_rows(csv_path)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: 質問CSVを読めません: {exc}", file=sys.stderr)
        return 2

    meta_path = CONFIG.index_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {}
    if int(meta.get("n_chunks", 0)) <= 0:
        print("ERROR: 検索索引がありません。先に preprocess を実行してください。",
              file=sys.stderr)
        return 2

    run_id = safe_component(
        args.run_id or f"csv-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "csv-run")
    output = Path(args.output).expanduser().resolve() if args.output else csv_path
    run_dir = CONFIG.logs_dir / run_id
    answers = _answers_from_run(run_dir, questions, only_existing=True)
    print(f"CSVの全{len(questions)}問に回答します。run_id={run_id}")

    import app.run.runner as runner

    if args.no_retry:
        runner.RETRY_UNFINISHED = 0

    def progress(state, record):
        answer = sanitize(record.get("answer") or "") or CONFIG.missing_answer
        answers[int(record["index"])] = answer
        write_answers_to_c_column(
            csv_path, answers, CONFIG.missing_answer, output=output)
        print(f"[{state['done']}/{state['total']}] "
              f"index={record['index']} stop={record.get('stop')} "
              f"-> {answer[:100]}（CSV保存済み）", flush=True)

    try:
        if answers:
            write_answers_to_c_column(
                csv_path, answers, CONFIG.missing_answer, output=output)
        runner.run_batch(
            csv_path, run_id, parallel=args.parallel,
            backend_name=args.backend, model=args.model,
            max_turns=args.max_turns, budget_hours=args.budget_hours,
            progress=progress, force=args.force)
        answers = _answers_from_run(run_dir, questions)
        target = write_answers_to_c_column(
            csv_path, answers, CONFIG.missing_answer, output=output)
    except Exception as exc:  # noqa: BLE001 - CLI境界でエラーを明示する
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        return 1

    print(f"全回答の書き込みが完了しました: {target}")
    print("各質問の回答ログと完全経路は logs/" + run_id + "/ に保存されています。")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser(
        "preprocess", help="share/共有ドライブを一括前処理する")
    preprocess.add_argument(
        "--skip-processed", action="store_true",
        help="前処理済みを省略し、新規・変更・削除分だけ処理する")
    preprocess.add_argument("--no-embed", action="store_true",
                            help="密ベクトル索引を構築しない")
    preprocess.add_argument("--extract-workers", type=int, default=4)
    preprocess.add_argument("--ocr-workers", type=int,
                            default=CONFIG.codex_concurrency)
    preprocess.add_argument("--traceback", action="store_true")
    preprocess.set_defaults(handler=_preprocess)

    status = sub.add_parser(
        "status", help="共有ドライブの前処理状態を表示する")
    status.set_defaults(handler=_status)

    answer = sub.add_parser(
        "answer-csv", help="CSVの全質問に回答し、最終回答をC列へ書く")
    answer.add_argument("csv", help="質問CSVのパス")
    answer.add_argument(
        "--output", default=None,
        help="別ファイルへ保存する場合のパス（省略時は入力CSVを上書き）")
    answer.add_argument("--run-id", default=None,
                        help="再開に使う実行ID（省略時は日時から生成）")
    answer.add_argument("--parallel", type=int, default=None)
    answer.add_argument("--backend", default=None)
    answer.add_argument("--model", default=None)
    answer.add_argument("--max-turns", type=int, default=None)
    answer.add_argument("--budget-hours", type=float, default=None)
    answer.add_argument("--force", action="store_true",
                        help="同じrun-idの実行中ロックを無視する")
    answer.add_argument("--no-retry", action="store_true")
    answer.add_argument("--traceback", action="store_true")
    answer.set_defaults(handler=_answer_csv)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
