"""`share/共有ドライブ` へ抽出・OCR・索引構築を一括適用する。"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from app.config import CONFIG
from app.extract.health import check as health_check
from app.extract.pipeline import run_extract
from app.index.build import build as build_index
from app.ocr.runner import apply_to_docs, run_batch
from app.ocr.targets import collect_targets
from submission.state import read_json, update_job
from submission.storage import mark_preprocessing_completed


def _health_summary(result: dict) -> dict:
    return {
        "n_docs": result.get("n_docs", 0),
        "n_files": result.get("n_files", 0),
        "errors": len(result.get("errors", [])),
        "empty": len(result.get("empties", [])),
        "missing": len(result.get("missing", [])),
        "media_mismatch": len(result.get("mismatch", [])),
        "ocr_unapplied": len(result.get("ocr_unapplied", [])),
        "warnings": len(result.get("warnings", [])),
        "by_type": result.get("by_type", {}),
    }


def _raise_on_health_errors(report: dict, *, check_ocr: bool = False) -> None:
    """致命的な抽出異常を成功扱いにせず、対象を含むメッセージで停止する。"""
    labels = {
        "errors": "抽出エラー",
        "empties": "空文書",
        "missing": "未抽出",
        "mismatch": "メディア不一致",
    }
    if check_ocr:
        labels["ocr_unapplied"] = "OCR未反映"

    failures = [(key, labels[key], report.get(key, []))
                for key in labels if report.get(key)]
    if not failures:
        return

    counts = "、".join(f"{label} {len(items)}件"
                       for _, label, items in failures)
    samples = []
    for _, label, items in failures:
        for item in items[:2]:
            relpath = item[0] if isinstance(item, (list, tuple)) else item
            samples.append(f"{label}: {relpath}")
    detail = " / ".join(samples[:4])
    raise RuntimeError(
        f"前処理結果に致命的なエラーがあるため停止します（{counts}）。"
        f"対象: {detail}")


def run_pipeline(request: dict) -> dict:
    job_id = str(request["job_id"])
    started = time.time()
    result: dict = {"mount": None, "extract": None, "ocr": None,
                    "health": None, "index": None}

    update_job(job_id, status="running", stage="register", progress=0.03,
               message="共有ドライブの未処理資料を確認中")
    root = CONFIG.corpus_dir
    if not root.is_dir():
        raise FileNotFoundError("データソースが見つかりません: share/共有ドライブ")
    result["mount"] = {"path": "share/共有ドライブ", "mount_as": ""}

    update_job(job_id, stage="extract", progress=0.08,
               message="Office・PDF・表・書式・画像を増分抽出中", result=result)

    def extract_progress(done: int, total: int, doc) -> None:
        ratio = done / max(1, total)
        update_job(job_id, stage="extract", progress=0.08 + ratio * 0.34,
                   message=f"抽出 {done}/{total}: {doc.relpath}", result=result)

    result["extract"] = run_extract(
        full=bool(request.get("full", False)),
        workers=int(request.get("extract_workers", 4)),
        progress=extract_progress)

    update_job(job_id, stage="health", progress=0.44,
               message="抽出結果の欠落・空文書・画像数を検査中", result=result)
    health_report = health_check()
    result["health"] = _health_summary(health_report)
    _raise_on_health_errors(health_report)

    # OCRは前処理の必須工程。画像があれば必ずキャッシュ確認とOCRを実行する。
    targets = collect_targets()
    update_job(job_id, stage="ocr", progress=0.48,
               message=f"画像OCRを実行中（対象 {len(targets)}件、キャッシュ利用）",
               result=result)

    def ocr_progress(stats, target, response) -> None:
        done = stats.cached + stats.done + stats.failed + stats.skipped
        ratio = done / max(1, stats.total)
        update_job(job_id, stage="ocr", progress=0.48 + ratio * 0.24,
                   message=f"画像OCR {done}/{stats.total}: {target.relpath}",
                   result=result)

    stats = run_batch(
        engine=str(request.get("ocr_engine") or CONFIG.model_ocr),
        workers=int(request.get("ocr_workers", CONFIG.codex_concurrency)),
        targets=targets, progress=ocr_progress)
    result["ocr"] = {
        "total": stats.total, "cached": stats.cached, "done": stats.done,
        "failed": stats.failed, "skipped": stats.skipped,
        "errors": list(stats.errors[:20]),
    }
    ocr_errors = stats.failed + stats.skipped
    if ocr_errors:
        detail = "; ".join(stats.errors[:3]) or "詳細は前処理ログを確認してください"
        raise RuntimeError(
            "画像OCRを完了できなかったため前処理を停止します"
            f"（失敗 {stats.failed}件、読取不能 {stats.skipped}件）: {detail}")
    update_job(job_id, stage="ocr_apply", progress=0.73,
               message="OCR結果を抽出本文へ統合中", result=result)
    result["ocr"]["apply"] = apply_to_docs(
        str(request.get("ocr_engine") or CONFIG.model_ocr))
    health_report = health_check()
    result["health"] = _health_summary(health_report)
    _raise_on_health_errors(health_report, check_ocr=True)

    update_job(job_id, stage="index", progress=0.76,
               message="チャンク化・BM25・密ベクトル索引を構築中", result=result)

    def index_progress(message: str) -> None:
        update_job(job_id, stage="index", progress=0.86,
                   message=message, result=result)

    result["index"] = build_index(
        embed=bool(request.get("do_embed", True)), progress=index_progress)
    result["preprocessing_state"] = mark_preprocessing_completed()

    elapsed = round(time.time() - started, 1)
    update_job(job_id, status="completed", stage="completed", progress=1.0,
               message=f"前処理が完了しました（{elapsed:.1f}秒）",
               elapsed_sec=elapsed, result=result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = read_json(Path(args.request))
    if not request:
        print(f"request JSONを読めません: {args.request}", file=sys.stderr)
        return 2
    job_id = str(request.get("job_id", ""))
    try:
        run_pipeline(request)
        return 0
    except Exception as exc:  # noqa: BLE001 - ワーカー境界で状態へ記録する
        traceback.print_exc()
        if job_id:
            update_job(job_id, status="failed", stage="failed",
                       message=f"前処理に失敗しました: {type(exc).__name__}: {exc}",
                       error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
