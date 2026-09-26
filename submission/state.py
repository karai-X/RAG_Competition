"""CLI前処理ジョブの状態をJSONへ保存する。"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import CONFIG


_STATE_LOCK = threading.RLock()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def jobs_dir() -> Path:
    path = CONFIG.rag_home / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (2 ** attempt))
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def create_job(job_id: str, kind: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "stage": "queued",
        "message": "開始待ち",
        "progress": 0.0,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        **fields,
    }
    with _STATE_LOCK:
        atomic_write_json(job_path(job_id), payload)
    return payload


def update_job(job_id: str, **fields: Any) -> dict[str, Any]:
    with _STATE_LOCK:
        payload = read_json(job_path(job_id)) or {"job_id": job_id}
        payload.update(fields)
        payload["updated_at"] = now_iso()
        atomic_write_json(job_path(job_id), payload)
    return payload
