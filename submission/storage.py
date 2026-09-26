"""CLI前処理の対象名検証と、`share/共有ドライブ` の処理状態管理。"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from app.config import CONFIG

SHARE_CATEGORIES = ("プロジェクト", "社内管理")

_INVALID_WINDOWS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_WINDOWS = {
    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_component(value: str, fallback: str = "item") -> str:
    """ログIDなどを単一の安全なWindowsパス要素へ変換する。"""
    value = unicodedata.normalize("NFC", str(value or "")).strip()
    value = _INVALID_WINDOWS.sub("_", value).rstrip(". ")
    if not value:
        value = fallback
    if value.upper() in _RESERVED_WINDOWS:
        value = f"_{value}"
    return value[:120]


def _content_md5(path: str, size: int) -> str:
    digest = hashlib.md5()
    try:
        with open(path, "rb") as source:
            if size <= 256 * 1024 * 1024:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    digest.update(chunk)
            else:
                digest.update(source.read(1 << 20))
                source.seek(-(1 << 20), 2)
                digest.update(source.read())
                digest.update(str(size).encode())
    except OSError:
        return ""
    return digest.hexdigest()


def _is_processed(corpus_file, previous: dict | None) -> bool:
    if not previous or previous.get("size") != corpus_file.size:
        return False
    if not (CONFIG.extracted_dir / f"{corpus_file.doc_id}.json").exists():
        return False
    try:
        mtime_ns = Path(corpus_file.raw_path).stat().st_mtime_ns
    except OSError:
        return False
    if previous.get("mtime_ns") == mtime_ns:
        return True
    return bool(previous.get("md5")
                and previous.get("md5")
                == _content_md5(corpus_file.raw_path, corpus_file.size))


def load_preprocessing_state() -> dict:
    try:
        return json.loads(
            CONFIG.preprocess_state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def mark_preprocessing_completed() -> dict:
    """抽出・OCR・索引がすべて成功した時点のファイル状態を保存する。"""
    from app.corpus.manifest import load_manifest

    payload = {
        "completed_at": datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "files": load_manifest(),
    }
    target = CONFIG.preprocess_state_file
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(target)
    return {
        "completed_at": payload["completed_at"],
        "n_files": len(payload["files"]),
    }


def preprocessing_status() -> dict:
    """直下のファイル／フォルダ単位で、前処理済みかを集計する。"""
    from app.corpus.walk import walk_corpus

    completed = load_preprocessing_state()
    completed_files = completed.get("files") or {}
    files = walk_corpus(with_md5=False)
    by_unit: dict[str, list] = {}
    for corpus_file in files:
        parts = corpus_file.relpath.split("/")
        if len(parts) < 2 or parts[0] not in SHARE_CATEGORIES:
            continue
        by_unit.setdefault("/".join(parts[:2]), []).append(corpus_file)

    units: dict[str, dict] = {}
    for category in SHARE_CATEGORIES:
        base = CONFIG.corpus_dir / category
        if not base.exists():
            continue
        for child in base.iterdir():
            key = f"{category}/{unicodedata.normalize('NFC', child.name)}"
            units[key] = {
                "category": category,
                "name": key.split("/", 1)[1],
            }

    rows = []
    for key in sorted(units, key=str.casefold):
        unit_files = by_unit.get(key, [])
        pending = [
            corpus_file for corpus_file in unit_files
            if not _is_processed(
                corpus_file, completed_files.get(corpus_file.relpath))
        ]
        if not unit_files:
            state = "対象ファイルなし"
        elif pending:
            state = "未処理"
        else:
            state = "前処理済み"
        rows.append({
            **units[key],
            "path": key,
            "status": state,
            "files": len(unit_files),
            "pending": len(pending),
        })

    current = {corpus_file.relpath for corpus_file in files}
    removed = sorted(set(completed_files) - current)
    try:
        index_meta = json.loads(
            (CONFIG.index_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        index_meta = {}
    index_ready = int(index_meta.get("n_chunks", 0)) > 0
    return {
        "rows": rows,
        "processed": sum(row["status"] == "前処理済み" for row in rows),
        "pending": sum(row["status"] == "未処理" for row in rows),
        "no_files": sum(row["status"] == "対象ファイルなし" for row in rows),
        "pending_files": sum(row["pending"] for row in rows),
        "removed": len(removed),
        "removed_paths": removed[:10],
        "has_pending": any(row["status"] == "未処理" for row in rows)
        or bool(removed) or (bool(files) and not index_ready),
        "index_ready": index_ready,
        "last_completed_at": completed.get("completed_at"),
    }
