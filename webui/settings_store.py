"""設定画面の保存先。`.env` を書き換え、同じプロセスへも即時に反映する。

APIキーは `.env`（.gitignore 対象）と環境変数にだけ置く。画面・ログには末尾4文字しか出さない。
"""
from __future__ import annotations

import dataclasses
import os

from app.config import CONFIG, ENV_OVERRIDABLE, REPO_DIR, Config

ENV_PATH = REPO_DIR / ".env"

# 画面で扱うAPIキー。codex exec が単発実行で読むのは CODEX_API_KEY。
SECRET_KEYS = {
    "GEMINI_API_KEY": "Gemini API キー",
    "ANTHROPIC_API_KEY": "Anthropic API キー（Claude Code を API で使う場合）",
    "CODEX_API_KEY": "OpenAI API キー（Codex を API で使う場合）",
}

# CONFIG の項目で、画面から変えられるもの（RAG_<大文字> で保存する）。モデルは固定。
CONFIG_KEYS = ENV_OVERRIDABLE

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}


def env_name(key: str) -> str:
    return f"RAG_{key.upper()}"


def current() -> dict[str, str]:
    """画面に出す現在値。CONFIG の項目は CONFIG から、キーは環境変数から読む。"""
    vals = {k: str(getattr(CONFIG, k)) for k in CONFIG_KEYS}
    vals.update({k: os.environ.get(k, "") for k in SECRET_KEYS})
    return vals


def _write_env(updates: dict[str, str | None]) -> None:
    """`.env` の該当行だけを書き換える。None は行を消す。コメントと他の行は残す。"""
    lines = (ENV_PATH.read_text(encoding="utf-8-sig").splitlines()
             if ENV_PATH.exists() else [])
    pending = dict(updates)
    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key and not line.lstrip().startswith("#") and key in pending:
            val = pending.pop(key)
            if val is not None:
                out.append(f"{key}={val}")
            continue
        out.append(line)
    out += [f"{k}={v}" for k, v in pending.items() if v is not None]
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def save(config_values: dict[str, str], secrets: dict[str, str | None]) -> None:
    """設定を保存して反映する。

    config_values: CONFIG の項目名 → 値。既定値と同じなら `.env` から消す。
    secrets: キー名 → 新しい値。None は削除。渡さなかったキーはそのまま。
    """
    updates: dict[str, str | None] = {}
    for key, val in config_values.items():
        val = (val or "").strip()
        default = str(_DEFAULTS[key])
        updates[env_name(key)] = None if (not val or val == default) else val
        setattr(CONFIG, key, val or _DEFAULTS[key])
    for key, val in secrets.items():
        updates[key] = val.strip() if val else None

    _write_env(updates)
    for key, val in updates.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val

    # 画像読み取りのバックエンドは生成時にモデル名を握るので作り直させる
    from app.ocr import engines
    engines._BACKENDS.clear()
