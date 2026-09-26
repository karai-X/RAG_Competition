"""画像エンジン: codex CLI と Gemini Flash の2本立て。

claude CLI は使わない方針のため、**フォールバックは Gemini Flash** が担う。
どちらも `LLMBackend.chat(messages, schema=...)` の単発呼び出しに落とす。
失敗時は空の結果を返す（推測させない）。
"""
from __future__ import annotations

import base64
import time

from app.agent.backend import LLMBackend, get_backend
from app.config import CONFIG
from app.ocr.cache import PROMPT_VERSION, OcrResult
from app.ocr.prompt import OCR_PROMPT, OCR_SCHEMA

_BACKENDS: dict[str, LLMBackend] = {}


def backend_for(engine: str) -> LLMBackend:
    if engine not in _BACKENDS:
        _BACKENDS[engine] = get_backend(engine)
    return _BACKENDS[engine]


def run_engine(engine: str, img: bytes, md5: str, source: str = "",
               timeout: int | None = None) -> OcrResult:
    t0 = time.time()
    res = OcrResult(md5=md5, engine=engine, prompt_version=PROMPT_VERSION,
                    source=source)
    if not img:
        res.error = "画像データが空"
        return res

    messages = [{"role": "user", "content": [
        {"type": "text", "text": OCR_PROMPT},
        {"type": "image", "media_type": "image/png",
         "data": base64.b64encode(img).decode()},
    ]}]
    try:
        be = backend_for(engine)
    except Exception as e:                        # noqa: BLE001
        res.error = f"backend生成失敗: {type(e).__name__}: {e}"
        return res

    # Codex CLI は有効な画像でも180秒を超えることがある。同じ制限で
    # 再試行しても同じ箇所で切れるため、タイムアウトのたびに待ち時間を
    # 2倍にし、読み取りに成功するまで続ける。画像不正や構造化出力エラー
    # など、タイムアウト以外は無限再試行しない。
    timeout_sec = timeout or CONFIG.codex_timeout_sec
    while True:
        reply = be.chat_with_retry(
            messages, schema=OCR_SCHEMA, max_tokens=4000,
            timeout=timeout_sec if engine == "codex" else timeout,
            # Codex のタイムアウトはこの外側で待ち時間を
            # 倍増させる。同じ時間制限で再試行すると、xhighの
            # 長時間処理で無駄に同じタイムアウトを繰り返す。
            retries=1 if engine == "codex" else 2)
        if (engine != "codex" or reply.ok
                or "timeout" not in (reply.error or "").lower()):
            break
        timeout_sec *= 2
    res.latency_sec = round(time.time() - t0, 2)
    if not reply.ok or not isinstance(reply.data, dict):
        res.error = reply.error or "構造化出力が得られなかった"
        return res

    d = reply.data
    res.content_type = str(d.get("content_type") or "")
    res.markdown = str(d.get("markdown") or "")
    res.figure_description = str(d.get("figure_description") or "")
    res.layout = str(d.get("layout") or "")
    res.confident = bool(d.get("confident"))
    return res
