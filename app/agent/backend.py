"""LLMBackend 抽象 — 頭脳の差し替え点。

`chat(messages, tools=None, schema=None) -> Reply` を満たす実装を並べる。
バックエンドの差し替えは、この抽象の実装を追加して行う。
グラフ・ツール・プロンプトは Reply しか知らない。

実装方針:
- GeminiFlash / ClaudeAPI / OpenAIAPI  … tools 対応（agentループを回せる）
- CodexCLI                             … schema のみ。**bind_tools 相当を実装しない**。
  CLI は自前でループを完結させる設計なので tool_calls の往復が取れず、
  二重ハーネスになる。単発ノード（画像・verify）専用に固定する。

安全性:
- Gemini の Google Search grounding / URL context はツール定義から外し、
  コードでも明示的に無効化する。
- APIキーは環境変数からのみ読む。Reply・ログ・トレースに載せない。
"""
from __future__ import annotations

import abc
import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import CONFIG


# --------------------------------------------------------------------------
# データ契約
# --------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    # Gemini 3.x は functionCall を送り返すとき thought_signature の同梱を要求する
    # （欠けると 400 INVALID_ARGUMENT）。プロバイダ固有の値なのでここで運ぶ。
    signature: bytes | None = None


@dataclass
class Reply:
    """全バックエンド共通の応答。プロバイダ固有の形はここで吸収し切る。"""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    data: dict | None = None            # schema 指定時の構造化出力
    usage: dict = field(default_factory=dict)
    model: str = ""
    stop_reason: str = ""
    error: str | None = None
    latency_sec: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


class BackendError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# 抽象
# --------------------------------------------------------------------------

class LLMBackend(abc.ABC):
    """messages は [{'role': 'system'|'user'|'assistant'|'tool', 'content': ...}]。

    content は str または content-block のリスト:
      {'type': 'text', 'text': str}
      {'type': 'image', 'media_type': 'image/jpeg', 'data': <base64 str>}
      {'type': 'tool_result', 'tool_call_id': str, 'content': str}
      {'type': 'tool_use', 'id': str, 'name': str, 'input': dict}
    """

    name: str = "abstract"
    supports_tools: bool = False
    supports_images: bool = False

    def _retry_override(self, attempt: int) -> dict:
        """リトライ n 回目に差し込む追加引数（既定は無し）。"""
        return {}

    @abc.abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             schema: dict | None = None, *, model: str | None = None,
             max_tokens: int | None = None,
             timeout: int | None = None) -> Reply:
        ...

    # 429・503などの一時障害を再試行し、ジッタで同時再試行を分散する。
    def chat_with_retry(self, *args, retries: int = 6,
                        backoff: tuple[int, ...] = (3, 8, 20, 45, 90, 150),
                        **kwargs) -> Reply:
        last: Reply | None = None
        for i in range(retries):
            reply = self.chat(*args, **kwargs, **self._retry_override(i))
            if reply.ok:
                return reply
            last = reply
            if not _is_retryable(reply.error or ""):
                return reply
            wait = backoff[min(i, len(backoff) - 1)]
            # ジッタ: 並列ワーカーの再試行が同時に固まるのを防ぐ
            time.sleep(wait * (0.7 + 0.6 * _jitter()))
        return last or Reply(error="retry exhausted")


def _jitter() -> float:
    """0..1 の擬似乱数（スレッドごとにばらけれてれば十分）。"""
    return (time.time_ns() % 1000) / 1000.0


def _is_retryable(err: str) -> bool:
    e = err.lower()
    return any(s in e for s in (
        "429", "rate limit", "resource_exhausted", "quota",
        "500", "502", "503", "504", "unavailable", "deadline", "timeout",
        "出力が空", "overloaded", "internal", "deadline_exceeded",
        # 通信の一時的な切断・名前解決の失敗。1回で問題全体を落とさない
        "readerror", "connecterror", "remoteprotocolerror", "connection reset",
        "10054", "getaddrinfo"))


# 思考モデル向けの最低出力枠。
MIN_OUTPUT_TOKENS = 512


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

class GeminiFlash(LLMBackend):
    """google-genai SDK 直叩き。grounding / URL context は使わない。

    503 "high demand" はモデル単位で起きるため、規定回数を超えたら
    同世代の兄弟モデルへ切り替える。使用モデルは Reply.model に載るので
    manifest とトレースから後追いできる（再現性の担保）。
    """

    name = "gemini"
    supports_tools = True
    supports_images = True
    # 混雑時のフォールバック順（同世代の Flash 系）
    SIBLINGS = ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                "gemini-3-flash-preview")

    # モデル単位の連続失敗カウンタ（プロセス共有）。
    _degraded: dict[str, float] = {}
    _fail_streak: dict[str, int] = {}
    _CB_LOCK = threading.Lock()
    DEGRADE_AFTER = 3          # 連続失敗がこの回数を超えたら降格
    DEGRADE_SEC = 600          # 降格の有効期間

    def __init__(self, model: str | None = None) -> None:
        self.model = model or CONFIG.model_main
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise BackendError("GEMINI_API_KEY が未設定です（.env を確認）")
        from google import genai
        from google.genai import types as gtypes
        self._genai = genai
        # タイムアウトを入れないと、障害時にワーカーが無期限に張り付く
        self._client = genai.Client(
            api_key=key,
            http_options=gtypes.HttpOptions(
                timeout=CONFIG.request_timeout_sec * 1000))

    # ---- サーキットブレーカ ----
    @classmethod
    def _note(cls, model: str, ok: bool) -> None:
        with cls._CB_LOCK:
            if ok:
                cls._fail_streak[model] = 0
                cls._degraded.pop(model, None)
                return
            n = cls._fail_streak.get(model, 0) + 1
            cls._fail_streak[model] = n
            if n >= cls.DEGRADE_AFTER:
                cls._degraded[model] = time.time() + cls.DEGRADE_SEC

    @classmethod
    def _is_degraded(cls, model: str) -> bool:
        with cls._CB_LOCK:
            until = cls._degraded.get(model)
            if until and time.time() < until:
                return True
            if until:
                cls._degraded.pop(model, None)
                cls._fail_streak[model] = 0
            return False

    def _healthy_model(self, want: str) -> str:
        """降格中のモデルは避けて兄弟へ回す。全部降格なら元のまま。"""
        if not self._is_degraded(want):
            return want
        for alt in self.SIBLINGS:
            if alt != want and not self._is_degraded(alt):
                return alt
        return want

    # ---- 変換 ----
    def _to_contents(self, messages: list[dict]) -> tuple[str | None, list]:
        from google.genai import types
        system: str | None = None
        contents: list = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system = m["content"] if isinstance(m["content"], str) \
                    else "\n".join(b.get("text", "") for b in m["content"])
                continue
            parts = []
            content = m["content"]
            if isinstance(content, str):
                if content:
                    parts.append(types.Part(text=content))
            else:
                for b in content:
                    t = b.get("type")
                    if t == "text":
                        parts.append(types.Part(text=b["text"]))
                    elif t == "image":
                        import base64
                        parts.append(types.Part.from_bytes(
                            data=base64.b64decode(b["data"]),
                            mime_type=b.get("media_type", "image/jpeg")))
                    elif t == "tool_use":
                        parts.append(types.Part(
                            function_call=types.FunctionCall(
                                name=b["name"], args=b.get("input") or {}),
                            thought_signature=b.get("signature")))
                    elif t == "tool_result":
                        parts.append(types.Part(
                            function_response=types.FunctionResponse(
                                name=b.get("name", "tool"),
                                response={"result": b["content"]})))
            if not parts:
                continue
            contents.append(types.Content(
                role="model" if role == "assistant" else "user", parts=parts))
        return system, contents

    def _retry_override(self, attempt: int) -> dict:
        # 2回失敗したら兄弟モデルへ逃がす（503 はモデル単位で起きるため）
        if attempt < 2:
            return {}
        alt = _sibling_after(self.model, attempt - 1)
        return {"model": alt} if alt else {}

    def chat(self, messages, tools=None, schema=None, *, model=None,
             max_tokens=None, timeout=None) -> Reply:
        from google.genai import types
        t0 = time.time()
        mdl = self._healthy_model(model or self.model)
        system, contents = self._to_contents(messages)

        # 思考モデルでは max_output_tokens に思考分も含まれる。
        # 小さすぎると text が空で返るため下限を確保する。
        want = max_tokens or CONFIG.max_tokens_per_turn
        cfg: dict[str, Any] = {
            "max_output_tokens": max(want, MIN_OUTPUT_TOKENS),
            # 再現性: 通常回答・tool calling・構造化出力のいずれも同じ設定で呼ぶ
            "temperature": CONFIG.generation_temperature,
            "seed": CONFIG.generation_seed,
            # 外部Web情報を使わないため、grounding/URL context は渡さない。
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True),
        }
        if system:
            cfg["system_instruction"] = system
        if tools:
            cfg["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"], description=t.get("description", ""),
                    parameters=_clean_schema(t["input_schema"]))
                for t in tools])]
        elif schema:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = _clean_schema(schema)

        try:
            resp = self._client.models.generate_content(
                model=mdl, contents=contents,
                config=types.GenerateContentConfig(**cfg))
        except Exception as e:
            self._note(mdl, ok=False)
            return Reply(model=mdl, error=f"{type(e).__name__}: {e}",
                         latency_sec=round(time.time() - t0, 2))

        text, calls = "", []
        try:
            for cand in resp.candidates or []:
                for part in (cand.content.parts or []) if cand.content else []:
                    if getattr(part, "thought", False):
                        continue        # 思考の生テキストは回答に混ぜない
                    if getattr(part, "text", None):
                        text += part.text
                    fc = getattr(part, "function_call", None)
                    if fc is not None and fc.name:
                        calls.append(ToolCall(
                            id=f"{fc.name}_{len(calls)}", name=fc.name,
                            args=dict(fc.args or {}),
                            signature=getattr(part, "thought_signature", None)))
        except Exception as e:
            return Reply(model=mdl, error=f"parse failed: {e}",
                         latency_sec=round(time.time() - t0, 2))

        data = None
        if schema and not tools and text.strip():
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                pass

        um = getattr(resp, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
            "thought_tokens": getattr(um, "thoughts_token_count", 0) or 0,
            # 入力のうち Gemini のキャッシュ（暗黙的キャッシュを含む）から読んだ分
            "cached_tokens": getattr(um, "cached_content_token_count", 0) or 0,
            "total_tokens": getattr(um, "total_token_count", 0) or 0,
        } if um else {}

        finish = str(getattr((resp.candidates or [None])[0], "finish_reason", ""))
        error = None
        # Gemini 3.x Flash は思考モデル。max_output_tokens が思考分に食われると
        # finish_reason=MAX_TOKENS のまま text も tool_calls も空で返る。
        # これを成功として扱うとループが黙って壊れるので、明示的に失敗にする。
        if not text.strip() and not calls and data is None:
            if "MAX_TOKENS" in finish:
                error = (f"出力が空 (finish={finish}, "
                         f"thought_tokens={usage.get('thought_tokens')})"
                         " — max_tokens が思考分に食われた")
            else:
                error = f"出力が空 (finish={finish})"

        self._note(mdl, ok=error is None)
        return Reply(text=text, tool_calls=calls, data=data, usage=usage,
                     model=mdl, stop_reason=finish, error=error,
                     latency_sec=round(time.time() - t0, 2))


def _cli_exe(name: str) -> str:
    """CLI の実行ファイル。見つからなければ名前をそのまま返す。

    Windows の CreateProcess は PATHEXT を解釈しないため、`"claude"` を
    そのまま渡すと PATH 上に claude.CMD があっても起動できない。
    shutil.which は PATHEXT を見るので .CMD / .EXE まで解決する。
    """
    import shutil
    return shutil.which(name) or name


def _strict_schema(schema: dict) -> dict:
    """OpenAI strict 形式へ正規化（codex CLI の --output-schema が要求する）。

    - 全 object に additionalProperties: false
    - 全 property を required に入れる
    Gemini は逆に additionalProperties を受け付けないので、
    **スキーマ正規化は各バックエンドが自分で行う**（抽象の存在意義）。
    """
    def walk(node):
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        out = {k: walk(v) for k, v in node.items()}
        if out.get("type") == "object" or "properties" in out:
            out["type"] = "object"
            out["additionalProperties"] = False
            out["required"] = list((out.get("properties") or {}).keys())
        return out
    return walk(schema)


def _sibling_after(model: str, attempt: int) -> str | None:
    """attempt 回失敗した後に切り替える兄弟モデル。"""
    sibs = GeminiFlash.SIBLINGS
    if model not in sibs:
        return None
    i = sibs.index(model)
    nxt = (i + attempt) % len(sibs)
    return sibs[nxt] if sibs[nxt] != model else None


def _clean_schema(schema: dict) -> dict:
    """Gemini が受け付けない JSON Schema キーを落とす。"""
    drop = {"additionalProperties", "$schema", "default", "examples", "title"}
    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items() if k not in drop}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(schema)


# --------------------------------------------------------------------------
# codex CLI（単発ノード専用: 画像 / verify）
# --------------------------------------------------------------------------

_CODEX_SEM = threading.Semaphore(CONFIG.codex_concurrency)


class CodexCLI(LLMBackend):
    """`codex exec` の単発呼び出し。**bind_tools 相当は実装しない**。

    CLI固有の制約:
      1. `-i` は可変長引数。`-i img.png "prompt"` はプロンプトを画像として食い
         stdin待ちでハングする → **プロンプトは stdin で渡す**
      2. 画像が workspace root の外にあると fs sandbox が弾く
         → **一時ディレクトリにコピーして `-C` でそこを root にする**
      3. 実行条件を固定するため **`--ignore-user-config`** を使う
    危険な sandbox bypass フラグ（--dangerously-*, -s danger-full-access）は使わない。
    """

    name = "codex"
    supports_tools = False
    supports_images = True

    def __init__(self, model: str | None = None) -> None:
        self.model = model or ""

    def chat(self, messages, tools=None, schema=None, *, model=None,
             max_tokens=None, timeout=None) -> Reply:
        if tools:
            raise BackendError(
                "CodexCLI は tool-calling に使わない（二重ハーネスになる）。"
                "単発ノード専用。")
        t0 = time.time()
        prompt, images = _flatten_for_cli(messages)

        with tempfile.TemporaryDirectory(prefix="codex_") as tmp:
            tmpdir = Path(tmp)
            cmd = [_cli_exe("codex"), "exec", "--ephemeral",
                   "--skip-git-repo-check",
                   "-s", "read-only", "--ignore-user-config",
                   "--disable", "browser_use", "--disable", "in_app_browser",
                   "-C", str(tmpdir)]
            chosen_model = model or self.model or CONFIG.codex_model
            if chosen_model:
                cmd += ["--model", chosen_model]
            if CONFIG.codex_reasoning_effort:
                cmd += [
                    "--config",
                    f'model_reasoning_effort="{CONFIG.codex_reasoning_effort}"',
                ]

            out_json = tmpdir / "out.json"
            if schema:
                schema_file = tmpdir / "schema.json"
                schema_file.write_text(
                    json.dumps(_strict_schema(schema), ensure_ascii=False),
                    encoding="utf-8")
                cmd += ["--output-schema", str(schema_file), "-o", str(out_json)]

            for i, (blob, ext) in enumerate(images):
                # 落とし穴2: 画像は必ず workspace root(=tmpdir) の中に置く
                img = tmpdir / f"image{i}{ext}"
                img.write_bytes(blob)
                cmd += ["-i", f"./{img.name}"]

            with _CODEX_SEM:
                try:
                    # 落とし穴1: プロンプトは stdin
                    proc = _run_cli_with_tree_timeout(
                        cmd, input_text=prompt,
                        timeout=timeout or CONFIG.codex_timeout_sec,
                        cwd=tmpdir, env=cli_env("codex"))
                except subprocess.TimeoutExpired:
                    return Reply(model="codex", error="codex timeout",
                                 latency_sec=round(time.time() - t0, 2))
                except FileNotFoundError:
                    return Reply(model="codex",
                                 error="codex CLI が見つかりません",
                                 latency_sec=round(time.time() - t0, 2))

            data, text = None, (proc.stdout or "").strip()
            if schema and out_json.exists():
                try:
                    data = json.loads(out_json.read_text(encoding="utf-8"))
                    text = json.dumps(data, ensure_ascii=False)
                except (json.JSONDecodeError, OSError) as e:
                    return Reply(model="codex", text=text,
                                 error=f"output-schema parse failed: {e}",
                                 latency_sec=round(time.time() - t0, 2))

            if proc.returncode != 0 and data is None:
                err = (proc.stderr or "")[-500:]
                return Reply(model="codex", text=text,
                             error=f"codex exit {proc.returncode}: {err}",
                             latency_sec=round(time.time() - t0, 2))

        return Reply(text=text, data=data, model="codex",
                     latency_sec=round(time.time() - t0, 2))


def _flatten_for_cli(messages: list[dict]) -> tuple[str, list[tuple[bytes, str]]]:
    """単発CLI向けに messages を 1本のプロンプト + 画像リストへ畳む。"""
    import base64
    parts: list[str] = []
    images: list[tuple[bytes, str]] = []
    for m in messages:
        content = m["content"]
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) \
            else content
        for b in blocks:
            if b.get("type") == "text" and b.get("text"):
                parts.append(b["text"])
            elif b.get("type") == "image":
                mt = b.get("media_type", "image/png")
                images.append((base64.b64decode(b["data"]),
                               ".jpg" if "jpeg" in mt else ".png"))
    return "\n\n".join(parts), images


# --------------------------------------------------------------------------
# 未実装のAPIバックエンド
# --------------------------------------------------------------------------


class ClaudeCLI(LLMBackend):
    """`claude -p` の単発呼び出し。**画像読み取り専用**。

    CodexCLI と同じ単発ノードで、エージェントループには使わない。

    codex と違い claude CLI は画像を引数で渡せないので、**一時ディレクトリへ置いて
    Read だけを許可して読ませる**。安全のため次を固定する:
      --allowedTools Read   … 読み取り以外を禁止（Bash・書き込み・Webは不可）
      --strict-mcp-config + 空の --mcp-config
                            … ユーザーのMCPサーバを遮断する
    """

    name = "claude"
    supports_tools = False
    supports_images = True

    def __init__(self, model: str | None = None) -> None:
        self.model = model or CONFIG.claude_cli_model

    def chat(self, messages, tools=None, schema=None, *, model=None,
             max_tokens=None, timeout=None) -> Reply:
        if tools:
            raise BackendError(
                "ClaudeCLI は tool-calling に使わない（二重ハーネスになる）。"
                "単発ノード専用。")
        t0 = time.time()
        prompt, images = _flatten_for_cli(messages)
        mdl = model or self.model

        with tempfile.TemporaryDirectory(prefix="claudecli_") as tmp:
            tmpdir = Path(tmp)
            refs = []
            for i, (blob, ext) in enumerate(images):
                img = tmpdir / f"image{i}{ext}"
                img.write_bytes(blob)
                refs.append(_cli_path(img))
            if refs:
                prompt = ("次のファイルを Read ツールで読んでください:\n"
                          + "\n".join(refs) + "\n\n" + prompt)
            if schema:
                # claude CLI に構造化出力の指定は無いので、JSON を本文で出させる
                prompt += ("\n\n出力は次の JSON Schema に従う JSON オブジェクト"
                           "**のみ**を、コードブロックなしで出力してください:\n"
                           + json.dumps(schema, ensure_ascii=False))

            mcp = tmpdir / "nomcp.json"
            mcp.write_text('{"mcpServers":{}}', encoding="utf-8")
            cmd = [_cli_exe("claude"), "-p", "--model", mdl,
                   "--allowedTools", "Read",
                   "--strict-mcp-config", "--mcp-config", str(mcp)]
            if CONFIG.claude_cli_effort:
                cmd += ["--effort", CONFIG.claude_cli_effort]
            try:
                proc = _run_cli_with_tree_timeout(
                    cmd, input_text=prompt,
                    timeout=timeout or CONFIG.claude_cli_timeout_sec,
                    cwd=tmpdir, env=cli_env("claude"))
            except subprocess.TimeoutExpired:
                return Reply(model=mdl, error="claude CLI timeout",
                             latency_sec=round(time.time() - t0, 2))
            except FileNotFoundError:
                return Reply(model=mdl, error="claude CLI が見つかりません",
                             latency_sec=round(time.time() - t0, 2))

        text = (proc.stdout or "").strip()
        if proc.returncode != 0 and not text:
            return Reply(model=mdl, text=text,
                         error=f"claude exit {proc.returncode}: "
                               f"{(proc.stderr or '')[-500:]}",
                         latency_sec=round(time.time() - t0, 2))
        data = None
        if schema:
            data = _first_json_object(text)
            if data is None:
                return Reply(model=mdl, text=text,
                             error="JSON を取り出せなかった",
                             latency_sec=round(time.time() - t0, 2))
        return Reply(text=text, data=data, model=mdl,
                     latency_sec=round(time.time() - t0, 2))


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """CLIラッパーと子プロセスを止め、パイプを保持した子を残さない。

    Windows の `claude.CMD` は node.exe を子に起動する。`Popen.kill()` で
    CMDだけを止めると子が stdout/stderr を保持し、タイムアウト後の
    `communicate()` が無期限に待つ。PIDを明示してその木だけを終了する。
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


# CLI ごとの APIキー環境変数。subscription のときは子プロセスから外し、
# CLI が自分のログイン情報を使うようにする（キーがあると CLI はそちらを優先する）。
_CLI_KEY_VARS = {"claude": ("ANTHROPIC_API_KEY",),
                 "codex": ("CODEX_API_KEY", "OPENAI_API_KEY")}


def cli_env(cli: str) -> dict[str, str]:
    """CLI 子プロセスへ渡す環境変数。課金経路（subscription / api）をここで決める。"""
    env = dict(os.environ)
    auth = CONFIG.claude_cli_auth if cli == "claude" else CONFIG.codex_auth
    if auth != "api":
        for var in _CLI_KEY_VARS[cli]:
            env.pop(var, None)
    elif cli == "codex" and not env.get("CODEX_API_KEY") and env.get("OPENAI_API_KEY"):
        # codex exec が単発実行で読むのは CODEX_API_KEY
        env["CODEX_API_KEY"] = env["OPENAI_API_KEY"]
    return env


def _run_cli_with_tree_timeout(cmd: list[str], *, input_text: str,
                               timeout: int, cwd: Path,
                               env: dict[str, str] | None = None
                               ) -> subprocess.CompletedProcess:
    """`subprocess.run` 相当。ただしタイムアウト時に子プロセスも回収する。"""
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", cwd=str(cwd),
        creationflags=flags, env=env)
    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            stdout, stderr = "", ""
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
    except BaseException:
        _terminate_process_tree(proc)
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)



def _cli_path(path: Path) -> str:
    """claude CLI に渡すパス。POSIX 側から Windows の CLI を呼ぶときだけ変換する。

    WSL の一時ディレクトリに置いた画像は Windows 側のCLIから開けないため、
    必要な場合だけパスを変換する。
    Windows で動かしているときは変換不要（pagemap._to_native と同じ判定）。
    """
    if os.name == "nt":
        return str(path)
    import shutil
    exe = shutil.which("claude") or ""
    if not exe.startswith("/mnt/"):      # POSIX 側の CLI を呼んでいる
        return str(path)
    try:
        out = subprocess.run(["wslpath", "-w", str(path)],
                             capture_output=True, text=True, timeout=10)
        w = (out.stdout or "").strip()
        return w or str(path)
    except (OSError, subprocess.SubprocessError):
        return str(path)


def _first_json_object(text: str):
    """本文から最初の JSON オブジェクトを取り出す（前後の地の文を許容する）。"""
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None

class ClaudeAPI(LLMBackend):
    name = "claude"
    supports_tools = True
    supports_images = True

    def chat(self, *a, **k) -> Reply:  # pragma: no cover - 未実装
        raise BackendError("ClaudeAPI は未実装です")


class OpenAIAPI(LLMBackend):
    name = "openai"
    supports_tools = True
    supports_images = True

    def chat(self, *a, **k) -> Reply:  # pragma: no cover - 未実装
        raise BackendError("OpenAIAPI は未実装です")


_REGISTRY: dict[str, type[LLMBackend]] = {
    "gemini": GeminiFlash, "codex": CodexCLI,
    "claude_cli": ClaudeCLI,
    "claude": ClaudeAPI, "openai": OpenAIAPI,
}


def get_backend(name: str | None = None, **kw) -> LLMBackend:
    key = (name or CONFIG.backend).lower()
    if key not in _REGISTRY:
        raise BackendError(f"未知のバックエンド: {key}（{list(_REGISTRY)}）")
    return _REGISTRY[key](**kw)
