"""設定画面: 回答に使う Gemini と、画像読み取りに使うエンジン（Claude Code / Codex / Gemini）。

保存先は `.env`。起動し直さなくても次の質問から反映される。
"""
from __future__ import annotations

import shutil

import streamlit as st

from app.config import CONFIG
from webui import settings_store as store
from webui.core import RUNNER

AUTH_LABELS = {"subscription": "サブスクリプション（CLI のログインを使う）",
               "api": "API キー（従量課金）"}
ENGINE_LABELS = {"claude_cli": "Claude Code", "codex": "Codex", "gemini": "Gemini（API）"}
CLAUDE_EFFORTS = ["low", "medium", "high", "xhigh", "max"]
CODEX_EFFORTS = ["minimal", "low", "medium", "high", "xhigh"]
TEST_TIMEOUT_SEC = 300

st.title("設定")
st.caption("API キーはこの画面で設定します。入力したキーはこのフォルダの `.env` に保存され、"
           "このPCの外へは送りません（各サービスへの問い合わせを除く）。")

cur = store.current()


def secret_input(key: str) -> str:
    """キー入力欄。APIキーはこの画面でだけ設定する。

    保存済みのキーを入れた状態で出し、ふだんは伏せ字、目のボタンで表示する。
    欄の値をそのまま保存するので、空にして保存すると削除になる。
    """
    val = st.text_input(store.SECRET_KEYS[key], cur[key], type="password",
                        key=f"in-{key}", placeholder="API キーを入力")
    if not cur[key]:
        st.caption("未設定です。入力して「保存する」を押してください。")
    return val


def pick(where, label: str, options: list[str], value: str, **kw) -> str:
    if value not in options:
        options = options + [value]
    return where.selectbox(label, options, index=options.index(value), **kw)


# ---------------------------------------------------------------- 入力

with st.container(border=True):
    st.subheader("回答生成", divider="gray")
    st.caption("資料の探索と回答の作成には Gemini API を使います。")
    gemini_key = secret_input("GEMINI_API_KEY")
    st.text_input("モデル（固定）", CONFIG.model_main, disabled=True, key="gemini-model")

with st.container(border=True):
    st.subheader("画像読み取り", divider="gray")
    st.caption("図・グラフ・スキャンページを、回答中に読み取らせるエンジンです。")
    engines = list(ENGINE_LABELS)
    engine = st.radio("エンジン", engines, horizontal=True,
                      index=engines.index(cur["model_image"]) if cur["model_image"] in engines else 0,
                      format_func=ENGINE_LABELS.get)

    anthropic_key = codex_key = None             # 欄を出さなかったキーは触らない
    claude_auth, codex_auth = cur["claude_cli_auth"], cur["codex_auth"]
    claude_effort, codex_effort = cur["claude_cli_effort"], cur["codex_reasoning_effort"]

    if engine == "claude_cli":
        claude_auth = st.radio("課金方法", list(AUTH_LABELS), format_func=AUTH_LABELS.get,
                               index=list(AUTH_LABELS).index(claude_auth))
        if claude_auth == "subscription":
            st.caption("事前に PowerShell で `claude` を起動し、Claude のアカウントでログインしておいてください。")
        else:
            anthropic_key = secret_input("ANTHROPIC_API_KEY")
        c1, c2 = st.columns(2)
        c1.text_input("モデル（固定）", CONFIG.claude_cli_model, disabled=True, key="claude-model")
        claude_effort = pick(c2, "推論の深さ（effort）", CLAUDE_EFFORTS, claude_effort, key="claude-effort")
        if not shutil.which("claude"):
            st.warning("claude CLI が見つかりません。インストールして PATH を通してください。")
    elif engine == "codex":
        codex_auth = st.radio("課金方法", list(AUTH_LABELS), format_func=AUTH_LABELS.get,
                              index=list(AUTH_LABELS).index(codex_auth))
        if codex_auth == "subscription":
            st.caption("事前に PowerShell で `codex login` を実行し、ChatGPT のアカウントでログインしておいてください。")
        else:
            codex_key = secret_input("CODEX_API_KEY")
        c1, c2 = st.columns(2)
        c1.text_input("モデル（固定）", CONFIG.codex_model, disabled=True, key="codex-model")
        codex_effort = pick(c2, "推論の深さ（reasoning effort）", CODEX_EFFORTS, codex_effort,
                            key="codex-effort")
        if not shutil.which("codex"):
            st.warning("codex CLI が見つかりません。インストールして PATH を通してください。")
    else:
        st.caption("回答生成と同じ Gemini API キーを使います。")

# ---------------------------------------------------------------- 保存

if RUNNER.busy:
    st.info("質問の実行中は保存できません。終了してから保存してください。")

if st.button("保存する", type="primary", icon=":material/save:", disabled=RUNNER.busy):
    secrets = {key: (val or "").strip() or None
               for key, val in (("GEMINI_API_KEY", gemini_key),
                                ("ANTHROPIC_API_KEY", anthropic_key),
                                ("CODEX_API_KEY", codex_key)) if val is not None}
    store.save({"model_image": engine,
                "claude_cli_auth": claude_auth, "claude_cli_effort": claude_effort,
                "codex_auth": codex_auth, "codex_reasoning_effort": codex_effort},
               secrets)
    for key in store.SECRET_KEYS:                     # 保存した値で入力欄を作り直す
        st.session_state.pop(f"in-{key}", None)
    st.toast("保存しました。次の質問から反映されます。", icon=":material/check:")
    st.rerun()

# ---------------------------------------------------------------- 接続テスト

st.subheader("接続テスト")
st.caption("保存済みの設定で短い問い合わせを送り、ログインやキーが有効か確かめます。")


def run_test(name: str, **kw) -> None:
    from app.agent.backend import get_backend
    messages = [{"role": "user", "content": "「OK」とだけ返してください。"}]
    with st.spinner(f"{name} に問い合わせています…"):
        try:
            reply = get_backend(kw.pop("backend"), **kw).chat(messages, timeout=TEST_TIMEOUT_SEC)
        except Exception as exc:                      # noqa: BLE001
            st.error(f"{name}: {type(exc).__name__}: {exc}")
            return
    if reply.ok:
        st.success(f"{name}: 応答あり（{reply.latency_sec}秒）— {reply.text.strip()[:80]}")
    else:
        st.error(f"{name}: {reply.error}")


c1, c2 = st.columns(2)
if c1.button("回答生成（Gemini）", icon=":material/network_check:"):
    run_test(f"Gemini {CONFIG.model_main}", backend="gemini")
if c2.button(f"画像読み取り（{ENGINE_LABELS.get(CONFIG.model_image, CONFIG.model_image)}）",
             icon=":material/network_check:"):
    run_test(ENGINE_LABELS.get(CONFIG.model_image, CONFIG.model_image), backend=CONFIG.model_image)
