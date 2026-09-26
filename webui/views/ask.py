"""質問画面: 新しい質問を1問投げ、ツール呼び出しが起きるたびに経路が伸びていくのを見る。"""
from __future__ import annotations

import time

import streamlit as st

from app.config import CONFIG
from webui.core import (MAX_QUESTION_CHARS, RUNNER, missing_prerequisites, read_jsonl,
                         steps_from_events)
from webui.render import render_answer, render_steps

IMAGE_ENGINE_LABELS = {"claude_cli": "Claude Code", "codex": "Codex", "gemini": "Gemini"}
POLL_SEC = 1.5

st.title("質問する")
st.caption("共有ドライブの資料を調べて、根拠つきで答えます。1問あたり数分かかることがあります。")

missing = missing_prerequisites()
engine = CONFIG.model_image
auth = {"claude_cli": CONFIG.claude_cli_auth, "codex": CONFIG.codex_auth}.get(engine)
st.caption(
    f"回答モデル: **{CONFIG.model_main}** ・ 画像読み取り: **{IMAGE_ENGINE_LABELS.get(engine, engine)}**"
    + (f"（{'サブスクリプション' if auth == 'subscription' else 'API'}）" if auth else ""))

if missing:
    with st.container(border=True):
        st.markdown("**実行するには、次のものが必要です。**")
        st.markdown("\n".join(f"- {m['what']} — {m['fix']}" for m in missing))
        st.page_link("webui/views/settings.py", label="設定を開く", icon=":material/settings:")

with st.form("ask", border=False):
    question = st.text_area("質問", height=110, max_chars=MAX_QUESTION_CHARS,
                            placeholder="例: 〇〇株式会社の契約書で定められた納品物を挙げてください。")
    submitted = st.form_submit_button("実行する", type="primary", icon=":material/send:",
                                      disabled=bool(missing) or RUNNER.busy)

if submitted:
    if not question.strip():
        st.warning("質問を入力してください。")
    else:
        try:
            RUNNER.start(question.strip())
            st.rerun()                     # 実行ボタンを無効にした状態で描き直す
        except RuntimeError as exc:
            st.warning(str(exc))


def show_run() -> None:
    run = RUNNER.current
    if run is None:
        return
    st.divider()
    st.markdown(f"#### {run.question}")
    elapsed = (run.finished or time.time()) - run.started
    if not run.done:
        st.info(f"{run.phase}… 経過 {int(elapsed // 60)}分{int(elapsed % 60):02d}秒",
                icon=":material/progress_activity:")

    steps = steps_from_events(read_jsonl(run.chain_path))
    if run.done:
        if run.error:
            st.error(f"実行に失敗しました: {run.error}", icon=":material/error:")
        else:
            r = run.result
            render_answer({"a": r["answer"], "conf": r["confidence"], "ev": r["evidence"],
                           "turns": r["turns"], "sec": r["sec"],
                           "tok": r["tok"], "model": r["model"], "run_id": run.run_id})
        st.markdown(f"**回答経路**（ツール呼び出し {sum(s['k'] == 'tool' for s in steps)} 回）")
    render_steps(steps)

    # 終わった瞬間に全体を描き直し、実行ボタンを戻して自動更新を止める
    if run.done and st.session_state.get("polling"):
        st.session_state.polling = False
        st.rerun()


if RUNNER.busy:
    st.session_state.polling = True
    st.fragment(run_every=POLL_SEC)(show_run)()
else:
    show_run()
