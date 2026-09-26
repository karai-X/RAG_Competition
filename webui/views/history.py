"""回答経路の画面: Web UI から投げた質問を選び、回答とツール呼び出しの経路を読む。"""
from __future__ import annotations

import streamlit as st

from webui.core import list_questions, live_group
from webui.render import question_browser

st.title("回答経路")
st.caption("この画面から投げた質問について、どのツールをどの順で呼び、何を根拠に答えたかを表示します。")

group = live_group()
rows = sorted(list_questions(group), key=lambda r: r.get("t") or 0, reverse=True)  # 新しい順
if not rows:
    st.info("まだ記録がありません。「質問する」画面から質問すると、ここに残ります。")
    st.stop()

question_browser(group, rows, index_label="time")
