"""100問の記録の画面: 評価用の100問を一括実行したときの回答と経路。実行は固定。"""
from __future__ import annotations

import streamlit as st

from webui.core import benchmark_group, list_questions
from webui.render import question_browser

st.title("100問の記録")
st.caption("評価用の100問を一括で実行したときの記録です。回答の正誤判定は含みません。")

group = benchmark_group()
rows = list_questions(group) if group else []
if not rows:
    st.info("100問の記録が見つかりません（portfolio/data/questions.js）。")
    st.stop()

st.caption(f"実行: {group.key} ・ {len(rows)}問")
question_browser(group, rows, index_label="index")
