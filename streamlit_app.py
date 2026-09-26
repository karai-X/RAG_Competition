"""Web UI の入口。

  uv run streamlit run streamlit_app.py      # → http://127.0.0.1:8501/

画面は4つ: 質問する / 回答経路（Web UI からの質問） / 100問の記録 / 設定。
前処理済みの artifacts/ と share/ の資料を同梱して配布する前提で、前処理の画面は持たない。

**127.0.0.1 にだけ bind する（.streamlit/config.toml）。** エージェントは run_python で
モデルが書いたコードを実行するため、外部から到達できるアドレスへ公開しない。
"""
import streamlit as st

st.set_page_config(page_title="Drive Trace", page_icon=":material/travel_explore:",
                   layout="wide")

st.navigation([
    st.Page("webui/views/ask.py", title="質問する", icon=":material/chat:", default=True),
    st.Page("webui/views/history.py", title="回答経路", icon=":material/route:"),
    st.Page("webui/views/benchmark.py", title="100問の記録", icon=":material/fact_check:"),
    st.Page("webui/views/settings.py", title="設定", icon=":material/settings:"),
]).run()
