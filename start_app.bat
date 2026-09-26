@echo off
rem Web UI を起動する（ダブルクリック可）。停止はこのウィンドウで Ctrl+C。
cd /d "%~dp0"
uv run streamlit run streamlit_app.py
