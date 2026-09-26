"""解説ページを手元で確認するためのサーバー（キャッシュさせない）。

  uv run python portfolio/serve.py            # → http://127.0.0.1:8000/

Python 標準の http.server はキャッシュの指示を送らないため、ブラウザが古い app.js や
data/*.js を使い続けることがある。確認用なので、毎回読み直させる。
"""
from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args) -> None:          # 1リクエスト1行のログは出さない
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    handler = functools.partial(NoCacheHandler, directory=str(HERE))
    server = ThreadingHTTPServer(("127.0.0.1", a.port), handler)
    print(f"http://127.0.0.1:{a.port}/  （停止は Ctrl+C）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
