"""画像OCRの共通プロンプトと出力スキーマ。

読み取れない要素を推測せず、空のまま返すよう制約する。
"""
from __future__ import annotations

OCR_PROMPT = """この画像は社内文書の一部です。写っている内容を漏れなく日本語で構造化してください。

厳守事項:
- **読み取れないもの・画像に書かれていないものは絶対に推測して補わない**。
  数値ラベルが無い箇所、潰れて読めない文字は空のままにする。
- 表は Markdown テーブルでそのまま再現する。セルが空なら空のままにする。
- 太字・下線・マーカー(ハイライト)・色付き文字があれば、次の記法で注記する:
  **太字** / <u>下線</u> / ==マーカー=={色名} / [c:色名]文字[/c]
- セルや図形が塗られている場合は、どの値がどの色で塗られているかを明記する。
- グラフ・図は figure_description に、種類・軸ラベル・凡例・目盛り・
  読み取れる数値を具体的に書く。読み取れない数値は書かない。
- 書き起こし以外の前置き・感想・説明は出力しない。
- すべて日本語で書く（原文が英語の識別子・列名などは原文のまま残す）。

confident は、画像の内容を確信を持って読み切れた場合のみ true にする。"""

OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {
            "type": "string",
            "description": "画像の種別。表 / グラフ / 図表 / スキャン文書 / 座席表 / 写真 / その他 のいずれか",
        },
        "markdown": {
            "type": "string",
            "description": "画像内のテキスト・表の書き起こし（Markdown、書式注記つき）。無ければ空文字列",
        },
        "figure_description": {
            "type": "string",
            "description": "グラフ・図の場合の説明（種類・軸・凡例・読み取れる数値）。無ければ空文字列",
        },
        "confident": {
            "type": "boolean",
            "description": "確信を持って読み切れたか",
        },
    },
    "required": ["content_type", "markdown", "figure_description",
                 "confident"],
}
