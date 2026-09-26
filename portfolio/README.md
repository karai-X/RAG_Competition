# Drive Trace

社内共有ドライブを調べて質問に答えるエージェントRAG（このリポジトリの `rag_cli.py`）が、
100問をどの経路でたどったかを読むための静的Webアプリです。

次のことを説明します。

- **コンペについて** — 何が渡され、何を求められたか（最初に開くページ）
- **実装** — システム構成、前処理でやっていることと工夫、エージェントのループ、用意したツール
- **探索先の構成** — エージェントが読む `share/共有ドライブ/` がどう分かれていて、その分け方が検索と
  プロンプト構成にどう効いているか
- **設問別の経路** — 100問それぞれについて、最初のツール呼び出しから回答の確定までの全ステップ
- **情報源を見つける工夫** — 文章になっていない情報（セルの塗り色、図の中の位置、docxのページ、
  暗号化された契約書）を、答えの根拠として使える形に変えるためにやったこと

その場で新しい質問を実行するのは、リポジトリの Web UI（`streamlit_app.py`）の役割です。
このページは記録を読むだけで、サーバーを持ちません。

## 公開する — GitHub Pages

`.github/workflows/pages.yml` が、`main` への push のたびに
`index.html` / `app.css` / `app.js` / `data/` / `materials/` だけを GitHub Pages へ載せます
（`build_data.py` などは公開しません）。

初回だけ、リポジトリの Settings → Pages → Build and deployment の Source を
**GitHub Actions** にしてください。公開URLは https://karai-x.github.io/RAG_Competition/ です（ユーザー名は小文字になります）。
無料プランの GitHub Pages は公開リポジトリでだけ使えます。

## 手元で見る

ビルド工程も依存パッケージもありません。

```powershell
start portfolio\index.html
```

GitHub Pages と同じく `http://` で見るなら、確認用のサーバーを使います
（ブラウザにキャッシュさせないので、データを作り直したあとも F5 だけで最新になります）。

```powershell
uv run python portfolio/serve.py        # → http://127.0.0.1:8000/
```

Webフォントは Google Fonts から読みますが、オフラインでも代替フォントで問題なく表示されます。

## 構成

```
portfolio/
  index.html        ページ本体（解説の文章はここに直接書いてある）
  app.css           配色・タイポグラフィ・レイアウト
  app.js            データの集計と描画。外部ライブラリなし
  build_data.py     実行ログ + カタログ → data/questions.js・data/corpus.js
  build_explain.py  explain/*.json → data/explain.js と materials/（資料の画像を書き出す）
  serve.py          手元確認用のサーバ（キャッシュさせない。公開しない）
  explain/          設問ごとの解説の原稿（q0.json 〜 q99.json）
  materials/        解説に載せる資料の画像（qNN/k.webp）
  data/
    questions.js    window.RAG_RUN     … 100問の経路（約1.1 MB）
    corpus.js       window.RAG_CORPUS  … 403ファイルのカタログ（約130 KB）
    explain.js      window.RAG_EXPLAIN … 設問ごとの解説（約250 KB）
```

`data/*.js` は JSON ではなく `window.RAG_* = {...}` を代入する **.js** です。
`file://` で開いたときに `fetch()` が同一オリジンポリシーで弾かれるのを避けるためで、
これにより「クローンしてダブルクリック」で動きます。

## データを作り直す

前処理と回答実行を済ませたリポジトリで、次を実行します。

```powershell
uv run python portfolio/build_data.py --run-id csv-20260830-072745
```

読むもの:

- `logs/<run-id>/q*.json` — 1問ごとの回答・根拠・確信度・停止理由・使用量・トレース
- `logs/<run-id>/manifest.json` — モデル・生成設定・実行集計
- `artifacts/catalog/catalog.jsonl` — 全ファイルの台帳（形式・ページ数・版違い・特記事項）

書くもの: `portfolio/data/questions.js` と `portfolio/data/corpus.js` だけ。

資料そのもの（`share/` 配下）と、切り詰めていない完全経路（`logs/<run-id>/chain/`）は
**含めません**。アプリに載るのは、トレース中のツール引数と結果の冒頭（表示用に切り詰め済み）です。

## 注意

- 経路は実行の記録です。回答が正しいかどうかの判定は含みません。
- コーパスは架空の企業・案件で構成された評価用データです。
