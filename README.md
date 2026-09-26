# Agentic RAG — 共有ドライブ調査エージェント

社内共有ドライブの資料を前処理し、自然言語の質問へ根拠付きで回答するエージェントRAGのCLIです。
提案書・契約書・スケジュール・分析ノートブック・会議録・最終報告といった実務の資料群を対象に、
**どのファイルのどこを見て答えを出したか**を経路として残します。

Streamlit の Web UI（[`streamlit_app.py`](streamlit_app.py)）では、新しい質問を投げる、
投げた質問と評価用100問の回答経路をツールの引数・戻り値つきで読む、APIキーと画像読み取りエンジンを設定する、
ができます。

**解説ページ（GitHub Pages）**: https://karai-x.github.io/RAG_Competition/
コンペの概要、前処理、100問それぞれの解き方、情報源を見つけるための工夫を
[`portfolio/`](portfolio/) の静的ページにまとめています。公開の手順は [portfolio/README.md](portfolio/README.md) にあります。

---

## 何が難しく、何をしたか

このコーパスで答えの根拠になるのは、文章として書かれていない情報でした。

| 答えの在り処 | 素のテキスト抽出では | 対策 |
| --- | --- | --- |
| セルの塗り色・ハイライト・太字 | 消える | 書式を記法として本文へ焼き込み、色名でgrepできるようにした |
| 図・グラフ・スキャンページ | 読めない | 前処理でOCRを焼き切り、本文へ統合した |
| 画像として貼られた表（EMF） | セル値が存在しない | EMFレコードをバイナリから決定的に解析した |
| docxの「何ページ目か」 | ページの概念がない | Word本体に組版させて改ページを確定させた |
| パスワード保護された契約書 | 開けない | 社内規定の導出規則を読ませ、候補を組み立てさせた |

加えて、**ドライブ全体のツリーをsystemプロンプトへ常駐させています。**
その結果、100問のうちベクトル検索を呼んだのは1問だけで、残りは最初のツール呼び出しで
ファイルを名指ししました。設計の詳細と、それが効いた設問は
[経路ビューア](portfolio/)の「情報源を見つける工夫」で読めます。

---

## リポジトリ構成

```
.
├── .github/workflows/pages.yml  portfolio/ を GitHub Pages へ公開
├── LICENSE                 MIT（コードのみ。末尾の「ライセンス」を参照）
├── rag_cli.py              CLI入口（preprocess / status / answer-csv）
├── streamlit_app.py        Web UI入口（質問する / 回答経路 / 100問の記録 / 設定）
├── start_app.bat           Web UIの起動（ダブルクリック可）
├── webui/                  Web UIの実装
│   ├── core.py             前提チェック・1問のバックグラウンド実行・経路の読み出し
│   ├── settings_store.py   設定画面の保存先（.env）
│   ├── render.py           回答と経路の描画
│   └── views/              ask.py / history.py / benchmark.py / settings.py
├── app/
│   ├── config.py           全設定の単一入口（環境変数 > デフォルト）
│   ├── corpus/             走査・台帳・増分判定
│   │   ├── walk.py         全ファイルの列挙。NFC/NFD混在パスの二重保持
│   │   ├── manifest.py     md5で新規/変更/削除を判定する
│   │   └── mounts.py       データソース share/共有ドライブ の解決
│   ├── extract/            文書 → 書式つきMarkdown
│   │   ├── docx.py xlsx.py pptx.py pdf.py notebook.py plain.py
│   │   ├── markup.py       書式アノテーションの単一定義
│   │   ├── pagemap.py      docxのページ位置確定（Word → LibreOffice → 停止）
│   │   ├── emf.py          EMF埋め込み表の決定的解析
│   │   ├── ooxml_chart.py  グラフのキャッシュ値から表を復元
│   │   ├── encrypted.py    暗号化Officeの検出と復号
│   │   └── catalog.py      catalog.jsonl とツリーの生成
│   ├── ocr/                画像・スキャンページのOCR（md5キャッシュ、冪等）
│   │   ├── targets.py      OCR対象の列挙（直置き画像/埋め込み/スキャンPDF）
│   │   ├── runner.py       並列バッチ実行と本文への統合
│   │   └── ask.py          回答中に画像へ質問する経路（空間関係の検証つき）
│   ├── index/              検索索引
│   │   ├── tokenize_ja.py  SudachiPy mode C。識別子を壊さない
│   │   ├── dense.py        ruri-v3-310m による密ベクトル
│   │   ├── search.py       BM25 + 密ベクトル → RRF融合 → リランク
│   │   └── build.py        チャンク化と索引構築
│   ├── agent/              探索ループ
│   │   ├── prompts.py      systemプロンプト（方針 + ツリー + 横断参照文書）
│   │   ├── tools.py        9ツールの定義と実装
│   │   ├── graph.py        LangGraph の agent ↔ tools ループ
│   │   ├── sandbox.py      run_python の読み取り専用サンドボックス
│   │   └── backend.py      LLMバックエンド抽象
│   ├── answer/             正規化・裁定・最終整形（LLMを使わない決定的処理）
│   ├── run/                バッチ実行と経路記録
│   │   ├── runner.py       質問CSV → 各問実行 → predictions.csv
│   │   └── chain.py        切り詰めない完全経路のJSONL記録
│   └── submit/             回答CSVの形式検証とzip化
├── portfolio/              解説ページ（GitHub Pages で公開。100問の記録を同梱）
├── share/共有ドライブ/       データソース。ここだけを読む（同梱）
├── artifacts/              抽出・OCR・索引の成果物（前処理済みのものを同梱）
├── logs/                   回答と経路（.gitignore 対象）
└── jobs/                   CLI前処理の状態（.gitignore 対象）
```

資料（`share/`）と前処理済みの成果物（`artifacts/`）はリポジトリに同梱しています。
クローンして「セットアップ」を済ませれば、前処理をせずに Web UI から質問できます。
同梱しないのは `.env`（APIキー）、`logs/`、`jobs/`、`artifacts/embed_cache/`（前処理をやり直すときだけ使う埋め込みのキャッシュ）です。
資料を含むため、リポジトリは約310MBあります。

---

## クイックスタート — clone して質問するまで

資料と前処理済みの索引は同梱しているので、Gemini の APIキーがあれば前処理なしで質問できます（Windows）。

```powershell
winget install --exact --id astral-sh.uv   # uv が無ければ。入れたら PowerShell を開き直す
git clone https://github.com/karai-X/RAG_Competition.git
cd RAG_Competition
uv sync --frozen                           # Python 3.13.3 と依存関係をそろえる
uv run streamlit run streamlit_app.py      # → http://127.0.0.1:8501/
```

1. 「設定」画面で Gemini APIキーを入力して保存する。
2. 画像読み取りエンジンを選ぶ。既定は Claude Code CLI（サブスクリプション）で、`claude` にログイン済みならそのまま使える。
   Codex CLI も選べる（どちらも APIキーでの従量課金に切り替えられる）。
   どちらも使わない場合は「Gemini」を選ぶと、同じ Gemini の APIキーで動く。
   Claude Code / Codex のインストールとログインは、下の「セットアップ」の1〜2を参照。
3. 「質問する」画面で質問を入力して実行する。最初の1問は、索引の読み込みと、
   密ベクトル・リランカのモデルのダウンロード（Hugging Face）に時間がかかる。

前処理をやり直す場合や、Codex / Claude Code を使う場合は、下の「セットアップ」を参照してください。

---

## セットアップ

Windowsで動かします。PowerShellを開いてリポジトリのフォルダへ移動してください。
すべてのパスは `rag_cli.py` を基準に決まるので、配置場所の絶対パスには依存しません。

### 1. 必要なCLIのインストール

環境管理に `uv`、画像OCRにCodex CLI、回答時の画像検証にClaude CLIを使います。
Codex CLI は前処理（画像OCR）をやり直すときだけ要ります。
Claude CLI は、画像読み取りエンジンを Gemini にするなら要りません。
Claude CLIがWindowsで使用するGit for Windowsも同時に用意します。

```powershell
winget install --exact --id astral-sh.uv
winget install --exact --id OpenAI.Codex
winget install --exact --id Anthropic.ClaudeCode
winget install --exact --id Git.Git
```

インストール後はPowerShellを開き直し、各コマンドを認識できることを確認します。

```powershell
uv --version
codex --version
claude --version
```

### 2. Codex CLI／Claude CLIへのログイン

```powershell
codex login
codex login status
claude auth login
claude auth status --text
```

CLIの認証情報は各利用者のユーザーディレクトリに保存され、リポジトリや `.env` には入りません。
詳細は [Codex CLI公式ドキュメント](https://developers.openai.com/codex/cli/) と
[Claude Code公式ドキュメント](https://code.claude.com/docs/en/overview) を参照してください。

### 3. Python環境と依存関係

検証時と同じ Python 3.13.3 と依存関係を `uv.lock` で固定しています。

```powershell
uv python install 3.13.3
uv sync --frozen
```

仮想環境 `.venv` は `uv` が自動で作成・管理するため、手動で作る必要はありません。

### 4. APIキー

APIキーは Web UI の設定画面で入力します（「使い方」の4を参照）。
入力したキーは `.env` に保存され、CLI（`rag_cli.py`）もそこから読みます。
Codex CLIとClaude CLIをサブスクリプションで使う場合、キーは要りません。

初回の索引構築では、次のモデルがHugging Faceから取得されます。

- `cl-nagoya/ruri-v3-310m`（密ベクトル）
- `hotchpotch/japanese-reranker-cross-encoder-small-v1`（リランカ）

---

## 使い方

### 1. 資料

データソースとして読むのは次の2フォルダだけです。どちらもリポジトリに同梱しています。

- `share/共有ドライブ/プロジェクト/` — 案件フォルダ
- `share/共有ドライブ/社内管理/` — 横断参照フォルダ

### 2. 前処理

同梱の `artifacts/` は前処理済みなので、そのまま質問できます。
資料を追加・変更したときだけ、次を実行します。

```powershell
uv run python rag_cli.py preprocess                  # 全資料
uv run python rag_cli.py preprocess --skip-processed # 差分だけ
uv run python rag_cli.py status                      # 状態の確認だけ
```

文書抽出・画像OCR・OCR結果の本文統合・検索索引構築が含まれます。画像OCRは必須で、
無効化するオプションはありません。抽出やOCRで致命的なエラーが起きた場合は、
索引を構築せず停止します。

### 3. 質問CSVへ一括回答

CSVは1行目をヘッダー、A列を `index`、B列を `question` とします。
C列が無ければ `answer` 列を自動的に追加します。

```powershell
uv run python rag_cli.py answer-csv "share/質問回答/questions_test.csv"
```

既定では1問終わるごとに入力CSVのC列へ保存するので、途中で中断しても完了分は残ります。
元ファイルを残す場合は `--output` を指定します。

```powershell
uv run python rag_cli.py answer-csv "share/質問回答/questions_test.csv" `
  --output "share/質問回答/questions_test_answered.csv"
```

### 4. Web UI

```powershell
uv run streamlit run streamlit_app.py    # → http://127.0.0.1:8501/
```

`start_app.bat` をダブルクリックしても同じです。停止は Ctrl+C。
リポジトリのフォルダで起動してください（`.streamlit/config.toml` をそこから読みます）。

| 画面 | できること |
| --- | --- |
| 質問する | 1問を実行し、ツールを呼ぶたびに経路が伸びていくのを見る。画面を移動しても実行は続く |
| 回答経路 | Web UIから投げた質問を選び、回答・根拠・ツール呼び出しを読む |
| 100問の記録 | 評価用100問の一括実行（`csv-20260830-072745`）の回答と経路を読む |
| 設定 | Gemini APIキー、画像読み取りエンジン（Claude Code / Codex / Gemini）・課金方法・effortを設定し、接続を試す（モデルは固定） |

画像読み取りの課金方法は、CLIごとに次の2つから選べます。

- **サブスクリプション** — `claude` / `codex login` でログインしたアカウントを使う。
  子プロセスには APIキーの環境変数を渡しません
- **APIキー** — `ANTHROPIC_API_KEY` / `CODEX_API_KEY` を子プロセスに渡して従量課金で使う

設定は `.env` に保存され、起動し直さなくても次の質問から反映されます。
質問は1問ずつ実行し、記録は `logs/live-<日時>/chain/q0.jsonl` に残ります。
100問の記録は `logs/` に完全な経路があればそれを、無ければ同梱の要約版（`portfolio/data/questions.js`）を表示します。

> **Web UI は `127.0.0.1` にだけ bind します（`.streamlit/config.toml`）。**
> エージェントは `run_python` でモデルが書いたコードを実行するため、
> 外部から到達できるアドレスへ公開しないでください。

---

## 配布

リポジトリそのものが、前処理を済ませた配布物です。受け取った側では前処理は不要です。

| リポジトリに入っている | 入っていない |
| --- | --- |
| コード一式、`share/共有ドライブ/`（資料）、`artifacts/`（抽出・OCR・索引） | `.env`（APIキー）、`.venv/`、`logs/`、`jobs/`、`artifacts/embed_cache/` |

受け取った側は「セットアップ」の1〜3を済ませてから Web UI を起動し、設定画面で APIキーを入力します。
初回の質問では、カタログと検索索引の読み込みと、密ベクトル・リランカのモデルのダウンロードに時間がかかります。

- `share/` と `artifacts/` は `.gitattributes` で改行コードの変換を止めています。
  変換されるとファイルの md5 が変わり、前処理の台帳と食い違うためです。
- 暗号化ファイル2つの抽出結果は、前処理直後の状態（本文を読めず、decrypt で開くよう案内する状態）で入っています。
- `logs/` は入れていません。100問の記録の画面は、同梱の要約版（`portfolio/data/questions.js`）を表示します。
  `logs/csv-20260830-072745/` を置くと、切り詰めない完全な経路を読めます。

---

## 探索対象のフォルダ構成

`share/共有ドライブ/` は2つに分かれ、この分け方がそのまま検索フィルタと
プロンプト構成に対応します。

```
share/共有ドライブ/
├── プロジェクト/            案件フォルダ（10案件）
│   └── <案件名>/
│       ├── 00.提案/         提案書・調査資料
│       ├── 01.契約/         契約書（暗号化されているものがある）
│       ├── 02.計画/         スケジュール（ガント期間はセルの塗り色）
│       ├── 03.データ/       学習データとカラム説明
│       ├── 04.分析/         分析プロジェクト一式
│       │   ├── analysis_outputs/    metrics.json / experiments/
│       │   └── analysis_project/    src/ notebooks/ reports/figures/ configs/
│       ├── 05.会議/         会議録/ と 報告資料/
│       └── 06.報告書/       最終報告（old版と並ぶことがある）
└── 社内管理/                横断参照フォルダ（案件に属さない）
    ├── 社内用語集.docx
    ├── データアステル社内規定_パスワード導出規則.docx
    ├── データアステル社内管理_決裁基準.md
    └── 座席表.pptx
```

**番号は工程の順序です。** 「提案時の見込みと最終請求の差額は」のような問いは、
この番号の両端を突き合わせることになります。
マウント相対パスの位置から `project` / `category` を導出しているため、
検索の絞り込みにも、サンドボックスの `files(project=, category=)` にもそのまま使えます。

`社内管理/` の4文書はどの質問からも参照されうるので、検索させずに
**systemプロンプトへ全文で常駐させています**。略語の展開やパスワードの導出規則は
「検索して見つける」ものではなく、調査を始める前から手元にあるべき前提だからです。

同梱の記録では403ファイル（`py` 100 / `png` 54 / `docx` 46 / `md` 31 / `csv` 29 /
`json` 28 / `pdf` 28 / `pptx` 25 / `xlsx` 20 / `ipynb` 11 ほか）を対象にしています。
案件×工程の分布とファイル単位の一覧は、経路ビューアの「探索先の構成」で見られます。

---

## 保存先

以下はすべて `rag_cli.py` と同じディレクトリを基準にした相対パスです。

| 内容 | 既定パス |
| --- | --- |
| 共有ドライブ資料 | `share/共有ドライブ/` |
| 抽出・OCR・索引 | `artifacts/` |
| 回答と完全経路 | `logs/` |
| CLI前処理状態 | `jobs/` |

主な回答ログ:

- `logs/<run-id>/q<index>.json` — 回答、根拠、確信度、停止理由、モデル、使用量、トレース
- `logs/<run-id>/chain/q<index>.jsonl` — LLM要求、応答、ツール引数・結果、最終回答の時系列
- `logs/<run-id>/manifest.json` — モデル、生成設定、実行集計

経路を読み直すコマンドもあります。

```powershell
uv run python -m app.run.chain --run-id <id> --index <n>   # 1問の経路
uv run python -m app.run.chain --run-id <id> --summary     # 全問を1行ずつ
```

---

## 外部ツールと固定設定

| ツール | 用途 |
| --- | --- |
| Microsoft Word | docxに保存済み改ページが無い場合のページ組版 |
| LibreOffice | Wordを使えない場合の組版フォールバック |
| Codex CLI | 前処理時の画像OCR |
| Claude CLI | 回答中の画像に対する追加検証 |

docxのページ位置は、使用するフォント・用紙サイズ・余白・組版エンジンによって変わります。
文書内にWordが保存した改ページ情報があればそれを使い、無ければWordで再組版します。
Wordを使えない場合はLibreOfficeで試みますが、組版結果がWordと一致するとは限らないため、
既存のページ情報を持つ文書との照合が設定した一致率を満たす場合だけ採用します。
どちらも使えない、または組版に失敗した場合は、対象を含むエラーを表示して前処理を停止します。

### CLIのモデル・effort

| CLI | 用途 | 固定モデル | 固定effort |
| --- | --- | --- | --- |
| Codex CLI | 前処理時の画像OCR | `gpt-5.6-sol` | `xhigh` |
| Claude CLI | 回答時の画像追加検証 | `fable` | `high` |

Codex CLIには `--ignore-user-config` を渡すため、利用者の設定ファイルにある
別モデルやeffortには影響されません。既定値はリポジトリ内で固定しています。
回答時の画像読み取りについては、Web UI の設定画面でエンジン・課金方法・effortを変えられます
（`.env` の `RAG_MODEL_IMAGE` などに保存されます）。CLIのモデルは固定で、画面からも `.env` からも変えられません。
前処理時のOCRの設定は変わりません。

---

## 注意

- 同梱のコーパスと質問は、**架空の企業・案件で構成された評価用データ**です。
- 回答経路の画面は実行の記録を表示するだけで、回答の正誤判定は含みません。
- `.env` は `.gitignore` 対象です。APIキーをコミットしないでください。

---

## ライセンス

コードは [MIT License](LICENSE) です。
ただし、`share/` の資料、そこから作った `artifacts/`、`portfolio/data/` と `portfolio/materials/` に含まれる質問文・資料の抜粋・資料の画像は、
[SIGNATE「AI Engineering Challenge ～煩雑な社内ドライブをハックせよ～」](https://user.competition.signate.jp/ja/competition/detail/?competition=098a4365f4514c2fa197d4e16548b3bb)
で提供されたデータに由来するもので、MIT License の対象外です。
