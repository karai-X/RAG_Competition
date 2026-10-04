# Agentic RAG — 共有ドライブ調査エージェント

社内共有ドライブの資料を前処理し、自然言語の質問へ根拠付きで回答するエージェントRAGのCLIです。
提案書・契約書・スケジュール・分析ノートブック・会議録・最終報告といった実務の資料群を対象に、
**どのファイルのどこを見て答えを出したか**を経路として残します。

Streamlit の Web UI（[`streamlit_app.py`](streamlit_app.py)）では、新しい質問を投げる、
投げた質問と評価用100問の回答経路をツールの引数・戻り値つきで読む、APIキーと画像読み取りエンジンを設定する、
ができます。

**解説ページ: https://karai-x.github.io/RAG_Competition/**

コンペの内容、システムの実装と前処理、100問それぞれの解き方、情報源を見つけるための工夫をまとめています。まずはこちらをご覧ください。

---

## リポジトリ構成

```
.
├── .github/workflows/pages.yml  portfolio/ を GitHub Pages へ公開
├── LICENSE                 MIT（コードのみ。末尾の「ライセンス」を参照）
├── rag_cli.py              CLI入口（preprocess / status / answer-csv）
├── streamlit_app.py        Web UI入口
├── start_app.bat           Web UIの起動（ダブルクリック可）
├── webui/                  Web UIの実装
├── app/
│   ├── config.py           全設定の単一入口（環境変数 > デフォルト）
│   ├── corpus/             走査・md5台帳・増分判定
│   ├── extract/            文書 → 書式つきMarkdown（EMF・グラフ・docxのページ・暗号化を含む）
│   ├── ocr/                画像・スキャンページのOCR
│   ├── index/              検索索引（BM25 + 密ベクトル + リランク）
│   ├── agent/              探索ループ（systemプロンプト・9ツール・サンドボックス）
│   ├── answer/             回答の最終整形（LLMを使わない決定的処理）
│   ├── run/                バッチ実行と経路記録
│   └── submit/             回答CSVの形式検証とzip化
├── portfolio/              解説ページ（GitHub Pages で公開）
├── share/共有ドライブ/       データソース。ここだけを読む（同梱）
├── artifacts/              抽出・OCR・索引の成果物（前処理済みのものを同梱）
├── logs/                   回答と経路（.gitignore 対象）
└── jobs/                   CLI前処理の状態（.gitignore 対象）
```

資料（`share/`）と前処理済みの成果物（`artifacts/`）はリポジトリに同梱しています。
クローンして「セットアップ」を済ませれば、前処理をせずに Web UI から質問できます。
同梱しないのは `.env`（APIキー）、`logs/`、`jobs/`、`artifacts/embed_cache/`（前処理をやり直すときだけ使う埋め込みのキャッシュ）です。
資料を含むため、リポジトリは約310MBあります。
`share/` と `artifacts/` は `.gitattributes` で改行コードの変換を止めています（変換されると md5 が変わり、前処理の台帳と食い違うため）。

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

フォルダ構成と案件×工程のファイル分布は、解説ページの[「探索先の構成」](https://karai-x.github.io/RAG_Competition/#corpus)で見られます。

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

docxのページ位置は、Wordが保存した改ページ情報があればそれを使い、無ければWord（使えなければLibreOffice）で組版して決めます。
どちらでも決められなければ、前処理を停止します。

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
