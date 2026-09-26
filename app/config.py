"""全設定の単一入口。

優先順位: 環境変数 > デフォルト。
設定値は個別の資料や質問に依存させない。

`.env` は import 時に一度だけ読み込む。APIキーは環境変数としてのみ扱い、
値をログ・トレース・サンドボックスへ渡さない。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """REPO_DIR/.env を環境変数へ。既存の環境変数は上書きしない。"""
    path = REPO_DIR / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _disable_external_tracing() -> None:
    """LangSmith等への外部トレース送信を停止する。"""
    for var in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING",
                "LANGSMITH_TRACING", "LANGCHAIN_ENDPOINT",
                "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
        os.environ.pop(var, None)
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["ANONYMIZED_TELEMETRY"] = "false"


_load_dotenv()
_disable_external_tracing()

# Web UI の設定画面から変えられる項目。RAG_<大文字> の環境変数で上書きする。
# CLI のモデル（claude_cli_model / codex_model）は再現性のため固定し、ここに含めない。
ENV_OVERRIDABLE = ("model_image", "claude_cli_effort", "claude_cli_auth",
                   "codex_reasoning_effort", "codex_auth")


@dataclass
class Config:
    # ---- パス ----
    # rag_cli.py と同じディレクトリへ artifacts / logs / jobs を置く。
    # 配布先の絶対パスには依存させない。
    rag_home: Path = field(default_factory=lambda: REPO_DIR)
    # 配布先に依存しないよう、CLI配置フォルダからの相対位置で固定する。
    # データソースは share/共有ドライブ 配下だけを読む。
    corpus_dir: Path = field(default_factory=lambda: Path(
        REPO_DIR / "share" / "共有ドライブ"))
    questions_csv: Path = field(default_factory=lambda: Path(
        os.environ.get("QUESTIONS_CSV",
                       REPO_DIR / "share" / "質問回答" / "questions_valid.csv")))

    # ---- モデル ----
    backend: str = "gemini"                       # gemini | claude | openai
    # 再現性のため -latest エイリアスは使わず、モデルIDを固定する。
    model_main: str = "gemini-3.7-flash"
    # 生成を決定的に寄せる設定。全リクエストに必ず付ける。
    # モデルの選択とは独立していて、兄弟モデルへ切り替わっても効く。
    # 注: Google 側のモデル更新や API 実装まで固定できるものではない。
    generation_seed: int = 42
    generation_temperature: float = 0.0
    # 図の読み取りに使う単発モデル。read_image が画像を見せるとき、
    # ここで指定したモデルにも同じ画像と質問を渡して読ませる。
    model_image: str = "claude_cli"               # claude_cli | codex | gemini
    claude_cli_model: str = "fable"
    # 回答時の画像検証は精度を優先しつつ、maxほど遅延を増やさない。
    claude_cli_effort: str = "high"
    claude_cli_timeout_sec: int = 900
    # CLI の課金経路。subscription はCLIのログイン（claude / codex login）を使い、
    # api は ANTHROPIC_API_KEY / CODEX_API_KEY を子プロセスへ渡す。
    claude_cli_auth: str = "subscription"         # subscription | api
    codex_auth: str = "subscription"              # subscription | api
    model_ocr: str = "codex"                      # codex | gemini
    # Codex CLI は --ignore-user-config で起動するため、OCR実行条件は
    # ユーザーの config.toml ではなくこの2値から明示的に渡す。
    codex_model: str = "gpt-5.6-sol"
    codex_reasoning_effort: str = "xhigh"
    embed_model: str = "cl-nagoya/ruri-v3-310m"
    rerank_model: str = "hotchpotch/japanese-reranker-cross-encoder-small-v1"

    # ---- 時間・並列 ----
    answer_budget_hours: float = 12.0             # ★ここ1箇所で 3h/12h を切替
    parallel_workers: int = 8
    # 画像検証を含む長時間処理を許容する。
    question_timeout_sec: int = 3600
    sandbox_timeout_sec: int = 120                # 横断走査を見込む
    codex_concurrency: int = 3
    codex_timeout_sec: int = 180
    request_timeout_sec: int = 120   # LLM APIのHTTPタイムアウト（未設定だとハングする）

    # ---- エージェント ----
    max_turns: int = 40
    max_tokens_per_turn: int = 2000
    tool_result_max_tokens: int = 4000

    # ---- 検索 ----
    search_top_k: int = 8
    rerank_candidates: int = 20
    rrf_k: int = 60
    chunk_target_tokens: int = 600
    chunk_overlap_tokens: int = 100

    # ---- 回答 ----
    answer_max_tokens: int = 1000                 # 回答の上限
    confidence_floor: float = 0.35
    missing_answer: str = "わかりません"

    # ---- 画像 ----
    image_max_edge_px: int = 1568

    # ---- docx のページ位置（app/extract/pagemap.py） ----
    soffice_path: str = ""              # 空なら自動探索。RAG_SOFFICE で上書き
    soffice_timeout_sec: int = 180
    # Word 本体に組版させるときの1ファイルあたりの見込み時間（バッチ全体の上限に使う）
    word_timeout_sec: int = 120
    # レンダラを採用する最低一致率。Word が改ページを記録している docx で照合する。
    # 0 なら照合結果によらず採用する（照合は診断として記録・表示のみ）。
    # 1 に近づけるほど厳しくなり、下回ればエラーとして前処理を停止する。
    pagemap_min_agreement: float = 0.8
    # 用紙サイズ(pgSz)を持たない docx を組版するときに仮定するページ設定。
    # OOXML では pgSz 省略時の用紙は「開いたアプリの既定」なので、どこかで決める必要がある。
    #
    # 既定値は一般的な用紙設定に基づく。
    #   A4 (210x297mm) … 日本で一般的な用紙規格
    #   余白 25.4mm    … Word の標準余白（1インチ）
    # 仮定を使用した場合は抽出本文に明記する。
    pagemap_assumed_paper: str = "A4"                       # A4 | Letter | B5 | A3 | Legal
    pagemap_assumed_margins_mm: str = "25.4,25.4,25.4,25.4"  # 上,下,左,右

    def __post_init__(self) -> None:
        self.apply_env()

    def apply_env(self) -> None:
        """環境変数による個別上書き(数値・文字列のみ)。

        Web UI の設定画面は .env と os.environ を書き換えたあとこれを呼び、
        起動し直さずに反映させる。
        """
        for key in ("backend", "model_main", "model_ocr", "embed_model",
                    "soffice_path", "pagemap_assumed_paper") + ENV_OVERRIDABLE:
            env = os.environ.get(f"RAG_{key.upper()}")
            if env:
                setattr(self, key, env)
        timeout_env = os.environ.get("RAG_CODEX_TIMEOUT_SEC")
        if timeout_env:
            self.codex_timeout_sec = int(timeout_env)
        concurrency_env = os.environ.get("RAG_CODEX_CONCURRENCY")
        if concurrency_env:
            self.codex_concurrency = int(concurrency_env)

    # ---- 派生パス(全てネイティブFS側) ----
    @property
    def artifacts_dir(self) -> Path:
        return self.rag_home / "artifacts"

    @property
    def extracted_dir(self) -> Path:
        return self.artifacts_dir / "extracted"

    @property
    def assets_dir(self) -> Path:
        return self.artifacts_dir / "assets"

    @property
    def catalog_dir(self) -> Path:
        return self.artifacts_dir / "catalog"

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "index"

    @property
    def ocr_cache_dir(self) -> Path:
        return self.artifacts_dir / "ocr_cache"

    @property
    def embed_cache_dir(self) -> Path:
        return self.artifacts_dir / "embed_cache"

    @property
    def logs_dir(self) -> Path:
        return self.rag_home / "logs"

    @property
    def manifest_file(self) -> Path:
        return self.artifacts_dir / "manifest.json"

    @property
    def preprocess_state_file(self) -> Path:
        return self.artifacts_dir / "preprocess_state.json"

    @property
    def checkpoints_db(self) -> Path:
        return self.rag_home / "checkpoints.sqlite"

    def ensure_dirs(self) -> None:
        for p in (self.extracted_dir, self.assets_dir, self.catalog_dir,
                  self.index_dir, self.ocr_cache_dir, self.embed_cache_dir,
                  self.logs_dir, self.corpus_dir,
                  self.corpus_dir / "プロジェクト",
                  self.corpus_dir / "社内管理"):
            p.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
