"""OCR結果のディスクキャッシュ。

**キーは画像の md5 + エンジン + プロンプト版**。
画像が変わらなければ再実行しないので、フォルダ追加や再インジェストのコストがゼロになる。
回答処理の前にバッチで埋めておく。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import CONFIG

# OCR結果とプロンプトの互換性を識別する版。
PROMPT_VERSION = "v1"

# 抽出文書へ差し込み済みかを判定するマーカー（二重追記の防止と health 検査で共用）
MARKER = "[画像読み取り]"


def applied_head(kind: str, ref: str) -> str:
    """差し込み済みかの判定に使う見出し。**画像ごとに固有**にする。

    同じページに複数画像がある場合も区別できるよう、画像参照を含める。
    """
    if kind == "asset" and ref:
        return f"{MARKER} {ref.split('/')[-1]}"
    return MARKER


@dataclass
class OcrResult:
    md5: str
    engine: str
    prompt_version: str = PROMPT_VERSION
    content_type: str = ""
    markdown: str = ""
    figure_description: str = ""
    layout: str = ""              # 配置図の並び・隣接・向き
    confident: bool = False
    error: str | None = None
    latency_sec: float = 0.0
    source: str = ""              # 由来（トレース用。キャッシュキーには含めない）

    @property
    def ok(self) -> bool:
        return self.error is None and bool(
            self.markdown.strip() or self.figure_description.strip()
            or self.layout.strip())

    def as_text(self) -> str:
        parts = []
        if self.content_type:
            parts.append(f"[画像種別: {self.content_type}]")
        if self.markdown.strip():
            parts.append(self.markdown.strip())
        if self.figure_description.strip():
            parts.append(f"[図の説明] {self.figure_description.strip()}")
        if self.layout.strip():
            parts.append(f"[配置] {self.layout.strip()}")
        if not self.confident:
            parts.append("[注: 読み取りに不確かな箇所あり]")
        return "\n".join(parts)


def _path(md5: str, engine: str) -> Path:
    return CONFIG.ocr_cache_dir / f"{md5}.{engine}.{PROMPT_VERSION}.json"


def get(md5: str, engine: str) -> OcrResult | None:
    p = _path(md5, engine)
    if not p.exists():
        return None
    try:
        return OcrResult(**json.loads(p.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def put(res: OcrResult) -> None:
    CONFIG.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
    p = _path(res.md5, res.engine)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(res), ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(p)


def stats() -> dict:
    files = list(CONFIG.ocr_cache_dir.glob(f"*.{PROMPT_VERSION}.json"))
    by_engine: dict[str, int] = {}
    for f in files:
        parts = f.name.split(".")
        if len(parts) >= 3:
            by_engine[parts[1]] = by_engine.get(parts[1], 0) + 1
    return {"total": len(files), "by_engine": by_engine}
