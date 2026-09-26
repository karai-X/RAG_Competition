"""ページ→検索チャンク分割。

ページ / シート / スライド / セルを基本単位にし、長いものだけ段落境界で割る。
**表は行境界でしか切らない**（行の途中で割ると値と列名が離れて検索が壊れる）。
"""
from __future__ import annotations

import functools

from app.extract.models import ExtractedDoc


@functools.lru_cache(maxsize=1)
def _enc():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_enc().encode(text, disallowed_special=()))


def split_text(text: str, target: int, overlap: int) -> list[str]:
    if n_tokens(text) <= target * 1.3:
        return [text]
    chunks: list[str] = []
    buf: list[str] = []
    buf_tok = 0
    for para in text.split("\n\n"):
        pt = n_tokens(para)
        if pt > target * 1.5:
            # 巨大段落（表など）は行単位で
            for line in para.split("\n"):
                lt = n_tokens(line)
                if buf_tok + lt > target and buf:
                    chunks.append("\n".join(buf))
                    keep = buf[-2:] if overlap else []
                    buf = list(keep)
                    buf_tok = sum(n_tokens(x) for x in buf)
                buf.append(line)
                buf_tok += lt
            continue
        if buf_tok + pt > target and buf:
            chunks.append("\n\n".join(buf))
            buf, buf_tok = [], 0
        buf.append(para)
        buf_tok += pt
    if buf:
        chunks.append("\n\n".join(buf))
    return [c for c in chunks if c.strip()]


def make_chunks(doc: ExtractedDoc, target: int = 600,
                overlap: int = 100) -> list[dict]:
    out = []
    for page in doc.pages:
        if not page.text.strip():
            continue
        # 検索ヒット時に「どのファイルか」が分かるよう見出しを添える
        header = f"[{doc.relpath}"
        if page.label:
            header += f" / {page.label}"
        header += f" p{page.no}]"
        for j, piece in enumerate(split_text(page.text, target, overlap)):
            out.append({
                "chunk_id": f"{doc.doc_id}:{page.no}:{j}",
                "doc_id": doc.doc_id, "relpath": doc.relpath,
                "mount": doc.mount, "project": doc.project,
                "category": doc.category, "filetype": doc.filetype,
                "page_no": page.no, "label": page.label,
                "text": f"{header}\n{piece}",
            })
    return out
