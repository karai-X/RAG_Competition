"""固定データソース `share/共有ドライブ` のマウント表現。

配布先の絶対パスには依存せず、`app.config.REPO_DIR` からの相対位置を
Config が解決する。旧来の追加マウントは使用せず、このルートだけを読む。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import CONFIG


@dataclass(frozen=True)
class Mount:
    path: str        # 実在する絶対パス（open用。NFC正規化しない）
    mount_as: str    # 名前空間上の接頭辞（NFC。"" は既定コーパス）
    added_at: str = ""

    @property
    def root(self) -> Path:
        return Path(self.path)

    @property
    def label(self) -> str:
        return self.mount_as or self.root.name


def default_mount() -> Mount:
    return Mount(path=str(CONFIG.corpus_dir), mount_as="",
                 added_at="(default)")


def load_mounts() -> list[Mount]:
    """データソースは常に1件だけ。"""
    return [default_mount()]
