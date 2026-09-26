"""暗号化Officeファイルの検出と復号。

パスワードは実行時に導出し、導出済みの文字列をコードに書かない。
このモジュールは「候補を順に試して復号する」だけを担い、
候補の作り方（社内規定の読解）は agent 側の仕事にする。
"""
from __future__ import annotations

import io
from pathlib import Path

OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def is_ole_encrypted(path: str | Path) -> bool:
    """OOXML(zip) のはずが OLE/CFB になっている = パスワード保護。"""
    try:
        with open(path, "rb") as f:
            return f.read(8) == OLE_MAGIC
    except OSError:
        return False


def try_decrypt(path: str | Path, passwords: list[str],
                limit: int = 50) -> tuple[bytes | None, str | None]:
    """候補を順に試し、(復号バイト列, 成功したパスワード) を返す。"""
    import msoffcrypto
    for pw in passwords[:limit]:
        if not pw:
            continue
        try:
            with open(path, "rb") as f:
                of = msoffcrypto.OfficeFile(f)
                of.load_key(password=pw)
                buf = io.BytesIO()
                of.decrypt(buf)
            return buf.getvalue(), pw
        except Exception:
            # msoffcrypto は失敗時に多様な例外を投げる（不正PW/形式差）。
            # ここは「候補が違った」以上の意味を持たないので次へ進む。
            continue
    return None, None
