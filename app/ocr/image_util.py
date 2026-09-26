"""画像の正規化。

**透過PNGは白背景に合成する**。matplotlib 等の「暗い文字 + 透明背景」は、
合成しないとビューア側の暗背景で文字が潰れて読めない。
"""
from __future__ import annotations

import io


def normalize(img_bytes: bytes, max_edge: int = 1568,
              fmt: str = "PNG", quality: int = 88) -> bytes:
    from PIL import Image, UnidentifiedImageError
    try:
        im = Image.open(io.BytesIO(img_bytes))
        im.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return img_bytes
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    if max(im.size) > max_edge:
        r = max_edge / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))))
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        im.save(buf, format="JPEG", quality=quality)
    else:
        im.save(buf, format="PNG")
    return buf.getvalue()


def crop_region(img_bytes: bytes, region: str) -> tuple[bytes, str | None]:
    """'x0,y0,x1,y1'（0-1の割合）で切り出す。細かい文字の拡大読み取り用。"""
    try:
        x0, y0, x1, y1 = (float(v) for v in region.split(","))
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError
    except ValueError:
        return img_bytes, ("[エラー] region は 'x0,y0,x1,y1'（0-1の割合, "
                           "x0<x1, y0<y1）で指定してください")
    from PIL import Image, UnidentifiedImageError
    try:
        im = Image.open(io.BytesIO(img_bytes))
        im.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return img_bytes, f"[エラー] 画像を開けません: {type(e).__name__}"
    box = (int(x0 * im.width), int(y0 * im.height),
           int(x1 * im.width), int(y1 * im.height))
    buf = io.BytesIO()
    im.crop(box).save(buf, format="PNG")
    return buf.getvalue(), None
