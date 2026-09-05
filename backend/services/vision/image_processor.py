"""
Satellite Image Preprocessing & Tiling Utility
Handles EXIF orientation, Lanczos resizing, format normalization, and optional 2x2 sub-grid tiling.
"""

import io
import base64
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image, ImageOps

MAX_VISION_IMAGE_DIMENSION = 3072


def decode_data_url(data_url: str) -> Tuple[bytes, str]:
    """Decodes a base64 data URL into raw bytes and MIME type."""
    if not data_url:
        raise ValueError("Empty image data provided.")

    if data_url.startswith("data:"):
        comma_idx = data_url.find(",")
        if comma_idx == -1:
            raise ValueError("Malformed data URL (missing comma).")
        header = data_url[:comma_idx]
        b64_data = data_url[comma_idx + 1 :]

        mime = "image/jpeg"
        if ";" in header:
            mime = header.split(";")[0].replace("data:", "").strip()
        elif header:
            mime = header.replace("data:", "").strip()

        img_bytes = base64.b64decode(b64_data)
        return img_bytes, mime
    else:
        # Raw base64 string
        img_bytes = base64.b64decode(data_url)
        return img_bytes, "image/jpeg"


def encode_image_base64(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Encodes raw bytes to a standard data URL."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def preprocess_image(
    image_bytes: bytes,
    max_dimension: int = MAX_VISION_IMAGE_DIMENSION,
) -> Tuple[bytes, str, Tuple[int, int]]:
    """
    Normalizes orientation, converts to RGB, and bounds maximum dimensions
    using high-quality Lanczos resampling without resizing already-small images.
    Returns: (processed_bytes, mime_type, (width, height))
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Unable to read image bytes: {e}")

    # 1. Correct EXIF orientation
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # 2. Normalize color mode to RGB
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    orig_w, orig_h = img.size

    # 3. Downscale only if exceeding max_dimension
    if max(orig_w, orig_h) > max_dimension:
        if orig_w >= orig_h:
            new_w = max_dimension
            new_h = int(orig_h * (max_dimension / orig_w))
        else:
            new_h = max_dimension
            new_w = int(orig_w * (max_dimension / orig_h))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    final_w, final_h = img.size

    out_buf = io.BytesIO()
    img.save(out_buf, format="JPEG", quality=92, optimize=True)
    return out_buf.getvalue(), "image/jpeg", (final_w, final_h)


def generate_tiles(
    image_bytes: bytes,
    grid: Tuple[int, int] = (2, 2),
) -> List[Dict[str, Any]]:
    """
    Splits image into a 2x2 grid (top-left, top-right, bottom-left, bottom-right).
    Returns list of dicts with:
      - 'label': spatial quadrant name
      - 'bytes': cropped JPEG bytes
      - 'mime': 'image/jpeg'
      - 'box': [x1, y1, x2, y2]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    quadrant_names = [
        ["top-left", "top-right"],
        ["bottom-left", "bottom-right"],
    ]

    tiles: List[Dict[str, Any]] = []
    rows, cols = grid

    tile_w = w // cols
    tile_h = h // rows

    for r in range(rows):
        for c in range(cols):
            x1 = c * tile_w
            y1 = r * tile_h
            x2 = (c + 1) * tile_w if c < cols - 1 else w
            y2 = (r + 1) * tile_h if r < rows - 1 else h

            crop = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=90, optimize=True)

            label = quadrant_names[r][c] if r < len(quadrant_names) and c < len(quadrant_names[r]) else f"tile-{r}-{c}"

            tiles.append({
                "label": label,
                "bytes": buf.getvalue(),
                "mime": "image/jpeg",
                "box": [x1, y1, x2, y2],
            })

    return tiles
