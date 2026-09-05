"""
Satellite Image Tiling Module for Small-Object Detection.
Enables high-resolution detection of small features (buildings, cars, boats, small structures)
without losing visual fidelity through full-scene downscaling.
"""

import os
import math
import logging
from typing import List, Dict, Tuple, Iterator, Optional, Any
from PIL import Image

logger = logging.getLogger("satquery.detection.tiler")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")


# =====================================================================
# CONFIGURABLE DEFAULT CONSTANTS
# =====================================================================
# Default tile size in pixels (1024x1024 captures fine satellite features while fitting VRAM)
TILE_SIZE: int = int(os.getenv("DINO_TILE_SIZE", "1024"))

# Overlap ratio between adjacent tiles (15% prevents boundary truncation)
TILE_OVERLAP: float = float(os.getenv("DINO_TILE_OVERLAP", "0.15"))

# Global toggle to enable/disable tiling
ENABLE_TILING: bool = os.getenv("DINO_ENABLE_TILING", "true").lower() in ("true", "1", "yes")

# Minimum image dimension required before tiling is activated
MIN_IMAGE_SIZE_FOR_TILING: int = int(os.getenv("DINO_MIN_IMAGE_SIZE_FOR_TILING", "1024"))

# Maximum number of tiles permitted to prevent runaway inference latency
MAX_TILES: int = int(os.getenv("DINO_MAX_TILES", "16"))


# =====================================================================
# TILING DECISION & GRID CALCULATION
# =====================================================================
def should_tile_image(
    image: Image.Image,
    min_size: int = MIN_IMAGE_SIZE_FOR_TILING,
    enable_tiling: bool = ENABLE_TILING,
) -> bool:
    """
    Determines whether an image warrants tiling.
    Small images (both width and height <= min_size) bypass tiling to avoid redundant overhead.
    """
    if not enable_tiling:
        return False
    width, height = image.size
    return width > min_size or height > min_size


def calculate_tile_grid(
    width: int,
    height: int,
    tile_size: int = TILE_SIZE,
    overlap: float = TILE_OVERLAP,
    max_tiles: int = MAX_TILES,
) -> List[Dict[str, int]]:
    """
    Calculates 2D tile offset coordinates [x_offset, y_offset, crop_width, crop_height]
    covering the entire image with overlap, without allocating image buffers in memory.

    If the natural grid would exceed max_tiles, adaptively increases tile_size
    to ensure full coverage within compute budget.
    """
    if width <= 0 or height <= 0:
        return []

    # If image is smaller than tile size in both dimensions, return single full tile
    if width <= tile_size and height <= tile_size:
        return [{
            "tile_id": "tile_0_0",
            "x_offset": 0,
            "y_offset": 0,
            "width": width,
            "height": height,
            "row": 0,
            "col": 0,
        }]

    # Convert overlap fraction to pixels
    overlap_px = int(tile_size * overlap) if overlap < 1.0 else int(overlap)
    overlap_px = max(0, min(overlap_px, tile_size // 2))
    step = max(1, tile_size - overlap_px)

    # Estimate preliminary tile counts
    cols_est = max(1, math.ceil((width - overlap_px) / float(step)))
    rows_est = max(1, math.ceil((height - overlap_px) / float(step)))
    total_est = cols_est * rows_est

    # Adaptive scaling if exceeding max_tiles
    effective_tile_size = tile_size
    if total_est > max_tiles:
        scale_factor = math.sqrt(float(total_est) / float(max_tiles))
        effective_tile_size = int(math.ceil(tile_size * scale_factor))
        effective_overlap_px = int(effective_tile_size * overlap) if overlap < 1.0 else int(overlap)
        effective_step = max(1, effective_tile_size - effective_overlap_px)
        logger.warning(
            f"[Tiler] Estimated tiles ({total_est}) exceeded max_tiles ({max_tiles}). "
            f"Adaptively expanded tile_size from {tile_size} to {effective_tile_size}px."
        )
    else:
        effective_overlap_px = overlap_px
        effective_step = step

    # Calculate precise tile bounding boxes
    tiles: List[Dict[str, int]] = []
    row = 0
    y = 0

    while y < height:
        col = 0
        x = 0
        bottom = min(y + effective_tile_size, height)
        top = max(0, bottom - effective_tile_size)

        while x < width:
            right = min(x + effective_tile_size, width)
            left = max(0, right - effective_tile_size)

            tiles.append({
                "tile_id": f"tile_{row}_{col}",
                "x_offset": left,
                "y_offset": top,
                "width": right - left,
                "height": bottom - top,
                "row": row,
                "col": col,
            })

            if right >= width:
                break
            x += effective_step
            col += 1

        if bottom >= height:
            break
        y += effective_step
        row += 1

        if len(tiles) >= max_tiles:
            logger.warning(f"[Tiler] Reached hard tile limit of {max_tiles} tiles.")
            break

    return tiles


# =====================================================================
# TILE GENERATOR (MEMORY SAFE)
# =====================================================================
def iter_tiles(
    image: Image.Image,
    tile_size: int = TILE_SIZE,
    overlap: float = TILE_OVERLAP,
    max_tiles: int = MAX_TILES,
    min_image_size: int = MIN_IMAGE_SIZE_FOR_TILING,
) -> Iterator[Dict[str, Any]]:
    """
    Memory-safe generator that crops and yields one tile at a time.
    Prevents holding multiple duplicate image copies in RAM simultaneously.
    """
    width, height = image.size

    # Direct mode for small images
    if not should_tile_image(image, min_size=min_image_size):
        yield {
            "tile_id": "tile_0_0",
            "image": image,
            "x_offset": 0,
            "y_offset": 0,
            "width": width,
            "height": height,
            "row": 0,
            "col": 0,
        }
        return

    grid = calculate_tile_grid(
        width=width,
        height=height,
        tile_size=tile_size,
        overlap=overlap,
        max_tiles=max_tiles,
    )

    for cell in grid:
        left = cell["x_offset"]
        top = cell["y_offset"]
        right = left + cell["width"]
        bottom = top + cell["height"]

        # Crop on demand
        crop = image.crop((left, top, right, bottom))

        yield {
            "tile_id": cell["tile_id"],
            "image": crop,
            "x_offset": left,
            "y_offset": top,
            "width": cell["width"],
            "height": cell["height"],
            "row": cell["row"],
            "col": cell["col"],
        }


def generate_tiles(
    image: Image.Image,
    tile_size: int = TILE_SIZE,
    overlap: float = TILE_OVERLAP,
    max_tiles: int = MAX_TILES,
    min_image_size: int = MIN_IMAGE_SIZE_FOR_TILING,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper returning a list of all tiles.
    Each item contains:
      tile_id, image (PIL crop), x_offset, y_offset, width, height, row, col.
    """
    return list(iter_tiles(
        image=image,
        tile_size=tile_size,
        overlap=overlap,
        max_tiles=max_tiles,
        min_image_size=min_image_size,
    ))


# =====================================================================
# COORDINATE SYSTEM CONVERSION
# =====================================================================
def tile_bbox_to_global(
    bbox: List[float],
    x_offset: int,
    y_offset: int,
    clip_max_w: Optional[int] = None,
    clip_max_h: Optional[int] = None,
) -> List[float]:
    """
    Transforms bounding box from local tile coordinates into the global original image coordinates.
    Formula:
        global_x1 = tile_x1 + x_offset
        global_y1 = tile_y1 + y_offset
        global_x2 = tile_x2 + x_offset
        global_y2 = tile_y2 + y_offset
    """
    x1, y1, x2, y2 = bbox
    gx1 = float(x1 + x_offset)
    gy1 = float(y1 + y_offset)
    gx2 = float(x2 + x_offset)
    gy2 = float(y2 + y_offset)

    if clip_max_w is not None:
        gx1 = max(0.0, min(float(clip_max_w), gx1))
        gx2 = max(0.0, min(float(clip_max_w), gx2))
    if clip_max_h is not None:
        gy1 = max(0.0, min(float(clip_max_h), gy1))
        gy2 = max(0.0, min(float(clip_max_h), gy2))

    return [round(gx1, 1), round(gy1, 1), round(gx2, 1), round(gy2, 1)]


def format_tile_metadata(
    tiles_info: List[Dict[str, Any]],
    enabled: bool,
    full_width: int,
    full_height: int,
    tile_size: int = TILE_SIZE,
    overlap: float = TILE_OVERLAP,
) -> Dict[str, Any]:
    """
    Constructs clean, lightweight tiling metadata for debugging without exposing raw binaries.
    """
    return {
        "enabled": enabled,
        "full_resolution": f"{full_width}x{full_height}",
        "tile_count": len(tiles_info),
        "tile_size": tile_size,
        "overlap_ratio": overlap,
        "tiles": [
            {
                "tile_id": t["tile_id"],
                "x_offset": t["x_offset"],
                "y_offset": t["y_offset"],
                "width": t["width"],
                "height": t["height"],
                "detections_count": t.get("detections_count", 0),
            }
            for t in tiles_info
        ],
    }

