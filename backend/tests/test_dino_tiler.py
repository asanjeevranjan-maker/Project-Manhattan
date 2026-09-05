"""
Unit tests for the Grounding DINO Satellite Image Tiling Module.
Verifies:
1. Small image bypass (images <= 1024px bypass tiling)
2. Large image multi-tile generation (e.g., 2048x2048)
3. 15% overlap verification (step size check)
4. Local-to-global bounding box coordinate conversion
5. Seam object deduplication across overlapping tiles
6. Non-square aspect ratios (e.g. 2000x800, 800x2500)
7. Adaptive tile expansion and MAX_TILES enforcement
8. Metadata generation without binary exposure
"""

import pytest
from PIL import Image
from services.detection.tiler import (
    TILE_SIZE,
    TILE_OVERLAP,
    ENABLE_TILING,
    MIN_IMAGE_SIZE_FOR_TILING,
    MAX_TILES,
    should_tile_image,
    calculate_tile_grid,
    iter_tiles,
    generate_tiles,
    tile_bbox_to_global,
    format_tile_metadata,
)
from services.detection.vocabulary import filter_and_format_detections


def test_tiling_constants_and_defaults():
    """Verify default parameters match satellite detection specifications."""
    assert TILE_SIZE == 1024
    assert TILE_OVERLAP == 0.15
    assert ENABLE_TILING is True
    assert MIN_IMAGE_SIZE_FOR_TILING == 1024
    assert MAX_TILES == 16


def test_should_tile_image_decision():
    """Verify images smaller than min_size bypass tiling, while larger ones trigger tiling."""
    # Small image (<= 1024)
    small_img = Image.new("RGB", (800, 600))
    assert should_tile_image(small_img, min_size=1024, enable_tiling=True) is False

    # Exact boundary (1024x1024)
    boundary_img = Image.new("RGB", (1024, 1024))
    assert should_tile_image(boundary_img, min_size=1024, enable_tiling=True) is False

    # Large image (> 1024 in width)
    wide_img = Image.new("RGB", (1200, 800))
    assert should_tile_image(wide_img, min_size=1024, enable_tiling=True) is True

    # Large image (> 1024 in height)
    tall_img = Image.new("RGB", (800, 1500))
    assert should_tile_image(tall_img, min_size=1024, enable_tiling=True) is True

    # Global disabled toggle overrides size
    assert should_tile_image(wide_img, min_size=1024, enable_tiling=False) is False


def test_single_tile_grid_for_small_images():
    """Images <= tile_size produce exactly 1 full tile without tiling overhead."""
    grid = calculate_tile_grid(width=800, height=600, tile_size=1024, overlap=0.15)
    assert len(grid) == 1
    tile = grid[0]
    assert tile["tile_id"] == "tile_0_0"
    assert tile["x_offset"] == 0
    assert tile["y_offset"] == 0
    assert tile["width"] == 800
    assert tile["height"] == 600
    assert tile["row"] == 0
    assert tile["col"] == 0


def test_multi_tile_grid_and_overlap_step():
    """
    Test 2048x2048 image with tile_size=1024 and 15% overlap.
    15% of 1024 = 153px. Step = 1024 - 153 = 871px.
    """
    grid = calculate_tile_grid(width=2048, height=2048, tile_size=1024, overlap=0.15)
    assert len(grid) > 1

    # Check tile dimensions
    for t in grid:
        assert t["width"] <= 1024
        assert t["height"] <= 1024
        assert t["x_offset"] + t["width"] <= 2048
        assert t["y_offset"] + t["height"] <= 2048

    # Verify column offsets step by 871px (or final right alignment)
    row0_tiles = [t for t in grid if t["row"] == 0]
    assert len(row0_tiles) >= 2
    assert row0_tiles[0]["x_offset"] == 0
    assert row0_tiles[1]["x_offset"] == 871

    # Verify coverage: last tile reaches the right/bottom edge
    max_right = max(t["x_offset"] + t["width"] for t in grid)
    max_bottom = max(t["y_offset"] + t["height"] for t in grid)
    assert max_right == 2048
    assert max_bottom == 2048


def test_non_square_aspect_ratios():
    """Test panoramic and portrait satellite scenes."""
    # Ultra-wide scene: 2500x800 (height <= 1024, width > 1024)
    wide_grid = calculate_tile_grid(width=2500, height=800, tile_size=1024, overlap=0.15)
    rows = {t["row"] for t in wide_grid}
    cols = {t["col"] for t in wide_grid}
    assert len(rows) == 1  # only 1 row needed
    assert len(cols) >= 3  # multiple columns needed
    for t in wide_grid:
        assert t["y_offset"] == 0
        assert t["height"] == 800

    # Tall portrait scene: 800x2500 (width <= 1024, height > 1024)
    tall_grid = calculate_tile_grid(width=800, height=2500, tile_size=1024, overlap=0.15)
    t_rows = {t["row"] for t in tall_grid}
    t_cols = {t["col"] for t in tall_grid}
    assert len(t_cols) == 1  # only 1 column
    assert len(t_rows) >= 3  # multiple rows


def test_adaptive_tile_expansion_for_massive_scenes():
    """
    If natural tiling for an 8192x8192 orthomosaic would exceed MAX_TILES (16),
    the tiler adaptively expands tile_size so tile_count <= MAX_TILES.
    """
    grid = calculate_tile_grid(
        width=8192,
        height=8192,
        tile_size=1024,
        overlap=0.15,
        max_tiles=16,
    )
    assert len(grid) <= 16
    # Tiles must be larger than original 1024px to cover 8192x8192 in <= 16 tiles
    assert grid[0]["width"] > 1024
    assert grid[0]["height"] > 1024

    # Full extent must still be covered
    assert max(t["x_offset"] + t["width"] for t in grid) == 8192
    assert max(t["y_offset"] + t["height"] for t in grid) == 8192


def test_iter_tiles_and_generate_tiles():
    """Verify memory-safe tile generation produces valid PIL crops and correct offsets."""
    img = Image.new("RGB", (1500, 1200), color=(100, 150, 200))
    tiles = generate_tiles(img, tile_size=1024, overlap=0.15)

    assert len(tiles) >= 2
    for t in tiles:
        assert "tile_id" in t
        assert "image" in t
        assert isinstance(t["image"], Image.Image)
        assert t["image"].size == (t["width"], t["height"])
        assert t["x_offset"] >= 0
        assert t["y_offset"] >= 0
        assert t["x_offset"] + t["width"] <= 1500
        assert t["y_offset"] + t["height"] <= 1200


def test_tile_bbox_to_global_coordinate_conversion():
    """
    Test coordinate translation from tile local coords to original full image coords:
    global_x = tile_x + x_offset
    global_y = tile_y + y_offset
    """
    # Detection at [50, 60, 200, 250] in a tile at offset (871, 0)
    tile_box = [50.0, 60.0, 200.0, 250.0]
    x_offset = 871
    y_offset = 500

    global_box = tile_bbox_to_global(
        bbox=tile_box,
        x_offset=x_offset,
        y_offset=y_offset,
        clip_max_w=2000,
        clip_max_h=2000,
    )

    assert global_box == [921.0, 560.0, 1071.0, 750.0]


def test_tile_bbox_to_global_clipping():
    """Ensure coordinates do not exceed maximum image bounds."""
    tile_box = [900.0, 950.0, 1100.0, 1050.0]
    x_offset = 1000
    y_offset = 1000

    clipped_box = tile_bbox_to_global(
        bbox=tile_box,
        x_offset=x_offset,
        y_offset=y_offset,
        clip_max_w=2000,
        clip_max_h=2000,
    )

    assert clipped_box[0] == 1900.0
    assert clipped_box[1] == 1950.0
    assert clipped_box[2] == 2000.0  # clipped from 2100 to 2000
    assert clipped_box[3] == 2000.0  # clipped from 2050 to 2000


def test_seam_duplicate_deduplication():
    """
    When an object straddles an overlap seam, both adjacent tiles detect it.
    Converting to global coordinates and passing to filter_and_format_detections
    must merge the duplicates via NMS IoU thresholding.
    """
    # Full image is 2000x1024
    # Tile 0 covers [0..1024], Tile 1 covers [871..1895]
    # An object (e.g. building) is located at global coordinates [900, 200, 980, 280]

    # In Tile 0 local coordinates (offset 0, 0): [900, 200, 980, 280]
    det_tile0 = {
        "box": tile_bbox_to_global([900, 200, 980, 280], x_offset=0, y_offset=0),
        "label": "building",
        "score": 0.88,
    }

    # In Tile 1 local coordinates (offset 871, 0): [29, 200, 109, 280]
    # In global coords: [29+871, 200, 109+871, 280] = [900, 200, 980, 280]
    det_tile1 = {
        "box": tile_bbox_to_global([29, 200, 109, 280], x_offset=871, y_offset=0),
        "label": "building",
        "score": 0.82,
    }

    combined_detections = [det_tile0, det_tile1]

    # Format and deduplicate
    final = filter_and_format_detections(
        raw_detections=combined_detections,
        width=2000,
        height=1024,
        iou_threshold=0.45,
    )

    # Exactly 1 building detection must remain (the one with higher score 0.88)
    assert len(final) == 1
    assert final[0]["label"] == "building"
    assert final[0]["score"] == 0.88
    assert final[0]["box"] == [900.0, 200.0, 980.0, 280.0]


def test_format_tile_metadata():
    """Ensure debugging tiling metadata is structured, informative, and contains no image bytes."""
    tiles_info = [
        {"tile_id": "tile_0_0", "x_offset": 0, "y_offset": 0, "width": 1024, "height": 1024, "detections_count": 4},
        {"tile_id": "tile_0_1", "x_offset": 871, "y_offset": 0, "width": 1024, "height": 1024, "detections_count": 2},
    ]
    meta = format_tile_metadata(
        tiles_info=tiles_info,
        enabled=True,
        full_width=1895,
        full_height=1024,
        tile_size=1024,
        overlap=0.15,
    )

    assert meta["enabled"] is True
    assert meta["full_resolution"] == "1895x1024"
    assert meta["tile_count"] == 2
    assert meta["tile_size"] == 1024
    assert meta["overlap_ratio"] == 0.15
    assert len(meta["tiles"]) == 2
    assert meta["tiles"][0]["tile_id"] == "tile_0_0"
    assert meta["tiles"][0]["detections_count"] == 4
    # Ensure no binary or base64 keys exist
    for t in meta["tiles"]:
        assert "image" not in t
        assert "base64" not in t
