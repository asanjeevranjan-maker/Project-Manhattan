"""
Grounding DINO Object Detector for Satellite Imagery.
Enhanced with concrete observable class vocabulary, presets, class-specific thresholds,
label normalization, geometry validation, and structured output formatting.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

# Ensure ai-service and backend are on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
root_backend = Path(__file__).resolve().parent.parent / "backend"
if str(root_backend) not in sys.path:
    sys.path.insert(0, str(root_backend))

from dino_vocabulary import (
    SATELLITE_CLASSES,
    ANALYSIS_PRESETS,
    DEFAULT_CLASS_THRESHOLDS,
    get_class_threshold,
    normalize_label,
    sanitize_prompt,
    map_score_to_confidence_level,
    compute_relative_location,
    format_detection,
    filter_and_format_detections,
    remove_duplicate_detections,
    box_iou,
)

logger = logging.getLogger("satquery.grounding_dino")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")


MODEL_ID = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Using compute device: {device}")
logger.info(f"Loading Grounding DINO model: {MODEL_ID}")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
model.eval()


# ------------------------------------------------
# Tuning constants & Base Thresholds
# ------------------------------------------------
BASE_BOX_THRESHOLD = 0.25      # Broad candidate collection; filtered by class thresholds
BASE_TEXT_THRESHOLD = 0.22     # Text matching threshold
NMS_IOU_THRESHOLD = 0.25       # IoU for deduplicating overlapping detections of the same class
TILE_SIZE = 512
TILE_OVERLAP = 96


# ------------------------------------------------
# Geometry Validation Guards
# ------------------------------------------------
def valid_box(box: List[float], image_width: int, image_height: int) -> bool:
    """Rejects bounding boxes that are non-positive or clear image-boundary noise."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return False

    # Reject tiny noise under 8x8 px
    if width < 8 or height < 8:
        return False

    # Reject boxes covering more than 40% of the entire image tile (unless water body)
    area = width * height
    image_area = max(1, image_width * image_height)
    if area / float(image_area) > 0.40:
        return False

    return True


def class_valid_box(
    box: List[float],
    canonical_label: str,
    image_width: int,
    image_height: int,
) -> bool:
    """
    Applies class-specific geometric constraints (aspect ratio, maximum tile coverage)
    to suppress hallucinated false-positive shapes.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return False

    aspect = width / float(height) if height > 0 else 0.0
    area = width * height
    image_area = max(1, image_width * image_height)
    area_ratio = area / float(image_area)

    thresh = get_class_threshold(canonical_label)

    # Area ratio ceiling
    if area_ratio > thresh.max_area_ratio:
        logger.debug(
            f"Filtered {canonical_label!r}: area_ratio={area_ratio:.3f} > max={thresh.max_area_ratio:.3f}"
        )
        return False

    # Aspect ratio bounds
    if aspect < thresh.min_aspect or aspect > thresh.max_aspect:
        logger.debug(
            f"Filtered {canonical_label!r}: aspect={aspect:.2f} outside [{thresh.min_aspect:.2f}, {thresh.max_aspect:.2f}]"
        )
        return False

    return True


# ------------------------------------------------
# Tile-Level Inference
# ------------------------------------------------
def run_model_on_image(
    image: Image.Image,
    prompt: str,
    box_threshold: float = BASE_BOX_THRESHOLD,
    text_threshold: float = BASE_TEXT_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Runs Grounding DINO on a single image or image crop.
    Normalizes labels, filters out truncated / stopword noise, and checks class-specific thresholds.
    """
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )

    result = results[0]
    raw_labels = result.get("text_labels", result.get("labels", []))
    boxes = result["boxes"]
    scores = result["scores"]

    detections: List[Dict[str, Any]] = []

    for box_tensor, score_tensor, raw_label_val in zip(boxes, scores, raw_labels):
        confidence = float(score_tensor)

        # 1. Normalize label and reject truncated or stopword junk (e.g. 'a', 'l', 'all')
        canonical_label, cleaned_raw = normalize_label(raw_label_val)
        if not canonical_label:
            logger.debug(f"Discarding invalid / truncated label: {raw_label_val!r}")
            continue

        # 2. Check class-specific score threshold
        thresh = get_class_threshold(canonical_label)
        if confidence < thresh.min_score:
            logger.debug(
                f"Discarding {canonical_label!r}: score {confidence:.3f} < min {thresh.min_score:.3f}"
            )
            continue

        box_list = [float(v) for v in box_tensor.tolist()]

        # 3. Geometric bounds check
        if not valid_box(box_list, image.width, image.height):
            continue

        # 4. Class-specific geometry check
        if not class_valid_box(box_list, canonical_label, image.width, image.height):
            continue

        detections.append({
            "label": canonical_label,
            "raw_label": cleaned_raw,
            "score": confidence,
            "confidence": confidence,
            "box": box_list,
        })

    return detections


# ------------------------------------------------
# Overlapping Tile Generation
# ------------------------------------------------
def create_tiles(
    image: Image.Image,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> List[Tuple[Image.Image, int, int]]:
    """Splits an image into overlapping grid tiles for high-resolution inspection."""
    width, height = image.size
    step = tile_size - overlap

    tiles: List[Tuple[Image.Image, int, int]] = []
    y = 0
    while y < height:
        x = 0
        bottom = min(y + tile_size, height)
        top = max(0, bottom - tile_size)

        while x < width:
            right = min(x + tile_size, width)
            left = max(0, right - tile_size)

            crop = image.crop((left, top, right, bottom))
            tiles.append((crop, left, top))

            if right >= width:
                break
            x += step

        if bottom >= height:
            break
        y += step

    return tiles


# ------------------------------------------------
# Main Object Detection Entrypoint
# ------------------------------------------------
def detect_objects(
    image: Image.Image,
    prompt: str,
    preset: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Main Grounding DINO detection function:
    1. Sanitizes prompt to short, concrete observable classes
    2. Runs tiled inference with class-specific thresholds
    3. Merges and removes duplicate overlapping boxes (NMS)
    4. Formats each detection into the clean unified output schema
    """
    image = image.convert("RGB")
    full_width, full_height = image.size

    # 1. Sanitize prompt (translate abstract queries or presets to observable classes)
    clean_prompt = sanitize_prompt(prompt, preset=preset)
    logger.info(
        f"[Grounding DINO] Processing Image ({full_width}x{full_height}) | "
        f"Raw query: {prompt!r} | Preset: {preset!r} | Prompt: {clean_prompt!r}"
    )

    # 2. Tile image if larger than tile size
    if full_width <= TILE_SIZE and full_height <= TILE_SIZE:
        raw_detections = run_model_on_image(image, clean_prompt)
        all_detections = raw_detections
    else:
        tiles = create_tiles(image, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
        logger.info(f"[Grounding DINO] Image split into {len(tiles)} tiles for detection.")
        all_detections = []

        for idx, (tile, offset_x, offset_y) in enumerate(tiles):
            tile_detections = run_model_on_image(tile, clean_prompt)
            for det in tile_detections:
                x1, y1, x2, y2 = det["box"]
                det["box"] = [
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                ]
                all_detections.append(det)

    logger.info(f"[Grounding DINO] Raw candidate detections collected: {len(all_detections)}")

    # 3. Format and deduplicate via NMS
    final_detections = filter_and_format_detections(
        raw_detections=all_detections,
        width=full_width,
        height=full_height,
        iou_threshold=NMS_IOU_THRESHOLD,
    )

    logger.info(f"[Grounding DINO] Final verified detections after NMS: {len(final_detections)}")
    return final_detections