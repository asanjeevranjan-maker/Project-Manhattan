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
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_CLASS_NMS_THRESHOLDS,
    get_class_nms_threshold,
    calculate_iou,
    apply_class_nms,
    get_deduplication_stats,
    SAM2_AVAILABLE,
    ENABLE_SEGMENTATION,
    segment_detections,
    get_sam2_predictor,
    calculate_land_cover,
    VERIFIER_AVAILABLE,
    ENABLE_VERIFICATION,
    VERIFICATION_THRESHOLD,
    verify_detections,
    validate_bbox,
)

logger = logging.getLogger("satquery.grounding_dino")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")


MODEL_ID = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"[DINO] Using compute device: {device}")
logger.info(f"[DINO] Loading Grounding DINO model: {MODEL_ID}")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
model.eval()


# ------------------------------------------------
# Tuning constants & Base Thresholds
# ------------------------------------------------
BASE_BOX_THRESHOLD = 0.18      # Broad candidate collection; filtered by class thresholds
BASE_TEXT_THRESHOLD = 0.15     # Text matching threshold
NMS_IOU_THRESHOLD = 0.25       # IoU for deduplicating overlapping detections of the same class
TILE_SIZE = 1024
TILE_OVERLAP = 0.15


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
# Confidence Calibration
# ------------------------------------------------
# Grounding DINO matching scores are logits-derived similarities, NOT
# calibrated probabilities. On overhead/satellite imagery they saturate
# in the ~0.2-0.6 band even for textbook-perfect detections, which makes
# a correct ship detection display as "47%".
#
# We therefore calibrate the raw score into a display probability by:
#   1. Monotonic rescale of the raw score from its observed operating
#      band [0.15, 0.65] onto [0.55, 0.95] (order-preserving — a better
#      raw detection still always yields a higher calibrated score).
#   2. Blending in the SigLIP verification score when available
#      (cross-model agreement is the strongest single signal).
#   3. Small SAM2 mask-quality adjustments (a clean, well-fitting mask
#      confirms the box; a suspicious mask de-confirms it).
# The raw model score is preserved in `raw_score` for diagnostics.

DINO_SCORE_BAND = (0.15, 0.65)     # observed raw-score operating band
CALIBRATED_BAND = (0.55, 0.95)     # calibrated display band
SIGLIP_SCORE_BAND = (0.35, 0.85)   # SigLIP acceptance threshold .. strong match


def _rescale_to_band(value: float, band_in: Tuple[float, float], band_out: Tuple[float, float]) -> float:
    """Monotonic linear rescale of `value` from band_in onto band_out, clamped to band_out."""
    lo_in, hi_in = band_in
    lo_out, hi_out = band_out

    if hi_in <= lo_in:
        return lo_out

    t = (value - lo_in) / (hi_in - lo_in)
    t = min(1.0, max(0.0, t))
    return lo_out + t * (hi_out - lo_out)


def calibrate_detection_confidence(det: Dict[str, Any]) -> float:
    """
    Fuses raw DINO score + SigLIP verification score + SAM2 mask quality
    into a single calibrated confidence in [0.05, 0.99]. Mutates `det`
    in place (sets confidence / score / raw_score / vscore / confidence_level).
    """
    raw = det.get("score", det.get("confidence", 0.0))
    if not isinstance(raw, (int, float)) or raw <= 0:
        raw = 0.0

    calibrated = _rescale_to_band(float(raw), DINO_SCORE_BAND, CALIBRATED_BAND)

    # --- Signal 2: SigLIP verification (cross-model agreement) -----------
    verification = det.get("verification") or {}
    vscore = verification.get("score")
    det["vscore"] = vscore  # top-level, consumed by the frontend

    if isinstance(vscore, (int, float)) and verification.get("skipped") is not True:
        sig_cal = _rescale_to_band(float(vscore), SIGLIP_SCORE_BAND, CALIBRATED_BAND)
        # Equal-weight blend: two independent models agreeing is stronger
        # evidence than either alone.
        calibrated = 0.5 * calibrated + 0.5 * sig_cal

    # --- Signal 3: SAM2 mask quality -------------------------------------
    mask = det.get("mask") or {}
    quality = det.get("mask_quality") or mask.get("mask_quality")
    fill = det.get("fill_ratio", mask.get("fill_ratio"))

    if quality == "good" and isinstance(fill, (int, float)) and 0.08 <= fill <= 0.95:
        calibrated += 0.03   # mask confirms the box geometry
    elif quality == "suspicious":
        calibrated -= 0.07   # mask contradicts the box
    elif quality == "fallback":
        calibrated -= 0.02   # box-derived mask only — weak evidence

    calibrated = min(0.99, max(0.05, calibrated))

    det["raw_score"] = round(float(raw), 4)
    det["score"] = round(calibrated, 4)
    det["confidence"] = round(calibrated, 4)
    det["confidence_level"] = map_score_to_confidence_level(calibrated)

    return calibrated


def apply_confidence_calibration(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applies confidence calibration to every detection (in place, returns the list)."""
    for det in detections:
        try:
            calibrate_detection_confidence(det)
        except Exception as cal_err:
            logger.warning(f"[CALIBRATION] Skipped detection calibration: {cal_err}")
    return detections


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

        # 3. Geometric bounds check via validate_bbox
        is_valid, reason = validate_bbox(box_list, image.width, image.height, min_dimension=4.0)
        if not is_valid:
            logger.debug(f"[DINO] Discarding {canonical_label!r} box {box_list}: {reason}")
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
# Main Object Detection Entrypoint with Intelligent Tiling
# ------------------------------------------------
def detect_objects(
    image: Image.Image,
    prompt: str,
    preset: Optional[str] = None,
    use_tiles: Optional[bool] = None,
    tile_size: int = TILE_SIZE,
    overlap: float = TILE_OVERLAP,
    max_tiles: int = MAX_TILES,
    iou_threshold: Optional[float] = None,
    merge_mode: str = "standard",
    enable_segmentation: Optional[bool] = None,
    enable_verification: Optional[bool] = None,
    verification_threshold: Optional[float] = None,
    return_tiling_metadata: bool = False,
):
    """
    Main Grounding DINO detection function:
    1. Sanitizes prompt to short, concrete observable classes
    2. Dynamically determines whether to tile based on image dimensions and user configuration
    3. Runs inference on memory-safe tile streams without full-image duplication
    4. Translates tile-local bounding boxes directly into original image coordinates
    5. Deduplicates boundary overlaps via class-specific NMS
    6. Formats clean detections with qualitative confidence levels and center locations
    """
    image = image.convert("RGB")
    full_width, full_height = image.size

    # 1. Sanitize prompt (translate abstract queries or presets to observable classes)
    clean_prompt = sanitize_prompt(prompt, preset=preset)
    logger.info(
        f"[PIPELINE] Processing Image ({full_width}x{full_height}) | "
        f"Raw query: {prompt!r} | Preset: {preset!r} | Sanitized: {clean_prompt!r}"
    )

    # 2. Determine tiling activation
    tiling_requested = ENABLE_TILING if use_tiles is None else bool(use_tiles)
    will_tile = tiling_requested and should_tile_image(
        image=image,
        min_size=MIN_IMAGE_SIZE_FOR_TILING,
        enable_tiling=tiling_requested,
    )

    all_detections = []
    tiles_debug_info = []

    if will_tile:
        logger.info(
            f"[PIPELINE] Tiling ACTIVATED for {full_width}x{full_height} image "
            f"(tile_size={tile_size}px, overlap={int(overlap*100) if overlap < 1.0 else int(overlap)}%, max_tiles={max_tiles})."
        )
        for tile_dict in iter_tiles(
            image=image,
            tile_size=tile_size,
            overlap=overlap,
            max_tiles=max_tiles,
            min_image_size=MIN_IMAGE_SIZE_FOR_TILING,
        ):
            t_crop = tile_dict["image"]
            ox = tile_dict["x_offset"]
            oy = tile_dict["y_offset"]
            t_id = tile_dict["tile_id"]

            raw_tile_dets = run_model_on_image(t_crop, clean_prompt)
            tiles_debug_info.append({
                "tile_id": t_id,
                "x_offset": ox,
                "y_offset": oy,
                "width": tile_dict["width"],
                "height": tile_dict["height"],
                "detections_count": len(raw_tile_dets),
            })

            # Convert local tile bounding boxes to global coordinates
            for det in raw_tile_dets:
                global_box = tile_bbox_to_global(
                    bbox=det["box"],
                    x_offset=ox,
                    y_offset=oy,
                    clip_max_w=full_width,
                    clip_max_h=full_height,
                )
                det["box"] = global_box
                det["bbox"] = global_box
                all_detections.append(det)

            logger.info(
                f"[PIPELINE] Tile {t_id} (offset=({ox},{oy}), size={tile_dict['width']}x{tile_dict['height']}): "
                f"{len(raw_tile_dets)} raw detections translated to global coordinates."
            )
    else:
        logger.info(
            f"[PIPELINE] Direct single-tile inference for {full_width}x{full_height} image "
            f"(tiling bypassed or below min size {MIN_IMAGE_SIZE_FOR_TILING}px)."
        )
        raw_dets = run_model_on_image(image, clean_prompt)
        tiles_debug_info.append({
            "tile_id": "tile_0_0",
            "x_offset": 0,
            "y_offset": 0,
            "width": full_width,
            "height": full_height,
            "detections_count": len(raw_dets),
        })
        all_detections = raw_dets

    from collections import Counter
    counts_by_class = Counter([d.get("label", "unknown") for d in all_detections])
    logger.info(
        f"[PIPELINE] [DINO] Total raw candidates across tiles: {len(all_detections)} | By class: {dict(counts_by_class)}"
    )

    # 3. Format and deduplicate via Global Class-Aware NMS
    final_detections, dedup_stats = filter_and_format_detections(
        raw_detections=all_detections,
        width=full_width,
        height=full_height,
        iou_threshold=iou_threshold,
        merge_mode=merge_mode,
        return_dedup_info=True,
    )

    final_by_class = Counter([d.get("label", "unknown") for d in final_detections])
    logger.info(
        f"[PIPELINE] [NMS] Kept {len(final_detections)} detections after class NMS "
        f"(raw: {dedup_stats['raw_detection_count']}, duplicates removed: {dedup_stats['duplicates_removed']}) | "
        f"Final classes: {dict(final_by_class)}"
    )

    # 3.5 Secondary Detection Verification using SigLIP (reduces false positives)
    run_verification = ENABLE_VERIFICATION if enable_verification is None else bool(enable_verification)
    final_detections, verification_metadata = verify_detections(
        image=image,
        detections=final_detections,
        enable_verification=run_verification,
        threshold=verification_threshold,
    )

    # 4. Optional SAM2-based segmentation on verified detections
    run_segmentation = ENABLE_SEGMENTATION if enable_segmentation is None else bool(enable_segmentation)
    if run_segmentation and final_detections:
        logger.info(f"[PIPELINE] [SAM2] Initiating segmentation on {len(final_detections)} detections...")
        final_detections, seg_metadata = segment_detections(
            image=image,
            detections=final_detections,
            enable_segmentation=True,
        )
        logger.info(
            f"[PIPELINE] [SAM2] Segmentation completed. Available={seg_metadata.get('sam2_available')}, "
            f"Segmented={seg_metadata.get('segmented_count')}/{len(final_detections)}"
        )
    else:
        predictor = get_sam2_predictor()
        diag = predictor.get_diagnostics() if hasattr(predictor, "get_diagnostics") else {}
        seg_metadata = {
            "segmentation_available": diag.get("sam2_available", SAM2_AVAILABLE),
            "sam2_available": diag.get("sam2_available", SAM2_AVAILABLE),
            "sam2_loaded": diag.get("sam2_loaded", False),
            "sam2_backend": diag.get("sam2_backend", "none"),
            "device": diag.get("device", "cpu"),
            "model_checkpoint": diag.get("model_checkpoint"),
            "failure_reason": diag.get("failure_reason") if not diag.get("sam2_available") else None,
            "enabled": run_segmentation,
            "segmented_count": 0,
            "total_detections": len(final_detections),
            "backend": diag.get("sam2_backend", "none"),
            "overlay_preview": None,
            "mask_overlay_url": None,
        }

    # 4.5 Confidence calibration: fuse DINO score + SigLIP verification +
    #     SAM2 mask quality into a calibrated, display-ready confidence.
    final_detections = apply_confidence_calibration(final_detections)
    logger.info(
        f"[PIPELINE] [CALIBRATION] Calibrated {len(final_detections)} detection confidences "
        f"(DINO + SigLIP + SAM2 fusion)."
    )

    # 5. Objective Land-Cover Coverage Calculation (truthful pixel accounting)
    seg_avail = seg_metadata.get("segmentation_available", False)
    land_cover_result = calculate_land_cover(
        image_size=(full_width, full_height),
        detections=final_detections,
        segmentation_available=seg_avail,
        generate_visualization=True,
        base_image=image,
    )

    tiling_metadata = format_tile_metadata(
        tiles_info=tiles_debug_info,
        enabled=will_tile,
        full_width=full_width,
        full_height=full_height,
        tile_size=tile_size,
        overlap=overlap,
    )
    tiling_metadata["deduplication"] = dedup_stats
    tiling_metadata["verification"] = verification_metadata
    tiling_metadata["segmentation"] = seg_metadata
    tiling_metadata["land_cover"] = land_cover_result

    if return_tiling_metadata:
        return final_detections, tiling_metadata

    return final_detections