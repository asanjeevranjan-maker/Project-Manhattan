"""
SAM2-Based Object Segmentation Module for Grounding DINO.

Architecture:
Grounding DINO -> Bounding Boxes -> SAM2 Predictor -> Pixel Masks & Overlay Preview

Design Principles:
1. Optional import safety: Serverless/Vercel safe, never crashes on startup if SAM2 is not installed.
2. Direct box prompt: Passes Grounding DINO [x1, y1, x2, y2] bounding boxes directly to SAM2.
3. Compact mask formats: Returns SVG/Canvas-friendly polygons and COCO RLE, avoiding raw NumPy in JSON.
4. Transparent overlay preview: Generates a lightweight, color-coded alpha composite for visual inspection.
5. Class filtering: Avoids running SAM2 blindly; focuses compute on concrete visual structures (buildings, water, vegetation, etc.).
6. Memory safety: Uses inference_mode, downsamples previews, and clears GPU cache.
"""

import os
import io
import math
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from PIL import Image, ImageDraw

logger = logging.getLogger("satquery.detection.segmentation")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")


# =====================================================================
# 1. RUNTIME DETECTION & OPTIONAL IMPORTS
# =====================================================================
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False

# Probe for SAM2 or legacy SAM packages safely
SAM2_AVAILABLE = False
SAM2_BACKEND: Optional[str] = None

try:
    import sam2  # type: ignore
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    SAM2_AVAILABLE = True
    SAM2_BACKEND = "sam2"
except Exception:
    try:
        import segment_anything  # type: ignore
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        SAM2_AVAILABLE = True
        SAM2_BACKEND = "segment_anything"
    except Exception:
        SAM2_AVAILABLE = False
        SAM2_BACKEND = None


# =====================================================================
# 2. CONFIGURATION & CONSTANTS
# =====================================================================
# Global toggle to enable/disable segmentation
ENABLE_SEGMENTATION: bool = os.getenv("DINO_ENABLE_SEGMENTATION", "true").lower() in ("true", "1", "yes")

# Checkpoint configurations
SAM2_CHECKPOINT: str = os.getenv("SAM2_CHECKPOINT", "checkpoints/sam2_hiera_tiny.pt")
SAM2_CONFIG: str = os.getenv("SAM2_CONFIG", "sam2_hiera_t.yaml")

# Preview image resolution limit for memory safety (pixels along longest dimension)
MAX_OVERLAY_DIMENSION: int = int(os.getenv("SAM2_MAX_OVERLAY_DIMENSION", "1280"))

# Concrete observable classes suitable for segmentation (avoids running on diffuse/abstract concepts)
DEFAULT_SEGMENTABLE_CLASSES: Set[str] = {
    # Dense structures
    "building", "house", "structure", "facility", "industrial facility",
    # Water features
    "water", "water body", "river", "lake", "ocean", "reservoir", "flooded area",
    # Vegetation & Land Cover
    "vegetation", "tree", "forest", "field", "agricultural field", "bare land",
    # Construction & Earthworks
    "construction", "construction site",
    # Infrastructure & Transport
    "road", "bridge", "runway", "vehicle", "car", "truck", "boat", "vessel", "airplane", "aircraft",
}

# Color palette for segmentation overlay masks (RGBA)
CLASS_OVERLAY_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "building": (59, 130, 246, 110),        # Vibrant Blue
    "house": (59, 130, 246, 110),
    "structure": (59, 130, 246, 110),
    "water": (6, 182, 212, 130),             # Cyan / Deep water
    "water body": (6, 182, 212, 130),
    "river": (14, 165, 233, 130),           # Sky blue
    "lake": (6, 182, 212, 130),
    "flooded area": (56, 189, 248, 140),
    "vegetation": (34, 197, 94, 110),       # Emerald Green
    "forest": (22, 163, 74, 120),
    "tree": (34, 197, 94, 110),
    "field": (132, 204, 22, 110),           # Lime
    "agricultural field": (132, 204, 22, 110),
    "bare land": (202, 138, 4, 110),        # Ochre / Earth
    "construction": (234, 88, 12, 120),     # Construction Orange
    "construction site": (234, 88, 12, 120),
    "road": (234, 179, 8, 120),             # Golden Yellow
    "bridge": (245, 158, 11, 120),          # Amber
    "vehicle": (249, 115, 22, 130),         # Orange-red
    "car": (249, 115, 22, 130),
    "truck": (249, 115, 22, 130),
    "boat": (168, 85, 247, 120),            # Purple
    "vessel": (168, 85, 247, 120),
    "airplane": (236, 72, 153, 120),        # Pink
    "aircraft": (236, 72, 153, 120),
    "default": (147, 51, 234, 110),         # Violet default
}


def is_class_segmentable(
    label: str,
    segmentable_classes: Optional[Set[str]] = None,
) -> bool:
    """Checks if a canonical or raw detection label is configured for SAM2 segmentation."""
    if not label:
        return False
    classes = segmentable_classes or DEFAULT_SEGMENTABLE_CLASSES
    normalized = label.strip().lower()
    if normalized in classes:
        return True
    for c in classes:
        if c in normalized or normalized in c:
            return True
    return False


# =====================================================================
# 3. COMPACT MASK ENCODING (POLYGONS & COCO RLE)
# =====================================================================
def mask_to_rle(binary_mask: Any) -> Dict[str, Any]:
    """
    Encodes a 2D boolean mask into standard Run-Length Encoding (RLE).
    Returns {"counts": [count_0, count_1, ...], "size": [height, width]}.
    Does not include giant NumPy arrays in JSON.
    """
    if binary_mask is None:
        return {"counts": [], "size": [0, 0]}

    if NUMPY_AVAILABLE and isinstance(binary_mask, np.ndarray):
        h, w = binary_mask.shape[:2]
        flat = binary_mask.flatten()
        # Vectorized or simple diff count
        changes = np.diff(flat)
        change_indices = np.where(changes)[0] + 1
        splits = np.split(flat, change_indices)
        counts = [len(s) for s in splits]
        # COCO RLE always starts with zeros count; if first element is 1, prepend 0
        if len(flat) > 0 and flat[0]:
            counts = [0] + counts
        return {"counts": [int(c) for c in counts], "size": [int(h), int(w)]}

    # Pure-Python fallback for list of lists
    if isinstance(binary_mask, list) and binary_mask:
        h = len(binary_mask)
        w = len(binary_mask[0]) if h > 0 else 0
        flat = [bool(val) for row in binary_mask for val in row]
        if not flat:
            return {"counts": [], "size": [h, w]}
        counts = []
        current_val = False
        current_count = 0
        for v in flat:
            if v == current_val:
                current_count += 1
            else:
                counts.append(current_count)
                current_val = v
                current_count = 1
        counts.append(current_count)
        return {"counts": counts, "size": [h, w]}

    return {"counts": [], "size": [0, 0]}


def mask_to_polygon(
    binary_mask: Any,
    box: Optional[List[float]] = None,
    max_points: int = 48,
) -> List[List[float]]:
    """
    Extracts outer boundary polygon vertices [[x, y], ...] from a binary mask.
    Uses cv2.findContours if available, or falls back to bounding box polygon.
    Limits points to max_points for compact JSON and fast SVG/Canvas rendering.
    """
    if binary_mask is None:
        if box and len(box) == 4:
            x1, y1, x2, y2 = [float(v) for v in box]
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return []

    # Use OpenCV for high-accuracy polygon contour extraction
    if OPENCV_AVAILABLE and NUMPY_AVAILABLE and isinstance(binary_mask, np.ndarray):
        mask_uint8 = (binary_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Pick largest contour
            largest = max(contours, key=cv2.contourArea)
            # Approximate polygon to reduce vertex count
            epsilon = 0.005 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            points = approx.reshape(-1, 2).tolist()
            # If still more than max_points, downsample uniformly
            if len(points) > max_points:
                step = math.ceil(len(points) / float(max_points))
                points = points[::step]
            return [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in points]

    # Fallback to bounding box polygon
    if box and len(box) == 4:
        x1, y1, x2, y2 = [round(float(v), 1) for v in box]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    return []


def compute_mask_bounds(polygon: List[List[float]]) -> List[float]:
    """Calculates [min_x, min_y, max_x, max_y] bounding box of a polygon."""
    if not polygon:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def compute_mask_area(binary_mask: Any, polygon: Optional[List[List[float]]] = None) -> int:
    """Computes exact foreground pixel count of the mask, or polygon area estimate."""
    if NUMPY_AVAILABLE and isinstance(binary_mask, np.ndarray):
        return int(np.count_nonzero(binary_mask))

    if isinstance(binary_mask, list) and binary_mask:
        return sum(sum(1 for v in row if v) for row in binary_mask)

    if polygon and len(polygon) >= 3:
        # Shoelace formula for polygon area
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return int(abs(area) / 2.0)

    return 0


# =====================================================================
# 4. TRANSPARENT OVERLAY PREVIEW GENERATOR
# =====================================================================
def generate_overlay_preview(
    image: Image.Image,
    segmented_detections: List[Dict[str, Any]],
    max_dimension: int = MAX_OVERLAY_DIMENSION,
) -> Optional[str]:
    """
    Creates a composite transparent overlay preview of the satellite image
    with color-coded, semi-transparent segmentation masks.
    Memory-safe: scales oversized images to max_dimension before compositing.
    Returns base64 PNG data URL ('data:image/png;base64,...').
    """
    try:
        orig_w, orig_h = image.size
        if orig_w <= 0 or orig_h <= 0:
            return None

        # Compute scaling factor to constrain memory consumption
        scale = 1.0
        longest = max(orig_w, orig_h)
        if longest > max_dimension:
            scale = float(max_dimension) / float(longest)
            target_w = max(1, int(orig_w * scale))
            target_h = max(1, int(orig_h * scale))
            preview_img = image.resize((target_w, target_h), Image.Resampling.BILINEAR).convert("RGBA")
        else:
            preview_img = image.copy().convert("RGBA")

        # RGBA canvas for transparent masks
        overlay_canvas = Image.new("RGBA", preview_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_canvas)

        for det in segmented_detections:
            mask_data = det.get("mask")
            if not mask_data or not isinstance(mask_data, dict):
                continue

            polygon = mask_data.get("polygon")
            if not polygon or len(polygon) < 3:
                continue

            label = det.get("label", "default").lower()
            color = CLASS_OVERLAY_COLORS.get(label, CLASS_OVERLAY_COLORS["default"])

            # Scale polygon coordinates to match preview image dimensions
            scaled_poly = [
                (float(pt[0]) * scale, float(pt[1]) * scale)
                for pt in polygon
            ]

            # Draw filled semi-transparent mask
            draw.polygon(scaled_poly, fill=color)

            # Draw crisp outline around the mask perimeter (higher opacity)
            outline_color = (color[0], color[1], color[2], min(255, color[3] + 90))
            draw.line(scaled_poly + [scaled_poly[0]], fill=outline_color, width=2)

        # Composite mask layer onto base satellite image
        composite = Image.alpha_composite(preview_img, overlay_canvas)

        # Encode to PNG base64 data URL
        buf = io.BytesIO()
        composite.save(buf, format="PNG", optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    except Exception as e:
        logger.warning(f"[Segmentation] Failed to generate overlay preview: {e}")
        return None


# =====================================================================
# 5. SAM2 MODEL SERVICE & PREDICTOR SINGLETON
# =====================================================================
class SAM2PredictorWrapper:
    """
    Encapsulates SAM2 / SAM predictor initialization and inference.
    Supports CUDA / CPU selection and proper memory clearing.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.checkpoint = checkpoint_path or SAM2_CHECKPOINT
        self.config = config_path or SAM2_CONFIG
        self.device = device or ("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
        self._predictor = None
        self._initialized = False

    def is_available(self) -> bool:
        if not SAM2_AVAILABLE or not TORCH_AVAILABLE:
            return False
        # If checkpoint file does not exist on disk, model weights are missing
        if self.checkpoint and not os.path.exists(self.checkpoint):
            return False
        return True

    def initialize(self) -> bool:
        """Loads weights lazily on first inference call."""
        if self._initialized:
            return True
        if not self.is_available():
            return False

        try:
            logger.info(f"[SAM2] Initializing model ({SAM2_BACKEND}) on {self.device}...")
            if SAM2_BACKEND == "sam2":
                model = build_sam2(self.config, self.checkpoint, device=self.device)
                self._predictor = SAM2ImagePredictor(model)
            elif SAM2_BACKEND == "segment_anything":
                model = sam_model_registry["vit_b"](checkpoint=self.checkpoint).to(self.device)
                self._predictor = SamPredictor(model)

            self._initialized = True
            logger.info("[SAM2] Model successfully initialized.")
            return True
        except Exception as e:
            logger.warning(f"[SAM2] Initialization failed: {e}")
            self._initialized = False
            return False

    def predict_mask(
        self,
        image_np: Any,
        box_xyxy: List[float],
    ) -> Tuple[Optional[Any], float]:
        """
        Runs box-prompted segmentation for a single detection box [x1, y1, x2, y2].
        Returns (binary_mask_2d, score).
        """
        if not self._initialized and not self.initialize():
            return None, 0.0

        try:
            if torch is not None:
                context = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
            else:
                context = None

            def _run():
                self._predictor.set_image(image_np)
                box_arr = np.array(box_xyxy, dtype=np.float32)
                if SAM2_BACKEND == "sam2":
                    masks, scores, _ = self._predictor.predict(
                        box=box_arr[None, :],
                        multimask_output=False,
                    )
                else:
                    masks, scores, _ = self._predictor.predict(
                        box=box_arr,
                        multimask_output=False,
                    )
                best_mask = masks[0] if len(masks) > 0 else None
                best_score = float(scores[0]) if len(scores) > 0 else 0.0
                return best_mask, best_score

            if context:
                with context:
                    return _run()
            return _run()

        except Exception as e:
            logger.warning(f"[SAM2] Predict error on box {box_xyxy}: {e}")
            return None, 0.0


# Lazy singleton instance
_GLOBAL_SAM2_PREDICTOR: Optional[SAM2PredictorWrapper] = None

def get_sam2_predictor() -> SAM2PredictorWrapper:
    global _GLOBAL_SAM2_PREDICTOR
    if _GLOBAL_SAM2_PREDICTOR is None:
        _GLOBAL_SAM2_PREDICTOR = SAM2PredictorWrapper()
    return _GLOBAL_SAM2_PREDICTOR


# =====================================================================
# 6. MAIN SEGMENTATION SERVICE INTERFACE
# =====================================================================
def segment_detections(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    enable_segmentation: bool = ENABLE_SEGMENTATION,
    segmentable_classes: Optional[Set[str]] = None,
    generate_overlay: bool = True,
    predictor_override: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main segmentation entrypoint converting Grounding DINO bounding boxes into pixel masks.

    Workflow:
    1. Checks if segmentation is enabled and available.
       If unavailable or disabled: preserves detections normally, returns segmentation_available=False.
    2. Filters detections to segmentable classes (e.g. building, water, vegetation).
    3. Prompts SAM2 with Grounding DINO bounding boxes [x1, y1, x2, y2].
    4. Serializes each mask into compact format:
       - polygon: List of contour coordinates [[x, y], ...]
       - bounds: [min_x, min_y, max_x, max_y]
       - rle: COCO-style counts and shape
       - mask_area_pixels: integer pixel count
    5. Optionally generates transparent color-coded overlay preview.
    6. Ensures memory safety (clears model tensors, catches errors gracefully).

    Returns:
        (segmented_detections, segmentation_metadata)
    """
    total_count = len(detections) if detections else 0

    # Fallback response template when segmentation cannot or should not run
    def _create_fallback_response(available: bool, enabled: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        meta = {
            "segmentation_available": available,
            "enabled": enabled,
            "segmented_count": 0,
            "total_detections": total_count,
            "backend": SAM2_BACKEND if available else None,
            "overlay_preview": None,
        }
        # Guarantee each detection has mask=None and mask_area_pixels=0 for schema consistency
        clean_dets = []
        for d in (detections or []):
            cd = dict(d)
            cd.setdefault("mask", None)
            cd.setdefault("mask_area_pixels", 0)
            clean_dets.append(cd)
        return clean_dets, meta

    # 1. Check if segmentation was explicitly disabled
    if not enable_segmentation:
        logger.info("[Segmentation] Segmentation explicitly disabled by caller.")
        return _create_fallback_response(available=SAM2_AVAILABLE, enabled=False)

    # 2. Check if predictor is available (or caller provided mock)
    predictor = predictor_override or get_sam2_predictor()
    is_avail = getattr(predictor, "is_available", lambda: True)()
    if not is_avail:
        logger.info("[Segmentation] SAM2 model/weights unavailable. Returning Grounding DINO detections normally.")
        return _create_fallback_response(available=False, enabled=True)

    if not detections:
        return _create_fallback_response(available=True, enabled=True)

    # 3. Prepare image for segmentation
    img_rgb = image.convert("RGB")
    np_img = np.array(img_rgb) if NUMPY_AVAILABLE else None

    allowed_classes = segmentable_classes or DEFAULT_SEGMENTABLE_CLASSES
    segmented_detections: List[Dict[str, Any]] = []
    segmented_count = 0

    try:
        for det in detections:
            det_copy = dict(det)
            box = det_copy.get("bbox") or det_copy.get("box")
            label = det_copy.get("label") or det_copy.get("raw_label") or ""

            # Check if class is configured for segmentation
            if not box or len(box) != 4 or not is_class_segmentable(label, allowed_classes):
                det_copy["mask"] = None
                det_copy["mask_area_pixels"] = 0
                segmented_detections.append(det_copy)
                continue

            # Run SAM2 prediction on bounding box prompt
            box_xyxy = [float(b) for b in box]
            raw_mask, _ = predictor.predict_mask(image_np=np_img, box_xyxy=box_xyxy)

            if raw_mask is not None:
                poly = mask_to_polygon(raw_mask, box=box_xyxy)
                rle = mask_to_rle(raw_mask)
                bounds = compute_mask_bounds(poly)
                area = compute_mask_area(raw_mask, polygon=poly)

                det_copy["mask"] = {
                    "format": "polygon",
                    "polygon": poly,
                    "bounds": bounds,
                    "rle": rle,
                }
                det_copy["mask_area_pixels"] = area
                segmented_count += 1
            else:
                # If SAM failed for this specific box, provide clean fallback without crashing
                det_copy["mask"] = None
                det_copy["mask_area_pixels"] = 0

            segmented_detections.append(det_copy)

        # 4. Generate transparent overlay preview
        overlay_preview = None
        if generate_overlay and segmented_count > 0:
            overlay_preview = generate_overlay_preview(
                image=img_rgb,
                segmented_detections=segmented_detections,
                max_dimension=MAX_OVERLAY_DIMENSION,
            )

        # 5. Clear GPU memory if CUDA is active
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        metadata = {
            "segmentation_available": True,
            "enabled": True,
            "segmented_count": segmented_count,
            "total_detections": total_count,
            "backend": getattr(predictor, "backend", SAM2_BACKEND or "custom"),
            "overlay_preview": overlay_preview,
        }

        logger.info(
            f"[Segmentation] Successfully segmented {segmented_count}/{total_count} detections. "
            f"Overlay generated: {overlay_preview is not None}."
        )

        return segmented_detections, metadata

    except Exception as error:
        logger.error(f"[Segmentation] Error during segmentation execution: {error}", exc_info=True)
        # Never crash: fall back to unsegmented detections
        return _create_fallback_response(available=True, enabled=True)
