"""
Secondary Detection Verification using SigLIP / CLIP
Reduces Grounding DINO false positives by verifying cropped candidate bounding boxes
against class-specific negative distractor concepts.

Architecture:
Grounding DINO candidate box -> Crop with 15% padding -> SigLIP zero-shot scoring -> Keep or Reject.

Graceful Fallback:
If PyTorch or Transformers is unavailable (e.g. in Vercel serverless environments),
VERIFIER_AVAILABLE is False and detections pass through unblocked without crashing.
"""

import os
import math
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from PIL import Image

logger = logging.getLogger("satquery.verifier")

# Optional ML imports with zero startup failure guarantee
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore

try:
    from transformers import AutoProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoProcessor = None  # type: ignore
    AutoModel = None  # type: ignore

VERIFIER_AVAILABLE = bool(TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE)

# Configurable verification environment flags
ENABLE_VERIFICATION: bool = os.getenv("ENABLE_VERIFICATION", "true").lower() in ("1", "true", "yes")
VERIFICATION_THRESHOLD: float = float(os.getenv("VERIFICATION_THRESHOLD", "0.35"))
VERIFIER_MODEL_NAME: str = os.getenv("VERIFIER_MODEL_NAME", "google/siglip-base-patch16-224")
DEFAULT_CROP_PADDING: float = 0.15
MIN_CROP_SIZE: int = 10

# Classes prone to false positives that benefit from verification
DEFAULT_VERIFICATION_CLASSES: Set[str] = {
    "vehicle",
    "car",
    "truck",
    "boat",
    "vessel",
    "aircraft",
    "airplane",
    "bridge",
    "building",
    "house",
    "structure",
}

# Class-specific negative distractor concepts
CLASS_NEGATIVE_ALTERNATIVES: Dict[str, List[str]] = {
    "vehicle": [
        "building roof",
        "road pavement",
        "tree shadow",
        "vegetation",
        "empty terrain background",
    ],
    "car": [
        "building roof",
        "road surface",
        "tree shadow",
        "vegetation",
        "empty terrain background",
    ],
    "truck": [
        "shipping container",
        "building roof",
        "road pavement",
        "ground shadow",
        "empty ground",
    ],
    "boat": [
        "water wave crest",
        "dock pier",
        "buoy",
        "sea foam",
        "open water surface",
    ],
    "vessel": [
        "water wave crest",
        "dock pier",
        "quay wall",
        "sea foam",
        "open water surface",
    ],
    "aircraft": [
        "runway tarmac",
        "hangar roof",
        "ground vehicle",
        "ground shadow",
        "open ground",
    ],
    "airplane": [
        "runway tarmac",
        "hangar roof",
        "ground vehicle",
        "ground shadow",
        "open ground",
    ],
    "bridge": [
        "river water",
        "highway road",
        "embankment",
        "dam",
        "natural terrain background",
    ],
    "building": [
        "bare ground",
        "parking lot",
        "agricultural field",
        "tree canopy",
        "empty terrain background",
    ],
    "house": [
        "bare ground",
        "parking lot",
        "agricultural field",
        "tree canopy",
        "empty terrain background",
    ],
    "structure": [
        "bare ground",
        "rock formation",
        "cleared land",
        "tree canopy",
        "empty terrain background",
    ],
}

DEFAULT_NEGATIVE_ALTERNATIVES: List[str] = [
    "natural terrain",
    "water surface",
    "vegetation",
    "road pavement",
    "empty background",
]


class SiglipDetectionVerifier:
    """
    Singleton zero-shot image-text classifier for secondary detection verification.
    Uses SigLIP pairwise sigmoid loss to test candidate crops against target and distractors.
    """
    _instance: Optional["SiglipDetectionVerifier"] = None
    _processor: Any = None
    _model: Any = None
    _device: str = "cpu"

    def __new__(cls, model_name: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(SiglipDetectionVerifier, cls).__new__(cls)
            cls._instance.model_name = model_name or VERIFIER_MODEL_NAME
            cls._instance.initialized = False
        return cls._instance

    def _ensure_loaded(self) -> bool:
        if self.initialized:
            return True
        if not VERIFIER_AVAILABLE:
            return False

        try:
            logger.info(f"[SigLIP Verifier] Loading model '{self.model_name}'...")
            self._device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
            self._model.eval()
            self.initialized = True
            logger.info(f"[SigLIP Verifier] Successfully loaded on device: {self._device}")
            return True
        except Exception as e:
            logger.warning(f"[SigLIP Verifier] Failed to load model '{self.model_name}': {e}")
            return False

    def predict_batch(
        self,
        crops: List[Image.Image],
        targets: List[str],
        alternatives_list: List[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Executes zero-shot scoring for a batch of crops.
        Returns for each crop:
          {"score": float, "passed": bool, "best_alternative": str, "best_alt_score": float}
        """
        if not crops or not self._ensure_loaded():
            return [{"passed": True, "score": 1.0, "reason": "Model unavailable"} for _ in crops]

        results: List[Dict[str, Any]] = []

        # Process each crop with its tailored candidate set
        for crop, target, alternatives in zip(crops, targets, alternatives_list):
            candidate_texts = [f"a satellite photo of a {target}"] + [
                f"a satellite photo of {alt}" for alt in alternatives
            ]

            try:
                inputs = self._processor(
                    text=candidate_texts,
                    images=crop.convert("RGB"),
                    padding="max_length",
                    return_tensors="pt",
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    logits = outputs.logits_per_image[0]  # shape: (num_texts,)
                    probs = torch.softmax(logits, dim=-1)

                target_prob = float(probs[0].item())
                alt_probs = [float(p.item()) for p in probs[1:]]
                best_alt_idx = int(torch.argmax(probs[1:]).item()) if alt_probs else 0
                best_alt_score = alt_probs[best_alt_idx] if alt_probs else 0.0
                best_alt_name = alternatives[best_alt_idx] if alt_probs else "none"

                # Verification score represents relative vision-language alignment
                # (NOTE: do not pretend verification score is a calibrated probability)
                results.append({
                    "score": round(target_prob, 3),
                    "best_alternative": best_alt_name,
                    "best_alt_score": round(best_alt_score, 3),
                })
            except Exception as e:
                logger.warning(f"[SigLIP Verifier] Inference error for target '{target}': {e}")
                results.append({
                    "score": 1.0,
                    "best_alternative": "none",
                    "best_alt_score": 0.0,
                    "error": str(e),
                })

        return results


def get_verifier(model_name: Optional[str] = None) -> SiglipDetectionVerifier:
    """Returns singleton SiglipDetectionVerifier instance."""
    return SiglipDetectionVerifier(model_name=model_name)


def crop_detection_box(
    image: Image.Image,
    bbox: List[float],
    padding: float = DEFAULT_CROP_PADDING,
    min_size: int = MIN_CROP_SIZE,
) -> Tuple[Optional[Image.Image], bool]:
    """
    Crops the bounding box from the image with contextual padding.
    Clamps bounds to image boundaries.
    Returns (crop_image, is_too_small).
    """
    if not image or not bbox or len(bbox) < 4:
        return None, True

    img_w, img_h = image.size
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    if w < min_size or h < min_size:
        return None, True

    pad_x = w * padding
    pad_y = h * padding

    crop_x1 = max(0, int(math.floor(x1 - pad_x)))
    crop_y1 = max(0, int(math.floor(y1 - pad_y)))
    crop_x2 = min(img_w, int(math.ceil(x2 + pad_x)))
    crop_y2 = min(img_h, int(math.ceil(y2 + pad_y)))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None, True

    try:
        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        return crop, False
    except Exception:
        return None, True


def get_alternatives_for_class(label: str) -> List[str]:
    """Retrieves class-specific negative distractor concepts for verification."""
    clean = (label or "").strip().lower()
    if clean in CLASS_NEGATIVE_ALTERNATIVES:
        return CLASS_NEGATIVE_ALTERNATIVES[clean]
    for k, v in CLASS_NEGATIVE_ALTERNATIVES.items():
        if k in clean or clean in k:
            return v
    return DEFAULT_NEGATIVE_ALTERNATIVES


def verify_detection(
    image_crop: Optional[Image.Image],
    label: str,
    alternative_labels: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    min_crop_size: int = MIN_CROP_SIZE,
) -> Dict[str, Any]:
    """
    Verifies a single cropped candidate against negative alternatives.
    Returns:
    {
      "verified": bool,
      "verification_score": float,
      "passed": bool,
      "original_detection_score": Optional[float],
      "best_alternative": Optional[str],
      "model": "siglip",
    }
    """
    target_thresh = threshold if threshold is not None else VERIFICATION_THRESHOLD

    if image_crop is None:
        return {
            "verified": False,
            "passed": False,
            "verification_score": 0.0,
            "reason": f"Crop size below minimum resolution threshold ({min_crop_size}px)",
            "model": "siglip",
        }

    if not VERIFIER_AVAILABLE:
        return {
            "verified": False,
            "passed": True,
            "verification_score": None,
            "skipped": True,
            "reason": "Verifier model unavailable",
            "model": "siglip",
        }

    verifier = get_verifier()
    alternatives = alternative_labels or get_alternatives_for_class(label)

    res_list = verifier.predict_batch([image_crop], [label], [alternatives])
    item = res_list[0] if res_list else {"score": 1.0, "best_alternative": "none", "best_alt_score": 0.0}

    score = item["score"]
    passed = bool(score >= target_thresh)

    return {
        "verified": True,
        "passed": passed,
        "verification_score": score,
        "target_label": label,
        "best_alternative": item.get("best_alternative"),
        "best_alternative_score": item.get("best_alt_score"),
        "model": "siglip",
    }


def verify_detections(
    image: Any,
    detections: List[Dict[str, Any]],
    enable_verification: Optional[bool] = None,
    threshold: Optional[float] = None,
    crop_padding: float = DEFAULT_CROP_PADDING,
    min_crop_size: int = MIN_CROP_SIZE,
    verification_classes: Optional[Set[str]] = None,
    batch_size: int = 16,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Performs secondary verification on a list of Grounding DINO detections.
    Filters out rejected false positives and attaches verification debug records.

    Returns:
      (verified_detections, verification_metadata)
    """
    should_verify = ENABLE_VERIFICATION if enable_verification is None else bool(enable_verification)
    active_classes = verification_classes or DEFAULT_VERIFICATION_CLASSES
    target_thresh = threshold if threshold is not None else VERIFICATION_THRESHOLD

    # Fast bypass when disabled or no detections
    if not should_verify or not detections:
        return detections, {
            "verification_available": VERIFIER_AVAILABLE,
            "enabled": should_verify,
            "verified_count": 0,
            "rejected_count": 0,
            "skipped_count": len(detections),
            "total_candidates": len(detections),
            "backend": "siglip" if VERIFIER_AVAILABLE else None,
        }

    # Normalize PIL Image
    pil_img: Optional[Image.Image] = None
    if isinstance(image, Image.Image):
        pil_img = image
    elif hasattr(image, "convert"):
        pil_img = image.convert("RGB")
    elif isinstance(image, str):
        try:
            pil_img = Image.open(image).convert("RGB")
        except Exception:
            pass

    # Fallback if image cannot be opened
    if pil_img is None:
        logger.warning("[SigLIP Verifier] Image input could not be decoded for cropping.")
        return detections, {
            "verification_available": VERIFIER_AVAILABLE,
            "enabled": should_verify,
            "verified_count": 0,
            "rejected_count": 0,
            "skipped_count": len(detections),
            "total_candidates": len(detections),
            "backend": "siglip" if VERIFIER_AVAILABLE else None,
            "reason": "Image decode error",
        }

    verified_list: List[Dict[str, Any]] = []
    rejected_list: List[Dict[str, Any]] = []
    skipped_count = 0

    # Categorize items into verification candidates vs bypass items
    candidates_to_run: List[Tuple[int, Dict[str, Any], Image.Image, str, List[str]]] = []

    for idx, det in enumerate(detections):
        raw_score = float(det.get("confidence") or det.get("score") or 0.85)
        det["dino_score"] = raw_score
        label = str(det.get("label", "")).strip().lower()

        # Check if class is configured for verification
        if label not in active_classes and not any(c in label for c in active_classes):
            det["verification"] = {
                "passed": True,
                "score": None,
                "target_label": label,
                "skipped": True,
                "reason": "Class not in verification_classes",
                "verified": False,
            }
            verified_list.append(det)
            skipped_count += 1
            continue

        # Extract bbox
        bbox = det.get("bbox") or det.get("box")
        if not bbox or len(bbox) < 4:
            det["verification"] = {
                "passed": False,
                "score": 0.0,
                "reason": "Missing or invalid bounding box",
                "verified": False,
            }
            rejected_list.append(det)
            continue

        crop, is_tiny = crop_detection_box(pil_img, bbox, padding=crop_padding, min_size=min_crop_size)
        if is_tiny or crop is None:
            # Reject very tiny crops where verification is meaningless
            det["verification"] = {
                "passed": False,
                "score": 0.0,
                "target_label": label,
                "reason": f"Crop size below minimum resolution threshold ({min_crop_size}px)",
                "verified": False,
            }
            rejected_list.append(det)
            continue

        alternatives = get_alternatives_for_class(label)
        candidates_to_run.append((idx, det, crop, label, alternatives))

    # If verifier is unavailable in environment: pass all candidates through
    if not VERIFIER_AVAILABLE:
        for idx, det, _, label, _ in candidates_to_run:
            det["verification"] = {
                "passed": True,
                "score": None,
                "target_label": label,
                "skipped": True,
                "reason": "Verifier model unavailable in environment",
                "verified": False,
            }
            verified_list.append(det)
            skipped_count += 1

        return verified_list, {
            "verification_available": False,
            "enabled": True,
            "verified_count": 0,
            "rejected_count": len(rejected_list),
            "skipped_count": skipped_count,
            "total_candidates": len(detections),
            "backend": None,
        }

    # Execute batched prediction
    verifier = get_verifier()
    num_candidates = len(candidates_to_run)

    for i in range(0, num_candidates, batch_size):
        batch = candidates_to_run[i : i + batch_size]
        crops = [b[2] for b in batch]
        targets = [b[3] for b in batch]
        alt_lists = [b[4] for b in batch]

        pred_results = verifier.predict_batch(crops, targets, alt_lists)

        for (idx, det, _, target_lbl, _), pred in zip(batch, pred_results):
            score = pred["score"]
            passed = bool(score >= target_thresh)

            v_record = {
                "passed": passed,
                "score": score,
                "target_label": target_lbl,
                "best_alternative": pred.get("best_alternative"),
                "best_alternative_score": pred.get("best_alt_score"),
                "model": "siglip",
                "verified": True,
            }
            det["verification"] = v_record

            if passed:
                verified_list.append(det)
            else:
                rejected_list.append(det)

    metadata = {
        "verification_available": VERIFIER_AVAILABLE,
        "enabled": should_verify,
        "verified_count": len(verified_list) - skipped_count,
        "rejected_count": len(rejected_list),
        "skipped_count": skipped_count,
        "total_candidates": len(detections),
        "backend": "siglip",
        "threshold": target_thresh,
    }

    logger.info(
        f"[SigLIP Verifier] Verification completed: {metadata['verified_count']} passed, "
        f"{metadata['rejected_count']} false-positives rejected, {metadata['skipped_count']} skipped"
    )

    return verified_list, metadata
