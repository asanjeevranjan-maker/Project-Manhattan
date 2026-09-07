"""
Radiometric and Spatial Resolution Normalization Module (Stages 4 & 5).
Normalizes sensor, illumination, atmospheric, and resolution variations between T1 and T2
without destroying genuine physical and structural changes.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image

logger = logging.getLogger("satquery.temporal.normalization")

CV2_AVAILABLE = False
NUMPY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore


def normalize_spatial_resolution(
    image_t1: Image.Image,
    image_t2: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """
    Normalizes spatial resolution between T1 and T2 to a common effective grid.
    Prefers comparison at the native coarser resolution rather than artificially
    hallucinating detail through upscaling.
    """
    w1, h1 = image_t1.size
    w2, h2 = image_t2.size

    if target_size:
        common_w, common_h = target_size
    else:
        # If dimensions differ, adopt the minimum common dimension (coarser resolution)
        common_w = min(w1, w2)
        common_h = min(h1, h2)
        # Cap at 1280 to ensure optimal memory/compute performance
        if max(common_w, common_h) > 1280:
            scale = 1280.0 / max(common_w, common_h)
            common_w = int(common_w * scale)
            common_h = int(common_h * scale)

    grid_size = (common_w, common_h)

    # Use LANCZOS / Area resampling for high-quality downsampling without aliasing
    resampled_t1 = image_t1.convert("RGB").resize(grid_size, Image.Resampling.LANCZOS)
    resampled_t2 = image_t2.convert("RGB").resize(grid_size, Image.Resampling.LANCZOS)

    meta = {
        "original_size_t1": (w1, h1),
        "original_size_t2": (w2, h2),
        "normalized_size": grid_size,
        "downsampled": bool((w1, h1) != grid_size or (w2, h2) != grid_size),
    }

    return resampled_t1, resampled_t2, meta


def _match_channel_histogram(
    source_channel: "np.ndarray",
    template_channel: "np.ndarray",
    valid_mask: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    """
    Matches cumulative distribution function (CDF) of source to template channel
    strictly within the valid overlapping mask.
    """
    if valid_mask is not None:
        src_vals = source_channel[valid_mask > 0]
        tmpl_vals = template_channel[valid_mask > 0]
    else:
        src_vals = source_channel.ravel()
        tmpl_vals = template_channel.ravel()

    if len(src_vals) == 0 or len(tmpl_vals) == 0:
        return source_channel

    # Calculate empirical CDFs
    s_values, s_bin_idx, s_counts = np.unique(src_vals, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(tmpl_vals, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64) / src_vals.size
    t_quantiles = np.cumsum(t_counts).astype(np.float64) / tmpl_vals.size

    # Interpolate mapping
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    matched_channel = np.interp(source_channel, s_values, interp_t_values)
    return np.clip(matched_channel, 0, 255).astype(np.uint8)


def normalize_radiometry(
    image_t1: Image.Image,
    image_t2: Image.Image,
    valid_mask: Optional["np.ndarray"] = None,
    clip_percentiles: Tuple[float, float] = (1.5, 98.5),
    enable_clahe: bool = True,
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """
    Normalizes brightness, contrast, gamma, and atmospheric illumination between T1 and T2.

    Steps:
    1. Robust percentile dynamic range adjustment.
    2. Lab space CLAHE on L-channel (enhances local contrast without altering chromatic hues).
    3. Valid-mask constrained per-channel histogram matching (T2 mapped to T1 reference).
    """
    if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
        return image_t1, image_t2, {"applied": False, "reason": "OpenCV unavailable"}

    try:
        t1_arr = np.array(image_t1.convert("RGB"))
        t2_arr = np.array(image_t2.convert("RGB"))

        h, w, c = t1_arr.shape
        if valid_mask is None:
            valid_mask = np.ones((h, w), dtype=np.uint8) * 255

        # 1. Percentile stretch to suppress specular glint and sensor saturation
        low_p, high_p = clip_percentiles
        t1_stretched = t1_arr.copy().astype(np.float32)
        t2_stretched = t2_arr.copy().astype(np.float32)

        for ch in range(3):
            v_t1 = t1_arr[:, :, ch][valid_mask > 0]
            v_t2 = t2_arr[:, :, ch][valid_mask > 0]
            if len(v_t1) > 50 and len(v_t2) > 50:
                p1_low, p1_high = np.percentile(v_t1, [low_p, high_p])
                p2_low, p2_high = np.percentile(v_t2, [low_p, high_p])

                if p1_high > p1_low:
                    t1_stretched[:, :, ch] = np.clip((t1_stretched[:, :, ch] - p1_low) / (p1_high - p1_low) * 255.0, 0, 255)
                if p2_high > p2_low:
                    t2_stretched[:, :, ch] = np.clip((t2_stretched[:, :, ch] - p2_low) / (p2_high - p2_low) * 255.0, 0, 255)

        t1_clean = t1_stretched.astype(np.uint8)
        t2_clean = t2_stretched.astype(np.uint8)

        # 2. CLAHE in Lab color space (Luminance equalization)
        if enable_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            lab1 = cv2.cvtColor(t1_clean, cv2.COLOR_RGB2LAB)
            lab2 = cv2.cvtColor(t2_clean, cv2.COLOR_RGB2LAB)

            lab1[:, :, 0] = clahe.apply(lab1[:, :, 0])
            lab2[:, :, 0] = clahe.apply(lab2[:, :, 0])

            t1_clean = cv2.cvtColor(lab1, cv2.COLOR_LAB2RGB)
            t2_clean = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

        # 3. Channel-by-Channel Histogram Matching (matching T2 to T1 on valid_mask)
        t2_matched = np.empty_like(t2_clean)
        for ch in range(3):
            t2_matched[:, :, ch] = _match_channel_histogram(
                source_channel=t2_clean[:, :, ch],
                template_channel=t1_clean[:, :, ch],
                valid_mask=valid_mask,
            )

        norm_t1_pil = Image.fromarray(t1_clean)
        norm_t2_pil = Image.fromarray(t2_matched)

        return norm_t1_pil, norm_t2_pil, {
            "applied": True,
            "clahe_applied": enable_clahe,
            "histogram_matching_applied": True,
            "clip_percentiles": [low_p, high_p],
        }

    except Exception as e:
        logger.warning(f"[Radiometric Normalization Exception]: {e}")
        return image_t1, image_t2, {"applied": False, "error": str(e)}

