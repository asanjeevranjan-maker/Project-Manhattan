"""
Multi-Signal Structural Change Detection Module (Stages 7 & 8).
Calculates multiple independent physical change signals:
1. Radiometric intensity difference (Lab color distance)
2. Structural dissimilarity (1 - SSIM)
3. Edge gradient difference (Sobel magnitude)
4. Texture / High-frequency local variance difference
5. Differential SAR backscatter (when SAR imagery is provided)

Applies adaptive thresholding, morphological opening/closing, minimum-area filtering,
and nuisance suppression to produce a noise-free, verified structural change map.
"""

import io
import base64
import logging
from typing import Dict, Any, Tuple, Optional
from PIL import Image

logger = logging.getLogger("satquery.temporal.change_detection")

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


def _compute_ssim_map(gray1: "np.ndarray", gray2: "np.ndarray", ksize: int = 11, sigma: float = 1.5) -> "np.ndarray":
    """
    Computes local windowed Structural Similarity Index (SSIM) map between two grayscale images.
    Returns 2D float32 array in [-1.0, 1.0], where 1.0 indicates identical structure.
    """
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    f1 = gray1.astype(np.float32)
    f2 = gray2.astype(np.float32)

    mu1 = cv2.GaussianBlur(f1, (ksize, ksize), sigma)
    mu2 = cv2.GaussianBlur(f2, (ksize, ksize), sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(f1 ** 2, (ksize, ksize), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(f2 ** 2, (ksize, ksize), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(f1 * f2, (ksize, ksize), sigma) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-7)
    return np.clip(ssim_map, -1.0, 1.0)


def _compute_edge_gradient(gray: "np.ndarray") -> "np.ndarray":
    """Computes Sobel edge gradient magnitude normalized to [0, 255]."""
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    return mag


def _compute_texture_energy(gray: "np.ndarray", ksize: int = 7) -> "np.ndarray":
    """Computes local standard deviation representing high-frequency textural complexity."""
    f = gray.astype(np.float32)
    mean = cv2.blur(f, (ksize, ksize))
    sqr_mean = cv2.blur(f ** 2, (ksize, ksize))
    var = np.maximum(0.0, sqr_mean - mean ** 2)
    return np.sqrt(var)


def detect_multisignal_changes(
    image_t1: Image.Image,
    image_t2: Image.Image,
    valid_mask: Optional["np.ndarray"] = None,
    nuisance_mask: Optional["np.ndarray"] = None,
    sar_t1: Optional[Image.Image] = None,
    sar_t2: Optional[Image.Image] = None,
    min_area_pixels: int = 35,
) -> Dict[str, Any]:
    """
    Executes multi-signal structural change detection.

    Returns:
        {
            "structural_change_map": np.ndarray (uint8 0 or 255),
            "raw_difference_map": np.ndarray (uint8 0 or 255),
            "reliable_change_percent": float,
            "raw_difference_percent": float,
            "signals": {
                "pixel_change_score": float (0-1),
                "ssim_change_score": float (0-1),
                "edge_change_score": float (0-1),
                "texture_change_score": float (0-1),
                "sar_change_score": float (0-1),
            },
            "overlay_data_url": str (base64 RGBA PNG),
            "changed_components_count": int,
        }
    """
    if NUMPY_AVAILABLE and np is not None:
        if isinstance(image_t1, np.ndarray):
            image_t1 = Image.fromarray(image_t1)
        if isinstance(image_t2, np.ndarray):
            image_t2 = Image.fromarray(image_t2)

    w, h = image_t1.size

    if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
        dummy = np.zeros((h, w), dtype=np.uint8) if NUMPY_AVAILABLE else None
        return {
            "structural_change_map": dummy,
            "raw_difference_map": dummy,
            "reliable_change_percent": 0.0,
            "raw_difference_percent": 0.0,
            "signals": {"pixel_change_score": 0.0, "ssim_change_score": 0.0, "edge_change_score": 0.0, "texture_change_score": 0.0, "sar_change_score": 0.0},
            "overlay_data_url": "",
            "changed_components_count": 0,
        }

    try:
        t1_np = np.array(image_t1.convert("RGB"))
        t2_np = np.array(image_t2.convert("RGB"))

        if valid_mask is None:
            valid_mask = np.ones((h, w), dtype=np.uint8) * 255

        valid_indices = valid_mask > 0
        total_valid_pixels = float(max(1, np.sum(valid_indices)))

        # Check for identical images
        if np.array_equal(t1_np, t2_np):
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return {
                "structural_change_map": empty_mask,
                "raw_difference_map": empty_mask,
                "reliable_change_percent": 0.0,
                "raw_difference_percent": 0.0,
                "signals": {
                    "pixel_change_score": 0.0,
                    "ssim_change_score": 0.0,
                    "edge_change_score": 0.0,
                    "texture_change_score": 0.0,
                    "sar_change_score": 0.0,
                },
                "overlay_data_url": "",
                "changed_components_count": 0,
            }

        # -------------------------------------------------------------
        # SIGNAL 1: RADIOMETRIC INTENSITY DELTA (Lab color space)
        # -------------------------------------------------------------
        lab1 = cv2.cvtColor(t1_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(t2_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        # Delta E color distance
        delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))
        sig_intensity = np.clip(delta_e / 70.0, 0.0, 1.0)

        # -------------------------------------------------------------
        # SIGNAL 2: STRUCTURAL DISSIMILARITY (1 - SSIM)
        # -------------------------------------------------------------
        gray1 = cv2.cvtColor(t1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(t2_np, cv2.COLOR_RGB2GRAY)

        ssim_raw = _compute_ssim_map(gray1, gray2)
        # Discard similarity below 0 -> normalized dissimilarity [0, 1]
        sig_ssim = np.clip((1.0 - ssim_raw) / 1.5, 0.0, 1.0)

        # -------------------------------------------------------------
        # SIGNAL 3: EDGE GRADIENT DELTA (Sobel magnitude)
        # -------------------------------------------------------------
        edge1 = _compute_edge_gradient(gray1)
        edge2 = _compute_edge_gradient(gray2)
        edge_diff = np.abs(edge1 - edge2)
        sig_edge = np.clip(edge_diff / 120.0, 0.0, 1.0)

        # -------------------------------------------------------------
        # SIGNAL 4: TEXTURAL ENERGY DELTA
        # -------------------------------------------------------------
        text1 = _compute_texture_energy(gray1)
        text2 = _compute_texture_energy(gray2)
        text_diff = np.abs(text1 - text2)
        sig_texture = np.clip(text_diff / 35.0, 0.0, 1.0)

        # -------------------------------------------------------------
        # SIGNAL 5: SAR BACKSCATTER DELTA (if available)
        # -------------------------------------------------------------
        sig_sar = np.zeros((h, w), dtype=np.float32)
        has_sar = False
        if sar_t1 is not None and sar_t2 is not None:
            try:
                s1 = np.array(sar_t1.convert("L").resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32)
                s2 = np.array(sar_t2.convert("L").resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32)
                # Speckle reduction box filter
                s1_blur = cv2.blur(s1, (5, 5))
                s2_blur = cv2.blur(s2, (5, 5))
                sar_diff = np.abs(s2_blur - s1_blur)
                sig_sar = np.clip(sar_diff / 50.0, 0.0, 1.0)
                has_sar = True
            except Exception as se:
                logger.debug(f"[ChangeDetection] SAR processing ignored: {se}")

        # -------------------------------------------------------------
        # WEIGHTED MULTI-SIGNAL FUSION
        # -------------------------------------------------------------
        if has_sar:
            fused_change = (
                0.20 * sig_intensity +
                0.25 * sig_ssim +
                0.25 * sig_edge +
                0.15 * sig_texture +
                0.15 * sig_sar
            )
        else:
            fused_change = (
                0.20 * sig_intensity +
                0.30 * sig_ssim +
                0.30 * sig_edge +
                0.20 * sig_texture
            )

        # Mask out invalid boundaries
        fused_change[~valid_indices] = 0.0

        # Mean signal metrics across valid overlap
        mean_intensity = float(np.mean(sig_intensity[valid_indices]))
        mean_ssim = float(np.mean(sig_ssim[valid_indices]))
        mean_edge = float(np.mean(sig_edge[valid_indices]))
        mean_texture = float(np.mean(sig_texture[valid_indices]))
        mean_sar = float(np.mean(sig_sar[valid_indices])) if has_sar else 0.0

        # -------------------------------------------------------------
        # ADAPTIVE STATISTICAL THRESHOLDING
        # -------------------------------------------------------------
        valid_vals = fused_change[valid_indices]
        if len(valid_vals) > 100:
            med = float(np.median(valid_vals))
            p75 = float(np.percentile(valid_vals, 75))
            iqr = p75 - med
            adaptive_thresh = max(0.28, min(0.55, med + 1.8 * iqr))
        else:
            adaptive_thresh = 0.35

        # Raw difference (simple intensity threshold without structural filtering)
        raw_mask = (sig_intensity > 0.30) & valid_indices
        raw_diff_bin = np.where(raw_mask, 255, 0).astype(np.uint8)
        raw_difference_percent = round(float(np.sum(raw_diff_bin > 0)) / total_valid_pixels * 100.0, 2)

        # Structural binary candidate
        structural_candidate = (fused_change >= adaptive_thresh) & valid_indices

        # Nuisance Discounting: suppress water, cloud, and deep shadow false positives
        if nuisance_mask is not None:
            structural_candidate = structural_candidate & (nuisance_mask == 0)

        candidate_bin = np.where(structural_candidate, 255, 0).astype(np.uint8)

        # -------------------------------------------------------------
        # MORPHOLOGICAL CLEANUP & MINIMUM AREA FILTERING
        # -------------------------------------------------------------
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        clean_mask = cv2.morphologyEx(candidate_bin, cv2.MORPH_OPEN, kernel_open)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
        final_structural_map = np.zeros_like(clean_mask)
        valid_components = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area_pixels:
                final_structural_map[labels == i] = 255
                valid_components += 1

        reliable_changed_pixels = float(np.sum(final_structural_map > 0))
        reliable_change_percent = round((reliable_changed_pixels / total_valid_pixels) * 100.0, 2)

        # -------------------------------------------------------------
        # VISUAL OVERLAY GENERATION (Glowing amber/coral overlay)
        # -------------------------------------------------------------
        overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        changed_idx = final_structural_map > 0

        # Coral / Amber highlight
        overlay_rgba[changed_idx, 0] = 244  # R
        overlay_rgba[changed_idx, 1] = 67   # G
        overlay_rgba[changed_idx, 2] = 54   # B
        overlay_rgba[changed_idx, 3] = 170  # Alpha

        # Soft outer edge glow
        edge_glow = cv2.dilate(final_structural_map, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) - final_structural_map
        overlay_rgba[edge_glow > 0, 0] = 255  # Amber gold
        overlay_rgba[edge_glow > 0, 1] = 179
        overlay_rgba[edge_glow > 0, 2] = 0
        overlay_rgba[edge_glow > 0, 3] = 120

        overlay_pil = Image.fromarray(overlay_rgba)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        overlay_data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        return {
            "structural_change_map": final_structural_map,
            "raw_difference_map": raw_diff_bin,
            "reliable_change_percent": reliable_change_percent,
            "raw_difference_percent": raw_difference_percent,
            "signals": {
                "pixel_change_score": round(mean_intensity, 3),
                "ssim_change_score": round(mean_ssim, 3),
                "edge_change_score": round(mean_edge, 3),
                "texture_change_score": round(mean_texture, 3),
                "sar_change_score": round(mean_sar, 3),
            },
            "overlay_data_url": overlay_data_url,
            "changed_components_count": valid_components,
        }

    except Exception as e:
        logger.warning(f"[Change Detection Exception]: {e}")
        dummy = np.zeros((h, w), dtype=np.uint8)
        return {
            "structural_change_map": dummy,
            "raw_difference_map": dummy,
            "reliable_change_percent": 0.0,
            "raw_difference_percent": 0.0,
            "signals": {"pixel_change_score": 0.0, "ssim_change_score": 0.0, "edge_change_score": 0.0, "texture_change_score": 0.0, "sar_change_score": 0.0},
            "overlay_data_url": "",
            "changed_components_count": 0,
        }


# Alias for backwards compatibility
detect_structural_changes = detect_multisignal_changes

