"""
Pixel-Level Change Detection Module
Calculates radiometric differences between aligned temporal images,
applies illumination normalization, thresholding, and morphological filtering
to produce a clean visual change mask and overlay.
"""

import cv2
import io
import base64
import numpy as np
from typing import Dict, Any, Tuple
from PIL import Image


def _match_histograms(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Matches the illumination / mean & std of source to template per channel."""
    matched = np.empty_like(source, dtype=np.float32)
    for channel in range(3):
        src_c = source[:, :, channel].astype(np.float32)
        tmpl_c = template[:, :, channel].astype(np.float32)

        src_mean, src_std = np.mean(src_c), np.std(src_c) + 1e-5
        tmpl_mean, tmpl_std = np.mean(tmpl_c), np.std(tmpl_c) + 1e-5

        norm = (src_c - src_mean) / src_std
        adjusted = norm * tmpl_std + tmpl_mean
        matched[:, :, channel] = np.clip(adjusted, 0, 255)

    return matched.astype(np.uint8)


def compute_pixel_change(
    image_t1: Image.Image,
    image_t2: Image.Image,
    sensitivity_threshold: int = 32,
) -> Dict[str, Any]:
    """
    Computes radiometric pixel differences between aligned image_t1 and image_t2.
    Returns:
        {
            "change_percentage": float,
            "overlay_data_url": str (base64 RGBA PNG),
            "changed_pixels": int,
            "total_pixels": int,
        }
    """
    # Resize to identical dimensions if needed
    w, h = image_t1.size
    if image_t2.size != (w, h):
        image_t2 = image_t2.resize((w, h), Image.Resampling.LANCZOS)

    t1_np = np.array(image_t1.convert("RGB"))
    t2_np = np.array(image_t2.convert("RGB"))

    # 1. Illumination normalization
    t2_matched = _match_histograms(t2_np, t1_np)

    # 2. Gaussian smoothing to reduce sensor speckle
    t1_blur = cv2.GaussianBlur(t1_np, (5, 5), 0)
    t2_blur = cv2.GaussianBlur(t2_matched, (5, 5), 0)

    # 3. Color difference in RGB space
    diff = cv2.absdiff(t1_blur, t2_blur)
    diff_mag = np.max(diff, axis=2)

    # 4. Thresholding to create binary mask
    _, mask = cv2.threshold(diff_mag, sensitivity_threshold, 255, cv2.THRESH_BINARY)

    # 5. Morphological cleaning (remove tiny noise spots, connect solid structures)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

    # Filter out tiny connected components (< 30 px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    filtered_mask = np.zeros_like(clean_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 35:
            filtered_mask[labels == i] = 255

    changed_pixels = int(np.sum(filtered_mask > 0))
    total_pixels = int(w * h)
    change_percentage = round((changed_pixels / total_pixels) * 100.0, 2)

    # 6. Generate transparent RGBA change overlay (Red/Amber with transparency)
    # Background is fully transparent, changed areas are glowing red/amber
    rgba_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    # Cyan/Amber highlight color: Red=244, Green=63, Blue=94, Alpha=160
    changed_indices = filtered_mask > 0
    rgba_overlay[changed_indices, 0] = 239  # R
    rgba_overlay[changed_indices, 1] = 68   # G
    rgba_overlay[changed_indices, 2] = 68   # B
    rgba_overlay[changed_indices, 3] = 165  # Alpha

    # Dilate slightly for a soft glowing edge
    edge_mask = cv2.dilate(filtered_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) - filtered_mask
    rgba_overlay[edge_mask > 0, 0] = 251  # Amber glow edge
    rgba_overlay[edge_mask > 0, 1] = 146
    rgba_overlay[edge_mask > 0, 2] = 60
    rgba_overlay[edge_mask > 0, 3] = 120

    overlay_img = Image.fromarray(rgba_overlay, mode="RGBA")
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG", optimize=True)
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    overlay_data_url = f"data:image/png;base64,{overlay_b64}"

    return {
        "changePercentage": change_percentage,
        "overlayDataUrl": overlay_data_url,
        "changedPixels": changed_pixels,
        "totalPixels": total_pixels,
    }

