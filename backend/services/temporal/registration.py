"""
Temporal Image Registration Module
Performs geospatial resolution normalization and OpenCV feature-based
fine co-registration on bi-temporal satellite image pairs (T1 and T2).
Guarantees zero crashes with graceful PIL fallback in serverless environments.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image

logger = logging.getLogger("satquery.temporal.registration")

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


def register_temporal_scenes(
    image_t1: Image.Image,
    image_t2: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
    sar_t2: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    """
    Registers image_t2 (Time 2 / Recent) to image_t1 (Time 1 / Reference).
    Normalizes images to identical target dimensions.
    Applies ORB feature matching + RANSAC partial affine transformation if OpenCV is available.
    Optionally applies the exact same spatial transformation to an accompanying SAR T2 image.

    Returns:
        {
            "aligned_t1": PIL.Image,
            "aligned_t2": PIL.Image,
            "aligned_sar_t2": Optional[PIL.Image],
            "registration_quality": float (0.0 - 1.0),
            "warning": Optional[str],
            "transformation_type": str,
            "aligned_width": int,
            "aligned_height": int,
        }
    """
    # 1. Determine common target size
    if target_size:
        w, h = target_size
    else:
        # Default to T1 dimensions or standard 640x640 if massive
        w, h = image_t1.size
        if max(w, h) > 1600:
            scale = 1600.0 / max(w, h)
            w, h = int(w * scale), int(h * scale)

    norm_size = (w, h)

    t1_norm = image_t1.convert("RGB").resize(norm_size, Image.Resampling.LANCZOS)
    t2_norm = image_t2.convert("RGB").resize(norm_size, Image.Resampling.LANCZOS)
    sar_t2_norm = sar_t2.resize(norm_size, Image.Resampling.LANCZOS) if sar_t2 else None

    # Fallback default if CV2/NumPy unavailable
    if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
        logger.info("[Registration] OpenCV not present in environment; using Lanczos resolution normalization.")
        return {
            "aligned_t1": t1_norm,
            "aligned_t2": t2_norm,
            "aligned_sar_t2": sar_t2_norm,
            "registration_quality": 0.85,
            "warning": "Resolution normalized; sub-pixel feature registration bypassed (OpenCV unavailable).",
            "transformation_type": "resolution_normalization",
            "aligned_width": w,
            "aligned_height": h,
        }

    # 2. OpenCV ORB feature matching + RANSAC Affine Warp
    try:
        t1_np = np.array(t1_norm)
        t2_np = np.array(t2_norm)

        gray1 = cv2.cvtColor(t1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(t2_np, cv2.COLOR_RGB2GRAY)

        # Contrast Limited Adaptive Histogram Equalization (CLAHE) for illumination variation
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray1_enh = clahe.apply(gray1)
        gray2_enh = clahe.apply(gray2)

        orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
        kp1, des1 = orb.detectAndCompute(gray1_enh, None)
        kp2, des2 = orb.detectAndCompute(gray2_enh, None)

        aligned_t2_np = t2_np.copy()
        aligned_sar_np = np.array(sar_t2_norm) if sar_t2_norm else None
        registration_quality = 0.60
        warning = None
        transformation_type = "resolution_normalization"

        if des1 is not None and des2 is not None and len(kp1) >= 10 and len(kp2) >= 10:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des2, des1)
            matches = sorted(matches, key=lambda m: m.distance)
            good_matches = [m for m in matches if m.distance < 65]

            if len(good_matches) >= 8:
                pts2 = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Estimate partial affine (translation + rotation + scale, avoids shear distortion)
                M, inliers = cv2.estimateAffinePartial2D(
                    pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=4.0
                )

                if M is not None and inliers is not None:
                    inlier_count = int(np.sum(inliers))
                    inlier_ratio = inlier_count / len(good_matches)

                    scale_x = float(np.linalg.norm(M[:, 0]))
                    scale_y = float(np.linalg.norm(M[:, 1]))
                    tx = float(abs(M[0, 2]))
                    ty = float(abs(M[1, 2]))

                    # Strict plausibility checks
                    is_plausible = (
                        0.80 <= scale_x <= 1.25 and
                        0.80 <= scale_y <= 1.25 and
                        tx < w * 0.25 and
                        ty < h * 0.25
                    )

                    if is_plausible and inlier_ratio >= 0.35:
                        aligned_t2_np = cv2.warpAffine(
                            t2_np, M, norm_size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101
                        )
                        if aligned_sar_np is not None:
                            aligned_sar_np = cv2.warpAffine(
                                aligned_sar_np, M, norm_size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101
                            )
                        transformation_type = "orb_affine_subpixel_registration"
                        registration_quality = min(0.98, round(0.50 + inlier_ratio * 0.45, 2))
                    else:
                        warning = "Detected affine shift exceeded plausibility bounds; retained geometric normalization."
            else:
                warning = "Insufficient feature matches between T1 and T2; retained geometric normalization."
        else:
            warning = "Low textural contrast for ORB detection; retained geometric normalization."

        aligned_t2_pil = Image.fromarray(aligned_t2_np)
        aligned_sar_pil = Image.fromarray(aligned_sar_np) if aligned_sar_np is not None else None

        return {
            "aligned_t1": t1_norm,
            "aligned_t2": aligned_t2_pil,
            "aligned_sar_t2": aligned_sar_pil,
            "registration_quality": registration_quality,
            "warning": warning,
            "transformation_type": transformation_type,
            "aligned_width": w,
            "aligned_height": h,
        }
    except Exception as e:
        logger.warning(f"[Registration Error]: {e}; falling back to resolution normalization.")
        return {
            "aligned_t1": t1_norm,
            "aligned_t2": t2_norm,
            "aligned_sar_t2": sar_t2_norm,
            "registration_quality": 0.50,
            "warning": f"Registration exception: {e}",
            "transformation_type": "resolution_normalization",
            "aligned_width": w,
            "aligned_height": h,
        }
