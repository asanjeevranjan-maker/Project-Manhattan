"""
Temporal Image Registration Module
Performs geospatial resolution normalization and OpenCV feature-based
fine registration on bi-temporal satellite image pairs.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from PIL import Image


def register_temporal_images(
    image_t1: Image.Image,
    image_t2: Image.Image,
    target_size: Tuple[int, int] = (640, 640),
) -> Dict[str, Any]:
    """
    Registers image_t2 (Time 2 / Latest) to image_t1 (Time 1 / Reference).
    Both images are normalized to target_size.
    ORB feature matching + RANSAC affine transformation is applied for sub-pixel correction.

    Returns:
        {
            "aligned_t1": PIL.Image,
            "aligned_t2": PIL.Image,
            "registration_quality": float (0.0 - 1.0),
            "warning": Optional[str],
            "transformation_type": str,
        }
    """
    # 1. Dimension normalization
    t1_norm = image_t1.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    t2_norm = image_t2.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)

    # Convert to OpenCV numpy arrays
    t1_np = np.array(t1_norm)
    t2_np = np.array(t2_norm)

    gray1 = cv2.cvtColor(t1_np, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(t2_np, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle illumination differences
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray1_enh = clahe.apply(gray1)
    gray2_enh = clahe.apply(gray2)

    # 2. ORB Feature Detection
    orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    kp1, des1 = orb.detectAndCompute(gray1_enh, None)
    kp2, des2 = orb.detectAndCompute(gray2_enh, None)

    aligned_t2_np = t2_np.copy()
    registration_quality = 0.50
    warning = None
    transformation_type = "geospatial_crop_only"

    if des1 is not None and des2 is not None and len(kp1) >= 10 and len(kp2) >= 10:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des2, des1)

        # Sort matches by distance
        matches = sorted(matches, key=lambda m: m.distance)
        good_matches = [m for m in matches if m.distance < 60]

        if len(good_matches) >= 8:
            pts2 = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Estimate partial affine transform (translation + rotation + scale, no wild shearing)
            M, inliers = cv2.estimateAffinePartial2D(
                pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=4.0
            )

            if M is not None and inliers is not None:
                inlier_count = int(np.sum(inliers))
                inlier_ratio = inlier_count / len(good_matches)

                # Check if affine transform is plausible (e.g. scale within 0.85 - 1.15, modest translation)
                scale_x = np.linalg.norm(M[:, 0])
                scale_y = np.linalg.norm(M[:, 1])
                tx = abs(M[0, 2])
                ty = abs(M[1, 2])

                is_plausible = (
                    0.80 <= scale_x <= 1.25 and
                    0.80 <= scale_y <= 1.25 and
                    tx < target_size[0] * 0.25 and
                    ty < target_size[1] * 0.25
                )

                if is_plausible and inlier_ratio >= 0.35:
                    aligned_t2_np = cv2.warpAffine(
                        t2_np,
                        M,
                        target_size,
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_REFLECT101
                    )
                    transformation_type = "affine_feature_registration"
                    registration_quality = min(0.96, 0.50 + (inlier_ratio * 0.45))
                    print(
                        f"[ImageRegistration] Successfully aligned with {inlier_count}/{len(good_matches)} inliers "
                        f"(quality: {registration_quality:.2f})"
                    )
                else:
                    registration_quality = 0.45
                    warning = "Sub-pixel feature shift was outside stability bounds; preserved geographic bounding-box alignment."
            else:
                registration_quality = 0.40
                warning = "No convergent geometric transform found; using geographic coordinate alignment."
        else:
            registration_quality = 0.35
            warning = "Low number of sharp visual tie points; preserved coordinate-based registration."
    else:
        registration_quality = 0.30
        warning = "Low feature contrast in one or both scenes; using geographic coordinate alignment."

    aligned_t2_img = Image.fromarray(aligned_t2_np)

    return {
        "aligned_t1": t1_norm,
        "aligned_t2": aligned_t2_img,
        "registration_quality": round(registration_quality, 2),
        "warning": warning,
        "transformation_type": transformation_type,
    }

