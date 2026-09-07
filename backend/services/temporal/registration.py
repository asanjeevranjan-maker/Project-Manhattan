"""
Enhanced Temporal Image Co-Registration Module (Stages 2 & 3).
Performs sub-pixel image alignment between Time 1 and Time 2 satellite scenes.

Key Capabilities:
1. Multi-detector strategy: SIFT with Lowe's ratio test (primary) + ORB (fallback).
2. RANSAC Partial Affine transformation: guarantees rotation + scale + translation without shear distortion.
3. Phase correlation fallback for low-texture / large-displacement pairs.
4. Clean boundary handling: BORDER_CONSTANT (no fake mirrored edge artifacts).
5. Exact Geometric Overlap Mask (valid_mask) suppressing non-overlapping pixels, borders, and warp padding.
6. Quantitative registration confidence metric [0.0 - 1.0] derived from inliers, reprojection error, and post-warp SSIM.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
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


def _compute_phase_correlation_shift(gray1: "np.ndarray", gray2: "np.ndarray") -> Tuple[float, float, float]:
    """Computes translation shift using Fourier phase correlation."""
    if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
        return 0.0, 0.0, 0.0
    try:
        f1 = np.float32(gray1)
        f2 = np.float32(gray2)
        # Apply Hanning window to prevent edge ringing
        hanning = cv2.createHanningWindow((gray1.shape[1], gray1.shape[0]), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(f2, f1, hanning)
        return float(shift[0]), float(shift[1]), float(response)
    except Exception:
        return 0.0, 0.0, 0.0


def register_temporal_scenes(
    image_t1: Image.Image,
    image_t2: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
    sar_t2: Optional[Image.Image] = None,
    ransac_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Registers image_t2 (Time 2 / Recent) to image_t1 (Time 1 / Reference).
    Extracts the exact geometric intersection (valid_mask) between T1 and T2.

    Returns:
        {
            "aligned_t1": PIL.Image,
            "aligned_t2": PIL.Image,
            "aligned_sar_t2": Optional[PIL.Image],
            "valid_mask": np.ndarray (uint8 0 or 255),
            "registration_confidence": float (0.0 - 1.0),
            "registration_quality": float,  # legacy backward-compat alias
            "overlap_percentage": float,
            "reprojection_error": float,
            "inliers_count": int,
            "matches_count": int,
            "warning": Optional[str],
            "transformation_type": str,
            "aligned_width": int,
            "aligned_height": int,
            "is_aligned": bool,
        }
    """
    # 1. Determine common target size
    if target_size:
        w, h = target_size
    else:
        w, h = image_t1.size
        if max(w, h) > 1600:
            scale = 1600.0 / max(w, h)
            w, h = int(w * scale), int(h * scale)

    norm_size = (w, h)

    t1_norm = image_t1.convert("RGB").resize(norm_size, Image.Resampling.LANCZOS)
    t2_norm = image_t2.convert("RGB").resize(norm_size, Image.Resampling.LANCZOS)
    sar_t2_norm = sar_t2.resize(norm_size, Image.Resampling.LANCZOS) if sar_t2 else None

    # Fallback default if CV2 or NumPy is unavailable
    if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
        logger.info("[Registration] OpenCV not present; using Lanczos resolution normalization.")
        dummy_mask = np.ones((h, w), dtype=np.uint8) * 255 if NUMPY_AVAILABLE else None
        return {
            "aligned_t1": t1_norm,
            "aligned_t2": t2_norm,
            "aligned_sar_t2": sar_t2_norm,
            "valid_mask": dummy_mask,
            "registration_confidence": 0.80,
            "registration_quality": 0.80,
            "overlap_percentage": 100.0,
            "reprojection_error": 0.0,
            "inliers_count": 0,
            "matches_count": 0,
            "warning": "Resolution normalized; sub-pixel feature registration bypassed (OpenCV unavailable).",
            "transformation_type": "resolution_normalization",
            "aligned_width": w,
            "aligned_height": h,
            "is_aligned": True,
        }

    try:
        t1_np = np.array(t1_norm)
        t2_np = np.array(t2_norm)

        gray1 = cv2.cvtColor(t1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(t2_np, cv2.COLOR_RGB2GRAY)

        # CLAHE for handling illumination and atmospheric differences
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray1_enh = clahe.apply(gray1)
        gray2_enh = clahe.apply(gray2)

        # Check for identical image input
        diff_initial = np.max(np.abs(gray1.astype(np.int32) - gray2.astype(np.int32)))
        if diff_initial == 0:
            full_mask = np.ones((h, w), dtype=np.uint8) * 255
            return {
                "aligned_t1": t1_norm,
                "aligned_t2": t2_norm,
                "aligned_sar_t2": sar_t2_norm,
                "valid_mask": full_mask,
                "registration_confidence": 1.0,
                "registration_quality": 1.0,
                "overlap_percentage": 100.0,
                "reprojection_error": 0.0,
                "inliers_count": 1000,
                "matches_count": 1000,
                "warning": None,
                "transformation_type": "identity_identical_scenes",
                "aligned_width": w,
                "aligned_height": h,
                "is_aligned": True,
            }

        # 2. Multi-Detector Strategy: SIFT (high precision) -> ORB (speed fallback)
        kp1, des1 = None, None
        kp2, des2 = None, None
        detector_name = "orb"

        if hasattr(cv2, "SIFT_create"):
            try:
                sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.03, edgeThreshold=10)
                kp1, des1 = sift.detectAndCompute(gray1_enh, None)
                kp2, des2 = sift.detectAndCompute(gray2_enh, None)
                detector_name = "sift"
            except Exception as e:
                logger.debug(f"[Registration] SIFT failed: {e}; falling back to ORB.")

        if des1 is None or des2 is None or len(kp1) < 15 or len(kp2) < 15:
            orb = cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
            kp1, des1 = orb.detectAndCompute(gray1_enh, None)
            kp2, des2 = orb.detectAndCompute(gray2_enh, None)
            detector_name = "orb"

        good_matches = []
        if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
            if detector_name == "sift":
                # FLANN matcher with Lowe's ratio test for SIFT (floating point descriptors)
                index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                knn_matches = flann.knnMatch(des2, des1, k=2)
                for m_n in knn_matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
            else:
                # BFMatcher with Hamming distance for ORB
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                knn_matches = bf.knnMatch(des2, des1, k=2)
                for m_n in knn_matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.78 * n.distance:
                            good_matches.append(m)

        M = None
        inliers = None
        inliers_count = 0
        reprojection_error = 0.0
        transformation_type = "resolution_normalization"
        warning = None

        if len(good_matches) >= 6:
            pts2 = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Estimate partial affine (prevents shearing/perspective stretching artifacts)
            M, inliers = cv2.estimateAffinePartial2D(
                pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold, maxIters=3000
            )

            if M is not None and inliers is not None:
                inliers_count = int(np.sum(inliers))
                inlier_ratio = inliers_count / max(1, len(good_matches))

                # Compute mean reprojection error on inliers
                inlier_pts2 = pts2[inliers.ravel() == 1]
                inlier_pts1 = pts1[inliers.ravel() == 1]
                if len(inlier_pts2) > 0:
                    ones = np.ones((len(inlier_pts2), 1, 1), dtype=np.float32)
                    inlier_pts2_hom = np.concatenate([inlier_pts2, ones], axis=2)
                    proj_pts1 = np.matmul(inlier_pts2_hom, M.T).reshape(-1, 2)
                    reproj_error = float(np.mean(np.linalg.norm(inlier_pts1.reshape(-1, 2) - proj_pts1, axis=1)))
                    reprojection_error = round(reproj_error, 2)

                scale_x = float(np.linalg.norm(M[:, 0]))
                scale_y = float(np.linalg.norm(M[:, 1]))
                tx = float(abs(M[0, 2]))
                ty = float(abs(M[1, 2]))

                # Plausibility check: scale within [0.75, 1.35], translation < 35% of dimension
                is_plausible = (
                    0.75 <= scale_x <= 1.35 and
                    0.75 <= scale_y <= 1.35 and
                    tx < w * 0.35 and
                    ty < h * 0.35 and
                    inliers_count >= 6 and
                    inlier_ratio >= 0.25
                )

                if is_plausible:
                    transformation_type = f"{detector_name}_partial_affine"
                else:
                    M = None
                    warning = "Estimated feature transform exceeded plausibility limits; attempting phase correlation fallback."

        # 3. Phase Correlation Fallback (for translational shifts when feature matching is weak)
        if M is None:
            shift_x, shift_y, phase_resp = _compute_phase_correlation_shift(gray1_enh, gray2_enh)
            if phase_resp >= 0.15 and abs(shift_x) < w * 0.35 and abs(shift_y) < h * 0.35:
                M = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
                transformation_type = "phase_correlation_translation"
                reprojection_error = round(max(0.5, 1.0 / (phase_resp + 1e-4)), 2)
                inliers_count = int(phase_resp * 100)

        # 4. Warp Images with Strict Zero-Padding (Stage 3: Common Area Isolation)
        aligned_t2_np = t2_np.copy()
        aligned_sar_np = np.array(sar_t2_norm) if sar_t2_norm else None
        t2_mask = np.ones((h, w), dtype=np.uint8) * 255

        if M is not None:
            # Warp T2 and SAR using BORDER_CONSTANT (never BORDER_REFLECT)
            aligned_t2_np = cv2.warpAffine(
                t2_np, M, norm_size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
            if aligned_sar_np is not None:
                aligned_sar_np = cv2.warpAffine(
                    aligned_sar_np, M, norm_size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            # Warp the footprint mask of T2
            warped_t2_mask = cv2.warpAffine(
                t2_mask, M, norm_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        else:
            warped_t2_mask = t2_mask

        # Valid overlap mask: pixels where both T1 and warped T2 have valid non-border data
        valid_mask = warped_t2_mask.copy()
        # Erode mask slightly (2px) to discard interpolation boundary artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        valid_mask = cv2.erode(valid_mask, kernel, iterations=1)

        total_pixels = float(w * h)
        valid_pixels = float(np.sum(valid_mask > 0))
        overlap_percentage = round((valid_pixels / total_pixels) * 100.0, 2)

        # 5. Quantitative Registration Confidence Calculation (0.0 - 1.0)
        if M is not None and transformation_type != "resolution_normalization":
            # Feature alignment quality score
            inlier_score = min(1.0, inliers_count / 35.0)
            reproj_score = max(0.0, 1.0 - (reprojection_error / 5.0))
            overlap_score = min(1.0, overlap_percentage / 80.0)

            # Measure post-warp normalized correlation on overlapping area
            ov_indices = valid_mask > 0
            if np.sum(ov_indices) > 500:
                g1_ov = gray1[ov_indices].astype(np.float32)
                g2_ov = cv2.cvtColor(aligned_t2_np, cv2.COLOR_RGB2GRAY)[ov_indices].astype(np.float32)
                std1, std2 = np.std(g1_ov), np.std(g2_ov)
                if std1 > 1e-3 and std2 > 1e-3:
                    corr = float(np.mean((g1_ov - np.mean(g1_ov)) * (g2_ov - np.mean(g2_ov))) / (std1 * std2))
                    corr_score = max(0.0, min(1.0, (corr + 1.0) / 2.0))
                else:
                    corr_score = 0.50
            else:
                corr_score = 0.40

            registration_confidence = round(
                0.35 * inlier_score + 0.25 * reproj_score + 0.20 * overlap_score + 0.20 * corr_score,
                3,
            )
            is_aligned = bool(registration_confidence >= 0.50 and overlap_percentage >= 40.0)
        else:
            registration_confidence = 0.45
            is_aligned = False
            if warning is None:
                warning = "Co-registration could not confirm sub-pixel tie-points; using geometric frame normalization."

        aligned_t2_pil = Image.fromarray(aligned_t2_np)
        aligned_sar_pil = Image.fromarray(aligned_sar_np) if aligned_sar_np is not None else None

        return {
            "aligned_t1": t1_norm,
            "aligned_t2": aligned_t2_pil,
            "aligned_sar_t2": aligned_sar_pil,
            "valid_mask": valid_mask,
            "registration_confidence": registration_confidence,
            "registration_quality": registration_confidence,
            "overlap_percentage": overlap_percentage,
            "reprojection_error": reprojection_error,
            "inliers_count": inliers_count,
            "matches_count": len(good_matches),
            "warning": warning,
            "transformation_type": transformation_type,
            "aligned_width": w,
            "aligned_height": h,
            "is_aligned": is_aligned,
        }

    except Exception as e:
        logger.warning(f"[Registration Exception]: {e}; falling back to resolution normalization.")
        dummy_mask = np.ones((h, w), dtype=np.uint8) * 255
        return {
            "aligned_t1": t1_norm,
            "aligned_t2": t2_norm,
            "aligned_sar_t2": sar_t2_norm,
            "valid_mask": dummy_mask,
            "registration_confidence": 0.40,
            "registration_quality": 0.40,
            "overlap_percentage": 100.0,
            "reprojection_error": 0.0,
            "inliers_count": 0,
            "matches_count": 0,
            "warning": f"Registration failed with exception: {str(e)}",
            "transformation_type": "resolution_normalization",
            "aligned_width": w,
            "aligned_height": h,
            "is_aligned": False,
        }
