"""
Main Bi-Temporal Multimodal Satellite Analysis Engine (Stages 1 through 24).
Orchestrates:
1. Input validation (dimensions, overlap, resolution, timestamps).
2. Spatial co-registration with SIFT/ORB + RANSAC partial affine and valid_mask isolation.
3. Spatial resolution & radiometric normalization (CLAHE, percentile clipping, histogram matching).
4. Nuisance masking (water/tide, clouds, shadows, seasonal vegetation).
5. Multi-signal structural change detection (SSIM, edge gradient, texture, intensity, SAR).
6. Single-epoch Optical + SAR representation fusion per epoch.
7. Crop-level visual verification & Grounding DINO detection dropout recovery.
8. Persistence filtering (TEMPORARY vs PERMANENT vs ENVIRONMENTAL).
9. Calibrated confidence engine & strict confidence gating (>= 90% requires multi-signal proof).
10. Multi-layer visual overlays (raw difference, reliable change, land cover delta).
11. Explainable BI_TEMPORAL_DEBUG logging.
"""

import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from PIL import Image

from .fusion import fuse_optical_and_sar
from .registration import register_temporal_scenes
from .normalization import normalize_spatial_resolution, normalize_radiometry
from .nuisance_masking import compute_nuisance_masks
from .change_detection import detect_multisignal_changes
from .bitemporal_matcher import match_bitemporal_detections
from .land_cover_change import calculate_land_cover_deltas
from .sar_change import analyze_sar_differential_scattering
from .overlay_generator import generate_bitemporal_overlays

logger = logging.getLogger("satquery.temporal.analyzer")

try:
    from ..detection.land_cover import calculate_land_cover
    from ..detection.vocabulary import filter_and_format_detections, sanitize_prompt
except ImportError:
    try:
        from services.detection.land_cover import calculate_land_cover  # type: ignore
        from services.detection.vocabulary import filter_and_format_detections, sanitize_prompt  # type: ignore
    except ImportError:
        calculate_land_cover = None  # type: ignore
        filter_and_format_detections = None  # type: ignore
        sanitize_prompt = lambda p, **kw: p  # type: ignore


class BiTemporalMultimodalAnalyzer:
    """
    Unified analyzer for bi-temporal satellite scenes across Optical and SAR modalities.
    """

    def analyze(
        self,
        t1_optical: Optional[Image.Image] = None,
        t1_sar: Optional[Image.Image] = None,
        t2_optical: Optional[Image.Image] = None,
        t2_sar: Optional[Image.Image] = None,
        prompt: str = "building, water, vegetation, road",
        date_t1: str = "Time 1",
        date_t2: str = "Time 2",
        detections_t1: Optional[List[Dict[str, Any]]] = None,
        detections_t2: Optional[List[Dict[str, Any]]] = None,
        land_cover_t1: Optional[Dict[str, Any]] = None,
        land_cover_t2: Optional[Dict[str, Any]] = None,
        enable_registration: bool = True,
        enable_normalization: bool = True,
        enable_nuisance_masking: bool = True,
        enable_sar_analysis: bool = True,
        enable_overlays: bool = True,
        aoi: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end bi-temporal multimodal change analysis with robust safeguards.
        """
        # =============================================================
        # STAGE 1: VALIDATE INPUT IMAGES
        # =============================================================
        has_t1 = bool(t1_optical is not None or t1_sar is not None)
        has_t2 = bool(t2_optical is not None or t2_sar is not None)

        if not (has_t1 and has_t2):
            raise ValueError(
                "Both Time T1 (optical or SAR) and Time T2 (optical or SAR) imagery must be provided."
            )

        base_t1 = t1_optical if t1_optical is not None else t1_sar
        base_t2 = t2_optical if t2_optical is not None else t2_sar

        w1, h1 = base_t1.size
        w2, h2 = base_t2.size

        # Check dimension ratio sanity
        dim_ratio = max(w1 / max(1, w2), w2 / max(1, w1))
        if dim_ratio > 10.0:
            logger.warning(f"[Validation] Extreme resolution mismatch between T1 ({w1}x{h1}) and T2 ({w2}x{h2}).")

        # =============================================================
        # STAGES 4 & 5: RESOLUTION NORMALIZATION (downsample to common grid)
        # =============================================================
        norm_t1, norm_t2, res_meta = normalize_spatial_resolution(base_t1, base_t2)

        # =============================================================
        # STAGES 2 & 3: CO-REGISTRATION & VALID OVERLAP MASKING
        # =============================================================
        reg_info: Dict[str, Any] = {}
        valid_mask = None

        if enable_registration:
            reg_result = register_temporal_scenes(
                image_t1=norm_t1,
                image_t2=norm_t2,
                sar_t2=t2_sar if (t2_optical is not None and t2_sar is not None) else None,
            )
            aligned_base_t1 = reg_result["aligned_t1"]
            aligned_base_t2 = reg_result["aligned_t2"]
            aligned_sar_t2 = reg_result.get("aligned_sar_t2") or t2_sar
            valid_mask = reg_result.get("valid_mask")
            reg_confidence = reg_result.get("registration_confidence", 0.80)
            overlap_pct = reg_result.get("overlap_percentage", 100.0)

            reg_info = {
                "quality": reg_confidence,
                "registration_confidence": reg_confidence,
                "transformation": reg_result.get("transformation_type", "resolution_normalization"),
                "overlap_percentage": overlap_pct,
                "reprojection_error": reg_result.get("reprojection_error", 0.0),
                "inliers_count": reg_result.get("inliers_count", 0),
                "matches_count": reg_result.get("matches_count", 0),
                "is_aligned": reg_result.get("is_aligned", True),
                "warning": reg_result.get("warning"),
            }
        else:
            aligned_base_t1 = norm_t1
            aligned_base_t2 = norm_t2
            aligned_sar_t2 = t2_sar
            reg_confidence = 0.80
            overlap_pct = 100.0
            reg_info = {
                "quality": 0.80,
                "registration_confidence": 0.80,
                "transformation": "bypassed",
                "overlap_percentage": 100.0,
                "is_aligned": True,
            }

        target_size = aligned_base_t1.size
        img_w, img_h = target_size

        # =============================================================
        # STAGE 4: RADIOMETRIC NORMALIZATION (Histogram Matching + CLAHE)
        # =============================================================
        norm_meta = {"applied": False}
        if enable_normalization and aligned_base_t1 is not None and aligned_base_t2 is not None:
            rad_t1, rad_t2, norm_meta = normalize_radiometry(
                image_t1=aligned_base_t1,
                image_t2=aligned_base_t2,
                valid_mask=valid_mask,
            )
        else:
            rad_t1, rad_t2 = aligned_base_t1, aligned_base_t2

        # =============================================================
        # STAGE 6: NUISANCE MASKING (Water, Shadows, Clouds, Vegetation)
        # =============================================================
        nuisance_data: Dict[str, Any] = {}
        if enable_nuisance_masking and rad_t1 is not None and rad_t2 is not None:
            nuisance_data = compute_nuisance_masks(
                image_t1=rad_t1,
                image_t2=rad_t2,
                valid_mask=valid_mask,
            )
        nuisance_comb = nuisance_data.get("combined_nuisance_mask")

        # =============================================================
        # STAGES 7 & 8: MULTI-SIGNAL STRUCTURAL CHANGE DETECTION
        # =============================================================
        change_res = detect_multisignal_changes(
            image_t1=rad_t1,
            image_t2=rad_t2,
            valid_mask=valid_mask,
            nuisance_mask=nuisance_comb,
            sar_t1=t1_sar,
            sar_t2=aligned_sar_t2,
        )

        structural_change_map = change_res.get("structural_change_map")
        raw_difference_percent = change_res.get("raw_difference_percent", 0.0)
        reliable_change_percent = change_res.get("reliable_change_percent", 0.0)
        signals = change_res.get("signals", {})

        # =============================================================
        # STAGE 13: SAR DIFFERENTIAL SCATTERING ANALYSIS
        # =============================================================
        sar_result: Dict[str, Any] = {"available": False}
        has_sar = bool(t1_sar is not None and aligned_sar_t2 is not None)
        sar_score = 0.0

        if enable_sar_analysis and has_sar:
            sar_result = analyze_sar_differential_scattering(
                sar_t1=t1_sar,
                sar_t2=aligned_sar_t2,
                target_size=target_size,
            )
            sar_score = min(1.0, (sar_result.get("bright_scatterers_percentage", 0.0) + sar_result.get("dark_scatterers_percentage", 0.0)) / 10.0)

        # =============================================================
        # REPRESENTATION FUSION (FOR OVERLAYS & LEGACY COMPAT)
        # =============================================================
        aligned_opt_t1 = rad_t1 if t1_optical is not None else None
        aligned_opt_t2 = rad_t2 if t2_optical is not None else None

        t1_rep, t1_fuse_meta = fuse_optical_and_sar(
            optical_image=aligned_opt_t1,
            sar_image=t1_sar,
            target_size=target_size,
        )
        t2_rep, t2_fuse_meta = fuse_optical_and_sar(
            optical_image=aligned_opt_t2,
            sar_image=aligned_sar_t2,
            target_size=target_size,
        )

        # =============================================================
        # STAGES 9, 10, 12, 16, 17: OBJECT MATCHING & CONFIDENCE ENGINE
        # =============================================================
        matching_result = match_bitemporal_detections(
            detections_t1=detections_t1 or [],
            detections_t2=detections_t2 or [],
            img_width=img_w,
            img_height=img_h,
            date_t1=date_t1,
            date_t2=date_t2,
            image_t1=rad_t1,
            image_t2=rad_t2,
            structural_change_map=structural_change_map,
            nuisance_masks=nuisance_data,
            registration_quality=reg_confidence,
            sar_score=sar_score,
            has_sar=has_sar,
            aoi=aoi,
        )

        objects_dict = matching_result["objects"]
        objects_summary = matching_result["summary"]

        # Flatten changes array matching user Stage 24 API standard
        changes_list: List[Dict[str, Any]] = []
        for cat_key, items in objects_dict.items():
            for itm in items:
                changes_list.append({
                    "id": itm.get("id"),
                    "type": itm.get("type", cat_key).upper(),
                    "class": itm.get("label", "object"),
                    "label": itm.get("label", "object"),
                    "confidence": itm.get("confidence", 0.75),
                    "verificationStatus": itm.get("verificationStatus", "POSSIBLE_CHANGE"),
                    "persistence": itm.get("persistence", "LIKELY_PERMANENT"),
                    "isGated": itm.get("isGated", False),
                    "gateReason": itm.get("gateReason"),
                    "bbox": itm.get("current_box") or itm.get("box_t2") or itm.get("box_t1"),
                    "boxT1": itm.get("box_t1"),
                    "boxT2": itm.get("box_t2"),
                    "currentBox": itm.get("current_box") or itm.get("box_t2") or itm.get("box_t1"),
                    "coordinates": {
                        "latitude": itm.get("latitude"),
                        "longitude": itm.get("longitude"),
                    },
                    "evidence": itm.get("evidence", {}),
                    "penalties": itm.get("penalties", {}),
                    "possibleArtifacts": list(itm.get("penalties", {}).keys()),
                    "description": itm.get("description") or itm.get("details", ""),
                    "details": itm.get("details") or itm.get("description", ""),
                    "location": itm.get("location", "center"),
                    "historicalDate": date_t1,
                    "latestDate": date_t2,
                })

        # =============================================================
        # LAND-COVER CHANGE CALCULATION
        # =============================================================
        final_lc_t1 = land_cover_t1
        if final_lc_t1 is None and calculate_land_cover is not None and detections_t1 is not None:
            final_lc_t1 = calculate_land_cover(
                image_size=(img_w, img_h),
                detections=detections_t1,
                segmentation_available=any("mask" in d for d in detections_t1),
                base_image=t1_rep,
            )

        final_lc_t2 = land_cover_t2
        if final_lc_t2 is None and calculate_land_cover is not None and detections_t2 is not None:
            final_lc_t2 = calculate_land_cover(
                image_size=(img_w, img_h),
                detections=detections_t2,
                segmentation_available=any("mask" in d for d in detections_t2),
                base_image=t2_rep,
            )

        lc_delta_result = calculate_land_cover_deltas(
            land_cover_t1=final_lc_t1,
            land_cover_t2=final_lc_t2,
            detections_t1=detections_t1,
            detections_t2=detections_t2,
            image_size=(img_w, img_h),
        )

        # =============================================================
        # MULTI-LAYER VISUAL OVERLAYS
        # =============================================================
        overlays: Dict[str, str] = {}
        if enable_overlays:
            overlays = generate_bitemporal_overlays(
                image_t1=t1_rep,
                image_t2=t2_rep,
                matched_objects=objects_dict,
                land_cover_change=lc_delta_result,
            )
            # Add multi-signal change overlay
            if change_res.get("overlay_data_url"):
                overlays["structural_change"] = change_res["overlay_data_url"]
                overlays["change"] = change_res["overlay_data_url"]

        # =============================================================
        # STAGE 23: EXPLAINABLE BI_TEMPORAL_DEBUG LOGGING
        # =============================================================
        logger.info(
            f"\n=======================================================\n"
            f"  BI_TEMPORAL_DEBUG REPORT\n"
            f"=======================================================\n"
            f"Image 1 Date        : {date_t1}\n"
            f"Image 2 Date        : {date_t2}\n"
            f"Registration Score  : {reg_confidence:.2f} ({reg_info.get('transformation')})\n"
            f"Inliers / Reproj Err: {reg_info.get('inliers_count')} inliers, {reg_info.get('reprojection_error')}px error\n"
            f"Valid Overlap       : {overlap_pct:.1f}%\n"
            f"Radiometric Norm    : {norm_meta.get('applied', False)}\n"
            f"Raw Difference      : {raw_difference_percent:.2f}%\n"
            f"Reliable Structural : {reliable_change_percent:.2f}%\n"
            f"Nuisance Areas      : Water: {nuisance_data.get('percentages', {}).get('water', 0)}%, Shadow: {nuisance_data.get('percentages', {}).get('shadow', 0)}%\n"
            f"Change Signals      : Intensity: {signals.get('pixel_change_score')}, SSIM: {signals.get('ssim_change_score')}, Edge: {signals.get('edge_change_score')}, Texture: {signals.get('texture_change_score')}\n"
            f"Objects Summary     : Appeared: {objects_summary.get('appeared_count')}, Persisted: {objects_summary.get('persisted_count')}, Disappeared: {objects_summary.get('disappeared_count')}\n"
            f"=======================================================\n"
        )

        # =============================================================
        # STAGE 24: FINAL STRUCTURED RESPONSE
        # =============================================================
        result = {
            "success": True,
            "prompt": prompt,
            "dates": {"t1": date_t1, "t2": date_t2},
            "dimensions": {"width": img_w, "height": img_h},
            "comparison": {
                "date1": date_t1,
                "date2": date_t2,
                "registrationConfidence": reg_confidence,
                "rawDifferencePercent": raw_difference_percent,
                "reliableChangePercent": reliable_change_percent,
                "validOverlapPercent": overlap_pct,
                "structuralChangeComponents": change_res.get("changed_components_count", 0),
            },
            "changes": changes_list,
            "summary": {
                "totalBefore": len(detections_t1 or []),
                "totalLatest": len(detections_t2 or []),
                "newCount": objects_summary.get("appeared_count", 0),
                "removedCount": objects_summary.get("disappeared_count", 0),
                "modifiedCount": objects_summary.get("possibly_changed_count", 0),
                "unchangedCount": objects_summary.get("persisted_count", 0),
                "totalChanges": objects_summary.get("appeared_count", 0) + objects_summary.get("disappeared_count", 0) + objects_summary.get("possibly_changed_count", 0),
                "rawDifferencePercent": raw_difference_percent,
                "reliableChangePercent": reliable_change_percent,
            },
            "pixelChange": {
                "change_percentage": reliable_change_percent,
                "raw_difference_percentage": raw_difference_percent,
                "overlay_data_url": change_res.get("overlay_data_url", ""),
            },
            "modalities": {
                "t1": t1_fuse_meta,
                "t2": t2_fuse_meta,
            },
            "registration": reg_info,
            "objects": objects_dict,
            "objects_summary": objects_summary,
            "land_cover_change": lc_delta_result["land_cover_change"],
            "coverage_t1": lc_delta_result["coverage_t1"],
            "coverage_t2": lc_delta_result["coverage_t2"],
            "change_regions": lc_delta_result["change_regions"],
            "primary_shift": lc_delta_result["primary_shift"],
            "sar_analysis": sar_result,
            "signals": signals,
            "nuisance": nuisance_data.get("percentages", {}),
            "overlays": overlays,
            "raw_counts": {
                "t1_detections": len(detections_t1 or []),
                "t2_detections": len(detections_t2 or []),
            },
        }

        return result


bitemporal_analyzer = BiTemporalMultimodalAnalyzer()
