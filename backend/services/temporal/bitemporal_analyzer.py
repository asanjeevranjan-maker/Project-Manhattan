"""
Main Bi-Temporal Multimodal Analyzer Module
Orchestrates:
1. Ingestion of Optical & SAR imagery for Time T1 and Time T2
2. Multi-modal spatial co-registration
3. Optical + SAR representation fusion per epoch
4. Independent Grounding DINO detection + SAM2 segmentation on T1 and T2
5. Spatial object matching (appeared, disappeared, persisted, possibly_changed)
6. Mask-level land-cover delta computation (water, built-up, vegetation, soil)
7. Standalone SAR differential scattering analysis (cloud-independent radar intelligence)
8. Multi-layer visual overlay generation
9. Structured evidence compilation for Gemini/GLM interpretation
"""

import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from PIL import Image

from .fusion import fuse_optical_and_sar
from .registration import register_temporal_scenes
from .bitemporal_matcher import match_bitemporal_detections
from .land_cover_change import calculate_land_cover_deltas
from .sar_change import analyze_sar_differential_scattering
from .overlay_generator import generate_bitemporal_overlays

logger = logging.getLogger("satquery.temporal.analyzer")

# Optional detection & segmentation imports
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
        enable_sar_analysis: bool = True,
        enable_overlays: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end bi-temporal multimodal change analysis.
        """
        has_t1 = bool(t1_optical is not None or t1_sar is not None)
        has_t2 = bool(t2_optical is not None or t2_sar is not None)

        if not (has_t1 and has_t2):
            raise ValueError(
                "Both Time T1 (optical or SAR) and Time T2 (optical or SAR) imagery must be provided."
            )

        # 1. Representative Base Images for Spatial Registration
        base_t1 = t1_optical if t1_optical is not None else t1_sar
        base_t2 = t2_optical if t2_optical is not None else t2_sar

        # 2. Spatial Co-Registration (T2 aligned to T1)
        reg_info: Dict[str, Any] = {}
        if enable_registration and base_t1 is not None and base_t2 is not None:
            reg_result = register_temporal_scenes(
                image_t1=base_t1,
                image_t2=base_t2,
                sar_t2=t2_sar if (t2_optical is not None and t2_sar is not None) else None,
            )
            aligned_base_t1 = reg_result["aligned_t1"]
            aligned_base_t2 = reg_result["aligned_t2"]
            aligned_sar_t2 = reg_result.get("aligned_sar_t2") or t2_sar
            reg_info = {
                "quality": reg_result.get("registration_quality", 0.85),
                "transformation": reg_result.get("transformation_type", "resolution_normalization"),
                "warning": reg_result.get("warning"),
            }
        else:
            w, h = base_t1.size if base_t1 else (640, 640)
            aligned_base_t1 = base_t1
            aligned_base_t2 = base_t2.resize((w, h), Image.Resampling.LANCZOS) if base_t2 else None
            aligned_sar_t2 = t2_sar.resize((w, h), Image.Resampling.LANCZOS) if t2_sar else None
            reg_info = {"quality": 0.80, "transformation": "bypassed"}

        aligned_opt_t1 = aligned_base_t1 if t1_optical is not None else None
        aligned_opt_t2 = aligned_base_t2 if t2_optical is not None else None
        target_size = aligned_base_t1.size

        # 3. Single-Epoch Multimodal Representation Fusion
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

        img_w, img_h = t1_rep.size

        # 4. Land-Cover Coverage Calculation (Fallback if not passed directly)
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

        # 5. Spatiotemporal Object Matching
        matching_result = match_bitemporal_detections(
            detections_t1=detections_t1 or [],
            detections_t2=detections_t2 or [],
            img_width=img_w,
            img_height=img_h,
            date_t1=date_t1,
            date_t2=date_t2,
        )

        objects_dict = matching_result["objects"]
        objects_summary = matching_result["summary"]

        # 6. Land-Cover Delta & Change Regions Calculation
        lc_delta_result = calculate_land_cover_deltas(
            land_cover_t1=final_lc_t1,
            land_cover_t2=final_lc_t2,
            detections_t1=detections_t1,
            detections_t2=detections_t2,
            image_size=(img_w, img_h),
        )

        # 7. Standalone SAR Differential Scattering Analysis
        sar_result: Dict[str, Any] = {"available": False}
        if enable_sar_analysis and (t1_sar is not None and aligned_sar_t2 is not None):
            sar_result = analyze_sar_differential_scattering(
                sar_t1=t1_sar,
                sar_t2=aligned_sar_t2,
                target_size=target_size,
            )

        # 8. Multi-Layer Visual Overlays
        overlays: Dict[str, str] = {}
        if enable_overlays:
            overlays = generate_bitemporal_overlays(
                image_t1=t1_rep,
                image_t2=t2_rep,
                matched_objects=objects_dict,
                land_cover_change=lc_delta_result,
            )

        # 9. Structure Final Output
        result = {
            "success": True,
            "prompt": prompt,
            "dates": {"t1": date_t1, "t2": date_t2},
            "dimensions": {"width": img_w, "height": img_h},
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
            "overlays": overlays,
            "raw_counts": {
                "t1_detections": len(detections_t1 or []),
                "t2_detections": len(detections_t2 or []),
            },
        }

        logger.info(
            f"[Bi-Temporal Analysis] Complete: {objects_summary['appeared_count']} appeared, "
            f"{objects_summary['disappeared_count']} disappeared, {objects_summary['persisted_count']} persisted. "
            f"Primary shift: {lc_delta_result['primary_shift']}"
        )

        return result


bitemporal_analyzer = BiTemporalMultimodalAnalyzer()
