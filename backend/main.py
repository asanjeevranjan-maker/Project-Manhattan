import sys
from pathlib import Path

# Ensure backend directory and project dependencies are in sys.path
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

_ai_service_dir = str(Path(__file__).resolve().parent.parent / "ai-service")
if _ai_service_dir not in sys.path:
    sys.path.append(_ai_service_dir)

import time
import logging
import base64
import json
import os
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.vision.vision_service import vision_service, _load_env_if_missing
from services.vision.response_parser import to_legacy_analysis_result
from services.vision.base_provider import (
    VisionProviderError,
    VisionProviderAuthError,
    VisionProviderRateLimitError,
)
from services.detection.vocabulary import (
    SATELLITE_CLASSES,
    ANALYSIS_PRESETS,
    DEFAULT_CLASS_THRESHOLDS,
    sanitize_prompt,
)
from services.temporal import bitemporal_analyzer
from services.vision.image_processor import decode_data_url
from PIL import Image
import io

_load_env_if_missing()

logger = logging.getLogger("satquery.api")

app = FastAPI(
    title="SatQuery AI Backend",
    description="SatQuery Grounding DINO and Multi-Temporal Satellite Analysis API with Enhanced Multimodal Vision Pipeline",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from services.geospatial import RASTERIO_AVAILABLE
    from services.geospatial.router import router as geospatial_router
    app.include_router(geospatial_router)
    GEOSPATIAL_ROUTER_AVAILABLE = True
except Exception as _geo_err:
    logger.warning(f"Could not mount geospatial router: {_geo_err}")
    RASTERIO_AVAILABLE = False
    GEOSPATIAL_ROUTER_AVAILABLE = False


class AnalyzeRequest(BaseModel):
    user_query: Optional[str] = Field(None, description="Analytical query for satellite imagery")
    query: Optional[str] = Field(None, description="Alternative field for user query")
    imageDataUrl: Optional[str] = Field(None, description="Base64 data URL or raw base64 of primary image")
    image_data: Optional[str] = Field(None, description="Alternative field for image data")
    secondImageDataUrl: Optional[str] = Field(None, description="Base64 data URL of second image for change detection")
    second_image_data: Optional[str] = Field(None, description="Alternative field for second image data")
    provider: Optional[str] = Field("auto", description="Model provider: auto | gemini | glm | ensemble")
    model: Optional[str] = Field(None, description="Alternative field for provider/model")
    analysis_mode: Optional[str] = Field("general", description="general | objects | changes | urban | agriculture | water | infrastructure | disaster | landcover | custom")
    analysis_depth: Optional[str] = Field("standard", description="standard | deep")
    use_detections: Optional[bool] = Field(True, description="Whether to include Grounding DINO detection context")
    use_change_context: Optional[bool] = Field(True, description="Whether to include change detection context")
    use_tiles: Optional[bool] = Field(False, description="Whether to run 2x2 spatial quadrant tiling")
    detectionContext: Optional[Dict[str, Any]] = Field(None, description="Machine-generated Grounding DINO detections")
    detection_context: Optional[Dict[str, Any]] = Field(None, description="Alternative field for detection context")
    changeContext: Optional[Dict[str, Any]] = Field(None, description="Machine-generated change detection context")
    change_context: Optional[Dict[str, Any]] = Field(None, description="Alternative field for change context")
    segmentation_summary: Optional[Dict[str, Any]] = Field(None, description="SAM2 segmentation metadata")
    segmentation: Optional[Dict[str, Any]] = Field(None, description="Alternative field for segmentation summary")
    land_cover: Optional[Dict[str, Any]] = Field(None, description="Objective mask-measured land cover statistics")
    landCover: Optional[Dict[str, Any]] = Field(None, description="Alternative field for land cover")
    image_metadata: Optional[Dict[str, Any]] = Field(None, description="Optional image metadata (resolution, date, sensor)")
    history: Optional[List[Any]] = Field(default_factory=list, description="Optional conversational history")


class BiTemporalMultimodalRequest(BaseModel):
    t1_optical: Optional[str] = Field(None, description="Base64 data URL of T1 optical imagery")
    t1_sar: Optional[str] = Field(None, description="Base64 data URL of T1 SAR radar imagery")
    t2_optical: Optional[str] = Field(None, description="Base64 data URL of T2 optical imagery")
    t2_sar: Optional[str] = Field(None, description="Base64 data URL of T2 SAR radar imagery")
    prompt: Optional[str] = Field("building, water, vegetation, road", description="Objects of interest")
    date_t1: Optional[str] = Field("Time 1", description="Date/epoch of T1")
    date_t2: Optional[str] = Field("Time 2", description="Date/epoch of T2")
    user_query: Optional[str] = Field("Analyze all land-cover and structural changes between Time 1 and Time 2.", description="User analytical query")
    provider: Optional[str] = Field("auto", description="VLM provider")
    detections_t1: Optional[List[Dict[str, Any]]] = Field(None, description="Pre-computed T1 detections")
    detections_t2: Optional[List[Dict[str, Any]]] = Field(None, description="Pre-computed T2 detections")
    land_cover_t1: Optional[Dict[str, Any]] = Field(None, description="T1 land cover")
    land_cover_t2: Optional[Dict[str, Any]] = Field(None, description="T2 land cover")
    enable_vlm_interpretation: Optional[bool] = Field(True, description="Whether to run Gemini/GLM interpretation")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "SatQuery AI API",
        "version": "2.1.0",
        "vision_providers": vision_service.get_available_providers(),
        "geospatial_available": RASTERIO_AVAILABLE,
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SatQuery AI API",
        "version": "2.1.0",
        "providers_available": vision_service.get_available_providers(),
        "geospatial_available": RASTERIO_AVAILABLE,
    }


@app.get("/classes")
@app.get("/api/classes")
def get_classes():
    """Returns the catalog of 16 observable satellite object classes and their aliases."""
    return {"classes": SATELLITE_CLASSES}


@app.get("/presets")
@app.get("/api/presets")
def get_presets():
    """Returns analysis-specific class presets (general, urban, water, agriculture, infrastructure, disaster, maritime)."""
    return {"presets": ANALYSIS_PRESETS}


@app.get("/thresholds")
@app.get("/api/thresholds")
def get_thresholds():
    """Returns class-specific configurable score and geometry thresholds."""
    return {
        "thresholds": {
            k: {
                "box_threshold": v.box_threshold,
                "text_threshold": v.text_threshold,
                "min_score": v.min_score,
                "max_area_ratio": v.max_area_ratio,
                "min_aspect": v.min_aspect,
                "max_aspect": v.max_aspect,
            }
            for k, v in DEFAULT_CLASS_THRESHOLDS.items()
        }
    }


try:
    from grounding_dino import detect_objects
    GROUNDING_DINO_AVAILABLE = True
except Exception as _dino_err:
    logger.warning(f"Could not import grounding_dino: {_dino_err}")
    GROUNDING_DINO_AVAILABLE = False
    detect_objects = None


@app.post("/detect")
@app.post("/api/detect")
async def detect_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    preset: Optional[str] = Form(None),
    use_tiles: Optional[bool] = Form(None),
    iou_threshold: Optional[float] = Form(None),
    merge_mode: Optional[str] = Form("standard"),
    enable_segmentation: Optional[bool] = Form(True),
    enable_verification: Optional[bool] = Form(None),
    verification_threshold: Optional[float] = Form(None),
):
    """
    Grounding DINO Object Detection Endpoint.
    Executes text-prompted zero-shot detection, tiling, NMS deduplication, SigLIP verification, and SAM2 segmentation.
    """
    if not GROUNDING_DINO_AVAILABLE or detect_objects is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Grounding DINO model is not currently available or failed to load."},
        )

    if file is None:
        return JSONResponse(status_code=400, content={"error": "No image file was provided."})

    if not prompt or not prompt.strip():
        return JSONResponse(status_code=400, content={"error": "Detection prompt is required."})

    image_bytes = await file.read()
    if not image_bytes:
        return JSONResponse(status_code=400, content={"error": "Uploaded image is empty."})

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as image_error:
        return JSONResponse(status_code=400, content={"error": "Unable to read the uploaded image.", "details": str(image_error)})

    try:
        detections, tiling_meta = detect_objects(
            image=image,
            prompt=prompt.strip(),
            preset=preset,
            use_tiles=use_tiles,
            iou_threshold=iou_threshold,
            merge_mode=merge_mode or "standard",
            enable_segmentation=enable_segmentation,
            enable_verification=enable_verification,
            verification_threshold=verification_threshold,
            return_tiling_metadata=True,
        )

        if detections is None:
            detections = []

        dedup_stats = tiling_meta.get("deduplication") if isinstance(tiling_meta, dict) else None
        seg_meta = tiling_meta.get("segmentation") if isinstance(tiling_meta, dict) else {}
        seg_avail = seg_meta.get("segmentation_available", False) if isinstance(seg_meta, dict) else False
        land_cover = tiling_meta.get("land_cover") if isinstance(tiling_meta, dict) else None
        verification_meta = tiling_meta.get("verification") if isinstance(tiling_meta, dict) else {}
        verification_avail = verification_meta.get("verification_available", False) if isinstance(verification_meta, dict) else False

        result = {
            "count": len(detections),
            "width": image.width,
            "height": image.height,
            "prompt": prompt.strip(),
            "preset": preset,
            "detections": detections,
            "tiling": tiling_meta,
            "deduplication": dedup_stats,
            "segmentation_available": seg_avail,
            "segmentation": seg_meta,
            "mask_overlay_url": seg_meta.get("mask_overlay_url") or seg_meta.get("overlay_preview"),
            "land_cover": land_cover,
            "verification_available": verification_avail,
            "verification": verification_meta,
        }

        return JSONResponse(status_code=200, content=result)
    except Exception as error:
        logger.error(f"[Grounding DINO Error]: {error}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Detection failed.", "details": str(error)})


@app.post("/analyze")
@app.post("/api/analyze")
@app.post("/api/backend/analyze")
async def analyze_image_endpoint(req: AnalyzeRequest):
    """
    Multimodal Satellite Image Analysis Endpoint.
    Supports:
      - Advanced satellite prompting & anti-hallucination rules
      - Grounding DINO & change detection context integration
      - Gemini, GLM, auto-fallback, and ensemble modes
      - Standard and deep two-stage analysis
      - Optional 2x2 spatial quadrant tiling
      - 100% backward-compatible responses for the Next.js frontend
    """
    image_data = req.imageDataUrl or req.image_data
    if not image_data:
        raise HTTPException(status_code=400, detail="Missing required image data (imageDataUrl or image_data).")

    user_query = (req.user_query or req.query or "").strip()
    if not user_query:
        user_query = "Provide a comprehensive remote-sensing assessment of this satellite image."

    provider = req.provider or req.model or "auto"
    second_image = req.secondImageDataUrl or req.second_image_data
    detection_context = req.detectionContext or req.detection_context
    change_context = req.changeContext or req.change_context
    segmentation_summary = req.segmentation_summary or req.segmentation
    land_cover = req.land_cover or req.landCover

    try:
        result = await vision_service.analyze_image(
            image_data=image_data,
            user_query=user_query,
            provider=provider,
            analysis_mode=req.analysis_mode or "general",
            analysis_depth=req.analysis_depth or "standard",
            use_detections=req.use_detections if req.use_detections is not None else True,
            use_change_context=req.use_change_context if req.use_change_context is not None else True,
            use_tiles=req.use_tiles if req.use_tiles is not None else False,
            detection_context=detection_context,
            change_context=change_context,
            image_metadata=req.image_metadata,
            second_image_data=second_image,
            segmentation_summary=segmentation_summary,
            land_cover=land_cover,
        )

        structured = result["structured_analysis"]
        legacy_analysis = to_legacy_analysis_result(
            structured,
            intent=req.analysis_mode or "image_understanding",
            land_cover_result=land_cover,
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "provider": result["provider_used"],
                "analysis_mode": result["analysis_mode"],
                "query": result["query"],
                "answer": structured.answer or structured.answer_to_query,
                "summary": structured.summary,
                "evidence": structured.evidence.model_dump(),
                "calculated_statistics": structured.calculated_statistics,
                "observations": [o.model_dump() for o in structured.observations],
                "uncertainties": structured.uncertainties,
                "context": {
                    "grounding_dino_used": detection_context is not None and bool(detection_context.get("detections")),
                    "change_detection_used": change_context is not None,
                    "segmentation_used": segmentation_summary is not None and bool(segmentation_summary.get("segmentation_available")),
                    "land_cover_used": land_cover is not None and bool(land_cover.get("available")),
                },
                "processing_time_ms": result["processing_time_ms"],
                "structured_analysis": structured.model_dump(),
                # Zero-regression backward compatibility:
                "analysis": legacy_analysis,
                "modelUsed": result["provider_used"],
                "fallbackUsed": result["fallback_used"],
            },
        )

    except VisionProviderAuthError as ae:
        logger.error(f"[API Auth Error]: {ae}")
        raise HTTPException(status_code=401, detail=str(ae))
    except VisionProviderRateLimitError as re:
        logger.error(f"[API Rate Limit]: {re}")
        raise HTTPException(status_code=429, detail=str(re))
    except VisionProviderError as ve:
        logger.error(f"[API Provider Error]: {ve}")
        raise HTTPException(status_code=ve.status_code or 502, detail=str(ve))
    except ValueError as ve:
        logger.error(f"[API Validation Error]: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[API Unexpected Error]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Satellite analysis failed: {str(e)}")


@app.post("/temporal/multimodal")
@app.post("/api/temporal/multimodal")
async def temporal_multimodal_endpoint(req: BiTemporalMultimodalRequest):
    """
    Bi-Temporal Multimodal Satellite Change Analysis Endpoint.
    Ingests T1 & T2 Optical and/or SAR imagery, runs co-registration, single-epoch fusion,
    detects changes across objects, calculates objective land-cover deltas,
    analyzes differential SAR scattering, produces overlays, and delivers grounded VLM interpretation.
    """
    try:
        def _parse_pil(data_url: Optional[str]) -> Optional[Image.Image]:
            if not data_url:
                return None
            try:
                b, _ = decode_data_url(data_url)
                return Image.open(io.BytesIO(b))
            except Exception:
                return None

        opt1 = _parse_pil(req.t1_optical)
        sar1 = _parse_pil(req.t1_sar)
        opt2 = _parse_pil(req.t2_optical)
        sar2 = _parse_pil(req.t2_sar)

        if (opt1 is None and sar1 is None) or (opt2 is None and sar2 is None):
            raise HTTPException(
                status_code=400,
                detail="At least one image (optical or SAR) for Time 1 and Time 2 must be provided.",
            )

        analysis = bitemporal_analyzer.analyze(
            t1_optical=opt1,
            t1_sar=sar1,
            t2_optical=opt2,
            t2_sar=sar2,
            prompt=req.prompt or "building, water, vegetation, road",
            date_t1=req.date_t1 or "Time 1",
            date_t2=req.date_t2 or "Time 2",
            detections_t1=req.detections_t1,
            detections_t2=req.detections_t2,
            land_cover_t1=req.land_cover_t1,
            land_cover_t2=req.land_cover_t2,
        )

        # Grounded VLM Interpretation if requested and imagery available
        if req.enable_vlm_interpretation:
            t2_for_vlm = req.t2_optical or req.t2_sar
            t1_for_vlm = req.t1_optical or req.t1_sar
            if t2_for_vlm:
                try:
                    vlm_res = await vision_service.analyze_image(
                        image_data=t2_for_vlm,
                        user_query=req.user_query or "Assess changes between T1 and T2.",
                        provider=req.provider or "auto",
                        analysis_mode="changes",
                        detection_context={"detections": req.detections_t2 or []},
                        change_context=analysis,
                        second_image_data=t1_for_vlm,
                    )
                    structured = vlm_res["structured_analysis"]
                    analysis["interpretation"] = {
                        "provider": vlm_res["provider_used"],
                        "answer": structured.answer,
                        "summary": structured.summary,
                        "observed_changes": structured.observed_changes,
                        "possible_causes": structured.possible_causes,
                        "observations": [o.model_dump() for o in structured.observations],
                        "uncertainties": structured.uncertainties,
                    }
                except Exception as ve:
                    logger.warning(f"[VLM Interpretation Warning]: {ve}")

        return JSONResponse(status_code=200, content=analysis)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Temporal Multimodal Error]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bi-temporal analysis failed: {str(e)}")


def _image_to_data_url(image: Image.Image, format: str = "JPEG") -> str:
    """Converts a PIL Image to a base64 data URL."""
    buffered = io.BytesIO()
    if format.upper() == "JPEG":
        image.convert("RGB").save(buffered, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        image.save(buffered, format="PNG")
        mime = "image/png"
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


@app.post("/temporal/manual")
@app.post("/api/temporal/manual")
async def temporal_manual_endpoint(
    file_t1: UploadFile = File(...),
    file_t2: UploadFile = File(...),
    prompt: str = Form("building"),
    date_t1: Optional[str] = Form("Time 1 (Reference)"),
    date_t2: Optional[str] = Form("Time 2 (Recent)"),
    aoi_json: Optional[str] = Form(None),
    enable_pixel_change: bool = Form(True),
):
    """
    Manual Bi-Temporal Comparison Endpoint (Stages 1 - 24).
    Compares two uploaded images with SIFT/ORB registration, Lab-CLAHE normalization,
    nuisance masking, multi-signal structural change detection, and calibrated confidence gating.
    """
    try:
        bytes1 = await file_t1.read()
        bytes2 = await file_t2.read()

        image_t1 = Image.open(io.BytesIO(bytes1)).convert("RGB")
        image_t2 = Image.open(io.BytesIO(bytes2)).convert("RGB")

        aoi_dict = None
        if aoi_json:
            try:
                aoi_dict = json.loads(aoi_json)
            except Exception:
                pass

        analysis = bitemporal_analyzer.analyze(
            t1_optical=image_t1,
            t2_optical=image_t2,
            prompt=prompt.strip() if prompt else "building",
            date_t1=date_t1 or "Time 1",
            date_t2=date_t2 or "Time 2",
            aoi=aoi_dict,
        )

        t1_data_url = _image_to_data_url(image_t1)
        t2_data_url = _image_to_data_url(image_t2)

        response_payload = {
            "success": True,
            "aoi": aoi_dict,
            "prompt": prompt,
            "comparison": analysis.get("comparison", {}),
            "changes": analysis.get("changes", []),
            "summary": analysis.get("summary", {}),
            "pixelChange": analysis.get("pixelChange", {}),
            "registration": analysis.get("registration", {}),
            "historical": {
                "sceneId": file_t1.filename,
                "provider": "User Upload (Time 1)",
                "satellite": "Uploaded Image",
                "acquisitionDate": date_t1 or "Time 1",
                "resolution": "Native",
            },
            "latest": {
                "sceneId": file_t2.filename,
                "provider": "User Upload (Time 2)",
                "satellite": "Uploaded Image",
                "acquisitionDate": date_t2 or "Time 2",
                "resolution": "Native",
            },
            "images": {
                "t1DataUrl": t1_data_url,
                "t2DataUrl": t2_data_url,
            },
            "overlays": analysis.get("overlays", {}),
            "modalities": analysis.get("modalities", {}),
            "objects": analysis.get("objects", {}),
            "signals": analysis.get("signals", {}),
            "nuisance": analysis.get("nuisance", {}),
        }

        return JSONResponse(status_code=200, content=response_payload)
    except Exception as e:
        logger.error(f"[Temporal Manual Endpoint Error]: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "error": f"Manual temporal analysis failed: {str(e)}"})


@app.post("/temporal/latest")
@app.post("/api/temporal/latest")
async def temporal_latest_endpoint(
    request: Request,
    aoi_north: Optional[float] = Form(None),
    aoi_south: Optional[float] = Form(None),
    aoi_east: Optional[float] = Form(None),
    aoi_west: Optional[float] = Form(None),
    prompt: Optional[str] = Form("building"),
    historical_mode: Optional[str] = Form("date"),
    historical_date: Optional[str] = Form(None),
    historical_file: Optional[UploadFile] = File(None),
    max_cloud_cover: Optional[float] = Form(20.0),
    search_days: Optional[int] = Form(30),
    provider: Optional[str] = Form(None),
    enable_pixel_change: Optional[bool] = Form(True),
):
    """
    Real-Time Bi-Temporal Satellite Comparison Endpoint (Stages 1 - 24).
    Compares historical image vs latest retrieved satellite pass with calibrated gating.
    """
    try:
        img_t1 = None
        if historical_file is not None:
            b1 = await historical_file.read()
            img_t1 = Image.open(io.BytesIO(b1)).convert("RGB")

        aoi_dict = None
        if aoi_north is not None and aoi_south is not None and aoi_east is not None and aoi_west is not None:
            aoi_dict = {
                "north": float(aoi_north),
                "south": float(aoi_south),
                "east": float(aoi_east),
                "west": float(aoi_west),
            }

        img_t2 = None
        try:
            import sys
            root_dir = Path(__file__).resolve().parent.parent
            if str(root_dir / "ai-service") not in sys.path:
                sys.path.insert(0, str(root_dir / "ai-service"))
            from satellite.provider_service import satellite_service
            from satellite.provider_base import AOIBoundingBox

            if aoi_dict:
                aoi_obj = AOIBoundingBox(**aoi_dict)
                img_t2, meta_t2 = satellite_service.get_latest(
                    aoi=aoi_obj,
                    max_cloud_cover=float(max_cloud_cover or 20.0),
                    search_days=int(search_days or 30),
                    provider_key=provider,
                )
        except Exception as sat_err:
            logger.warning(f"[Satellite Service Warning]: {sat_err}")

        if img_t2 is None:
            img_t2 = img_t1

        if img_t1 is None or img_t2 is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "Unable to acquire temporal imagery pair."})

        analysis = bitemporal_analyzer.analyze(
            t1_optical=img_t1,
            t2_optical=img_t2,
            prompt=prompt or "building",
            date_t1=historical_date or "Time 1",
            date_t2="Latest",
            aoi=aoi_dict,
        )

        t1_data_url = _image_to_data_url(img_t1)
        t2_data_url = _image_to_data_url(img_t2)

        return JSONResponse(status_code=200, content={
            "success": True,
            "aoi": aoi_dict,
            "prompt": prompt,
            "comparison": analysis.get("comparison", {}),
            "changes": analysis.get("changes", []),
            "summary": analysis.get("summary", {}),
            "pixelChange": analysis.get("pixelChange", {}),
            "registration": analysis.get("registration", {}),
            "images": {"t1DataUrl": t1_data_url, "t2DataUrl": t2_data_url},
            "overlays": analysis.get("overlays", {}),
            "objects": analysis.get("objects", {}),
        })
    except Exception as e:
        logger.error(f"[Temporal Latest Endpoint Error]: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "error": f"Latest temporal analysis failed: {str(e)}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
