from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import base64
import json
import traceback
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Ensure parent directory is in python path so satellite, image_registration, etc. can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure root/backend is in python path for enhanced multimodal vision service
root_backend = Path(__file__).resolve().parent.parent.parent / "backend"
if str(root_backend) not in sys.path:
    sys.path.insert(0, str(root_backend))

import uvicorn
from grounding_dino import detect_objects
from satellite.provider_base import AOIBoundingBox, SatelliteSceneMetadata
from satellite.provider_service import satellite_service
from image_registration import register_temporal_images
from temporal_matcher import match_temporal_detections
from pixel_change import compute_pixel_change

try:
    from services.temporal import bitemporal_analyzer
    TEMPORAL_ANALYZER_AVAILABLE = True
except Exception as _te:
    print(f"[Warning] Failed to import temporal analyzer in ai-service: {_te}")
    bitemporal_analyzer = None
    TEMPORAL_ANALYZER_AVAILABLE = False

try:
    from services.vision.vision_service import vision_service
    from services.vision.response_parser import to_legacy_analysis_result
    from services.vision.base_provider import (
        VisionProviderError,
        VisionProviderAuthError,
        VisionProviderRateLimitError,
    )
    VISION_SERVICE_AVAILABLE = True
except Exception as _e:
    print(f"[Warning] Failed to import vision service in ai-service: {_e}")
    VISION_SERVICE_AVAILABLE = False


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="SatQuery Grounding DINO & Temporal AI API",
    description="Multimodal Grounding DINO & Real-Time Bi-Temporal Satellite Analysis Service",
    version="2.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HELPERS
# =========================================================

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


def _format_time_difference(date1_str: str, date2_str: str) -> str:
    """Calculates human-readable time difference between two dates."""
    try:
        d1 = datetime.strptime(date1_str[:10], "%Y-%m-%d")
        d2 = datetime.strptime(date2_str[:10], "%Y-%m-%d")
        delta_days = abs((d2 - d1).days)

        years = delta_days // 365
        remaining_days = delta_days % 365
        months = remaining_days // 30
        days = remaining_days % 30

        parts = []
        if years > 0:
            parts.append(f"{years} year{'s' if years > 1 else ''}")
        if months > 0:
            parts.append(f"{months} month{'s' if months > 1 else ''}")
        if not parts or (years == 0 and days > 0):
            parts.append(f"{days} day{'s' if days != 1 else ''}")

        return ", ".join(parts) if parts else "Same date"
    except Exception:
        return "Temporal interval"


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def home():
    return {
        "status": "running",
        "service": "SatQuery Grounding DINO & Bi-Temporal API",
        "model": "Grounding DINO + Temporal Matcher",
        "endpoints": {
            "detect": "/detect",
            "temporal_latest": "/temporal/latest",
            "temporal_manual": "/temporal/manual",
            "providers": "/temporal/providers",
        },
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Grounding DINO & Bi-Temporal",
        "satelliteService": "online",
    }


# =========================================================
# SATELLITE PROVIDERS
# =========================================================

@app.get("/temporal/providers")
async def list_providers():
    return {
        "default": satellite_service.default_provider_key,
        "providers": satellite_service.list_providers(),
    }


# =========================================================
# SINGLE-IMAGE DETECTION
# =========================================================

@app.post("/detect")
async def detect(
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
    try:
        if file is None:
            return JSONResponse(status_code=400, content={"error": "No image file was provided."})

        if not prompt or not prompt.strip():
            return JSONResponse(status_code=400, content={"error": "Detection prompt is required."})

        image_bytes = await file.read()
        if not image_bytes:
            return JSONResponse(status_code=400, content={"error": "Uploaded image is empty."})

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            return JSONResponse(status_code=400, content={"error": "The uploaded file is not a valid image."})
        except Exception as image_error:
            return JSONResponse(status_code=400, content={"error": "Unable to read the uploaded image.", "details": str(image_error)})

        print(f"\n[SINGLE-IMAGE] Filename: {file.filename} | Prompt: {prompt} | Preset: {preset} | use_tiles: {use_tiles} | merge_mode: {merge_mode} | seg: {enable_segmentation} | verif: {enable_verification}")
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
            "land_cover": land_cover,
            "verification_available": verification_avail,
            "verification": verification_meta,
        }

        return JSONResponse(status_code=200, content=result)

    except Exception as error:
        print("\n[FASTAPI DETECTION ERROR]")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Detection service failed.", "details": str(error)})


# =========================================================
# MULTIMODAL SATELLITE IMAGE ANALYSIS (GEMINI & GLM)
# =========================================================

@app.post("/analyze")
@app.post("/api/analyze")
async def analyze_image(req: dict):
    if not VISION_SERVICE_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "Vision service not loaded on backend."})

    image_data = req.get("imageDataUrl") or req.get("image_data")
    if not image_data:
        return JSONResponse(status_code=400, content={"error": "Missing required image data."})

    user_query = (req.get("user_query") or req.get("query") or "").strip()
    if not user_query:
        user_query = "Provide a comprehensive remote-sensing assessment of this satellite image."

    provider = req.get("provider") or req.get("model") or "auto"
    second_image = req.get("secondImageDataUrl") or req.get("second_image_data")
    detection_context = req.get("detectionContext") or req.get("detection_context")
    change_context = req.get("changeContext") or req.get("change_context")

    try:
        result = await vision_service.analyze_image(
            image_data=image_data,
            user_query=user_query,
            provider=provider,
            analysis_mode=req.get("analysis_mode", "general"),
            analysis_depth=req.get("analysis_depth", "standard"),
            use_detections=req.get("use_detections", True),
            use_change_context=req.get("use_change_context", True),
            use_tiles=req.get("use_tiles", False),
            detection_context=detection_context,
            change_context=change_context,
            image_metadata=req.get("image_metadata"),
            second_image_data=second_image,
        )

        structured = result["structured_analysis"]
        legacy_analysis = to_legacy_analysis_result(structured, intent=req.get("analysis_mode", "image_understanding"))

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "provider": result["provider_used"],
                "analysis_mode": result["analysis_mode"],
                "query": result["query"],
                "answer": structured.answer_to_query,
                "summary": structured.summary,
                "observations": [o.model_dump() for o in structured.observations],
                "uncertainties": structured.uncertainties,
                "context": {
                    "grounding_dino_used": detection_context is not None and bool(detection_context.get("detections")),
                    "change_detection_used": change_context is not None,
                },
                "processing_time_ms": result["processing_time_ms"],
                "structured_analysis": structured.model_dump(),
                "analysis": legacy_analysis,
                "modelUsed": result["provider_used"],
                "fallbackUsed": result["fallback_used"],
            },
        )
    except VisionProviderAuthError as ae:
        return JSONResponse(status_code=401, content={"error": str(ae)})
    except VisionProviderRateLimitError as re:
        return JSONResponse(status_code=429, content={"error": str(re)})
    except VisionProviderError as ve:
        return JSONResponse(status_code=ve.status_code or 502, content={"error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Satellite analysis failed: {str(e)}"})


# =========================================================
# REAL-TIME / LATEST SATELLITE BI-TEMPORAL COMPARISON
# =========================================================

@app.post("/temporal/latest")
async def temporal_latest(
    aoi_north: float = Form(...),
    aoi_south: float = Form(...),
    aoi_east: float = Form(...),
    aoi_west: float = Form(...),
    prompt: str = Form(...),
    historical_mode: str = Form("date"),  # 'date' or 'upload'
    historical_date: Optional[str] = Form(None),
    historical_file: Optional[UploadFile] = File(None),
    max_cloud_cover: float = Form(20.0),
    search_days: int = Form(30),
    provider: Optional[str] = Form(None),
    enable_pixel_change: bool = Form(True),
):
    """
    Real-Time Bi-Temporal Satellite Comparison:
    Compares Historical Image (Time 1, via date or upload) vs
    Latest Available Satellite Image (Time 2, auto-retrieved for exact AOI).
    """
    try:
        # 1. Parse & validate Area of Interest (AOI)
        aoi = AOIBoundingBox(
            north=float(aoi_north),
            south=float(aoi_south),
            east=float(aoi_east),
            west=float(aoi_west),
        )
        try:
            aoi.validate(max_span_degrees=0.25)
        except ValueError as ve:
            return JSONResponse(status_code=400, content={"error": str(ve)})

        clean_prompt = prompt.strip() if prompt else "building"
        if not clean_prompt:
            return JSONResponse(status_code=400, content={"error": "Comparison prompt is required."})

        print("\n========================================================")
        print("  REAL-TIME BI-TEMPORAL COMPARISON REQUEST")
        print("========================================================")
        print(f"AOI        : North={aoi.north}, South={aoi.south}, East={aoi.east}, West={aoi.west}")
        print(f"Prompt     : {clean_prompt}")
        print(f"Time 1 Mode: {historical_mode} (Date: {historical_date})")
        print(f"Max Cloud  : {max_cloud_cover}%")
        print("========================================================\n")

        # 2. Obtain Time 1 (Historical Imagery)
        meta_t1_dict = {}
        if historical_mode == "upload" and historical_file is not None:
            t1_bytes = await historical_file.read()
            if not t1_bytes:
                return JSONResponse(status_code=400, content={"error": "Historical uploaded image is empty."})
            image_t1 = Image.open(io.BytesIO(t1_bytes)).convert("RGB")
            hist_date_label = historical_date.strip() if (historical_date and historical_date.strip()) else "Reference Image"
            meta_t1_dict = {
                "sceneId": "user-uploaded-reference",
                "provider": "User Reference Upload",
                "satellite": "External / User Provided",
                "acquisitionDate": hist_date_label,
                "acquisitionTime": None,
                "cloudCoverage": None,
                "resolution": "Native",
                "dataFreshnessDays": None,
            }
        else:
            hist_date_str = historical_date.strip() if (historical_date and historical_date.strip()) else "2024-01-01"
            print(f"[Temporal] Fetching Historical Satellite Scene for {hist_date_str}...")
            image_t1, scene_t1 = satellite_service.get_historical(
                aoi=aoi,
                target_date=hist_date_str,
                date_range_days=14,
                max_cloud_cover=max(max_cloud_cover, 25.0),
                provider_key=provider,
            )
            meta_t1_dict = scene_t1.to_dict()

        # 3. Obtain Time 2 (Latest Available Satellite Imagery)
        print(f"[Temporal] Fetching Latest Available Satellite Scene (search window: {search_days} days)...")
        image_t2, scene_t2 = satellite_service.get_latest(
            aoi=aoi,
            max_cloud_cover=max_cloud_cover,
            search_days=search_days,
            provider_key=provider,
        )
        meta_t2_dict = scene_t2.to_dict()

        # 4. Perform Geospatial & Feature Image Registration
        print("[Temporal] Registering and aligning temporal image pair...")
        reg_result = register_temporal_images(image_t1, image_t2, target_size=(640, 640))
        aligned_t1 = reg_result["aligned_t1"]
        aligned_t2 = reg_result["aligned_t2"]
        reg_quality = reg_result["registration_quality"]
        reg_warning = reg_result["warning"]

        # 5. Dual-Temporal Grounding DINO Object Detection (with exact same prompt)
        print(f"[Temporal] Running Grounding DINO on Historical Scene (T1) for prompt: '{clean_prompt}'...")
        detections_t1 = detect_objects(aligned_t1, clean_prompt)

        print(f"[Temporal] Running Grounding DINO on Latest Scene (T2) for prompt: '{clean_prompt}'...")
        detections_t2 = detect_objects(aligned_t2, clean_prompt)

        # 6. Temporal Spatial Object Matching & Geolocation
        print("[Temporal] Matching objects across timestamps & calculating geolocation...")
        hist_date_disp = meta_t1_dict.get("acquisitionDate", "Historical")
        latest_date_disp = meta_t2_dict.get("acquisitionDate", "Latest")

        match_res = match_temporal_detections(
            detections_t1=detections_t1,
            detections_t2=detections_t2,
            img_width=640,
            img_height=640,
            aoi=aoi,
            historical_date=hist_date_disp,
            latest_date=latest_date_disp,
        )

        # 7. Pixel-Level Change Detection
        pixel_change_res = None
        if enable_pixel_change:
            print("[Temporal] Computing pixel-level radiometric change mask...")
            pixel_change_res = compute_pixel_change(aligned_t1, aligned_t2)

        # 8. Encode images for frontend visualization
        t1_data_url = _image_to_data_url(aligned_t1)
        t2_data_url = _image_to_data_url(aligned_t2)

        # 9. Format response
        time_diff_str = _format_time_difference(hist_date_disp, latest_date_disp)
        freshness_days = meta_t2_dict.get("dataFreshnessDays")
        freshness_label = f"{freshness_days} days ago" if freshness_days is not None else "Recent pass"

        response_payload = {
            "success": True,
            "aoi": aoi.to_dict(),
            "prompt": clean_prompt,
            "timeDifference": time_diff_str,
            "latestImageryAge": freshness_label,
            "registration": {
                "quality": reg_quality,
                "warning": reg_warning,
                "transformation": reg_result.get("transformation_type", "geospatial"),
            },
            "historical": meta_t1_dict,
            "latest": meta_t2_dict,
            "summary": match_res["summary"],
            "changes": match_res["changes"],
            "pixelChange": pixel_change_res,
            "images": {
                "t1DataUrl": t1_data_url,
                "t2DataUrl": t2_data_url,
            },
        }

        print(f"\n[Temporal] Completed comparison! Summary: {match_res['summary']}")
        return JSONResponse(status_code=200, content=response_payload)

    except Exception as error:
        print("\n[FASTAPI TEMPORAL LATEST ERROR]")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Real-time temporal comparison failed.",
            "details": str(error)
        })


# =========================================================
# MANUAL TWO-IMAGE TEMPORAL COMPARISON
# =========================================================

@app.post("/temporal/manual")
async def temporal_manual(
    file_t1: UploadFile = File(...),
    file_t2: UploadFile = File(...),
    prompt: str = Form(...),
    date_t1: Optional[str] = Form("Time 1 (Reference)"),
    date_t2: Optional[str] = Form("Time 2 (Recent)"),
    aoi_json: Optional[str] = Form(None),
    enable_pixel_change: bool = Form(True),
):
    """
    Manual Bi-Temporal Comparison:
    Compares two user-uploaded images (Time 1 and Time 2)
    using Grounding DINO, registration, and change classification.
    """
    try:
        if file_t1 is None or file_t2 is None:
            return JSONResponse(status_code=400, content={"error": "Both Time 1 and Time 2 images are required."})

        bytes1 = await file_t1.read()
        bytes2 = await file_t2.read()

        image_t1 = Image.open(io.BytesIO(bytes1)).convert("RGB")
        image_t2 = Image.open(io.BytesIO(bytes2)).convert("RGB")

        aoi = None
        if aoi_json:
            try:
                ad = json.loads(aoi_json)
                aoi = AOIBoundingBox(
                    north=float(ad["north"]),
                    south=float(ad["south"]),
                    east=float(ad["east"]),
                    west=float(ad["west"]),
                )
            except Exception:
                pass

        clean_prompt = prompt.strip() if prompt else "building"

        # 1. Image alignment
        reg_result = register_temporal_images(image_t1, image_t2, target_size=(640, 640))
        aligned_t1 = reg_result["aligned_t1"]
        aligned_t2 = reg_result["aligned_t2"]

        # 2. Dual Grounding DINO detection
        detections_t1 = detect_objects(aligned_t1, clean_prompt)
        detections_t2 = detect_objects(aligned_t2, clean_prompt)

        # 3. Match objects
        match_res = match_temporal_detections(
            detections_t1=detections_t1,
            detections_t2=detections_t2,
            img_width=640,
            img_height=640,
            aoi=aoi,
            historical_date=date_t1 or "Time 1",
            latest_date=date_t2 or "Time 2",
        )

        # 4. Pixel change
        pixel_change_res = None
        if enable_pixel_change:
            pixel_change_res = compute_pixel_change(aligned_t1, aligned_t2)

        t1_data_url = _image_to_data_url(aligned_t1)
        t2_data_url = _image_to_data_url(aligned_t2)

        time_diff_str = _format_time_difference(date_t1 or "2024-01-01", date_t2 or "2024-02-01")

        response_payload = {
            "success": True,
            "aoi": aoi.to_dict() if aoi else None,
            "prompt": clean_prompt,
            "timeDifference": time_diff_str,
            "latestImageryAge": "Manual pair",
            "registration": {
                "quality": reg_result["registration_quality"],
                "warning": reg_result["warning"],
                "transformation": reg_result["transformation_type"],
            },
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
            "summary": match_res["summary"],
            "changes": match_res["changes"],
            "pixelChange": pixel_change_res,
            "images": {
                "t1DataUrl": t1_data_url,
                "t2DataUrl": t2_data_url,
            },
        }

        return JSONResponse(status_code=200, content=response_payload)

    except Exception as error:
        print("\n[FASTAPI TEMPORAL MANUAL ERROR]")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Manual temporal comparison failed.",
            "details": str(error)
        })


# =========================================================
# BI-TEMPORAL MULTIMODAL COMPARISON (OPTICAL + SAR)
# =========================================================

@app.post("/temporal/multimodal")
async def temporal_multimodal(
    file_t1_optical: Optional[UploadFile] = File(None),
    file_t1_sar: Optional[UploadFile] = File(None),
    file_t2_optical: Optional[UploadFile] = File(None),
    file_t2_sar: Optional[UploadFile] = File(None),
    file_t1: Optional[UploadFile] = File(None),
    file_t2: Optional[UploadFile] = File(None),
    prompt: str = Form("building, water, vegetation, road"),
    date_t1: Optional[str] = Form("Time 1"),
    date_t2: Optional[str] = Form("Time 2"),
    user_query: Optional[str] = Form("Analyze all land-cover and structural changes between Time 1 and Time 2."),
    provider: Optional[str] = Form("auto"),
    enable_vlm_interpretation: bool = Form(True),
):
    try:
        # Resolve files with generic fallback
        f_t1_opt = file_t1_optical or file_t1
        f_t2_opt = file_t2_optical or file_t2
        f_t1_sar = file_t1_sar
        f_t2_sar = file_t2_sar

        if (f_t1_opt is None and f_t1_sar is None) or (f_t2_opt is None and f_t2_sar is None):
            return JSONResponse(
                status_code=400,
                content={"error": "At least one image (optical or SAR) for Time 1 and Time 2 must be provided."}
            )

        async def _read_img(f: Optional[UploadFile]) -> Optional[Image.Image]:
            if f is None:
                return None
            b = await f.read()
            if not b:
                return None
            return Image.open(io.BytesIO(b))

        opt1 = await _read_img(f_t1_opt)
        opt2 = await _read_img(f_t2_opt)
        sar1 = await _read_img(f_t1_sar)
        sar2 = await _read_img(f_t2_sar)

        # 1. Independent Grounding DINO detection on available optical/fused representations
        clean_prompt = prompt.strip() if prompt else "building, water, vegetation, road"

        dets_t1, meta_t1 = [], {}
        t1_input = opt1 or sar1
        if t1_input:
            dets_t1, meta_t1 = detect_objects(
                image=t1_input,
                prompt=clean_prompt,
                enable_segmentation=True,
                return_tiling_metadata=True,
            )

        dets_t2, meta_t2 = [], {}
        t2_input = opt2 or sar2
        if t2_input:
            dets_t2, meta_t2 = detect_objects(
                image=t2_input,
                prompt=clean_prompt,
                enable_segmentation=True,
                return_tiling_metadata=True,
            )

        # 2. Run Bi-Temporal Multimodal Analyzer
        if bitemporal_analyzer is None:
            return JSONResponse(status_code=503, content={"error": "BiTemporalMultimodalAnalyzer service unavailable."})

        analysis = bitemporal_analyzer.analyze(
            t1_optical=opt1,
            t1_sar=sar1,
            t2_optical=opt2,
            t2_sar=sar2,
            prompt=clean_prompt,
            date_t1=date_t1 or "Time 1",
            date_t2=date_t2 or "Time 2",
            detections_t1=dets_t1,
            detections_t2=dets_t2,
            land_cover_t1=meta_t1.get("land_cover") if isinstance(meta_t1, dict) else None,
            land_cover_t2=meta_t2.get("land_cover") if isinstance(meta_t2, dict) else None,
        )

        # 3. Optional Grounded Gemini/GLM Interpretation
        interpretation_data = None
        if enable_vlm_interpretation and VISION_SERVICE_AVAILABLE:
            t2_img_for_vlm = opt2 or sar2
            t1_img_for_vlm = opt1 or sar1
            if t2_img_for_vlm:
                try:
                    vlm_res = await vision_service.analyze_image(
                        image_data=_image_to_data_url(t2_img_for_vlm),
                        user_query=user_query or "Assess changes between T1 and T2.",
                        provider=provider or "auto",
                        analysis_mode="changes",
                        detection_context={"detections": dets_t2},
                        change_context=analysis,
                        second_image_data=_image_to_data_url(t1_img_for_vlm) if t1_img_for_vlm else None,
                    )
                    structured = vlm_res["structured_analysis"]
                    interpretation_data = {
                        "provider": vlm_res["provider_used"],
                        "answer": structured.answer,
                        "summary": structured.summary,
                        "observed_changes": structured.observed_changes,
                        "possible_causes": structured.possible_causes,
                        "observations": [o.model_dump() for o in structured.observations],
                        "uncertainties": structured.uncertainties,
                    }
                except Exception as _ve:
                    print(f"[Warning] VLM interpretation failed during multimodal comparison: {_ve}")

        analysis["interpretation"] = interpretation_data
        return JSONResponse(status_code=200, content=analysis)

    except Exception as error:
        print("\n[FASTAPI MULTIMODAL TEMPORAL ERROR]")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Multimodal temporal comparison failed.",
            "details": str(error)
        })



# =========================================================
# SERVER START
# =========================================================

if __name__ == "__main__":
    print("\n========================================")
    print("  SATQUERY GROUNDING DINO & TEMPORAL SERVER")
    print("========================================")
    print("Server : http://127.0.0.1:8000")
    print("Docs   : http://127.0.0.1:8000/docs")
    print("Health : http://127.0.0.1:8000/health")
    print("Detect : http://127.0.0.1:8000/detect")
    print("Latest : http://127.0.0.1:8000/temporal/latest")
    print("Manual : http://127.0.0.1:8000/temporal/manual")
    print("========================================\n")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )