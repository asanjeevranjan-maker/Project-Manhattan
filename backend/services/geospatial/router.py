"""
FastAPI router for geospatial raster operations.
Provides clean HTTP endpoints for metadata extraction, clipping, alignment,
georeferenced mask export, and QGIS-ready GeoJSON feature generation.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import json
import uuid
import io
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any

from . import (
    RASTERIO_AVAILABLE,
    RASTERIO_VERSION,
    GDAL_VERSION,
    read_raster,
    write_geotiff,
    get_raster_metadata,
    reproject_raster,
    align_raster_to_reference,
    clip_raster,
    detections_to_geojson,
    mask_to_geojson_polygons,
    export_qgis_package,
    get_qgis_export_manifest,
    LANDCOVER_CLASS_MAP,
    to_json_safe,
)

router = APIRouter(prefix="/api/geospatial", tags=["geospatial"])

OUTPUTS_DIR = Path("outputs/geospatial")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _check_rasterio():
    if not RASTERIO_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Geospatial processing unavailable in this deployment (rasterio missing)",
        )


@router.get("/status")
def get_geospatial_status():
    """Returns availability of rasterio, GDAL, and geospatial runtime."""
    return {
        "available": RASTERIO_AVAILABLE,
        "rasterio_version": RASTERIO_VERSION,
        "gdal_version": GDAL_VERSION,
        "service": "Project-Manhattan Geospatial Engine",
    }


@router.post("/metadata")
async def extract_metadata_endpoint(file: UploadFile = File(...)):
    """
    POST /api/geospatial/metadata
    Uploads a GeoTIFF and returns JSON-safe geospatial metadata (CRS, bounds, resolution, transform).
    """
    _check_rasterio()
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        meta = get_raster_metadata(content)
        return JSONResponse(status_code=200, content=meta)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract geospatial metadata: {str(e)}")


@router.post("/clip")
async def clip_raster_endpoint(
    file: UploadFile = File(...),
    bbox: Optional[str] = Form(None),
    geometry: Optional[str] = Form(None),
    save_output: Optional[bool] = Form(False),
):
    """
    POST /api/geospatial/clip
    Clips a GeoTIFF using a bounding box [min_x, min_y, max_x, max_y] or GeoJSON geometry.
    """
    _check_rasterio()
    try:
        content = await file.read()
        bbox_list = None
        geom_dict = None

        if bbox:
            try:
                bbox_list = json.loads(bbox)
            except Exception:
                # Comma separated fallback
                bbox_list = [float(x.strip()) for x in bbox.split(",")]

        if geometry:
            geom_dict = json.loads(geometry)

        clipped_arr, clipped_trans, clipped_prof = clip_raster(
            source=content,
            bbox=bbox_list,
            geometry=geom_dict,
        )

        output_bytes = write_geotiff(
            output_dest=None,
            array=clipped_arr,
            reference_profile=clipped_prof,
        )

        if save_output:
            out_id = uuid.uuid4().hex[:8]
            out_path = OUTPUTS_DIR / f"clipped_{out_id}.tif"
            out_path.write_bytes(output_bytes)
            return JSONResponse(status_code=200, content={
                "success": True,
                "saved_path": str(out_path).replace("\\", "/"),
                "width": clipped_arr.shape[2],
                "height": clipped_arr.shape[1],
                "bands": clipped_arr.shape[0],
            })

        return Response(
            content=output_bytes,
            media_type="image/tiff",
            headers={"Content-Disposition": 'attachment; filename="clipped.tif"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Raster clipping failed: {str(e)}")


@router.post("/align")
async def align_raster_endpoint(
    source_file: UploadFile = File(...),
    reference_file: UploadFile = File(...),
    is_mask: Optional[bool] = Form(False),
    save_output: Optional[bool] = Form(False),
):
    """
    POST /api/geospatial/align
    Spatially aligns a source raster (e.g. SAR or T2) to match the reference raster's pixel grid.
    """
    _check_rasterio()
    try:
        src_bytes = await source_file.read()
        ref_bytes = await reference_file.read()

        aligned_arr, aligned_prof, alignment_meta = align_raster_to_reference(
            source=src_bytes,
            reference=ref_bytes,
            is_mask=is_mask,
        )

        aligned_geotiff_bytes = write_geotiff(
            output_dest=None,
            array=aligned_arr,
            reference_profile=aligned_prof,
        )

        if save_output:
            out_id = uuid.uuid4().hex[:8]
            out_path = OUTPUTS_DIR / f"aligned_{out_id}.tif"
            out_path.write_bytes(aligned_geotiff_bytes)
            return JSONResponse(status_code=200, content={
                "success": True,
                "saved_path": str(out_path).replace("\\", "/"),
                "alignment": alignment_meta,
            })

        return Response(
            content=aligned_geotiff_bytes,
            media_type="image/tiff",
            headers={"Content-Disposition": 'attachment; filename="aligned.tif"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Raster alignment failed: {str(e)}")


@router.post("/export-mask")
async def export_mask_geotiff_endpoint(
    reference_file: UploadFile = File(...),
    mask_file: Optional[UploadFile] = File(None),
    mask_data_url: Optional[str] = Form(None),
    output_name: Optional[str] = Form(None),
    save_output: Optional[bool] = Form(False),
):
    """
    POST /api/geospatial/export-mask
    Exports an AI segmentation / change / water mask as a georeferenced GeoTIFF
    inheriting the exact CRS and affine transform of the reference image.
    """
    _check_rasterio()
    try:
        ref_bytes = await reference_file.read()
        ref_data = read_raster(ref_bytes)

        # Decode mask
        mask_np = None
        if mask_file is not None:
            mask_bytes = await mask_file.read()
            pil_img = Image.open(io.BytesIO(mask_bytes))
            mask_np = np.array(pil_img)
        elif mask_data_url:
            import base64
            b64_data = mask_data_url.split(",")[-1] if "," in mask_data_url else mask_data_url
            raw = base64.b64decode(b64_data)
            pil_img = Image.open(io.BytesIO(raw))
            mask_np = np.array(pil_img)
        else:
            raise HTTPException(status_code=400, detail="Either 'mask_file' or 'mask_data_url' must be provided.")

        # Ensure single channel mask
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        # Resample mask to match reference dimensions if necessary
        if mask_np.shape[0] != ref_data.height or mask_np.shape[1] != ref_data.width:
            pil_mask = Image.fromarray(mask_np)
            pil_mask = pil_mask.resize((ref_data.width, ref_data.height), resample=Image.NEAREST)
            mask_np = np.array(pil_mask)

        out_prof = dict(ref_data.profile)
        out_prof.update({
            "count": 1,
            "dtype": str(mask_np.dtype),
            "nodata": 0,
        })

        mask_geotiff_bytes = write_geotiff(
            output_dest=None,
            array=mask_np,
            reference_profile=out_prof,
        )

        name = output_name or f"mask_{uuid.uuid4().hex[:8]}"
        if not name.endswith(".tif") and not name.endswith(".tiff"):
            name = f"{name}.tif"

        if save_output:
            out_path = OUTPUTS_DIR / name
            out_path.write_bytes(mask_geotiff_bytes)
            return JSONResponse(status_code=200, content={
                "success": True,
                "saved_path": str(out_path).replace("\\", "/"),
                "width": ref_data.width,
                "height": ref_data.height,
            })

        return Response(
            content=mask_geotiff_bytes,
            media_type="image/tiff",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Mask GeoTIFF export failed: {str(e)}")


@router.post("/detections-geojson")
async def detections_geojson_endpoint(
    reference_file: UploadFile = File(...),
    detections: str = Form(...),
    save_output: Optional[bool] = Form(False),
):
    """
    POST /api/geospatial/detections-geojson
    Converts Grounding DINO detection bounding boxes into a QGIS-compatible
    GeoJSON FeatureCollection georeferenced with the reference raster.
    """
    _check_rasterio()
    try:
        ref_bytes = await reference_file.read()
        ref_data = read_raster(ref_bytes)

        detections_list = json.loads(detections)
        if not isinstance(detections_list, list):
            raise ValueError("'detections' must be a JSON array of detection objects.")

        geojson_data = detections_to_geojson(
            detections=detections_list,
            transform=ref_data.transform,
            crs=ref_data.crs,
        )

        if save_output:
            out_id = uuid.uuid4().hex[:8]
            out_path = OUTPUTS_DIR / f"detections_{out_id}.geojson"
            out_path.write_text(json.dumps(geojson_data, indent=2), encoding="utf-8")
            return JSONResponse(status_code=200, content={
                "success": True,
                "saved_path": str(out_path).replace("\\", "/"),
                "feature_count": len(geojson_data.get("features", [])),
                "geojson": geojson_data,
            })

        return JSONResponse(status_code=200, content=geojson_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate detections GeoJSON: {str(e)}")


async def _process_mask_upload(
    upload: Optional[UploadFile], target_w: int, target_h: int
) -> Optional[np.ndarray]:
    if upload is None:
        return None
    data = await upload.read()
    if not data:
        return None
    pil_img = Image.open(io.BytesIO(data))
    arr = np.array(pil_img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape[0] != target_h or arr.shape[1] != target_w:
        resized_pil = Image.fromarray(arr).resize((target_w, target_h), resample=Image.NEAREST)
        arr = np.array(resized_pil)
    return arr


@router.post("/export/package")
async def export_package_endpoint(
    reference_file: UploadFile = File(...),
    analysis_id: Optional[str] = Form(None),
    detections: Optional[str] = Form(None),
    sam2_masks: Optional[UploadFile] = File(None),
    sam2_masks_json: Optional[str] = Form(None),
    change_mask: Optional[UploadFile] = File(None),
    landcover_mask: Optional[UploadFile] = File(None),
    aligned_sar: Optional[UploadFile] = File(None),
    aligned_optical: Optional[UploadFile] = File(None),
    create_zip: Optional[bool] = Form(True),
):
    """
    POST /api/geospatial/export/package
    Generates an all-in-one QGIS-compatible export package containing:
      - detections.geojson (Grounding DINO bounding box polygons)
      - sam2_masks.geojson (SAM2 instance segmentation polygons)
      - change_mask.tif (Change detection difference GeoTIFF)
      - landcover.tif (Integer class land-cover segmentation GeoTIFF)
      - aligned_sar.tif / aligned_optical.tif (Georeferenced rasters, if provided)
      - original_metadata.json & README.txt
      - Bundled analysis_{id}_qgis.zip
    """
    _check_rasterio()
    try:
        ref_bytes = await reference_file.read()
        ref_data = read_raster(ref_bytes)

        # Parse detections if provided
        detections_list = None
        if detections:
            try:
                detections_list = json.loads(detections)
                if not isinstance(detections_list, list):
                    raise ValueError("'detections' must be a JSON array.")
            except Exception as ex:
                raise HTTPException(status_code=400, detail=f"Invalid detections JSON: {str(ex)}")

        # Parse SAM2 masks
        sam2_input = None
        if sam2_masks is not None:
            sam2_input = await _process_mask_upload(sam2_masks, ref_data.width, ref_data.height)
        elif sam2_masks_json:
            try:
                sam2_input = json.loads(sam2_masks_json)
            except Exception as ex:
                raise HTTPException(status_code=400, detail=f"Invalid sam2_masks_json: {str(ex)}")

        # Parse change mask
        change_np = await _process_mask_upload(change_mask, ref_data.width, ref_data.height)

        # Parse land-cover mask
        landcover_np = await _process_mask_upload(landcover_mask, ref_data.width, ref_data.height)

        # Parse aligned SAR & optical
        sar_bytes = await aligned_sar.read() if aligned_sar is not None else None
        optical_bytes = await aligned_optical.read() if aligned_optical is not None else None

        manifest = export_qgis_package(
            reference_source=ref_data,
            output_base_dir=OUTPUTS_DIR,
            analysis_id=analysis_id,
            detections=detections_list,
            sam2_masks=sam2_input,
            change_mask=change_np,
            landcover_mask=landcover_np,
            aligned_sar=sar_bytes,
            aligned_optical=optical_bytes,
            create_zip=bool(create_zip),
        )

        aid = manifest["analysis_id"]
        file_urls = {
            fname: f"/api/geospatial/export/{aid}/{fname}"
            for fname in manifest.get("files", [])
        }
        manifest["file_urls"] = file_urls
        manifest["zip_url"] = f"/api/geospatial/export/{aid}/zip" if manifest.get("zip_path") else None

        return JSONResponse(status_code=200, content=manifest)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create QGIS export package: {str(e)}")


@router.get("/export/{analysis_id}")
def get_export_manifest_endpoint(analysis_id: str):
    """
    GET /api/geospatial/export/{analysis_id}
    Returns the file manifest and metadata for a previously generated QGIS package.
    """
    _check_rasterio()
    safe_aid = Path(analysis_id).name
    manifest = get_qgis_export_manifest(analysis_id=safe_aid, output_base_dir=OUTPUTS_DIR)
    if not manifest.get("exists", False):
        raise HTTPException(status_code=404, detail=f"Export package '{analysis_id}' not found.")
    return JSONResponse(status_code=200, content=manifest)


@router.get("/export/{analysis_id}/zip")
def download_export_zip_endpoint(analysis_id: str):
    """
    GET /api/geospatial/export/{analysis_id}/zip
    Downloads the bundled analysis_{analysis_id}_qgis.zip package.
    """
    _check_rasterio()
    safe_aid = Path(analysis_id).name
    zip_path = OUTPUTS_DIR / f"analysis_{safe_aid}_qgis.zip"
    if not zip_path.is_file():
        raise HTTPException(status_code=404, detail=f"ZIP archive for analysis '{analysis_id}' not found.")
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"analysis_{safe_aid}_qgis.zip",
    )


@router.get("/export/{analysis_id}/{filename}")
def download_export_file_endpoint(analysis_id: str, filename: str):
    """
    GET /api/geospatial/export/{analysis_id}/{filename}
    Downloads an individual layer file (GeoTIFF, GeoJSON, metadata, or README)
    from the analysis export package.
    """
    _check_rasterio()
    safe_aid = Path(analysis_id).name
    safe_fname = Path(filename).name
    file_path = OUTPUTS_DIR / safe_aid / safe_fname

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' in export package '{analysis_id}' not found.")

    media_types = {
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".geojson": "application/geo+json",
        ".json": "application/json",
        ".txt": "text/plain",
    }
    ext = file_path.suffix.lower()
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=safe_fname,
    )

