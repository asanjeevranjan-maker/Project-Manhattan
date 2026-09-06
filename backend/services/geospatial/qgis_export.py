"""
QGIS-Compatible Export Packager for Project-Manhattan AI Satellite Outputs.
Generates open GIS-standard GeoTIFF and GeoJSON layers and bundles them into
an all-in-one QGIS ready zip package.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timezone
import json
import uuid
import zipfile
import numpy as np

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    RASTERIO_AVAILABLE = False

from .raster_io import read_raster, write_geotiff, RasterData, image_to_bands_layout
from .metadata import get_raster_metadata, to_json_safe
from .coordinates import detections_to_geojson, mask_to_geojson_polygons


# Canonical Land-Cover Class Mapping (Step 4)
LANDCOVER_CLASS_MAP: Dict[int, str] = {
    0: "other",
    1: "water",
    2: "vegetation",
    3: "built-up",
    4: "bare soil",
}

# Inverted mapping for string label -> integer ID
CATEGORY_TO_ID: Dict[str, int] = {
    "other": 0,
    "background": 0,
    "unassigned": 0,
    "water": 1,
    "vegetation": 2,
    "built-up": 3,
    "built_up": 3,
    "urban": 3,
    "structure": 3,
    "bare soil": 4,
    "bare_soil": 4,
    "soil": 4,
}


def build_qgis_readme(analysis_id: str, crs_str: str, generated_files: List[str]) -> str:
    """Generates a clear, informative README.txt guide for QGIS desktop users."""
    files_str = "\n".join(f"  - {f}" for f in generated_files)
    return f"""======================================================================
PROJECT-MANHATTAN QGIS EXPORT PACKAGE
======================================================================
Analysis ID : {analysis_id}
CRS         : {crs_str}
Created At  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

EXPORTED FILES & GIS LAYERS:
{files_str}

----------------------------------------------------------------------
HOW TO LOAD IN QGIS:
----------------------------------------------------------------------
1. Open QGIS (Desktop GIS).
2. To load raster layers (*.tif):
   Go to top menu: Layer -> Add Layer -> Add Raster Layer...
   Select:
     - change_mask.tif   (Change detection difference raster)
     - landcover.tif     (Integer class land-cover segmentation)
     - aligned_sar.tif   (Radar backscatter grid, if present)
     - aligned_optical.tif (Optical imagery grid, if present)

3. To load vector layers (*.geojson):
   Go to top menu: Layer -> Add Layer -> Add Vector Layer...
   Select:
     - detections.geojson  (Grounding DINO bounding box polygons)
     - sam2_masks.geojson  (SAM2 instance segmentation polygons)

----------------------------------------------------------------------
LAND-COVER INTEGER CLASS VALUES:
----------------------------------------------------------------------
  Value 0 = Other (Unassigned background)
  Value 1 = Water (Ocean, rivers, lakes, flooded zones)
  Value 2 = Vegetation (Forest, agriculture, greenery)
  Value 3 = Built-Up (Urban structures, roads, buildings)
  Value 4 = Bare Soil (Exposed ground, earth, sand)

Tip: In QGIS, right-click 'landcover.tif' -> Properties -> Symbology.
     Set Render Type to 'Paletted/Unique values' and click 'Classify'
     to assign distinct colors to each land-cover category.
======================================================================
"""


def export_qgis_package(
    reference_source: Union[str, Path, bytes, RasterData],
    output_base_dir: Union[str, Path] = "outputs/geospatial",
    analysis_id: Optional[str] = None,
    detections: Optional[List[Dict[str, Any]]] = None,
    sam2_masks: Optional[Union[np.ndarray, List[Dict[str, Any]]]] = None,
    change_mask: Optional[np.ndarray] = None,
    landcover_mask: Optional[np.ndarray] = None,
    aligned_sar: Optional[Union[np.ndarray, bytes]] = None,
    aligned_optical: Optional[Union[np.ndarray, bytes]] = None,
    create_zip: bool = True,
) -> Dict[str, Any]:
    """
    Exports Grounding DINO detections, SAM2 segmentation, change-detection masks,
    and land-cover maps into standard QGIS layers (GeoTIFF + GeoJSON).

    Returns a manifest dictionary summarizing all exported files and their paths.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    aid = analysis_id or f"analysis_{uuid.uuid4().hex[:10]}"
    base_dir = Path(output_base_dir)
    package_dir = base_dir / aid
    package_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest reference raster
    if isinstance(reference_source, RasterData):
        ref = reference_source
    else:
        ref = read_raster(reference_source)

    crs_str = ref.crs.to_string() if ref.crs else "UNSPECIFIED"
    generated_files: List[str] = []

    # 2. Export Grounding DINO Detections -> detections.geojson
    if detections and len(detections) > 0:
        det_geojson = detections_to_geojson(
            detections=detections,
            transform=ref.transform,
            crs=ref.crs,
        )
        det_path = package_dir / "detections.geojson"
        det_path.write_text(json.dumps(det_geojson, indent=2), encoding="utf-8")
        generated_files.append("detections.geojson")

    # 3. Export SAM2 Masks -> sam2_masks.geojson
    if sam2_masks is not None:
        if isinstance(sam2_masks, np.ndarray):
            mask_geojson = mask_to_geojson_polygons(
                mask=sam2_masks,
                transform=ref.transform,
                crs=ref.crs,
                label="sam2_segmentation",
                min_area_pixels=10,
            )
        elif isinstance(sam2_masks, list):
            # List of individual mask objects: {"mask": ndarray, "label": str, "score": float}
            all_features = []
            for m_item in sam2_masks:
                sub_arr = m_item.get("mask")
                if sub_arr is None:
                    continue
                sub_label = m_item.get("label", "object")
                sub_score = m_item.get("score")
                fc = mask_to_geojson_polygons(
                    mask=sub_arr,
                    transform=ref.transform,
                    crs=ref.crs,
                    label=sub_label,
                    score=sub_score,
                    min_area_pixels=10,
                )
                all_features.extend(fc.get("features", []))
            mask_geojson = {
                "type": "FeatureCollection",
                "features": all_features,
            }
            if ref.crs:
                mask_geojson["crs"] = {"type": "name", "properties": {"name": crs_str}}
        else:
            mask_geojson = None

        if mask_geojson and len(mask_geojson.get("features", [])) > 0:
            sam2_path = package_dir / "sam2_masks.geojson"
            sam2_path.write_text(json.dumps(mask_geojson, indent=2), encoding="utf-8")
            generated_files.append("sam2_masks.geojson")

    # 4. Export Change-Detection Mask -> change_mask.tif
    if change_mask is not None:
        ch_prof = dict(ref.profile)
        ch_prof.update({
            "count": 1,
            "dtype": str(change_mask.dtype),
            "nodata": 0,
        })
        ch_path = package_dir / "change_mask.tif"
        write_geotiff(
            output_dest=ch_path,
            array=change_mask,
            reference_profile=ch_prof,
        )
        generated_files.append("change_mask.tif")

    # 5. Export Land-Cover Segmentation -> landcover.tif
    if landcover_mask is not None:
        lc_arr = landcover_mask.astype(np.uint8)
        lc_prof = dict(ref.profile)
        lc_prof.update({
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
        })
        lc_path = package_dir / "landcover.tif"
        write_geotiff(
            output_dest=lc_path,
            array=lc_arr,
            reference_profile=lc_prof,
        )
        generated_files.append("landcover.tif")

    # 6. Export Aligned SAR & Optical if available
    if aligned_sar is not None:
        sar_path = package_dir / "aligned_sar.tif"
        if isinstance(aligned_sar, bytes):
            sar_path.write_bytes(aligned_sar)
        else:
            write_geotiff(output_dest=sar_path, array=aligned_sar, reference_profile=ref.profile)
        generated_files.append("aligned_sar.tif")

    if aligned_optical is not None:
        opt_path = package_dir / "aligned_optical.tif"
        if isinstance(aligned_optical, bytes):
            opt_path.write_bytes(aligned_optical)
        else:
            write_geotiff(output_dest=opt_path, array=aligned_optical, reference_profile=ref.profile)
        generated_files.append("aligned_optical.tif")

    # 7. Metadata JSON -> original_metadata.json
    metadata_dict = to_json_safe({
        "analysis_id": aid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "driver": ref.driver,
        "crs": crs_str,
        "width": ref.width,
        "height": ref.height,
        "bands": ref.bands,
        "resolution": {"x": ref.resolution[0], "y": ref.resolution[1]},
        "bounds": {
            "left": ref.bounds.left,
            "bottom": ref.bounds.bottom,
            "right": ref.bounds.right,
            "top": ref.bounds.top,
        },
        "transform": [float(x) for x in ref.transform[:6]],
        "landcover_classes": LANDCOVER_CLASS_MAP,
        "generated_files": generated_files,
    })
    meta_path = package_dir / "original_metadata.json"
    meta_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
    generated_files.insert(0, "original_metadata.json")

    # 8. README.txt
    readme_text = build_qgis_readme(analysis_id=aid, crs_str=crs_str, generated_files=generated_files)
    readme_path = package_dir / "README.txt"
    readme_path.write_text(readme_text, encoding="utf-8")
    generated_files.append("README.txt")

    # 9. Create ZIP bundle: analysis_{id}_qgis.zip
    zip_path = None
    if create_zip:
        zip_filename = f"analysis_{aid}_qgis.zip"
        zip_path = base_dir / zip_filename
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in generated_files:
                fpath = package_dir / fname
                if fpath.exists():
                    zf.write(fpath, arcname=f"{aid}/{fname}")

    return {
        "success": True,
        "analysis_id": aid,
        "package_dir": str(package_dir).replace("\\", "/"),
        "zip_path": str(zip_path).replace("\\", "/") if zip_path else None,
        "zip_filename": f"analysis_{aid}_qgis.zip" if zip_path else None,
        "files": generated_files,
        "crs": crs_str,
        "landcover_classes": LANDCOVER_CLASS_MAP,
        "total_files": len(generated_files),
    }


def get_qgis_export_manifest(analysis_id: str, output_base_dir: Union[str, Path] = "outputs/geospatial") -> Dict[str, Any]:
    """
    Retrieves the file manifest and metadata for a previously generated QGIS package.
    """
    base_dir = Path(output_base_dir)
    package_dir = base_dir / analysis_id
    zip_path = base_dir / f"analysis_{analysis_id}_qgis.zip"

    if not package_dir.exists():
        return {
            "success": False,
            "analysis_id": analysis_id,
            "exists": False,
            "detail": f"Export package '{analysis_id}' not found.",
        }

    file_items = []
    for p in package_dir.iterdir():
        if p.is_file():
            file_items.append({
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "url": f"/api/geospatial/export/{analysis_id}/{p.name}",
            })

    meta_content = {}
    meta_path = package_dir / "original_metadata.json"
    if meta_path.exists():
        try:
            meta_content = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "success": True,
        "analysis_id": analysis_id,
        "exists": True,
        "package_dir": str(package_dir).replace("\\", "/"),
        "files": file_items,
        "zip_available": zip_path.exists(),
        "zip_size_bytes": zip_path.stat().st_size if zip_path.exists() else 0,
        "zip_url": f"/api/geospatial/export/{analysis_id}/zip" if zip_path.exists() else None,
        "metadata": meta_content,
        "landcover_classes": LANDCOVER_CLASS_MAP,
    }

