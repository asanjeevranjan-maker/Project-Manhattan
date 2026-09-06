"""
Coordinate transformations, QGIS-ready GeoJSON feature generation, and SAM2 mask polygonization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import rasterio
    from rasterio.transform import xy, rowcol
    from rasterio.features import shapes
    from rasterio.crs import CRS
    from affine import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    xy = None
    rowcol = None
    shapes = None
    CRS = None
    Affine = None
    RASTERIO_AVAILABLE = False

try:
    from shapely.geometry import shape, mapping
    SHAPELY_AVAILABLE = True
except ImportError:
    shape = None
    mapping = None
    SHAPELY_AVAILABLE = False

from .metadata import to_json_safe


def pixel_to_geo(row: float, col: float, transform: Any, offset: str = "center") -> Tuple[float, float]:
    """
    Transforms pixel coordinates (row, col) into geographic coordinates (x, y)
    using the raster affine transform.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    trans_obj = Affine(*transform[:6]) if not isinstance(transform, Affine) else transform
    x, y = xy(trans_obj, row, col, offset=offset)
    return float(x), float(y)


def geo_to_pixel(x: float, y: float, transform: Any, op=round) -> Tuple[int, int]:
    """
    Transforms geographic coordinates (x, y) into pixel coordinates (row, col).
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    trans_obj = Affine(*transform[:6]) if not isinstance(transform, Affine) else transform
    r, c = rowcol(trans_obj, x, y, op=op)
    return int(r), int(c)


def bbox_pixel_to_geo(pixel_bbox: Union[List[float], Tuple[float, float, float, float]], transform: Any) -> Dict[str, float]:
    """
    Converts a pixel bounding box [x1, y1, x2, y2] (col_min, row_min, col_max, row_max)
    into geographic bounds {'left': ..., 'bottom': ..., 'right': ..., 'top': ...}.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    trans_obj = Affine(*transform[:6]) if not isinstance(transform, Affine) else transform
    col_min, row_min, col_max, row_max = pixel_bbox

    a, b, c, d, e, f = trans_obj[:6]
    # Corner 1: top-left (col_min, row_min)
    x1 = a * col_min + b * row_min + c
    y1 = d * col_min + e * row_min + f
    # Corner 2: bottom-right (col_max, row_max)
    x2 = a * col_max + b * row_max + c
    y2 = d * col_max + e * row_max + f

    return {
        "left": min(float(x1), float(x2)),
        "bottom": min(float(y1), float(y2)),
        "right": max(float(x1), float(x2)),
        "top": max(float(y1), float(y2)),
    }


def detections_to_geojson(
    detections: List[Dict[str, Any]],
    transform: Any,
    crs: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Converts Grounding DINO detections into a standard GeoJSON FeatureCollection
    ready for direct loading into QGIS and web map platforms.

    Each detection polygon is georeferenced using the raster transform.
    Raw model scores are preserved uncalibrated (never modified into pseudo-probabilities).
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    features = []
    crs_str = None
    if crs:
        if hasattr(crs, "to_string"):
            crs_str = crs.to_string()
        else:
            crs_str = str(crs)

    for idx, det in enumerate(detections):
        # Determine bbox: support [x1, y1, x2, y2] in 'box', 'bbox', or 'pixel_bbox'
        raw_box = det.get("box") or det.get("bbox") or det.get("pixel_bbox")
        if not raw_box or len(raw_box) != 4:
            continue

        geo_b = bbox_pixel_to_geo(raw_box, transform)
        min_x = geo_b["left"]
        min_y = geo_b["bottom"]
        max_x = geo_b["right"]
        max_y = geo_b["top"]

        # Polygon coordinates in GeoJSON standard: [ [x, y], ... ] closed ring (counter-clockwise)
        coordinates = [
            [
                [min_x, max_y],  # Top-Left
                [max_x, max_y],  # Top-Right
                [max_x, min_y],  # Bottom-Right
                [min_x, min_y],  # Bottom-Left
                [min_x, max_y],  # Closed
            ]
        ]

        det_id = det.get("detection_id") or det.get("id") or f"det_{idx + 1}"
        label = det.get("label") or det.get("class_name") or "object"
        raw_score = det.get("score", 0.0)

        # Preserve uncalibrated score as float
        try:
            score_val = float(raw_score)
        except (ValueError, TypeError):
            score_val = 0.0

        properties = {
            "id": str(det_id),
            "detection_id": str(det_id),
            "label": str(label),
            "score": score_val,
            "pixel_bbox": [float(x) for x in raw_box],
        }

        # Include any extra metadata from detection (e.g. verified, verification_score)
        if "verified" in det:
            properties["verified"] = bool(det["verified"])
        if "verification_score" in det:
            properties["verification_score"] = float(det["verification_score"])

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates,
            },
            "properties": properties,
        }
        features.append(feature)

    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    if crs_str:
        feature_collection["crs"] = {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:OGC:1.3:{crs_str}" if "EPSG" in crs_str else crs_str},
        }

    return to_json_safe(feature_collection)


def mask_to_geojson_polygons(
    mask: np.ndarray,
    transform: Any,
    crs: Optional[Any] = None,
    label: Optional[str] = None,
    score: Optional[float] = None,
    min_area_pixels: int = 10,
    simplify_tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Vectorizes a 2D binary or categorical mask into a GeoJSON FeatureCollection
    using rasterio.features.shapes.

    Features:
      - Ignores background (value 0).
      - Filters small noisy fragments (< min_area_pixels).
      - Applies optional polygon simplification (via Shapely if available).
      - Preserves detection properties (label, score, mask_area).
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            raise ValueError(f"Mask must be 2D or single-channel 3D, got {mask.shape}")

    trans_obj = Affine(*transform[:6]) if not isinstance(transform, Affine) else transform

    # Ensure integer type for shapes
    mask_int = mask.astype(np.int32)

    features = []
    poly_idx = 1

    for geom, val in shapes(mask_int, transform=trans_obj):
        if val == 0:
            continue  # Skip background

        # Calculate pixel area of this mask value
        pixel_count = int(np.sum(mask_int == val))
        if pixel_count < min_area_pixels:
            continue

        # Optional polygon simplification to prevent noisy/huge GeoJSONs
        final_geom = geom
        if SHAPELY_AVAILABLE and simplify_tolerance is not None and simplify_tolerance > 0:
            try:
                poly = shape(geom)
                simplified = poly.simplify(simplify_tolerance, preserve_topology=True)
                if not simplified.is_empty:
                    final_geom = mapping(simplified)
            except Exception:
                final_geom = geom

        geo_pixel_area = abs(trans_obj.a * trans_obj.e - trans_obj.b * trans_obj.d)
        geo_area = round(float(pixel_count * geo_pixel_area), 3)

        properties = {
            "id": f"sam2_{poly_idx}",
            "polygon_id": f"poly_{poly_idx}",
            "label": str(label) if label else f"class_{val}",
            "score": float(score) if score is not None else None,
            "source_detection_score": float(score) if score is not None else None,
            "mask_area_pixels": pixel_count,
            "geographic_area": geo_area,
            "mask_value": int(val),
        }

        features.append({
            "type": "Feature",
            "geometry": final_geom,
            "properties": properties,
        })
        poly_idx += 1

    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    if crs:
        crs_str = crs.to_string() if hasattr(crs, "to_string") else str(crs)
        feature_collection["crs"] = {
            "type": "name",
            "properties": {"name": crs_str},
        }

    return to_json_safe(feature_collection)
