"""
Metadata extraction and JSON serialization for geospatial raster datasets.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import io
import numpy as np

try:
    import rasterio
    from rasterio.io import MemoryFile
    from affine import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    MemoryFile = None
    Affine = None
    RASTERIO_AVAILABLE = False


def to_json_safe(obj: Any) -> Any:
    """
    Recursively converts NumPy, Affine, and Rasterio types to standard Python types
    for seamless JSON serialization.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if Affine is not None and isinstance(obj, Affine):
        return [float(x) for x in obj[:6]]
    if hasattr(obj, "to_string"):
        return obj.to_string()
    if hasattr(obj, "to_epsg") and obj.to_epsg() is not None:
        return f"EPSG:{obj.to_epsg()}"
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(item) for item in obj]
    return str(obj)


def get_raster_metadata(source: Union[str, Path, bytes, io.BytesIO, Any]) -> Dict[str, Any]:
    """
    Extracts comprehensive, JSON-safe metadata from a GeoTIFF dataset or source.

    Supports:
      - File path (str or Path)
      - Raw bytes or io.BytesIO buffer
      - Already opened rasterio.DatasetReader

    Returns:
      Dict matching Step 5 specification:
      {
        "success": True,
        "driver": "GTiff",
        "width": int,
        "height": int,
        "bands": int,
        "dtype": str,
        "crs": str,
        "bounds": {"left": float, "bottom": float, "right": float, "top": float},
        "resolution": {"x": float, "y": float},
        "nodata": Optional[float/int],
        "transform": list of 6 affine coefficients
      }
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    def _extract_from_dataset(ds: Any) -> Dict[str, Any]:
        crs_str = None
        if ds.crs:
            epsg = ds.crs.to_epsg()
            crs_str = f"EPSG:{epsg}" if epsg else ds.crs.to_string()

        bounds_dict = {
            "left": float(ds.bounds.left),
            "bottom": float(ds.bounds.bottom),
            "right": float(ds.bounds.right),
            "top": float(ds.bounds.top),
        }

        res_dict = {
            "x": float(ds.res[0]),
            "y": float(ds.res[1]),
        }

        transform_list = [float(x) for x in ds.transform[:6]]
        dtype_str = str(ds.dtypes[0]) if ds.dtypes else "unknown"

        meta = {
            "success": True,
            "driver": str(ds.driver),
            "width": int(ds.width),
            "height": int(ds.height),
            "bands": int(ds.count),
            "dtype": dtype_str,
            "crs": crs_str or "UNSPECIFIED",
            "bounds": bounds_dict,
            "resolution": res_dict,
            "nodata": float(ds.nodata) if ds.nodata is not None else None,
            "transform": transform_list,
        }
        return to_json_safe(meta)

    # Already an open rasterio dataset
    if hasattr(source, "read") and hasattr(source, "bounds"):
        return _extract_from_dataset(source)

    # In-memory bytes or BytesIO
    if isinstance(source, bytes):
        with MemoryFile(source) as memfile:
            with memfile.open() as ds:
                return _extract_from_dataset(ds)

    if isinstance(source, io.BytesIO):
        with MemoryFile(source.getvalue()) as memfile:
            with memfile.open() as ds:
                return _extract_from_dataset(ds)

    # File path
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as ds:
        return _extract_from_dataset(ds)

