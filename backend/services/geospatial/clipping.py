"""
Raster spatial clipping by geographic bounding box or GeoJSON geometry via rasterio.mask.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

try:
    import rasterio
    from rasterio.mask import mask
    from rasterio.io import MemoryFile
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    mask = None
    MemoryFile = None
    RASTERIO_AVAILABLE = False

from .raster_io import read_raster, RasterData


def clip_raster(
    source: Union[str, Path, bytes, RasterData],
    bbox: Optional[Union[List[float], Tuple[float, float, float, float]]] = None,
    geometry: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    crop: bool = True,
    all_touched: bool = False,
    nodata: Optional[float] = None,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Clips a geospatial raster using a geographic bounding box [min_x, min_y, max_x, max_y]
    or GeoJSON geometry specification.

    Returns:
      (clipped_array, clipped_transform, clipped_profile)
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    shapes = []
    if geometry is not None:
        if isinstance(geometry, dict):
            # Check if Feature or Geometry
            geom = geometry.get("geometry", geometry)
            shapes = [geom]
        elif isinstance(geometry, list):
            shapes = [g.get("geometry", g) if isinstance(g, dict) else g for g in geometry]
    elif bbox is not None:
        if len(bbox) != 4:
            raise ValueError("Bounding box must be [min_x, min_y, max_x, max_y]")
        min_x, min_y, max_x, max_y = bbox
        box_polygon = {
            "type": "Polygon",
            "coordinates": [
                [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                    [min_x, min_y],
                ]
            ],
        }
        shapes = [box_polygon]
    else:
        raise ValueError("Either 'bbox' or 'geometry' must be provided for clipping.")

    def _do_clip(ds: Any) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        clip_nodata = nodata if nodata is not None else ds.nodata
        out_image, out_transform = mask(
            ds,
            shapes=shapes,
            crop=crop,
            all_touched=all_touched,
            nodata=clip_nodata,
        )
        out_profile = dict(ds.profile)
        out_profile.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "count": out_image.shape[0],
            "dtype": str(out_image.dtype),
        })
        if clip_nodata is not None:
            out_profile["nodata"] = clip_nodata
        return out_image, out_transform, out_profile

    if isinstance(source, bytes):
        with MemoryFile(source) as memfile:
            with memfile.open() as ds:
                return _do_clip(ds)

    if isinstance(source, RasterData):
        from .raster_io import write_geotiff
        raw = write_geotiff(output_dest=None, array=source.array, reference_profile=source.profile)
        with MemoryFile(raw) as memfile:
            with memfile.open() as ds:
                return _do_clip(ds)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as ds:
        return _do_clip(ds)

