"""
Project-Manhattan Geospatial Raster Processing Service Layer.
Powered by Rasterio with guaranteed graceful fallback for serverless/Vercel environments.
"""

from typing import Optional

try:
    import rasterio
    RASTERIO_AVAILABLE = True
    RASTERIO_VERSION: Optional[str] = getattr(rasterio, "__version__", None)
    GDAL_VERSION: Optional[str] = getattr(rasterio, "__gdal_version__", None)
except ImportError:
    rasterio = None
    RASTERIO_AVAILABLE = False
    RASTERIO_VERSION = None
    GDAL_VERSION = None


class GeospatialUnavailableError(RuntimeError):
    """Raised when a geospatial operation is invoked in an environment lacking rasterio/GDAL."""
    def __init__(self, message: str = "Geospatial processing unavailable in this deployment"):
        super().__init__(message)


if RASTERIO_AVAILABLE:
    from .raster_io import (
        read_raster,
        write_geotiff,
        bands_to_image_layout,
        image_to_bands_layout,
        RasterData,
    )
    from .metadata import (
        get_raster_metadata,
        to_json_safe,
    )
    from .reprojection import (
        reproject_raster,
    )
    from .alignment import (
        align_raster_to_reference,
    )
    from .clipping import (
        clip_raster,
    )
    from .coordinates import (
        pixel_to_geo,
        geo_to_pixel,
        bbox_pixel_to_geo,
        detections_to_geojson,
        mask_to_geojson_polygons,
    )
    from .qgis_export import (
        export_qgis_package,
        get_qgis_export_manifest,
        build_qgis_readme,
        LANDCOVER_CLASS_MAP,
        CATEGORY_TO_ID,
    )
else:
    # Safe stubs that raise GeospatialUnavailableError if invoked when rasterio is absent
    def _unavailable_stub(*args, **kwargs):
        raise GeospatialUnavailableError()

    read_raster = _unavailable_stub
    write_geotiff = _unavailable_stub
    bands_to_image_layout = _unavailable_stub
    image_to_bands_layout = _unavailable_stub
    RasterData = None
    get_raster_metadata = _unavailable_stub
    to_json_safe = lambda x: x
    reproject_raster = _unavailable_stub
    align_raster_to_reference = _unavailable_stub
    clip_raster = _unavailable_stub
    pixel_to_geo = _unavailable_stub
    geo_to_pixel = _unavailable_stub
    bbox_pixel_to_geo = _unavailable_stub
    detections_to_geojson = _unavailable_stub
    mask_to_geojson_polygons = _unavailable_stub
    export_qgis_package = _unavailable_stub
    get_qgis_export_manifest = _unavailable_stub
    build_qgis_readme = _unavailable_stub
    LANDCOVER_CLASS_MAP = {}
    CATEGORY_TO_ID = {}


__all__ = [
    "RASTERIO_AVAILABLE",
    "RASTERIO_VERSION",
    "GDAL_VERSION",
    "GeospatialUnavailableError",
    "read_raster",
    "write_geotiff",
    "bands_to_image_layout",
    "image_to_bands_layout",
    "RasterData",
    "get_raster_metadata",
    "to_json_safe",
    "reproject_raster",
    "align_raster_to_reference",
    "clip_raster",
    "pixel_to_geo",
    "geo_to_pixel",
    "bbox_pixel_to_geo",
    "detections_to_geojson",
    "mask_to_geojson_polygons",
    "export_qgis_package",
    "get_qgis_export_manifest",
    "build_qgis_readme",
    "LANDCOVER_CLASS_MAP",
    "CATEGORY_TO_ID",
]

