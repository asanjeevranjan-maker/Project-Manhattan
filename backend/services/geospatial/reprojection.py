"""
Reprojection and resampling with strict separation of continuous and categorical data.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import array_bounds
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    calculate_default_transform = None
    array_bounds = None
    reproject = None
    Resampling = None
    CRS = None
    RASTERIO_AVAILABLE = False

from .raster_io import read_raster, RasterData, image_to_bands_layout


def reproject_raster(
    source: Union[RasterData, str, bytes, np.ndarray],
    destination_crs: Union[str, Any],
    src_crs: Optional[Any] = None,
    src_transform: Optional[Any] = None,
    resampling: Optional[str] = None,
    is_mask: bool = False,
    src_nodata: Optional[float] = None,
    dst_nodata: Optional[float] = None,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Reprojects a raster or mask array to a target coordinate reference system (CRS).

    Resampling Rules (Step 6):
      - Continuous data (e.g. optical reflectance, SAR dB backscatter): bilinear
      - Categorical / Mask data (e.g. SAM2 masks, land-cover classes): nearest
      - NEVER use bilinear for categorical masks to avoid interpolating discrete class IDs.

    Returns:
      (reprojected_array, destination_transform, destination_profile)
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    # Normalize destination CRS
    if isinstance(destination_crs, str):
        dst_crs_obj = CRS.from_string(destination_crs)
    else:
        dst_crs_obj = destination_crs

    # Extract source array, transform, crs, and dimensions
    profile = {}
    if isinstance(source, RasterData):
        arr = source.array
        source_crs = source.crs
        source_transform = source.transform
        width = source.width
        height = source.height
        profile = dict(source.profile)
        src_nodata = src_nodata if src_nodata is not None else source.nodata
    elif isinstance(source, (str, bytes)):
        rdata = read_raster(source)
        arr = rdata.array
        source_crs = rdata.crs
        source_transform = rdata.transform
        width = rdata.width
        height = rdata.height
        profile = dict(rdata.profile)
        src_nodata = src_nodata if src_nodata is not None else rdata.nodata
    elif isinstance(source, np.ndarray):
        arr = image_to_bands_layout(source)
        source_crs = src_crs
        source_transform = src_transform
        if source_crs is None or source_transform is None:
            raise ValueError("src_crs and src_transform must be provided when source is a NumPy array")
        height, width = arr.shape[1], arr.shape[2]
    else:
        raise TypeError(f"Unsupported source type for reprojection: {type(source)}")

    if isinstance(source_crs, str):
        source_crs = CRS.from_string(source_crs)

    num_bands = arr.shape[0]

    # Select resampling algorithm
    if is_mask or np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        # Strictly enforce nearest neighbor for masks/labels
        chosen_resampling = Resampling.nearest
    else:
        if resampling == "nearest":
            chosen_resampling = Resampling.nearest
        elif resampling == "cubic":
            chosen_resampling = Resampling.cubic
        else:
            chosen_resampling = Resampling.bilinear

    # Calculate default transform and target dimensions
    left, bottom, right, top = array_bounds(height, width, source_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        source_crs,
        dst_crs_obj,
        width,
        height,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
    )

    dst_arr = np.zeros((num_bands, dst_height, dst_width), dtype=arr.dtype)

    reproject(
        source=arr,
        destination=dst_arr,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs_obj,
        resampling=chosen_resampling,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
    )

    out_profile = dict(profile)
    out_profile.update({
        "crs": dst_crs_obj,
        "transform": dst_transform,
        "width": dst_width,
        "height": dst_height,
        "count": num_bands,
        "dtype": str(dst_arr.dtype),
    })

    return dst_arr, dst_transform, out_profile
