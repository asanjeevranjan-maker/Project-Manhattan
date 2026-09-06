"""
Spatial alignment of secondary/temporal rasters (e.g. SAR -> Optical, T2 -> T1) to reference pixel grids.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    reproject = None
    Resampling = None
    CRS = None
    RASTERIO_AVAILABLE = False

from .raster_io import read_raster, RasterData, image_to_bands_layout
from .metadata import to_json_safe


def align_raster_to_reference(
    source: Union[RasterData, str, bytes, np.ndarray],
    reference: Union[RasterData, str, bytes],
    src_crs: Optional[Any] = None,
    src_transform: Optional[Any] = None,
    is_mask: bool = False,
    resampling: Optional[str] = None,
    fill_value: Optional[float] = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Reprojects and resamples a source raster directly onto the pixel grid
    (CRS, affine transform, width, height) of a reference raster.

    Critical for:
      - Optical + SAR fusion (overlaying SAR backscatter onto optical geometry)
      - Bi-temporal change analysis (aligning T2 onto T1)

    Returns:
      (aligned_array, aligned_profile, alignment_metadata)
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    # Ingest reference raster metadata
    if not isinstance(reference, RasterData):
        ref_data = read_raster(reference)
    else:
        ref_data = reference

    ref_crs = ref_data.crs
    ref_transform = ref_data.transform
    ref_width = ref_data.width
    ref_height = ref_data.height
    ref_profile = dict(ref_data.profile)

    # Ingest source raster
    if isinstance(source, RasterData):
        src_arr = source.array
        source_crs = source.crs
        source_transform = source.transform
        src_nodata = source.nodata
    elif isinstance(source, (str, bytes)):
        src_data = read_raster(source)
        src_arr = src_data.array
        source_crs = src_data.crs
        source_transform = src_data.transform
        src_nodata = src_data.nodata
    elif isinstance(source, np.ndarray):
        src_arr = image_to_bands_layout(source)
        source_crs = src_crs
        source_transform = src_transform
        src_nodata = None
        if source_crs is None or source_transform is None:
            raise ValueError("src_crs and src_transform must be provided when source is a NumPy array")
    else:
        raise TypeError(f"Unsupported source type for alignment: {type(source)}")

    if isinstance(source_crs, str):
        source_crs = CRS.from_string(source_crs)
    if isinstance(ref_crs, str):
        ref_crs = CRS.from_string(ref_crs)

    num_bands = src_arr.shape[0]

    # Resampling choice: nearest for masks/classes, bilinear for continuous
    if is_mask or np.issubdtype(src_arr.dtype, np.integer) or np.issubdtype(src_arr.dtype, np.bool_):
        chosen_resampling = Resampling.nearest
    else:
        if resampling == "nearest":
            chosen_resampling = Resampling.nearest
        elif resampling == "cubic":
            chosen_resampling = Resampling.cubic
        else:
            chosen_resampling = Resampling.bilinear

    aligned_arr = np.full((num_bands, ref_height, ref_width), fill_value=fill_value or 0, dtype=src_arr.dtype)

    reproject(
        source=src_arr,
        destination=aligned_arr,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=chosen_resampling,
        src_nodata=src_nodata,
        dst_nodata=ref_data.nodata,
    )

    out_profile = dict(ref_profile)
    out_profile.update({
        "count": num_bands,
        "dtype": str(aligned_arr.dtype),
        "height": ref_height,
        "width": ref_width,
        "transform": ref_transform,
        "crs": ref_crs,
    })

    src_crs_str = source_crs.to_string() if hasattr(source_crs, "to_string") else str(source_crs)
    ref_crs_str = ref_crs.to_string() if hasattr(ref_crs, "to_string") else str(ref_crs)

    alignment_metadata = to_json_safe({
        "aligned": True,
        "source_crs": src_crs_str,
        "reference_crs": ref_crs_str,
        "width": ref_width,
        "height": ref_height,
        "bands": num_bands,
        "transform": [float(x) for x in ref_transform[:6]],
        "resampling": chosen_resampling.name,
    })

    return aligned_arr, out_profile, alignment_metadata

