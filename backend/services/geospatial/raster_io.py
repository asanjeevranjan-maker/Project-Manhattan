"""
GeoTIFF reading, writing, in-memory buffering, and array layout transformations.
"""

from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import io
import numpy as np

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.crs import CRS
    from affine import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    MemoryFile = None
    CRS = None
    Affine = None
    RASTERIO_AVAILABLE = False


@dataclass
class RasterData:
    """Encapsulates in-memory raster data, coordinates, and spatial metadata."""
    array: np.ndarray             # Layout: (bands, height, width)
    width: int
    height: int
    bands: int
    dtype: str
    crs: Any                      # rasterio.crs.CRS or None
    transform: Any                # affine.Affine
    bounds: Any                   # rasterio.coords.BoundingBox
    resolution: Tuple[float, float]
    nodata: Optional[float]
    driver: str
    profile: Dict[str, Any]
    tags: Dict[str, Any]

    def to_image(self) -> np.ndarray:
        """Convenience method to get (H, W, C) or 2D (H, W) layout."""
        return bands_to_image_layout(self.array)


def bands_to_image_layout(array: np.ndarray, squeeze_single_band: bool = True) -> np.ndarray:
    """
    Converts raster array from Rasterio band-first layout (bands, height, width)
    to standard Computer Vision / PIL / PyTorch layout (height, width, channels)
    or single-band 2D (height, width) for masks.
    """
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        # If it's already (H, W, C) where C in (1, 3, 4) and H, W > C:
        if array.shape[2] in (1, 3, 4) and array.shape[0] > 4:
            return array.squeeze(axis=2) if (squeeze_single_band and array.shape[2] == 1) else array
        
        # Band first (C, H, W) -> (H, W, C)
        transposed = np.transpose(array, (1, 2, 0))
        if squeeze_single_band and transposed.shape[2] == 1:
            return transposed[:, :, 0]
        return transposed
    raise ValueError(f"Unsupported array dimensions for image conversion: {array.shape}")


def image_to_bands_layout(array: np.ndarray) -> np.ndarray:
    """
    Converts standard 2D (H, W) mask or 3D (H, W, C) image to Rasterio
    band-first layout (bands, height, width).
    """
    if array.ndim == 2:
        return np.expand_dims(array, axis=0)
    if array.ndim == 3:
        # If shape is already (C, H, W) where C in (1, 3, 4) and H, W > C:
        if array.shape[0] in (1, 3, 4) and array.shape[2] > 4:
            return array
        # Transpose (H, W, C) -> (C, H, W)
        return np.transpose(array, (2, 0, 1))
    raise ValueError(f"Unsupported array dimensions for bands conversion: {array.shape}")


def read_raster(source: Union[str, Path, bytes, io.BytesIO, Any]) -> RasterData:
    """
    Reads a geospatial raster from a filepath, raw bytes, or MemoryFile.
    Preserves original metadata, CRS, affine transform, nodata, and band array.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    def _from_dataset(ds: Any) -> RasterData:
        array = ds.read()  # (bands, height, width)
        profile = dict(ds.profile)
        tags = ds.tags()
        res = (float(ds.res[0]), float(ds.res[1]))
        return RasterData(
            array=array,
            width=ds.width,
            height=ds.height,
            bands=ds.count,
            dtype=str(array.dtype),
            crs=ds.crs,
            transform=ds.transform,
            bounds=ds.bounds,
            resolution=res,
            nodata=ds.nodata,
            driver=ds.driver or "GTiff",
            profile=profile,
            tags=tags,
        )

    if hasattr(source, "read") and hasattr(source, "bounds"):
        return _from_dataset(source)

    if isinstance(source, bytes):
        with MemoryFile(source) as memfile:
            with memfile.open() as ds:
                return _from_dataset(ds)

    if isinstance(source, io.BytesIO):
        with MemoryFile(source.getvalue()) as memfile:
            with memfile.open() as ds:
                return _from_dataset(ds)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as ds:
        return _from_dataset(ds)


def write_geotiff(
    output_dest: Optional[Union[str, Path, io.BytesIO]] = None,
    array: Optional[np.ndarray] = None,
    reference_profile: Optional[Dict[str, Any]] = None,
    crs: Optional[Any] = None,
    transform: Optional[Any] = None,
    nodata: Optional[float] = None,
    dtype: Optional[Union[str, np.dtype]] = None,
    driver: str = "GTiff",
    compress: str = "lzw",
) -> Union[Path, bytes]:
    """
    Writes a NumPy array to GeoTIFF preserving or updating CRS, transform, dimensions, and nodata.
    Safely adapts between 2D (H, W) masks and 3D (bands, H, W) multi-spectral rasters.

    If output_dest is a file path, writes to disk and returns the Path.
    If output_dest is None or io.BytesIO, returns the GeoTIFF bytes.
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("Geospatial processing unavailable in this deployment (rasterio missing)")

    if array is None:
        raise ValueError("array must be provided to write_geotiff")

    # Ensure band-first layout: (bands, H, W)
    bands_arr = image_to_bands_layout(array)
    num_bands, height, width = bands_arr.shape

    out_dtype = dtype or str(bands_arr.dtype)
    bands_arr = bands_arr.astype(out_dtype)

    # Build profile
    profile = {}
    if reference_profile:
        profile.update(reference_profile)

    profile.update({
        "driver": driver,
        "count": num_bands,
        "height": height,
        "width": width,
        "dtype": out_dtype,
    })

    if compress and driver == "GTiff":
        profile["compress"] = compress

    if crs is not None:
        profile["crs"] = crs
    if transform is not None:
        profile["transform"] = transform
    if nodata is not None:
        profile["nodata"] = nodata

    # Validate mandatory georeferencing
    if "transform" not in profile or profile["transform"] is None:
        profile["transform"] = Affine.identity()

    if output_dest is None or isinstance(output_dest, io.BytesIO):
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(bands_arr)
            raw_bytes = memfile.read()
        if isinstance(output_dest, io.BytesIO):
            output_dest.write(raw_bytes)
            output_dest.seek(0)
        return raw_bytes

    out_path = Path(output_dest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(bands_arr)

    return out_path

