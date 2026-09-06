"""
Comprehensive unit tests for Geospatial Raster Support via Rasterio.
Covers metadata extraction, coordinate transformations, reprojection, grid alignment,
clipping, GeoTIFF export, QGIS-ready GeoJSON features, and fallback behavior.
"""

import io
import json
import pytest
import numpy as np
from PIL import Image

import rasterio
from rasterio.io import MemoryFile
from rasterio.crs import CRS
from affine import Affine
from fastapi.testclient import TestClient

from services.geospatial import (
    RASTERIO_AVAILABLE,
    read_raster,
    write_geotiff,
    bands_to_image_layout,
    image_to_bands_layout,
    get_raster_metadata,
    reproject_raster,
    align_raster_to_reference,
    clip_raster,
    pixel_to_geo,
    geo_to_pixel,
    bbox_pixel_to_geo,
    detections_to_geojson,
    mask_to_geojson_polygons,
    to_json_safe,
)
from main import app

client = TestClient(app)


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def synthetic_geotiff_bytes() -> bytes:
    """Generates a synthetic 3-band RGB GeoTIFF in memory with EPSG:32633."""
    width, height = 64, 64
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)
    crs = CRS.from_epsg(32633)

    # 3 bands: (3, 64, 64)
    data = np.zeros((3, height, width), dtype=np.uint8)
    data[0, :, :] = 100  # Red
    data[1, :, :] = 150  # Green
    data[2, :, :] = 200  # Blue

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)
        return memfile.read()


@pytest.fixture
def synthetic_mask_bytes() -> bytes:
    """Generates a synthetic 1-band binary mask GeoTIFF."""
    width, height = 64, 64
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)
    crs = CRS.from_epsg(32633)

    # Central square is foreground (1), background is (0)
    mask = np.zeros((1, height, width), dtype=np.uint8)
    mask[0, 20:44, 20:44] = 1

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(mask)
        return memfile.read()


@pytest.fixture
def synthetic_sar_bytes() -> bytes:
    """Generates a synthetic 1-band float32 SAR GeoTIFF with 20m resolution."""
    width, height = 32, 32
    # 20m pixel size, covering roughly the same area
    transform = Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 6000000.0)
    crs = CRS.from_epsg(32633)

    sar_data = np.random.uniform(-25.0, 5.0, (1, height, width)).astype(np.float32)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(sar_data)
        return memfile.read()


# =====================================================================
# UNIT TESTS
# =====================================================================

def test_status_endpoint():
    """Verifies the geospatial status endpoint reports availability and versions."""
    res = client.get("/api/geospatial/status")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is True
    assert "rasterio_version" in data
    assert "gdal_version" in data


def test_metadata_extraction(synthetic_geotiff_bytes):
    """Tests metadata extraction via Python function and HTTP endpoint."""
    meta = get_raster_metadata(synthetic_geotiff_bytes)
    assert meta["success"] is True
    assert meta["driver"] == "GTiff"
    assert meta["width"] == 64
    assert meta["height"] == 64
    assert meta["bands"] == 3
    assert meta["dtype"] == "uint8"
    assert "32633" in meta["crs"]
    assert meta["resolution"]["x"] == 10.0
    assert meta["resolution"]["y"] == 10.0
    assert meta["bounds"]["left"] == 500000.0
    assert meta["bounds"]["top"] == 6000000.0

    # Test HTTP endpoint
    files = {"file": ("test.tif", synthetic_geotiff_bytes, "image/tiff")}
    res = client.post("/api/geospatial/metadata", files=files)
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["width"] == 64
    assert res_data["bands"] == 3


def test_pixel_and_geo_conversions():
    """Tests pixel_to_geo and geo_to_pixel bi-directional round-trip."""
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)

    # Pixel (row=10, col=20)
    x, y = pixel_to_geo(row=10, col=20, transform=transform, offset="center")
    assert x == 500000.0 + (20 + 0.5) * 10.0
    assert y == 6000000.0 - (10 + 0.5) * 10.0

    # Inverse round-trip
    row, col = geo_to_pixel(x=x, y=y, transform=transform)
    assert row == 10
    assert col == 20


def test_bbox_pixel_to_geo():
    """Tests bounding box pixel to geographic bounds conversion."""
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)
    # Box [x1, y1, x2, y2] = [10, 5, 30, 25]
    geo_bbox = bbox_pixel_to_geo([10, 5, 30, 25], transform)

    assert geo_bbox["left"] == 500000.0 + 10 * 10.0
    assert geo_bbox["right"] == 500000.0 + 30 * 10.0
    assert geo_bbox["top"] == 6000000.0 - 5 * 10.0
    assert geo_bbox["bottom"] == 6000000.0 - 25 * 10.0


def test_reprojection_continuous(synthetic_geotiff_bytes):
    """Tests continuous raster reprojection to WGS84 (EPSG:4326)."""
    dst_arr, dst_trans, dst_prof = reproject_raster(
        source=synthetic_geotiff_bytes,
        destination_crs="EPSG:4326",
        is_mask=False,
    )
    assert dst_arr.ndim == 3
    assert dst_arr.shape[0] == 3
    assert "4326" in str(dst_prof["crs"])


def test_reprojection_categorical_mask(synthetic_mask_bytes):
    """Verifies that mask reprojection enforces nearest-neighbor to prevent class blurring."""
    dst_arr, dst_trans, dst_prof = reproject_raster(
        source=synthetic_mask_bytes,
        destination_crs="EPSG:4326",
        is_mask=True,
    )
    # Binary mask must only contain 0 and 1, no intermediate interpolation decimals
    unique_vals = set(np.unique(dst_arr))
    assert unique_vals.issubset({0, 1})


def test_alignment(synthetic_geotiff_bytes, synthetic_sar_bytes):
    """Tests spatial grid alignment of SAR onto Optical reference grid."""
    aligned_arr, aligned_prof, meta = align_raster_to_reference(
        source=synthetic_sar_bytes,
        reference=synthetic_geotiff_bytes,
    )
    # Optical is 64x64; aligned SAR must match 64x64
    assert aligned_arr.shape == (1, 64, 64)
    assert aligned_prof["width"] == 64
    assert aligned_prof["height"] == 64
    assert meta["aligned"] is True

    # Test HTTP endpoint
    files = {
        "source_file": ("sar.tif", synthetic_sar_bytes, "image/tiff"),
        "reference_file": ("opt.tif", synthetic_geotiff_bytes, "image/tiff"),
    }
    res = client.post("/api/geospatial/align", files=files)
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/tiff"


def test_clipping(synthetic_geotiff_bytes):
    """Tests clipping raster using a geographic bounding box."""
    # Bounding box in middle: left=500200, bottom=5999600, right=500400, top=5999800
    bbox = [500200.0, 5999600.0, 500400.0, 5999800.0]
    clipped_arr, clipped_trans, clipped_prof = clip_raster(
        source=synthetic_geotiff_bytes,
        bbox=bbox,
    )
    assert clipped_arr.shape[1] < 64
    assert clipped_arr.shape[2] < 64

    # Test HTTP endpoint
    files = {"file": ("test.tif", synthetic_geotiff_bytes, "image/tiff")}
    data = {"bbox": json.dumps(bbox)}
    res = client.post("/api/geospatial/clip", files=files, data=data)
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/tiff"


def test_mask_geotiff_export(synthetic_geotiff_bytes):
    """Tests exporting an AI mask as a georeferenced GeoTIFF retaining reference profile."""
    # Create simple 64x64 PIL binary image
    mask_img = Image.new("L", (64, 64), color=0)
    for x in range(20, 40):
        for y in range(20, 40):
            mask_img.putpixel((x, y), 1)

    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    files = {
        "reference_file": ("ref.tif", synthetic_geotiff_bytes, "image/tiff"),
        "mask_file": ("mask.png", png_bytes, "image/png"),
    }
    res = client.post("/api/geospatial/export-mask", files=files)
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/tiff"

    # Verify georeferencing of exported GeoTIFF
    rdata = read_raster(res.content)
    assert rdata.width == 64
    assert rdata.height == 64
    assert "32633" in str(rdata.crs)
    assert rdata.transform[0] == 10.0


def test_detections_to_geojson(synthetic_geotiff_bytes):
    """Tests Grounding DINO detections conversion to QGIS-compatible GeoJSON."""
    detections = [
        {
            "detection_id": "vessel_01",
            "label": "cargo vessel",
            "score": 0.892,  # Raw uncalibrated score
            "box": [10.0, 15.0, 35.0, 40.0],
        },
        {
            "detection_id": "dock_02",
            "label": "pier",
            "score": 0.941,
            "box": [40.0, 10.0, 60.0, 30.0],
        },
    ]

    rdata = read_raster(synthetic_geotiff_bytes)
    geojson = detections_to_geojson(detections, rdata.transform, rdata.crs)

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2
    f1 = geojson["features"][0]
    assert f1["geometry"]["type"] == "Polygon"
    assert f1["properties"]["detection_id"] == "vessel_01"
    assert f1["properties"]["score"] == 0.892
    assert f1["properties"]["label"] == "cargo vessel"

    # Test HTTP endpoint
    files = {"reference_file": ("ref.tif", synthetic_geotiff_bytes, "image/tiff")}
    data = {"detections": json.dumps(detections)}
    res = client.post("/api/geospatial/detections-geojson", files=files, data=data)
    assert res.status_code == 200
    res_json = res.json()
    assert res_json["type"] == "FeatureCollection"
    assert len(res_json["features"]) == 2


def test_sam2_mask_to_geojson_polygons():
    """Tests SAM2 mask vectorization into GeoJSON polygons."""
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)
    crs = CRS.from_epsg(32633)

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[15:35, 15:35] = 1  # 20x20 foreground square

    geojson = mask_to_geojson_polygons(
        mask=mask,
        transform=transform,
        crs=crs,
        label="oil_storage_tank",
        score=0.915,
        min_area_pixels=10,
    )

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) >= 1
    feat = geojson["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    assert feat["properties"]["label"] == "oil_storage_tank"
    assert feat["properties"]["score"] == 0.915
    assert feat["properties"]["mask_area_pixels"] == 400


def test_invalid_raster_error():
    """Verifies that corrupted / invalid raster uploads return a clean 400 error."""
    invalid_bytes = b"Not a valid GeoTIFF file content"
    files = {"file": ("corrupt.tif", invalid_bytes, "image/tiff")}
    res = client.post("/api/geospatial/metadata", files=files)
    assert res.status_code == 400
    assert "Failed to extract geospatial metadata" in res.json()["detail"]


def test_rasterio_unavailable_fallback(monkeypatch):
    """
    Simulates environment where rasterio is not available (e.g. serverless Vercel).
    Verifies:
      1. Non-geospatial endpoints continue to return 200 OK.
      2. Geospatial endpoints return HTTP 503 with a clean explanation.
    """
    import services.geospatial.router as geo_router
    monkeypatch.setattr(geo_router, "RASTERIO_AVAILABLE", False)

    # Health check must still be 200 OK
    health_res = client.get("/health")
    assert health_res.status_code == 200

    # Classes endpoint must still be 200 OK
    classes_res = client.get("/classes")
    assert classes_res.status_code == 200

    # Geospatial endpoint must return 503
    files = {"file": ("test.tif", b"some-bytes", "image/tiff")}
    geo_res = client.post("/api/geospatial/metadata", files=files)
    assert geo_res.status_code == 503
    assert "Geospatial processing unavailable in this deployment" in geo_res.json()["detail"]

