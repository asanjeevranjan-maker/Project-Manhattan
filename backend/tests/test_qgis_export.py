"""
Unit test suite for QGIS-Compatible Export Packager.
Tests GeoJSON features, GeoTIFF masks, integer land-cover classes,
manifest generation, zip packaging, and FastAPI export endpoints.
"""

import io
import json
import zipfile
import shutil
from pathlib import Path
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
    export_qgis_package,
    get_qgis_export_manifest,
    build_qgis_readme,
    LANDCOVER_CLASS_MAP,
    read_raster,
)
from main import app

client = TestClient(app)


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def synthetic_ref_geotiff_bytes() -> bytes:
    """Generates a synthetic 64x64 GeoTIFF in EPSG:32633 (UTM Zone 33N)."""
    width, height = 64, 64
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)
    crs = CRS.from_epsg(32633)
    data = np.full((3, height, width), fill_value=128, dtype=np.uint8)

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
def sample_detections():
    return [
        {
            "id": "det_ship_001",
            "label": "ship",
            "score": 0.94,
            "box": [10, 15, 30, 45],  # ymin, xmin, ymax, xmax
        },
        {
            "id": "det_dock_002",
            "label": "dock",
            "score": 0.88,
            "box": [40, 5, 60, 25],
        },
    ]


@pytest.fixture
def sample_binary_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[15:35, 20:40] = 255
    return mask


@pytest.fixture
def sample_landcover_mask():
    # Classes: 0=other, 1=water, 2=vegetation, 3=built-up, 4=bare soil
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[0:20, :] = 1    # water
    mask[20:40, :] = 2   # vegetation
    mask[40:50, :] = 3   # built-up
    mask[50:64, :] = 4   # bare soil
    return mask


@pytest.fixture(autouse=True)
def cleanup_test_outputs():
    test_dir = Path("outputs/geospatial_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)


# =====================================================================
# CORE QGIS EXPORT PACKAGER TESTS
# =====================================================================

def test_export_qgis_package_layers_and_zip(
    synthetic_ref_geotiff_bytes,
    sample_detections,
    sample_binary_mask,
    sample_landcover_mask,
    cleanup_test_outputs,
):
    """Verifies that export_qgis_package creates all expected layers and zip archive."""
    if not RASTERIO_AVAILABLE:
        pytest.skip("Rasterio not installed in this environment.")

    aid = "test_qgis_001"
    manifest = export_qgis_package(
        reference_source=synthetic_ref_geotiff_bytes,
        output_base_dir=cleanup_test_outputs,
        analysis_id=aid,
        detections=sample_detections,
        sam2_masks=sample_binary_mask,
        change_mask=sample_binary_mask,
        landcover_mask=sample_landcover_mask,
        create_zip=True,
    )

    assert manifest["success"] is True
    assert manifest["analysis_id"] == aid
    pkg_path = Path(manifest["package_dir"])
    assert pkg_path.exists()

    # 1. detections.geojson
    det_path = pkg_path / "detections.geojson"
    assert det_path.exists()
    det_geojson = json.loads(det_path.read_text(encoding="utf-8"))
    assert det_geojson["type"] == "FeatureCollection"
    assert len(det_geojson["features"]) == 2
    f0 = det_geojson["features"][0]
    assert f0["properties"]["label"] == "ship"
    assert f0["properties"]["score"] == 0.94
    assert f0["properties"]["id"] == "det_ship_001"
    # Verify coordinates are georeferenced (UTM coords around 500,000 / 6,000,000)
    coords = f0["geometry"]["coordinates"][0]
    assert len(coords) == 5  # Closed polygon
    assert coords[0][0] >= 500000.0
    assert coords[0][1] <= 6000000.0

    # 2. sam2_masks.geojson
    sam2_path = pkg_path / "sam2_masks.geojson"
    assert sam2_path.exists()
    sam2_geojson = json.loads(sam2_path.read_text(encoding="utf-8"))
    assert sam2_geojson["type"] == "FeatureCollection"
    assert len(sam2_geojson["features"]) >= 1
    sf0 = sam2_geojson["features"][0]
    assert "mask_area_pixels" in sf0["properties"]
    assert "geographic_area" in sf0["properties"]
    assert sf0["properties"]["geographic_area"] > 0
    # Geographic coordinates
    scoords = sf0["geometry"]["coordinates"][0]
    assert scoords[0][0] >= 500000.0

    # 3. change_mask.tif
    ch_path = pkg_path / "change_mask.tif"
    assert ch_path.exists()
    with rasterio.open(ch_path) as ch_ds:
        assert ch_ds.width == 64
        assert ch_ds.height == 64
        assert ch_ds.count == 1
        assert ch_ds.crs == CRS.from_epsg(32633)
        ch_arr = ch_ds.read(1)
        assert np.max(ch_arr) > 0

    # 4. landcover.tif
    lc_path = pkg_path / "landcover.tif"
    assert lc_path.exists()
    with rasterio.open(lc_path) as lc_ds:
        assert lc_ds.width == 64
        assert lc_ds.height == 64
        assert lc_ds.count == 1
        assert str(lc_ds.dtypes[0]) == "uint8"
        assert lc_ds.crs == CRS.from_epsg(32633)
        lc_arr = lc_ds.read(1)
        unique_vals = set(np.unique(lc_arr))
        assert {1, 2, 3, 4}.issubset(unique_vals)

    # 5. original_metadata.json & README.txt
    meta_path = pkg_path / "original_metadata.json"
    assert meta_path.exists()
    meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta_json["analysis_id"] == aid
    assert "landcover_classes" in meta_json
    assert meta_json["landcover_classes"]["1"] == "water"

    readme_path = pkg_path / "README.txt"
    assert readme_path.exists()
    readme_content = readme_path.read_text(encoding="utf-8")
    assert "Layer -> Add Layer -> Add Raster Layer" in readme_content
    assert "Layer -> Add Layer -> Add Vector Layer" in readme_content

    # 6. Zip archive
    zip_path = Path(manifest["zip_path"])
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert f"{aid}/detections.geojson" in namelist
        assert f"{aid}/sam2_masks.geojson" in namelist
        assert f"{aid}/change_mask.tif" in namelist
        assert f"{aid}/landcover.tif" in namelist
        assert f"{aid}/original_metadata.json" in namelist
        assert f"{aid}/README.txt" in namelist


def test_get_qgis_export_manifest(
    synthetic_ref_geotiff_bytes,
    sample_detections,
    cleanup_test_outputs,
):
    """Verifies get_qgis_export_manifest retrieves existing packages and handles missing ones."""
    if not RASTERIO_AVAILABLE:
        pytest.skip("Rasterio not installed in this environment.")

    aid = "test_manifest_pkg"
    export_qgis_package(
        reference_source=synthetic_ref_geotiff_bytes,
        output_base_dir=cleanup_test_outputs,
        analysis_id=aid,
        detections=sample_detections,
    )

    # Found
    manifest = get_qgis_export_manifest(analysis_id=aid, output_base_dir=cleanup_test_outputs)
    assert manifest["success"] is True
    assert manifest["exists"] is True
    assert manifest["analysis_id"] == aid
    assert manifest["zip_available"] is True
    assert len(manifest["files"]) >= 3  # detections, metadata, readme

    # Not found
    missing_manifest = get_qgis_export_manifest(analysis_id="nonexistent_id", output_base_dir=cleanup_test_outputs)
    assert missing_manifest["success"] is False
    assert missing_manifest["exists"] is False


# =====================================================================
# API ENDPOINT TESTS
# =====================================================================

def test_api_export_package_and_download(
    synthetic_ref_geotiff_bytes,
    sample_detections,
    sample_binary_mask,
    sample_landcover_mask,
):
    """Verifies POST /api/geospatial/export/package and subsequent GET endpoints."""
    if not RASTERIO_AVAILABLE:
        pytest.skip("Rasterio not installed in this environment.")

    # Prepare mask PNGs
    ch_img = Image.fromarray(sample_binary_mask)
    ch_buf = io.BytesIO()
    ch_img.save(ch_buf, format="PNG")
    ch_bytes = ch_buf.getvalue()

    lc_img = Image.fromarray(sample_landcover_mask)
    lc_buf = io.BytesIO()
    lc_img.save(lc_buf, format="PNG")
    lc_bytes = lc_buf.getvalue()

    files = [
        ("reference_file", ("ref.tif", synthetic_ref_geotiff_bytes, "image/tiff")),
        ("change_mask", ("change.png", ch_bytes, "image/png")),
        ("landcover_mask", ("landcover.png", lc_bytes, "image/png")),
    ]
    data = {
        "analysis_id": "test_api_qgis_export",
        "detections": json.dumps(sample_detections),
        "create_zip": "true",
    }

    resp = client.post("/api/geospatial/export/package", files=files, data=data)
    assert resp.status_code == 200, resp.text
    res = resp.json()
    assert res["success"] is True
    assert res["analysis_id"] == "test_api_qgis_export"
    assert "zip_url" in res
    assert "detections.geojson" in res["files"]
    assert "change_mask.tif" in res["files"]
    assert "landcover.tif" in res["files"]

    # Test GET /api/geospatial/export/{analysis_id}
    resp_get = client.get(f"/api/geospatial/export/{res['analysis_id']}")
    assert resp_get.status_code == 200
    m = resp_get.json()
    assert m["exists"] is True
    assert m["zip_available"] is True

    # Test GET /api/geospatial/export/{analysis_id}/zip
    resp_zip = client.get(f"/api/geospatial/export/{res['analysis_id']}/zip")
    assert resp_zip.status_code == 200
    assert resp_zip.headers["content-type"] == "application/zip"
    assert len(resp_zip.content) > 100

    # Test GET /api/geospatial/export/{analysis_id}/detections.geojson
    resp_det = client.get(f"/api/geospatial/export/{res['analysis_id']}/detections.geojson")
    assert resp_det.status_code == 200
    det_json = resp_det.json()
    assert det_json["type"] == "FeatureCollection"
    assert len(det_json["features"]) == 2

    # Test GET /api/geospatial/export/{analysis_id}/landcover.tif
    resp_lc = client.get(f"/api/geospatial/export/{res['analysis_id']}/landcover.tif")
    assert resp_lc.status_code == 200
    assert resp_lc.headers["content-type"] == "image/tiff"
    with rasterio.open(io.BytesIO(resp_lc.content)) as lc_ds:
        assert lc_ds.count == 1
        assert lc_ds.crs == CRS.from_epsg(32633)

    # Test 404 on missing analysis or missing file
    resp_404_pkg = client.get("/api/geospatial/export/nonexistent_xyz_999")
    assert resp_404_pkg.status_code == 404

    resp_404_file = client.get(f"/api/geospatial/export/{res['analysis_id']}/missing_file.geojson")
    assert resp_404_file.status_code == 404

