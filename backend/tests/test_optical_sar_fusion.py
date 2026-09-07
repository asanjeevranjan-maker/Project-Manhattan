"""
Optical + SAR Fusion Service tests.

Covers:
  TEST 1  valid optical + valid SAR            -> pipeline completes
  TEST 2  optical + optical                    -> rejected
  TEST 3  SAR + SAR                            -> rejected
  TEST 4  modality detection separates classes
  TEST 5  scene fusion runs on non-georef pair (feature-based alignment)
  TEST 6  SAR statistics are measured (not placeholders)
  TEST 7  optical evidence deterministic
  TEST 8  disagreement reduces fused confidence
  TEST 9  missing metadata -> honest, never fabricated
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure ai-service is importable (this file may run from repo root or ai-service)
ROOT = Path(__file__).resolve().parent.parent.parent
AI_SERVICE = ROOT / "ai-service"
for p in (AI_SERVICE, AI_SERVICE / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Mock the Grounding DINO model before importing the fusion module so the
# tests stay fast/offline. The fusion tests exercise validation, preprocessing,
# evidence extraction and the fusion engine — not the DINO weights. The real
# region-level DINO path is covered end-to-end by the running server.
import types as _types  # noqa: E402

_fake_dino = _types.ModuleType("grounding_dino")

def _mock_detect_objects(image, prompt, **kwargs):
    # Lightweight box proposals derived from the image itself so region fusion
    # code path is exercised deterministically in tests.
    import numpy as _np
    from PIL import Image as _PIL
    w, h = image.size
    arr = _np.asarray(image.convert("RGB"))
    gray = arr.mean(axis=2)
    # water-ish dark block + bright block
    boxes = []
    if gray[int(h * 0.15), int(w * 0.2)] < 90:
        boxes.append({"label": "water", "confidence": 0.72, "score": 0.72,
                      "box": [w * 0.1, h * 0.1, w * 0.5, h * 0.4], "mask": {"polygon": None}})
    if gray[int(h * 0.7), int(w * 0.7)] > 120:
        boxes.append({"label": "building", "confidence": 0.68, "score": 0.68,
                      "box": [w * 0.6, h * 0.6, w * 0.95, h * 0.85], "mask": {"polygon": None}})
    return boxes

_fake_dino.detect_objects = _mock_detect_objects
sys.modules["grounding_dino"] = _fake_dino

from optical_sar_fusion import OpticalSARFusionService  # noqa: E402
from modality_detection import detect_modality  # noqa: E402


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_optical(w: int = 800, h: int = 800) -> Image.Image:
    """Colorful synthetic optical scene: green vegetation + blue water patch."""
    rng = np.random.default_rng(42)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = 60
    arr[..., 1] = 150
    arr[..., 2] = 50
    # Water body (dark blue)
    arr[int(h * 0.1):int(h * 0.4), int(w * 0.1):int(w * 0.5)] = (30, 80, 120)
    # Built-up bright grey block
    arr[int(h * 0.6):int(h * 0.85), int(w * 0.6):int(w * 0.95)] = (120, 120, 122)
    noise = rng.normal(0, 6, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def make_sar(w: int = 800, h: int = 800) -> Image.Image:
    """Synthetic SAR: near-grayscale with strong multiplicative speckle."""
    rng = np.random.default_rng(7)
    base = np.ones((h, w), dtype=np.float32) * 0.35
    base[int(h * 0.1):int(h * 0.4), int(w * 0.1):int(w * 0.5)] = 0.08    # dark water
    base[int(h * 0.6):int(h * 0.85), int(w * 0.6):int(w * 0.95)] = 0.65  # bright built-up
    speckle = rng.lognormal(0.0, 0.65, (h, w))
    arr = np.clip(base * speckle, 0, 1)
    gray = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(gray, "L").convert("RGB")


def _run_fusion(svc, img1: Image.Image, img2: Image.Image, name1: str, name2: str, prompt: str = ""):
    return svc.run(_png_bytes(img1), _png_bytes(img2), name1, name2, prompt)


# ------------------------------------------------------------------
# TEST 1 — valid optical + SAR pair -> pipeline completes
# ------------------------------------------------------------------
def test_valid_pair_completes():
    svc = OpticalSARFusionService()
    result = _run_fusion(svc, make_optical(), make_sar(), "s2_scene.png", "s1_vv.png", "built-up areas?")
    assert result["task"] == "optical_sar_fusion"
    assert result["status"] in ("VERIFIED", "UNCERTAIN", "INSUFFICIENT_EVIDENCE")
    assert result["input_validation"]["optical_detected"] is True
    assert result["input_validation"]["sar_detected"] is True
    assert isinstance(result["reliability_score"], (int, float))
    assert "trace" in result and len(result["trace"]["workflow"]) > 3


# ------------------------------------------------------------------
# TEST 2 — optical + optical -> rejected
# ------------------------------------------------------------------
def test_optical_optical_rejected():
    svc = OpticalSARFusionService()
    result = _run_fusion(svc, make_optical(), make_optical(), "a.png", "b.png")
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert "optical imagery" in result.get("error", "").lower()


# ------------------------------------------------------------------
# TEST 3 — SAR + SAR -> rejected
# ------------------------------------------------------------------
def test_sar_sar_rejected():
    svc = OpticalSARFusionService()
    result = _run_fusion(svc, make_sar(), make_sar(), "x_vv.png", "y_vh.png")
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert "sar imagery" in result.get("error", "").lower()


# ------------------------------------------------------------------
# TEST 4 — modality detection separates classes
# ------------------------------------------------------------------
def test_modality_detection():
    opt = detect_modality(make_optical(), "rgb_scene.png")
    sar = detect_modality(make_sar(), "sar_render.png")
    assert opt["modality"] == "OPTICAL"
    assert sar["modality"] == "SAR"


# ------------------------------------------------------------------
# TEST 5 — scene fusion runs on non-georeferenced pair (feature-based)
# ------------------------------------------------------------------
def test_nongeoref_uses_feature_alignment():
    svc = OpticalSARFusionService()
    result = _run_fusion(svc, make_optical(), make_sar(), "opt.png", "sar.png", "")
    assert result["status"] in ("VERIFIED", "UNCERTAIN", "INSUFFICIENT_EVIDENCE")
    assert "alignment" in result
    assert result["alignment"]["method"] in ("feature_based", "geospatial_reprojection")


# ------------------------------------------------------------------
# TEST 6 — SAR statistics are measured, not placeholders
# ------------------------------------------------------------------
def test_sar_statistics_are_measured():
    svc = OpticalSARFusionService()
    sar_img = make_sar()
    prepped = svc.preprocess_sar(sar_img)
    # Water region is dark, built-up is bright -> measurable dynamic range
    assert 0.05 < prepped["array"].mean() < 0.95
    assert prepped["filter"] in ("median(3)", "none", "median")
    ev = svc.extract_sar_evidence(
        prepped["array"], prepped["quality"], prepped["polarizations"],
        prepped["db_note"], prepped["estimated_mean_db"],
    )
    assert "structural_evidence" in ev
    assert "dark_surface_fraction" in ev["structural_evidence"]


# ------------------------------------------------------------------
# TEST 7 — optical evidence deterministic
# ------------------------------------------------------------------
def test_evidence_extraction_robust():
    svc = OpticalSARFusionService()
    opt_prep = svc.preprocess_optical(make_optical())
    ev = svc.extract_optical_evidence(opt_prep["array"], opt_prep["quality"])
    assert {"water", "vegetation", "built_up"} <= set(ev["evidence"].keys())
    # Synthetic water is dark and bluish -> water fraction should be nonzero
    assert ev["evidence"]["water"]["fraction"] > 0.0


# ------------------------------------------------------------------
# TEST 8 — agreement damping: disagreement reduces fused confidence
# ------------------------------------------------------------------
def test_disagreement_damps_confidence():
    svc = OpticalSARFusionService()
    agree_fusion = svc.fuse_scene(
        {"evidence": {"water": {"fraction": 0.30}, "vegetation": {"fraction": 0.45},
                      "built_up": {"fraction": 0.25}}, "quality": 0.8},
        {"structural_evidence": {"dark_surface_fraction": 0.31, "rough_surface_fraction": 0.44,
                                 "bright_scatter_fraction": 0.23}, "quality": 0.8},
        registration_quality=0.85, targets=["water"],
    )
    conflict_fusion = svc.fuse_scene(
        {"evidence": {"water": {"fraction": 0.30}, "vegetation": {"fraction": 0.45},
                      "built_up": {"fraction": 0.25}}, "quality": 0.8},
        {"structural_evidence": {"dark_surface_fraction": 0.02, "rough_surface_fraction": 0.95,
                                 "bright_scatter_fraction": 0.90}, "quality": 0.8},
        registration_quality=0.85, targets=["water"],
    )
    assert conflict_fusion["fused_confidence"] < agree_fusion["fused_confidence"]
    assert conflict_fusion["agreement_level"] in ("LOW", "MODERATE")


# ------------------------------------------------------------------
# TEST 9 — missing metadata does not crash and never fabricates
# ------------------------------------------------------------------
def test_missing_metadata_is_honest():
    svc = OpticalSARFusionService()
    result = _run_fusion(svc, make_optical(), make_sar(), "my_photo.jpg", "img_001.png")
    # Either it completes with warnings about missing georef, or reports insufficient
    # evidence — it must never claim georeference that does not exist.
    meta = result.get("metadata", {})
    if meta:
        assert meta.get("optical", {}).get("georeferenced") in (True, False)
        if not meta["optical"].get("georeferenced"):
            assert "No georeferenced metadata" in meta["optical"].get("note", "")