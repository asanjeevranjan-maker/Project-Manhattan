"""
Modality Detection Module (OPTICAL / SAR / UNKNOWN).

Deterministic, multi-signal classifier that decides whether an uploaded
remote-sensing image is OPTICAL (multispectral / RGB) or SAR (radar
backscatter). Signals used, in decreasing reliability:

1. Raster band descriptions (GeoTIFF: "VV", "VH", "HH", "HV")  -> SAR
2. Band count (1-2 bands leans SAR, >=3 leans OPTICAL)
3. Colour structure (true optical RGB has inter-channel variance;
   SAR renders are near-grayscale with correlated channels)
4. Amplitude distribution (SAR speckle -> strong right skew,
   mean/median ratio well above optical)
5. Filename hints (weakest signal, only breaks ties)

The classifier NEVER fabricates certainty: when signals conflict or are
absent it returns UNKNOWN with an explanation of what metadata is missing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("satquery.modality_detection")

# --------------------------------------------------------------
# Configurable thresholds
# --------------------------------------------------------------
COLOR_SATURATION_SAR_MAX = 0.02      # mean |R-G|+|G-B| (0..1) below this => grayscale-like
SKEW_RATIO_SAR_MIN = 1.18            # mean/median above this => speckle-like amplitude
BAND_COUNT_SAR_MAX = 2               # <=2 bands leans SAR
CONFIDENT_THRESHOLD = 0.39           # below this => UNKNOWN

SAR_FILENAME_RE = re.compile(r"(^|[^a-z])(vv|vh|hh|hv|sar|s1[ab]?(_|-)?|sentinel-?1|sigma0|gamma0)([^a-z]|$)", re.IGNORECASE)
OPTICAL_FILENAME_RE = re.compile(r"(^|[^a-z])(s2|sentinel-?2|l8|l9|landsat|rgb|optical|true.?color|multispectral)([^a-z]|$)", re.IGNORECASE)
SAR_BAND_NAMES = {"vv", "vh", "hh", "hv", "vv_db", "vh_db", "sigma0", "gamma0", "beta0"}


def _to_numpy(image: Image.Image) -> np.ndarray:
    """Converts any PIL image to a float32 HxWxC array in [0, 1]."""
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _channel_signals(arr: np.ndarray) -> Dict[str, float]:
    """Colour structure + amplitude statistics of an RGB-rendered image."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = arr.mean(axis=2)
    median = float(np.median(gray))
    mean = float(gray.mean())
    return {
        "colorfulness": round(float(np.mean(np.abs(r - g) + np.abs(g - b))), 5),
        "channel_diff": round(float((np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b))) / 3.0), 5),
        "skew_ratio": round((mean / median) if median > 1e-6 else 1.0, 4),
        "dark_fraction": round(float(np.mean(gray < 0.02)), 4),
        "mean": round(mean, 4),
        "std": round(float(gray.std()), 4),
    }


def detect_modality(
    image: Optional[Image.Image],
    filename: str = "",
    raster_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classifies an image as OPTICAL, SAR or UNKNOWN.

    Returns:
        {
          "modality": "OPTICAL" | "SAR" | "UNKNOWN",
          "confidence": float,          # 0..1 signal score, NOT a detection confidence
          "signals": {...},             # every measured signal, for transparency
          "reason": str,                # human-readable justification / missing info
        }
    """
    signals: Dict[str, Any] = {}
    score = 0.0  # >0 leans SAR, <0 leans OPTICAL
    reasons: list = []

    # ---- Signal 1: raster band descriptions (strongest) ------------------
    band_descriptions = []
    if raster_meta:
        band_descriptions = [str(b).strip().lower() for b in (raster_meta.get("band_descriptions") or []) if b]
    signals["band_descriptions"] = band_descriptions
    if band_descriptions:
        if any(b in SAR_BAND_NAMES for b in band_descriptions):
            score += 0.60
            reasons.append(f"SAR polarization band descriptions found: {band_descriptions}")
        else:
            score -= 0.30
            reasons.append("GeoTIFF band descriptions look non-SAR (spectral bands)")

    # ---- Signal 2: band count --------------------------------------------
    # Only meaningful for REAL georeferenced rasters. Plain PNG/JPEG renders
    # always count 3 (or 1) RGB bands regardless of modality and GDAL exposes
    # DPI as "resolution" — trusting those would misclassify grayscale SAR
    # renders as optical.
    band_count = None
    has_geo = bool(raster_meta and raster_meta.get("crs"))
    if has_geo and raster_meta and raster_meta.get("band_count"):
        band_count = int(raster_meta["band_count"])
    signals["band_count"] = band_count
    if band_count is not None:
        if band_count <= BAND_COUNT_SAR_MAX:
            score += 0.35
            reasons.append(f"Only {band_count} raster band(s) — consistent with single/dual-pol SAR")
        else:
            score -= 0.35
            reasons.append(f"{band_count} spectral bands — consistent with optical imagery")
    elif raster_meta and not has_geo:
        signals["band_count"] = "n/a (plain image render)"

    # ---- Signal 3: colour structure ---------------------------------------
    if image is not None:
        try:
            arr = _to_numpy(image)
            cs = _channel_signals(arr)
            signals["colour"] = {k: cs[k] for k in ("colorfulness", "channel_diff", "skew_ratio", "dark_fraction")}

            if cs["channel_diff"] <= COLOR_SATURATION_SAR_MAX:
                score += 0.40
                reasons.append(f"Near-grayscale channels (diff {cs['channel_diff']:.4f}) — SAR-like")
            elif cs["channel_diff"] >= 0.06:
                score -= 0.45
                reasons.append(f"Strong colour separation (diff {cs['channel_diff']:.4f}) — optical-like")
            else:
                reasons.append(f"Weak colour signal (diff {cs['channel_diff']:.4f}) — inconclusive")

            if cs["skew_ratio"] >= SKEW_RATIO_SAR_MIN and cs["channel_diff"] <= 0.05:
                score += 0.15
                reasons.append(f"Amplitude skew (mean/median {cs['skew_ratio']:.2f}) — speckle-like")
            elif cs["skew_ratio"] < 1.10 and cs["channel_diff"] > 0.05:
                score -= 0.10
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[MODALITY] Pixel signal extraction failed: {e}")
            reasons.append("Pixel statistics unavailable")
    else:
        reasons.append("No decodable image pixels available")

    # ---- Signal 4: filename hint (weakest) ---------------------------------
    fname = (filename or "").lower()
    signals["filename"] = filename
    if fname:
        if SAR_FILENAME_RE.search(fname):
            score += 0.10
            reasons.append("Filename suggests SAR product")
        if OPTICAL_FILENAME_RE.search(fname):
            score -= 0.10
            reasons.append("Filename suggests optical product")

    # ---- Decision ----------------------------------------------------------
    abs_score = abs(score)
    if abs_score < CONFIDENT_THRESHOLD:
        modality = "UNKNOWN"
        confidence = round(abs_score, 3)
        reason = (
            "Modality could not be reliably determined: signals were inconclusive or conflicting. "
            "Missing helpful metadata: raster band descriptions, band count, CRS/georeferencing, "
            "or a product filename (e.g. Sentinel-1 VV/VH vs Sentinel-2 L1C)."
        )
    elif score > 0:
        modality = "SAR"
        confidence = round(min(0.99, 0.5 + abs_score / 2.0), 3)
        reason = "; ".join(reasons)
    else:
        modality = "OPTICAL"
        confidence = round(min(0.99, 0.5 + abs_score / 2.0), 3)
        reason = "; ".join(reasons)

    result = {"modality": modality, "confidence": confidence, "signals": signals, "reason": reason}
    logger.info(f"[MODALITY] {modality} (score={score:.2f}, conf={confidence}) :: {reason[:160]}")
    return result
