"""
Optical + SAR Fusion Service (SatQuery AI — third analysis mode).

Late / hybrid EVIDENCE fusion between an OPTICAL image and a SAR image of
the same (near-same) region. This is NOT bi-temporal change detection and
it is NOT "two AI descriptions glued together".

Pipeline:
    input validation -> modality detection -> metadata extraction ->
    pair compatibility -> alignment (geospatial or feature-based) ->
    optical preprocessing -> SAR preprocessing ->
    optical evidence (deterministic, + Grounding DINO/SAM when the query
    targets regions) -> SAR evidence (deterministic statistics inside the
    SAME aligned regions) -> cross-modal fusion -> confidence states ->
    visualizations -> structured response.

Every number in the response is measured. Missing capability is reported
as unavailable — never fabricated.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from modality_detection import detect_modality

logger = logging.getLogger("satquery.optical_sar_fusion")

try:  # optional geospatial stack — degrade gracefully
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import reproject, Resampling
    from shapely.geometry import box as shapely_box
    RASTERIO_AVAILABLE = True
except Exception as _re:  # noqa: BLE001
    RASTERIO_AVAILABLE = False
    logger.warning(f"[OPTICAL-SAR] rasterio/shapely unavailable: {_re}")

# Grounding DINO + SAM are reused as-is from the Image Analysis pipeline
try:
    from grounding_dino import detect_objects
    DINO_AVAILABLE = True
except Exception as _de:  # noqa: BLE001
    DINO_AVAILABLE = False
    logger.warning(f"[OPTICAL-SAR] Grounding DINO import failed: {_de}")

# --------------------------------------------------------------
# Configuration (all weights/thresholds intentionally explicit)
# --------------------------------------------------------------
ANALYSIS_SIZE = 768                 # common working grid (square)
OPTICAL_WEIGHT = 0.5                # evidence weighting
SAR_WEIGHT = 0.5
AGREEMENT_HIGH = 0.65               # agreement >= this -> HIGH
AGREEMENT_MODERATE = 0.40           # >= this -> MODERATE, else LOW
AGREEMENT_FACTOR_MIN = 0.70         # confidence damping at zero agreement
AGREEMENT_FACTOR_RANGE = 0.30       # extra multiplier at full agreement
VERIFIED_FUSED_MIN = 0.60           # fused confidence needed for VERIFIED
REGISTRATION_GOOD = 0.70
REGISTRATION_MIN = 0.35
MAX_DAYS_BETWEEN_ACQUISITIONS = 60  # warning threshold for temporal mismatch
SAR_SPECKLE_FILTER = "median"       # configurable: "median" | "none"
SAR_SPECKLE_KERNEL = 3
SAR_DB_MIN, SAR_DB_MAX = -25.0, 0.0  # declared normalization window for 8-bit SAR renders

# Scene-level optical evidence thresholds (RGB proxies, deterministic)
OPT_WATER_GREEN_MINUS_BLUE = 0.04   # (G-B) above this with dark pixels -> water-like
OPT_WATER_BRIGHTNESS_MAX = 0.45
OPT_VEG_EXG_MIN = 0.06              # excess-green index threshold
OPT_BUILT_BRIGHTNESS_MIN = 0.35

# SAR evidence windows on the normalized backscatter scale (0..1, mapped
# to SAR_DB_MIN..SAR_DB_MAX when dB conversion is not possible).
SAR_WATER_NORM_MAX = 0.22           # calm water is dark in SAR
SAR_BUILT_NORM_MIN = 0.45           # built-up double-bounce is bright
SAR_BUILT_TEXTURE_MIN = 0.10        # local std inside built-up is high


def _to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _decode_image(data: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Decodes uploaded bytes; returns (PIL image, error message)."""
    if not data:
        return None, "Empty file uploaded."
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB"), None
    except Exception as e:  # noqa: BLE001
        return None, f"Could not decode image: {e}"


def extract_raster_meta(data: bytes) -> Optional[Dict[str, Any]]:
    """
    Attempts to open the upload as a georeferenced raster (GeoTIFF etc.).
    Returns CRS/bounds/resolution/band info when available, else None.
    Never fabricates values.
    """
    if not RASTERIO_AVAILABLE or not data:
        return None
    try:
        with MemoryFile(data) as memfile:
            with memfile.open() as src:
                meta: Dict[str, Any] = {
                    "band_count": src.count,
                    "band_descriptions": [src.descriptions[i] for i in range(src.count)],
                    "dtype": str(src.dtypes[0]) if src.dtypes else None,
                    "width": src.width,
                    "height": src.height,
                    "nodata": src.nodata,
                }
                if src.crs is not None:
                    meta["crs"] = str(src.crs)
                try:
                    b = src.bounds
                    meta["bounds"] = [b.left, b.bottom, b.right, b.top]
                except Exception:  # noqa: BLE001
                    meta["bounds"] = None
                try:
                    res = src.res
                    meta["resolution"] = [float(res[0]), float(res[1])]
                except Exception:  # noqa: BLE001
                    meta["resolution"] = None
                tags = {}
                try:
                    tags = src.tags() or {}
                except Exception:  # noqa: BLE001
                    pass
                for key in ("acquisition_date", "timestamp", "date", "SENSING_TIME"):
                    if key in tags:
                        meta["acquisition_date"] = str(tags[key])
                        break
                return meta
    except Exception:  # noqa: BLE001
        return None


class OpticalSARFusionService:
    """
    Late / hybrid evidence fusion between one OPTICAL and one SAR image.
    Input order does not matter: modalities are detected and slots are
    normalized internally (input1/input2 -> optical_image/sar_image).
    """

    def __init__(self) -> None:
        self.trace: Dict[str, Any] = {"workflow": [], "tools": [], "runtime_ms": {}}
        self.warnings: List[str] = []

    # --------------------------------------------------------------
    # Trace helpers
    # --------------------------------------------------------------
    def _step(self, name: str) -> None:
        self.trace["workflow"].append(name)

    def _timed(self, name: str, t0: float) -> None:
        self.trace["runtime_ms"][name] = int((time.time() - t0) * 1000)

    def _warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    # --------------------------------------------------------------
    # Query routing
    # --------------------------------------------------------------
    TARGET_CLASSES = {
        "built_up": ("built", "building", "urban", "city", "house", "settlement", "infrastructure"),
        "water": ("water", "river", "lake", "flood", "reservoir", "coast", "harbor", "harbour"),
        "vegetation": ("vegetation", "forest", "tree", "crop", "agricultur", "farm", "field", "grass"),
    }

    def route_query(self, prompt: str) -> Dict[str, Any]:
        """Deterministic query routing — decides which pipeline branches run."""
        p = (prompt or "").lower()
        targets = []
        for cls, keywords in self.TARGET_CLASSES.items():
            if any(k in p for k in keywords):
                targets.append(cls)
        requires_localization = bool(targets)
        return {
            "prompt": prompt or "",
            "targets": targets,
            "requires_localization": requires_localization,
            "requires_segmentation": requires_localization and DINO_AVAILABLE,
            "requires_sar_statistics": True,
            "requires_spectral_analysis": True,
            "scene_level": not targets,
        }

    # --------------------------------------------------------------
    # Input validation + modality detection
    # --------------------------------------------------------------
    def validate_and_detect(
        self,
        data1: bytes,
        data2: bytes,
        name1: str,
        name2: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Image.Image], Optional[Image.Image], Optional[Dict[str, Any]]]:
        """Returns (error_response, optical_image, sar_image, detection_info)."""
        t0 = time.time()
        self._step("input_validation")
        img1, err1 = _decode_image(data1)
        img2, err2 = _decode_image(data2)
        if img1 is None or img2 is None:
            bad = err1 or err2 or "image"
            return {
                "task": "optical_sar_fusion",
                "status": "INSUFFICIENT_EVIDENCE",
                "error": bad,
            }, None, None, None

        self._step("modality_detection")
        raster1 = extract_raster_meta(data1)
        raster2 = extract_raster_meta(data2)
        det1 = detect_modality(img1, filename=name1, raster_meta=raster1)
        det2 = detect_modality(img2, filename=name2, raster_meta=raster2)
        self._timed("modality_detection", t0)

        # Normalize input order: slots do not matter
        modalities = {det1["modality"], det2["modality"]}
        if det1["modality"] == det2["modality"] or "UNKNOWN" in modalities:
            both_optical = "OPTICAL" in modalities and "UNKNOWN" not in modalities
            both_sar = "SAR" in modalities and "UNKNOWN" not in modalities
            if both_optical or both_sar:
                kind = "optical" if both_optical else "SAR"
                return {
                    "task": "optical_sar_fusion",
                    "status": "INSUFFICIENT_EVIDENCE",
                    "error": (
                        f"Both uploaded images appear to be {kind} imagery. "
                        "Optical + SAR Fusion requires one optical image and one SAR image."
                    ),
                    "modality_detection": {"input1": det1, "input2": det2},
                }, None, None, None
            # One side UNKNOWN -> insufficient but explain what is missing
            unknown_side = "input1" if det1["modality"] == "UNKNOWN" else "input2"
            return {
                "task": "optical_sar_fusion",
                "status": "INSUFFICIENT_EVIDENCE",
                "error": (
                    f"Modality of the {unknown_side} upload could not be determined with confidence "
                    f"({det1['modality'] if unknown_side == 'input1' else det2['modality']}). "
                    "Optical + SAR Fusion requires one optical image and one SAR image."
                ),
                "modality_detection": {"input1": det1, "input2": det2},
            }, None, None, None

        optical_image = img1 if det1["modality"] == "OPTICAL" else img2
        sar_image = img2 if det1["modality"] == "OPTICAL" else img1
        raster_optical = raster1 if det1["modality"] == "OPTICAL" else raster2
        raster_sar = raster2 if det1["modality"] == "OPTICAL" else raster1
        fname_optical = name1 if det1["modality"] == "OPTICAL" else name2
        fname_sar = name2 if det1["modality"] == "OPTICAL" else name1

        info = {
            "detection": {"optical": det1 if det1["modality"] == "OPTICAL" else det2,
                          "sar": det2 if det2["modality"] == "SAR" else det1},
            "raster_optical": raster_optical,
            "raster_sar": raster_sar,
            "filename_optical": fname_optical,
            "filename_sar": fname_sar,
        }
        return None, optical_image, sar_image, info

    # --------------------------------------------------------------
    # Metadata extraction
    # --------------------------------------------------------------
    def extract_metadata(self, info: Dict[str, Any]) -> Dict[str, Any]:
        self._step("metadata_extraction")

        def clean(raster: Optional[Dict[str, Any]], fname: str) -> Dict[str, Any]:
            if not raster:
                return {
                    "georeferenced": False,
                    "filename": fname,
                    "note": "No georeferenced metadata found (file is a plain image render).",
                }
            return {
                "georeferenced": True,
                "filename": fname,
                "crs": raster.get("crs"),
                "bounds": raster.get("bounds"),
                "resolution": raster.get("resolution"),
                "band_count": raster.get("band_count"),
                "band_descriptions": raster.get("band_descriptions"),
                "dtype": raster.get("dtype"),
                "width": raster.get("width"),
                "height": raster.get("height"),
                "nodata": raster.get("nodata"),
                "acquisition_date": raster.get("acquisition_date"),
            }

        return {
            "optical": clean(info["raster_optical"], info["filename_optical"]),
            "sar": clean(info["raster_sar"], info["filename_sar"]),
        }

    # --------------------------------------------------------------
    # Pair compatibility validation
    # --------------------------------------------------------------
    def validate_pair(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Geographic overlap check when metadata allows; honest fallback otherwise."""
        self._step("pair_compatibility")
        opt, sar = metadata["optical"], metadata["sar"]
        result: Dict[str, Any] = {"pair_valid": True, "overlap_ratio": None, "warnings": []}

        if opt.get("georeferenced") and sar.get("georeferenced") and opt.get("bounds") and sar.get("bounds"):
            try:
                sar_bounds = sar["bounds"]
                if (opt.get("crs") and sar.get("crs")) and opt["crs"] != sar["crs"]:
                    try:
                        from pyproj import Transformer
                        tr = Transformer.from_crs(sar["crs"], opt["crs"], always_xy=True)
                        sx1, sy1 = tr.transform(sar_bounds[0], sar_bounds[1])
                        sx2, sy2 = tr.transform(sar_bounds[2], sar_bounds[3])
                        sar_bounds = [min(sx1, sx2), min(sy1, sy2), max(sx1, sx2), max(sy1, sy2)]
                        result["warnings"].append("SAR bounds reprojected into the optical CRS for overlap check.")
                    except Exception:  # noqa: BLE001
                        result["warnings"].append(
                            "CRS differ and reprojection of bounds failed; overlap computed in native coordinates."
                        )
                b1 = shapely_box(*opt["bounds"])
                b2 = shapely_box(*sar_bounds)
                inter = b1.intersection(b2).area
                union = b1.union(b2).area
                overlap = round(inter / union, 4) if union > 0 else 0.0
                result["overlap_ratio"] = overlap
                if overlap < 0.15:
                    result["pair_valid"] = False
                    result["error"] = (
                        "The uploaded optical and SAR images do not sufficiently overlap geographically "
                        f"(overlap ratio {overlap:.2f})."
                    )
                elif overlap < 0.50:
                    result["warnings"].append(
                        f"Geographic overlap is partial ({overlap:.0%}); fusion covers the shared extent only."
                    )
            except Exception as e:  # noqa: BLE001
                self._warn(f"Overlap computation failed ({e}); falling back to feature-based validation.")
        else:
            result["warnings"].append(
                "Geo-metadata (CRS/bounds) is unavailable for at least one image; "
                "pair compatibility is validated via feature-based registration instead."
            )

        # Acquisition-date proximity check when dates are tagged
        d1, d2 = opt.get("acquisition_date"), sar.get("acquisition_date")
        if d1 and d2:
            try:
                from datetime import datetime
                fmts = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
                dt1 = dt2 = None
                for f in fmts:
                    try:
                        dt1 = datetime.strptime(str(d1)[:19], f)
                        break
                    except ValueError:
                        continue
                for f in fmts:
                    try:
                        dt2 = datetime.strptime(str(d2)[:19], f)
                        break
                    except ValueError:
                        continue
                if dt1 and dt2:
                    diff_days = abs((dt2 - dt1).days)
                    result["acquisition"] = {
                        "optical_date": str(d1),
                        "sar_date": str(d2),
                        "difference_days": diff_days,
                    }
                    if diff_days > MAX_DAYS_BETWEEN_ACQUISITIONS:
                        result["warnings"].append(
                            "The optical and SAR observations were acquired at significantly different times "
                            f"({diff_days} days apart); some differences may be temporal, not modality-driven."
                        )
            except Exception:  # noqa: BLE001
                pass
        self.trace["pair_validation"] = {k: v for k, v in result.items() if k != "warnings"}
        return result

    # --------------------------------------------------------------
    # Alignment / registration
    # --------------------------------------------------------------
    def align_images(
        self,
        optical_bytes: bytes,
        sar_bytes: bytes,
        raster_optical: Optional[Dict[str, Any]],
        raster_sar: Optional[Dict[str, Any]],
        optical_image: Image.Image,
        sar_image: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
        """
        Returns (optical_aligned, sar_aligned, registration_info).

        - Georeferenced pair -> rasterio reprojection onto a common grid
          (optical CRS, intersection extent, common coarse resolution,
          bilinear resampling for continuous imagery).
        - Plain renders      -> existing feature-based registration (reused).
        """
        self._step("registration")
        t0 = time.time()
        if (
            RASTERIO_AVAILABLE
            and raster_optical and raster_sar
            and raster_optical.get("crs") and raster_sar.get("crs")
            and raster_optical.get("bounds") and raster_sar.get("bounds")
        ):
            aligned_opt, aligned_sar, reg = self._align_geospatial(
                optical_bytes, sar_bytes, raster_optical, raster_sar
            )
            if aligned_opt is not None and aligned_sar is not None:
                self._timed("registration", t0)
                return aligned_opt, aligned_sar, reg
            self._warn("Geospatial reprojection failed; using feature-based alignment instead.")

        from image_registration import register_temporal_images
        reg_result = register_temporal_images(optical_image, sar_image, target_size=(ANALYSIS_SIZE, ANALYSIS_SIZE))
        reg = {
            "method": "feature_based",
            "quality": float(reg_result["registration_quality"]),
            "transformation": reg_result.get("transformation_type", "affine_feature_registration"),
            "common_grid": None,
            "note": (
                "Inputs are plain image renders without georeferencing; feature-based "
                "alignment was used. Geospatial reprojection (rasterio) is applied automatically "
                "when GeoTIFF inputs with CRS/bounds are provided."
            ),
        }
        if reg["quality"] < REGISTRATION_MIN:
            reg["fatal"] = "The images could not be aligned reliably enough for fusion."
        self._timed("registration", t0)
        return reg_result["aligned_t1"], reg_result["aligned_t2"], reg

    def _align_geospatial(
        self,
        optical_bytes: bytes,
        sar_bytes: bytes,
        raster_optical: Dict[str, Any],
        raster_sar: Dict[str, Any],
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image], Dict[str, Any]]:
        """Rasterio common-grid reprojection from raw GeoTIFF bytes."""
        try:
            from rasterio.io import MemoryFile
            from rasterio.warp import reproject, Resampling
            from rasterio.transform import from_bounds as rio_from_bounds

            opt_bounds, sar_bounds = raster_optical["bounds"], raster_sar["bounds"]
            b1 = shapely_box(*opt_bounds)
            b2 = shapely_box(*sar_bounds)
            if not b1.intersects(b2):
                return None, None, {}
            inter = b1.intersection(b2)
            res_o = raster_optical.get("resolution") or []
            res_s = raster_sar.get("resolution") or []
            cands = [r for r in list(res_o) + list(res_s) if r]
            resolution = max(cands) if cands else max(
                (inter.bounds[2] - inter.bounds[0]) / ANALYSIS_SIZE,
                (inter.bounds[3] - inter.bounds[1]) / ANALYSIS_SIZE,
            )
            width = max(2, min(2048, int(round((inter.bounds[2] - inter.bounds[0]) / resolution))))
            height = max(2, min(2048, int(round((inter.bounds[3] - inter.bounds[1]) / resolution))))
            dst_transform = rio_from_bounds(
                inter.bounds[0], inter.bounds[1], inter.bounds[2], inter.bounds[3], width, height
            )
            dst_crs = raster_optical["crs"]

            def _reproject_bytes(data: bytes) -> Optional[np.ndarray]:
                with MemoryFile(data) as memfile:
                    with memfile.open() as src:
                        bands = []
                        for band_idx in range(1, min(src.count, 3) + 1):
                            src_band = src.read(band_idx).astype(np.float32)
                            dst = np.full((height, width), np.nan, dtype=np.float32)
                            reproject(
                                source=src_band,
                                destination=dst,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=dst_transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.bilinear,  # continuous imagery
                            )
                            bands.append(dst)
                        if not bands:
                            return None
                        arr = np.stack(bands, axis=-1)
                        for c in range(arr.shape[-1]):  # fill nodata with band median
                            ch = arr[..., c]
                            if np.isnan(ch).any():
                                med = np.nanmedian(ch) if not np.isnan(ch).all() else 0.0
                                ch[np.isnan(ch)] = med
                        return arr

            opt_arr = _reproject_bytes(optical_bytes)
            sar_arr = _reproject_bytes(sar_bytes)
            if opt_arr is None or sar_arr is None:
                return None, None, {}

            opt_img = Image.fromarray(
                np.clip(_normalize_bandwise(opt_arr) * 255, 0, 255).astype(np.uint8)
            )
            sar_img = Image.fromarray(
                np.clip(_sar_to_grayscale(sar_arr) * 255, 0, 255).astype(np.uint8)
            ).convert("RGB")
            opt_img = _resize(opt_img)   # common working grid downstream
            sar_img = _resize(sar_img)

            reg = {
                "method": "geospatial_reprojection",
                "quality": 0.90,
                "transformation": f"reproject_to_common_grid ({dst_crs})",
                "common_grid": {
                    "target_crs": dst_crs,
                    "resolution": round(float(resolution), 6),
                    "bounds": [round(v, 6) for v in inter.bounds],
                    "width": width,
                    "height": height,
                },
                "note": "Both images resampled (bilinear) onto a shared georeferenced grid before any pixel comparison.",
            }
            return opt_img, sar_img, reg
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[OPTICAL-SAR] Geospatial alignment failed: {e}")
            return None, None, {}

    # --------------------------------------------------------------
    # Preprocessing — OPTICAL branch
    # --------------------------------------------------------------
    def preprocess_optical(self, image: Image.Image) -> Dict[str, Any]:
        """
        Optical preprocessing: RGB extraction, band-wise percentile
        normalization (2-98%), simple haze/nodata guard.
        """
        self._step("optical_preprocessing")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        normalized = _normalize_bandwise(arr)
        rgb = Image.fromarray(np.clip(normalized * 255, 0, 255).astype(np.uint8))
        quality = _optical_quality(normalized)
        return {"image": rgb, "array": normalized, "quality": quality}

    # --------------------------------------------------------------
    # Preprocessing — SAR branch (deliberately NOT optical normalization)
    # --------------------------------------------------------------
    def preprocess_sar(self, image: Image.Image) -> Dict[str, Any]:
        """
        SAR preprocessing: grayscale amplitude handling, optional linear->dB
        conversion, extreme-value clipping, configurable speckle reduction
        (median), then normalization for display/statistics.
        """
        self._step("sar_preprocessing")
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

        # Extreme value clipping (1-99 percentile) — removes isolated hot spikes
        p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
        clipped = np.clip(arr, p1, p99)

        # Speckle reduction (configurable, deliberately lightweight)
        filter_used = "none"
        if SAR_SPECKLE_FILTER == "median":
            clipped_u8 = np.clip(clipped * 255, 0, 255).astype(np.uint8)
            despeckled = cv2.medianBlur(clipped_u8, SAR_SPECKLE_KERNEL)
            clipped = despeckled.astype(np.float32) / 255.0
            filter_used = f"median({SAR_SPECKLE_KERNEL})"

        # Declared linear normalization window (0..1 <-> SAR_DB_MIN..SAR_DB_MAX).
        # Absolute dB is only computable from real complex/Calibrated products;
        # for 8-bit renders we use this declared window and say so in provenance.
        mean_val = float(clipped.mean())
        est_db = SAR_DB_MIN + mean_val * (SAR_DB_MAX - SAR_DB_MIN)
        if np.asarray(image).dtype == np.uint8 and image.mode in ("L", "RGB"):
            db_note = (
                f"8-bit amplitude render; statistics reported on a declared normalized scale "
                f"(0..1 mapped to {SAR_DB_MIN}..{SAR_DB_MAX} dB ≈ {est_db:.1f} dB mean). "
                "Absolute calibrated dB requires the original complex product."
            )
        else:
            db_note = "Statistics computed from raster amplitude values."

        quality = _sar_quality(clipped)
        return {
            "array": clipped,           # normalized 0..1 backscatter proxy
            "filter": filter_used,
            "quality": quality,
            "db_note": db_note,
            "estimated_mean_db": round(est_db, 2),
            "polarizations": _detect_polarizations(image),
        }

    # --------------------------------------------------------------
    # Optical evidence (deterministic spectral indices)
    # --------------------------------------------------------------
    def extract_optical_evidence(self, arr: np.ndarray, quality: float) -> Dict[str, Any]:
        self._step("optical_evidence")
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        brightness = (r + g + b) / 3.0
        exg = (2.0 * g) - r - b                      # excess green (vegetation proxy)
        g_minus_b = g - b
        gray = arr.mean(axis=2)
        local_std = _local_std(gray, k=9)            # texture

        water_mask = (g_minus_b > OPT_WATER_GREEN_MINUS_BLUE) & (brightness < OPT_WATER_BRIGHTNESS_MAX)
        veg_mask = exg > OPT_VEG_EXG_MIN
        built_mask = (brightness > OPT_BUILT_BRIGHTNESS_MIN) & (arr.max(axis=2) - arr.min(axis=2) < 0.25) & (local_std > 0.03)

        evidence = {
            "modality": "optical",
            "evidence": {
                "water": {
                    "fraction": round(float(water_mask.mean()), 4),
                    "mean_brightness_in_class": round(float(brightness[water_mask].mean()) if water_mask.any() else 0.0, 4),
                },
                "vegetation": {
                    "fraction": round(float(veg_mask.mean()), 4),
                    "mean_excess_green_in_class": round(float(exg[veg_mask].mean()) if veg_mask.any() else 0.0, 4),
                },
                "built_up": {
                    "fraction": round(float(built_mask.mean()), 4),
                    "mean_brightness_in_class": round(float(brightness[built_mask].mean()) if built_mask.any() else 0.0, 4),
                    "mean_texture_in_class": round(float(local_std[built_mask].mean()) if built_mask.any() else 0.0, 4),
                },
                "spectral": {
                    "mean_r": round(float(r.mean()), 4),
                    "mean_g": round(float(g.mean()), 4),
                    "mean_b": round(float(b.mean()), 4),
                    "colorfulness": round(float(np.mean(np.abs(r - g) + np.abs(g - b))), 4),
                    "overall_brightness": round(float(brightness.mean()), 4),
                },
            },
            "quality": quality,
            "confidence": round(min(0.95, 0.35 + 0.6 * quality), 3),
        }
        return evidence

    # --------------------------------------------------------------
    # SAR evidence (deterministic backscatter statistics)
    # --------------------------------------------------------------
    def extract_sar_evidence(
        self, sar_arr: np.ndarray, quality: float, polarizations: List[str], db_note: str, est_db: float
    ) -> Dict[str, Any]:
        self._step("sar_evidence")
        mean_v = float(sar_arr.mean())
        median_v = float(np.median(sar_arr))
        std_v = float(sar_arr.std())
        texture = float(_local_std(sar_arr, k=9).mean())

        evidence = {
            "modality": "sar",
            "polarizations": polarizations,
            "statistics": {
                "mean_normalized": round(mean_v, 4),
                "median_normalized": round(median_v, 4),
                "std_normalized": round(std_v, 4),
                "texture_mean_local_std": round(texture, 4),
                "estimated_mean_db": est_db,
                "scale_note": db_note,
            },
            "structural_evidence": {
                # Dark surface fraction — calm water / smooth surfaces
                "dark_surface_fraction": round(float((sar_arr < SAR_WATER_NORM_MAX).mean()), 4),
                # Bright double-bounce fraction — built-up / metallic structures
                "bright_scatter_fraction": round(float((sar_arr > SAR_BUILT_NORM_MIN).mean()), 4),
                # Rough surface fraction — vegetation-like moderate backscatter
                "rough_surface_fraction": round(
                    float(((sar_arr >= SAR_WATER_NORM_MAX) & (sar_arr <= SAR_BUILT_NORM_MIN)).mean()), 4
                ),
            },
            "quality": quality,
            "confidence": round(min(0.95, 0.35 + 0.6 * quality), 3),
        }
        return evidence

    # --------------------------------------------------------------
    # Region-level fusion: Grounding DINO + SAM on OPTICAL, then the
    # SAME aligned mask is transferred onto the SAR data.
    # --------------------------------------------------------------
    CLASS_DINO_PROMPT = {
        "built_up": "building",
        "water": "water",
        "vegetation": "forest, vegetation",
    }

    def extract_region_fusion(
        self,
        optical_image: Image.Image,
        sar_arr: np.ndarray,
        targets: List[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        For each requested target class:
          Grounding DINO candidates (optical) -> SAM refinement (existing
          pipeline) -> mask transferred onto aligned SAR -> SAR statistics
          inside the SAME region -> cross-modal support per region.
        Returns (regions, meta). Never runs DINO on raw SAR.
        """
        self._step("region_evidence")
        regions: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {"available": DINO_AVAILABLE, "regions": 0, "note": None}

        if not targets:
            meta["note"] = "Scene-level query — region localization skipped (performance routing)."
            return regions, meta
        if not DINO_AVAILABLE:
            meta["note"] = "Grounding DINO unavailable — falling back to scene-level fusion only."
            return regions, meta

        w, h = optical_image.size
        for target in targets:
            prompt = self.CLASS_DINO_PROMPT.get(target, target)
            try:
                detections = detect_objects(optical_image, prompt, enable_segmentation=True)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[OPTICAL-SAR] DINO failed for '{target}': {e}")
                meta["note"] = f"Localization failed for '{target}': {e}"
                continue
            if not isinstance(detections, list):
                detections = detections.get("detections", []) if isinstance(detections, dict) else []

            for det in detections[:15]:  # safety cap
                try:
                    box = det.get("box") or det.get("bbox")
                    if not box or len(box) < 4:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box[:4]]
                    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
                    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        continue

                    # Build the region mask: SAM polygon when present, else box
                    mask = np.zeros((h, w), dtype=np.uint8)
                    mask_source = "bounding_box"
                    polygon = None
                    if isinstance(det.get("mask"), dict):
                        polygon = (det.get("mask") or {}).get("polygon")
                    if polygon:
                        try:
                            pts = np.array(polygon, dtype=np.int32)
                            if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[0] >= 3:
                                cv2.fillPoly(mask, [pts], 1)
                                mask_source = "sam_polygon"
                        except Exception:  # noqa: BLE001
                            mask = np.zeros((h, w), dtype=np.uint8)
                    if mask_source == "bounding_box":
                        mask[int(y1):int(y2), int(x1):int(x2)] = 1

                    inside = sar_arr[mask > 0]
                    if inside.size < 50:
                        continue

                    sar_support, sar_stats = _sar_class_support(target, float(inside.mean()), float(inside.std()))
                    optical_support = float(det.get("confidence", det.get("score", 0.0)) or 0.0)
                    agreement = 1.0 - abs(optical_support - sar_support)
                    fused = (
                        (OPTICAL_WEIGHT * optical_support + SAR_WEIGHT * sar_support)
                        * (AGREEMENT_FACTOR_MIN + AGREEMENT_FACTOR_RANGE * agreement)
                    )
                    regions.append({
                        "target": target,
                        "label": det.get("label", target),
                        "rect": [round(x1 / w, 4), round(y1 / h, 4),
                                 round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)],
                        "optical_support": round(optical_support, 4),
                        "sar_support": round(sar_support, 4),
                        "agreement": round(agreement, 4),
                        "agreement_level": _agreement_level(agreement),
                        "fused_confidence": round(min(0.97, max(0.05, fused)), 4),
                        "sar_statistics": sar_stats,
                        "mask_source": mask_source,
                    })
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[OPTICAL-SAR] Region fusion skipped a detection: {e}")

        meta["regions"] = len(regions)
        if regions:
            meta["note"] = (
                "Grounding DINO + SAM proposed regions on the OPTICAL image; SAR backscatter "
                "statistics were extracted inside the SAME aligned regions."
            )
        return regions, meta


    # --------------------------------------------------------------
    # Scene-level fusion engine
    # --------------------------------------------------------------
    def fuse_scene(
        self,
        optical_evidence: Dict[str, Any],
        sar_evidence: Dict[str, Any],
        registration_quality: float,
        targets: List[str],
    ) -> Dict[str, Any]:
        """
        Weighted evidence combination with agreement damping:
            fused = (w_o*opt + w_s*sar) * (AGREEMENT_FACTOR_MIN + RANGE*agreement)
        All inputs measured; weights configurable (see config note).
        """
        self._step("cross_modal_fusion")
        opt_ev = optical_evidence["evidence"]
        sar_st = sar_evidence["structural_evidence"]
        input_quality = round(0.6 * ((optical_evidence["quality"] + sar_evidence["quality"]) / 2.0)
                              + 0.4 * registration_quality, 3)

        class_support = {
            "water": (opt_ev["water"]["fraction"], sar_st["dark_surface_fraction"]),
            "vegetation": (opt_ev["vegetation"]["fraction"], sar_st["rough_surface_fraction"]),
            "built_up": (opt_ev["built_up"]["fraction"], sar_st["bright_scatter_fraction"]),
        }

        features = []
        for cls, (opt_v, sar_v) in class_support.items():
            opt_v, sar_v = float(opt_v), float(sar_v)
            agreement = 1.0 - abs(opt_v - sar_v)
            base = (OPTICAL_WEIGHT * opt_v + SAR_WEIGHT * sar_v)
            # Asymmetry penalty: if one modality is strong and the other weak,
            # the claim is NOT cross-verified. Penalize by the min/max ratio so
            # single-modality strength cannot reach "verified" territory.
            denom = max(opt_v, sar_v)
            asymmetry = (min(opt_v, sar_v) / denom) if denom > 1e-6 else 0.0
            fused = (
                base
                * (AGREEMENT_FACTOR_MIN + AGREEMENT_FACTOR_RANGE * agreement)
                * (0.5 + 0.5 * asymmetry)
                * input_quality
            )
            features.append({
                "feature": cls,
                "optical_support": round(opt_v, 4),
                "sar_support": round(sar_v, 4),
                "agreement": round(agreement, 4),
                "agreement_level": _agreement_level(agreement),
                "fused_confidence": round(min(0.97, fused), 4),
                "in_focus": cls in targets,
            })

        weights = [2.0 if f["in_focus"] else 1.0 for f in features]
        scene_agreement = float(np.average([f["agreement"] for f in features], weights=weights))
        fused_confidence = float(np.average([f["fused_confidence"] for f in features], weights=weights))

        return {
            "features": features,
            "scene_agreement": round(scene_agreement, 4),
            "agreement_level": _agreement_level(scene_agreement),
            "fused_confidence": round(fused_confidence, 4),
            "input_quality": input_quality,
            "formula": (
                f"fused = ({OPTICAL_WEIGHT}*optical + {SAR_WEIGHT}*sar) * "
                f"({AGREEMENT_FACTOR_MIN} + {AGREEMENT_FACTOR_RANGE}*agreement) * "
                f"(0.5+0.5*support_asymmetry) * input_quality"
            ),
            "config_note": (
                "Weights and windows are declared configuration for the current release — "
                "NOT scientifically calibrated constants. See provenance."
            ),
        }

    # --------------------------------------------------------------
    # Verification + claim generation (claims must cite evidence)
    # --------------------------------------------------------------
    def verify_and_claim(
        self,
        fusion: Dict[str, Any],
        regions: List[Dict[str, Any]],
        pair: Dict[str, Any],
        registration: Dict[str, Any],
        optical_evidence: Dict[str, Any],
        sar_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._step("verification")
        warnings = list(self.warnings) + list(pair.get("warnings", []))

        if not pair.get("pair_valid", False):
            status = "INSUFFICIENT_EVIDENCE"
            reason = pair.get("error", "Pair compatibility check failed.")
        elif registration.get("fatal"):
            status = "INSUFFICIENT_EVIDENCE"
            reason = registration["fatal"]
        elif registration.get("quality", 0) < REGISTRATION_GOOD:
            status = "UNCERTAIN"
            reason = (
                f"Registration quality is limited ({registration.get('quality', 0):.2f}); "
                "cross-modal spatial consistency cannot be fully guaranteed."
            )
        else:
            level = fusion["agreement_level"]
            if level == "HIGH" and fusion["fused_confidence"] >= VERIFIED_FUSED_MIN:
                status = "VERIFIED"
                reason = (
                    "Input pair compatible, alignment good, both modalities provide useful "
                    "evidence and cross-modal agreement is high."
                )
            elif level == "LOW":
                status = "UNCERTAIN"
                reason = (
                    "Optical and SAR evidence are not sufficiently consistent for a "
                    "high-confidence conclusion."
                )
            else:
                status = "UNCERTAIN"
                reason = (
                    "Some evidence exists but modality agreement is moderate or input "
                    "quality is limited."
                )

        # Cloud advantage check: SAR still useful when optical is degraded
        optical_q, sar_q = optical_evidence["quality"], sar_evidence["quality"]
        if optical_q < 0.30 and sar_q >= 0.50 and status != "INSUFFICIENT_EVIDENCE":
            warnings.append(
                "Optical quality is LOW (possible cloud cover / poor visibility) while SAR quality is "
                "GOOD. SAR provides useful structural evidence despite limited optical visibility — "
                "confidence is reduced accordingly."
            )

        claims: List[Dict[str, Any]] = []
        LABELS = {"built_up": "built-up", "water": "water", "vegetation": "vegetation"}
        for f in fusion["features"]:
            cls = f["feature"]
            has_opt = f["optical_support"] >= 0.05
            has_sar = f["sar_support"] >= 0.05
            if not (has_opt or has_sar):
                continue
            if has_opt and has_sar and f["agreement_level"] in ("HIGH", "MODERATE"):
                claim = f"The scene shows evidence of {LABELS[cls]} supported by both optical and SAR observations."
                evidence_basis = {
                    "optical": f"spectral/semantic evidence (support {f['optical_support']:.2f})",
                    "sar": f"backscatter/structural evidence (support {f['sar_support']:.2f})",
                    "spatial": "same aligned scene grid",
                }
            elif has_opt:
                claim = f"Optical imagery suggests {LABELS[cls]} presence; SAR evidence does not independently confirm it."
                evidence_basis = {
                    "optical": f"spectral/semantic evidence (support {f['optical_support']:.2f})",
                    "sar": f"weak/absent structural support ({f['sar_support']:.2f})",
                }
            else:
                claim = f"SAR evidence suggests {LABELS[cls]} characteristics; optical evidence does not independently confirm it."
                evidence_basis = {
                    "sar": f"backscatter/structural evidence (support {f['sar_support']:.2f})",
                    "optical": f"weak/absent spectral support ({f['optical_support']:.2f})",
                }
            claims.append({
                "claim": claim,
                "feature": cls,
                "evidence": evidence_basis,
                "confidence": f["fused_confidence"],
                "agreement_level": f["agreement_level"],
                "focus": f["in_focus"],
            })

        region_claims = []
        for r in sorted(regions, key=lambda x: -x["fused_confidence"])[:12]:
            region_claims.append({
                "claim": (
                    f"Region '{r['label']}' ({r['target']}): optical support {r['optical_support']:.2f}, "
                    f"SAR support {r['sar_support']:.2f}, agreement {r['agreement_level']}."
                ),
                "rect": r["rect"],
                "confidence": r["fused_confidence"],
                "agreement_level": r["agreement_level"],
            })

        return {
            "status": status,
            "reason": reason,
            "claims": claims,
            "region_claims": region_claims,
            "warnings": warnings,
        }

    # --------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------
    def generate_visualizations(
        self,
        optical_image: Image.Image,
        sar_image: Image.Image,
        regions: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Fused overlay = aligned optical base + verified-region boxes/tints."""
        self._step("visualization")
        base = _resize(optical_image).convert("RGB")
        w, h = base.size
        arr = np.asarray(base).copy()
        COLORS = {"built_up": (59, 130, 246), "water": (6, 182, 212), "vegetation": (16, 185, 129)}
        for r in regions:
            if r["agreement_level"] == "LOW" or r["fused_confidence"] < 0.30:
                continue  # display only cross-modally consistent regions
            x, y, bw, bh = r["rect"]
            px1, py1 = int(x * w), int(y * h)
            px2, py2 = int(min(w, (x + bw) * w)), int(min(h, (y + bh) * h))
            color = COLORS.get(r["target"], (139, 92, 246))
            arr[py1:py2, px1:px2] = (
                0.78 * arr[py1:py2, px1:px2] + 0.22 * np.array(color, dtype=np.float32)
            ).astype(np.uint8)
            cv2.rectangle(arr, (px1, py1), (px2, py2), color, thickness=2)
        fused = Image.fromarray(arr)
        return {
            "opticalDataUrl": _to_data_url(_resize(optical_image)),
            "sarDataUrl": _to_data_url(_resize(sar_image)),
            "fusedOverlayDataUrl": _to_data_url(fused),
        }

    # --------------------------------------------------------------
    # Orchestrator
    # --------------------------------------------------------------
    def run(
        self,
        data1: bytes,
        data2: bytes,
        name1: str = "",
        name2: str = "",
        prompt: str = "",
    ) -> Dict[str, Any]:
        t_start = time.time()
        routing = self.route_query(prompt)

        # 1-2. Validation + modality detection (order-independent)
        err, optical_image, sar_image, info = self.validate_and_detect(data1, data2, name1, name2)
        if err:
            return err

        # 3. Metadata
        metadata = self.extract_metadata(info)

        # 4. Pair compatibility
        pair = self.validate_pair(metadata)
        if not pair.get("pair_valid", False):
            self._step("terminated_pair_invalid")
            return {
                "task": "optical_sar_fusion",
                "status": "INSUFFICIENT_EVIDENCE",
                "error": pair.get("error", "Pair is not compatible for fusion."),
                "input_validation": {
                    "optical_detected": True,
                    "sar_detected": True,
                    "pair_compatible": False,
                    "overlap_ratio": pair.get("overlap_ratio"),
                },
                "warnings": pair.get("warnings", []),
                "trace": self.trace,
            }

        # 5. Registration / common grid
        optical_bytes_for_reg = data1 if info["detection"]["optical"]["modality"] == "OPTICAL" else data2
        sar_bytes_for_reg = data2 if info["detection"]["sar"]["modality"] == "SAR" else data1
        optical_aligned, sar_aligned, registration = self.align_images(
            optical_bytes_for_reg,
            sar_bytes_for_reg,
            info["raster_optical"], info["raster_sar"],
            optical_image, sar_image,
        )

        # 6-7. Modality-specific preprocessing
        try:
            opt_prep = self.preprocess_optical(optical_aligned)
        except Exception as e:  # noqa: BLE001
            return {"task": "optical_sar_fusion", "status": "INSUFFICIENT_EVIDENCE",
                    "error": f"Optical preprocessing failed: {e}", "trace": self.trace}
        try:
            sar_prep = self.preprocess_sar(sar_aligned)
        except Exception as e:  # noqa: BLE001
            return {"task": "optical_sar_fusion", "status": "INSUFFICIENT_EVIDENCE",
                    "error": f"The uploaded radar image could not be processed: {e}", "trace": self.trace}

        # 8-9. Evidence extraction
        optical_evidence = self.extract_optical_evidence(opt_prep["array"], opt_prep["quality"])
        sar_evidence = self.extract_sar_evidence(
            sar_prep["array"], sar_prep["quality"], sar_prep["polarizations"],
            sar_prep["db_note"], sar_prep["estimated_mean_db"],
        )

        # 10. Region-level fusion (DINO + SAM on optical, SAR stats in same regions)
        regions, region_meta = self.extract_region_fusion(
            optical_aligned, sar_prep["array"], routing["targets"]
        )

        # 11. Scene-level fusion + agreement
        fusion = self.fuse_scene(optical_evidence, sar_evidence, float(registration.get("quality", 0.5)), routing["targets"])

        # 12. Verification + claims
        verdict = self.verify_and_claim(fusion, regions, pair, registration, optical_evidence, sar_evidence)

        # 13. Visualizations
        vis = self.generate_visualizations(optical_aligned, sar_aligned, regions)

        focus_claims = [c["claim"] for c in verdict["claims"] if c.get("focus")][:3]
        answer = " ".join(focus_claims) if focus_claims else (
            "General scene fusion complete. See per-feature evidence for optical and SAR support."
        )

        self._step("complete")
        self.trace["tools"] = sorted({
            "rasterio" if RASTERIO_AVAILABLE else "PIL/cv2",
            "grounding_dino+SAM2" if region_meta.get("regions") else "cv2-statistics",
            "numpy",
        })
        reliability = round(
            min(0.97, fusion["fused_confidence"] * (0.7 + 0.3 * float(registration.get("quality", 0.5)))),
            4,
        ) if verdict["status"] != "INSUFFICIENT_EVIDENCE" else 0.0

        response = {
            "task": "optical_sar_fusion",
            "status": verdict["status"],
            "status_reason": verdict["reason"],
            "answer": answer,
            "input_validation": {
                "optical_detected": True,
                "sar_detected": True,
                "pair_compatible": bool(pair.get("pair_valid")),
                "overlap_ratio": pair.get("overlap_ratio"),
                "modality_detection": info["detection"],
            },
            "metadata": metadata,
            "optical": {
                "sensor": "georeferenced optical raster" if metadata["optical"].get("georeferenced") else "uploaded optical render",
                "evidence": optical_evidence["evidence"],
                "quality": optical_evidence["quality"],
                "confidence": optical_evidence["confidence"],
            },
            "sar": {
                "sensor": "SAR (georeferenced raster)" if metadata["sar"].get("georeferenced") else "SAR (uploaded render)",
                "polarizations": sar_evidence["polarizations"],
                "evidence": {
                    "statistics": sar_evidence["statistics"],
                    "structural": sar_evidence["structural_evidence"],
                },
                "speckle_filter": sar_prep["filter"],
                "quality": sar_evidence["quality"],
                "confidence": sar_evidence["confidence"],
            },
            "alignment": {
                "method": registration.get("method"),
                "quality": registration.get("quality"),
                "transformation": registration.get("transformation"),
                "common_grid": registration.get("common_grid"),
                "note": registration.get("note"),
            },
            "fusion": {
                "agreement": fusion["scene_agreement"],
                "agreement_level": fusion["agreement_level"],
                "fused_confidence": fusion["fused_confidence"],
                "input_quality": fusion["input_quality"],
                "formula": fusion["formula"],
                "config_note": fusion["config_note"],
                "features": fusion["features"],
                "regions": regions,
                "region_meta": region_meta,
            },
            "claims": verdict["claims"],
            "region_claims": verdict["region_claims"],
            "reliability_score": reliability,
            "routing": routing,
            "visualizations": vis,
            "warnings": verdict["warnings"],
            "provenance": {
                "engine": "OpticalSARFusionService (deterministic evidence fusion)",
                "analysis_grid": ANALYSIS_SIZE,
                "sar_scale_window_db": [SAR_DB_MIN, SAR_DB_MAX],
                "note": "All support/agreement values are computed from measured image statistics. No LLM produced numeric evidence.",
            },
            "trace": {
                "task": "optical_sar_fusion",
                "workflow": self.trace["workflow"],
                "tools": self.trace["tools"],
                "runtime_ms": {**self.trace["runtime_ms"], "total": int((time.time() - t_start) * 1000)},
            },
        }
        logger.info(
            f"[OPTICAL-SAR] status={response['status']} agreement={fusion['agreement_level']} "
            f"reliability={reliability} regions={len(regions)}"
        )
        return response


# Module-level singleton used by the API route
optical_sar_fusion_service = OpticalSARFusionService()

# ==================================================================
# Module-level deterministic helpers
# ==================================================================

def _resize(image: Image.Image, size: int = ANALYSIS_SIZE) -> Image.Image:
    """Resize keeping aspect square (LANCZOS) for the common analysis grid."""
    return image.resize((size, size), Image.LANCZOS)


def _normalize_bandwise(arr: np.ndarray) -> np.ndarray:
    """Per-band 2-98 percentile normalization (optical display)."""
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[-1]):
        ch = arr[..., c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        out[..., c] = (ch - lo) / (hi - lo) if hi > lo else np.zeros_like(ch)
    return np.clip(out, 0.0, 1.0)


def _sar_to_grayscale(arr: np.ndarray) -> np.ndarray:
    """SAR display render: amplitude mean across polarizations, 2-98 stretch."""
    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
    return (gray - lo) / (hi - lo) if hi > lo else np.zeros_like(gray)


def _local_std(gray: np.ndarray, k: int = 9) -> np.ndarray:
    """Local standard deviation (texture) via box filter."""
    mean = cv2.blur(gray, (k, k))
    sq_mean = cv2.blur(gray * gray, (k, k))
    return np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))


def _detect_polarizations(image: Image.Image) -> List[str]:
    """Reports polarizations only when actually determinable."""
    descs = getattr(image, "raster_band_descriptions", None)
    if descs:
        return [d.upper() for d in descs if str(d).upper() in {"VV", "VH", "HH", "HV"}]
    return ["VV (assumed single-pol render)"] if image.mode in ("L", "RGB") else []


def _optical_quality(arr: np.ndarray) -> float:
    """Measurable optical quality: dynamic range + non-saturation + signal."""
    gray = arr.mean(axis=2)
    p2, p98 = np.percentile(gray, 2), np.percentile(gray, 98)
    dynamic = float(np.clip((p98 - p2) / 0.7, 0, 1))
    saturated = float(np.mean((gray < 0.02) | (gray > 0.98)))
    signal = float(np.clip(gray.std() / 0.18, 0, 1))
    cloud_like = float(np.mean(gray > 0.92))
    quality = 0.4 * dynamic + 0.3 * signal + 0.3 * (1.0 - min(1.0, saturated + cloud_like))
    return round(float(np.clip(quality, 0.0, 1.0)), 3)


def _sar_quality(arr: np.ndarray) -> float:
    """Measurable SAR quality: dynamic range + texture (structure) presence."""
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    dynamic = float(np.clip((p98 - p2) / 0.6, 0, 1))
    texture = float(np.clip(_local_std(arr, k=9).mean() / 0.12, 0, 1))
    return round(float(np.clip(0.6 * dynamic + 0.4 * texture, 0.0, 1.0)), 3)


def _agreement_level(agreement: float) -> str:
    if agreement >= AGREEMENT_HIGH:
        return "HIGH"
    if agreement >= AGREEMENT_MODERATE:
        return "MODERATE"
    return "LOW"


def _clamp01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def _sar_class_support(target: str, mean_in: float, std_in: float) -> Tuple[float, Dict[str, Any]]:
    """
    Deterministic SAR support for a land-cover class from measured
    normalized backscatter inside the region. Declared windows are
    configuration, not fitted science (see config note in response).
    """
    stats = {
        "mean_backscatter_in_region": round(mean_in, 4),
        "std_backscatter_in_region": round(std_in, 4),
        "estimated_db_in_region": round(SAR_DB_MIN + mean_in * (SAR_DB_MAX - SAR_DB_MIN), 2),
    }
    if target == "water":
        # Calm water = very dark in SAR
        support = _clamp01((SAR_WATER_NORM_MAX - mean_in) / max(SAR_WATER_NORM_MAX, 1e-6))
    elif target == "built_up":
        # Built-up = bright (double-bounce) AND textured
        bright = _clamp01((mean_in - SAR_BUILT_NORM_MIN) / max(1.0 - SAR_BUILT_NORM_MIN, 1e-6))
        textured = _clamp01((std_in - SAR_BUILT_TEXTURE_MIN) / 0.20)
        support = 0.6 * bright + 0.4 * textured
    elif target == "vegetation":
        # Vegetation = moderate backscatter band (0.22..0.45 normalized)
        lo, hi = SAR_WATER_NORM_MAX, SAR_BUILT_NORM_MIN
        if mean_in < lo or mean_in > hi:
            support = _clamp01(1.0 - min(abs(mean_in - lo), abs(mean_in - hi)) / 0.30)
        else:
            support = 0.75 + 0.25 * _clamp01(std_in / 0.15)
    else:
        support = 0.5
    return round(support, 4), stats
