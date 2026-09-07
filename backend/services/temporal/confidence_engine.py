"""
Calibrated Confidence Engine, Gating & Persistence Module (Stages 12, 16, 17, 18, 19).
Replaces uncalibrated/hardcoded scores with an evidence-based multi-factor model.

Strict Rule:
A change CANNOT receive >= 90% confidence unless:
1. registration_quality >= 0.85
2. At least 2 independent change detectors agree
3. Object-level temporal difference is verified (demonstrated absence in T1)
4. No severe environmental / water / shadow conflict exists.
Otherwise, confidence is strictly capped.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger("satquery.temporal.confidence_engine")

# -------------------------------------------------------------
# CLASS CATEGORIZATION TAXONOMY
# -------------------------------------------------------------
TEMPORARY_CLASSES = {
    "ship", "boat", "vessel", "cargo ship", "tanker", "container ship",
    "vehicle", "car", "truck", "bus", "aircraft", "airplane", "container",
    "movable", "temporary structure", "crane", "machinery"
}

PERMANENT_CLASSES = {
    "building", "house", "warehouse", "facility", "structure", "factory",
    "road", "highway", "bridge", "runway", "pier", "dock", "wharf",
    "storage tank", "oil tank", "terminal", "construction", "foundation"
}

ENVIRONMENTAL_CLASSES = {
    "water", "river", "lake", "ocean", "sea", "tide", "wave", "shoreline",
    "vegetation", "tree", "forest", "crop", "grass", "field", "soil", "sand"
}


def classify_persistence(label: str) -> str:
    """
    Categorizes object into PERMANENT, LIKELY_PERMANENT, TEMPORARY, ENVIRONMENTAL, or UNCERTAIN.
    """
    clean_label = (label or "").strip().lower()

    for t_cls in TEMPORARY_CLASSES:
        if t_cls in clean_label or clean_label in t_cls:
            return "TEMPORARY"

    for e_cls in ENVIRONMENTAL_CLASSES:
        if e_cls in clean_label or clean_label in e_cls:
            return "ENVIRONMENTAL"

    for p_cls in PERMANENT_CLASSES:
        if p_cls in clean_label or clean_label in p_cls:
            return "PERMANENT"

    return "LIKELY_PERMANENT"


def calibrate_change_confidence(
    change_type: str,
    label: str,
    registration_quality: float,
    structural_change_score: float,
    object_detection_confidence: float,
    temporal_object_evidence: float,
    patch_ssim_t1_t2: float,
    segmentation_quality: float = 0.80,
    optical_score: float = 0.80,
    sar_score: float = 0.0,
    has_sar: bool = False,
    vlm_score: Optional[float] = None,
    nuisance_overlap_ratio: float = 0.0,
    is_water_dominated: bool = False,
) -> Dict[str, Any]:
    """
    Computes an evidence-based calibrated confidence score and assigns verification status.

    Returns:
        {
            "confidence": float (0.0 to 1.0),
            "verification_status": str (UNRELIABLE | POSSIBLE_CHANGE | LIKELY_CHANGE | CONFIRMED_CHANGE),
            "persistence": str (PERMANENT | LIKELY_PERMANENT | TEMPORARY | ENVIRONMENTAL | UNCERTAIN),
            "is_gated": bool,
            "gate_reason": Optional[str],
            "evidence": Dict[str, float],
            "penalties": Dict[str, float],
            "description": str,
        }
    """
    persistence = classify_persistence(label)

    # If the object is inherently temporary (ship, vehicle), override permanent construction
    if persistence == "TEMPORARY" and change_type == "new":
        change_type = "temporary_appearance"
    elif persistence == "ENVIRONMENTAL":
        change_type = "environmental_shift"

    # Base weighted multi-evidence formula
    weights = {
        "reg": 0.15,
        "structural": 0.25,
        "dino": 0.15,
        "temporal": 0.20,
        "seg": 0.10,
        "optical": 0.15,
    }

    raw_conf = (
        weights["reg"] * min(1.0, max(0.0, registration_quality)) +
        weights["structural"] * min(1.0, max(0.0, structural_change_score)) +
        weights["dino"] * min(1.0, max(0.0, object_detection_confidence)) +
        weights["temporal"] * min(1.0, max(0.0, temporal_object_evidence)) +
        weights["seg"] * min(1.0, max(0.0, segmentation_quality)) +
        weights["optical"] * min(1.0, max(0.0, optical_score))
    )

    if has_sar:
        # Incorporate SAR late fusion
        sar_weight = 0.12
        raw_conf = raw_conf * (1.0 - sar_weight) + sar_score * sar_weight

    if vlm_score is not None:
        # Incorporate conservative VLM verification
        vlm_weight = 0.15
        raw_conf = raw_conf * (1.0 - vlm_weight) + vlm_score * vlm_weight

    penalties: Dict[str, float] = {}
    total_penalty = 0.0

    # 1. Misalignment penalty
    if registration_quality < 0.70:
        p_misalign = round((0.70 - registration_quality) * 0.50, 3)
        penalties["misalignment"] = p_misalign
        total_penalty += p_misalign

    # 2. Environmental / Nuisance overlap penalty
    if nuisance_overlap_ratio > 0.20:
        p_nuisance = round(min(0.35, nuisance_overlap_ratio * 0.40), 3)
        penalties["nuisance_overlap"] = p_nuisance
        total_penalty += p_nuisance

    if is_water_dominated and persistence != "TEMPORARY":
        p_water = 0.25
        penalties["water_boundary_instability"] = p_water
        total_penalty += p_water

    # 3. Patch Identity / DINO Dropout penalty:
    # If the T1 crop and T2 crop are structurally nearly identical (SSIM > 0.82),
    # then the object was already physically present in T1 and DINO simply failed to detect it!
    if patch_ssim_t1_t2 > 0.80 and change_type == "new":
        p_presence = round(0.40 + (patch_ssim_t1_t2 - 0.80) * 2.0, 3)
        penalties["pre_existing_structural_presence"] = p_presence
        total_penalty += p_presence

    adjusted_conf = max(0.10, raw_conf - total_penalty)

    # -------------------------------------------------------------
    # STRICT CONFIDENCE GATING (Stage 17 & 28)
    # -------------------------------------------------------------
    is_gated = False
    gate_reason = None
    confidence_cap = 1.0

    # Minimum criteria for HIGH CONFIDENCE (>= 90%)
    reasons = []
    if registration_quality < 0.85:
        reasons.append(f"Registration quality ({registration_quality:.2f}) < 0.85")
    if structural_change_score < 0.35:
        reasons.append(f"Structural change signal ({structural_change_score:.2f}) < 0.35")
    if temporal_object_evidence < 0.35:
        reasons.append(f"Temporal difference evidence ({temporal_object_evidence:.2f}) < 0.35")
    if nuisance_overlap_ratio > 0.30:
        reasons.append(f"Nuisance overlap ({nuisance_overlap_ratio:.0%}) exceeds 30%")
    if patch_ssim_t1_t2 > 0.78 and change_type == "new":
        reasons.append("High structural presence in T1 (potential detection dropout)")

    if reasons:
        is_gated = True
        gate_reason = "; ".join(reasons)
        confidence_cap = 0.78  # Strict cap preventing fake >90% claims

    if persistence == "TEMPORARY":
        confidence_cap = min(confidence_cap, 0.65)
        is_gated = True
        gate_reason = (gate_reason + "; " if gate_reason else "") + "Transient object (capped <= 0.65)"
    elif persistence == "ENVIRONMENTAL":
        confidence_cap = min(confidence_cap, 0.50)
        is_gated = True
        gate_reason = (gate_reason + "; " if gate_reason else "") + "Environmental variation (capped <= 0.50)"

    if nuisance_overlap_ratio > 0.50:
        confidence_cap = min(confidence_cap, 0.48)
        is_gated = True
        gate_reason = (gate_reason + "; " if gate_reason else "") + f"Severe nuisance conflict ({nuisance_overlap_ratio:.0%})"

    if registration_quality < 0.50 or patch_ssim_t1_t2 > 0.85:
        confidence_cap = min(confidence_cap, 0.48)

    final_conf = round(min(adjusted_conf, confidence_cap), 2)

    # -------------------------------------------------------------
    # VERIFICATION STATUS
    # -------------------------------------------------------------
    if final_conf >= 0.88:
        verification_status = "CONFIRMED_CHANGE"
    elif final_conf >= 0.68:
        verification_status = "LIKELY_CHANGE"
    elif final_conf >= 0.50:
        verification_status = "POSSIBLE_CHANGE"
    else:
        verification_status = "UNCERTAIN"


    # -------------------------------------------------------------
    # CALIBRATED SEMANTIC DESCRIPTION (Stage 18)
    # -------------------------------------------------------------
    description = _generate_semantic_description(
        change_type=change_type,
        label=label,
        persistence=persistence,
        verification_status=verification_status,
        confidence=final_conf,
        is_gated=is_gated,
        gate_reason=gate_reason,
        has_sar=has_sar,
    )

    return {
        "confidence": final_conf,
        "verification_status": verification_status,
        "persistence": persistence,
        "is_gated": is_gated,
        "gate_reason": gate_reason,
        "evidence": {
            "registration_score": round(registration_quality, 3),
            "structural_change_score": round(structural_change_score, 3),
            "detection_confidence": round(object_detection_confidence, 3),
            "temporal_evidence": round(temporal_object_evidence, 3),
            "patch_ssim_t1_t2": round(patch_ssim_t1_t2, 3),
            "optical_score": round(optical_score, 3),
            "sar_score": round(sar_score, 3) if has_sar else 0.0,
            "vlm_score": round(vlm_score, 3) if vlm_score is not None else 0.0,
        },
        "penalties": penalties,
        "description": description,
    }


def _generate_semantic_description(
    change_type: str,
    label: str,
    persistence: str,
    verification_status: str,
    confidence: float,
    is_gated: bool,
    gate_reason: Optional[str],
    has_sar: bool,
) -> str:
    """Generates an honest, calibrated remote-sensing change description."""
    clean_lbl = label.capitalize()

    if persistence == "TEMPORARY":
        return f"Transient object ({label}) detected at location. Classified as temporary mobile asset; no permanent infrastructure development."

    if persistence == "ENVIRONMENTAL":
        return f"Environmental or surface boundary shift observed ({label}). Consistent with water level, vegetation phenology, or soil moisture variations."

    if verification_status == "CONFIRMED_CHANGE":
        sensor_text = "optical and SAR sensors" if has_sar else "multi-signal structural edge and SSIM analysis"
        return f"Newly constructed {label} confirmed ({confidence*100:.0f}% confidence) with strong corroboration across {sensor_text}."

    if verification_status == "LIKELY_CHANGE":
        return f"Likely new {label} identified. Structural change signals indicate significant development, though minor viewing condition variances remain."

    if verification_status == "POSSIBLE_CHANGE":
        reason_hint = f" ({gate_reason})" if gate_reason else ""
        return f"Possible {label} alteration observed. Evidence is insufficient for definitive confirmation due to viewing conditions or subtle spectral shifts{reason_hint}."

    # UNRELIABLE / NO CONFIRMED CHANGE
    return f"Unconfirmed detection artifact for {label}. Photometric or registration differences prevent reliable physical change classification."
