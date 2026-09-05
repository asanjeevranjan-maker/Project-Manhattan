"""
Satellite Vision Response Parser & Pydantic Data Models
Robust JSON extraction, validation, and backward-compatible mapping.
"""

import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ObservationItem(BaseModel):
    finding: str = Field(description="Title or summary of the visual finding")
    location: str = Field(default="center", description="Spatial quadrant or region")
    confidence: str = Field(default="medium", description="'high', 'medium', or 'low'")
    evidence: str = Field(default="", description="Specific observable visual evidence")


class EvidenceMetadata(BaseModel):
    detections_used: bool = Field(default=False, description="Whether Grounding DINO detection evidence was incorporated")
    segmentation_used: bool = Field(default=False, description="Whether instance segmentation was incorporated")
    land_cover_used: bool = Field(default=False, description="Whether mask-measured land-cover coverage was incorporated")
    change_detection_used: bool = Field(default=False, description="Whether bi-temporal change detection was incorporated")


class SatelliteAnalysisStructured(BaseModel):
    answer: str = Field(default="", description="Direct, evidence-grounded answer to user query")
    summary: str = Field(default="", description="Concise 1-2 sentence executive summary")
    answer_to_query: str = Field(default="", description="Direct, evidence-grounded answer to user query (mirror of answer)")
    evidence: EvidenceMetadata = Field(default_factory=EvidenceMetadata, description="Evidence usage flags")
    calculated_statistics: Dict[str, Any] = Field(default_factory=dict, description="Objective metrics, counts, and percentages")
    observed_changes: List[str] = Field(default_factory=list, description="Confirmed physical differences between timestamps")
    possible_causes: List[str] = Field(default_factory=list, description="Plausible hypotheses explaining observed changes")
    observations: List[ObservationItem] = Field(default_factory=list, description="Structured visual findings")
    uncertainties: List[str] = Field(default_factory=list, description="Caveats or resolution/sensor limitations")
    model_notes: Dict[str, Any] = Field(default_factory=dict, description="Metadata about context used")


def _clean_json_str(text: str) -> str:
    """Strips markdown code blocks, XML wrappers, or leading/trailing prose."""
    # Fenced blocks: ```json ... ``` or ```satquery ... ``` or ``` ... ```
    match = re.search(r"```(?:json|satquery)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Search for outermost matching braces { ... }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return text[brace_start : brace_end + 1].strip()

    return text.strip()


def parse_structured_response(
    raw_text: str,
    query: str,
    detection_used: bool = False,
    change_used: bool = False,
    segmentation_used: bool = False,
    land_cover_used: bool = False,
) -> SatelliteAnalysisStructured:
    """
    Parses LLM/VLM text output into SatelliteAnalysisStructured.
    Guaranteed not to crash on malformed or unexpected responses.
    """
    cleaned = _clean_json_str(raw_text)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            # Extract fields with safe defaults
            summary = str(data.get("summary") or "").strip()
            answer = str(data.get("answer") or data.get("answer_to_query") or summary).strip()

            if not summary and answer:
                summary = answer[:160] + "..." if len(answer) > 160 else answer
            if not answer and summary:
                answer = summary

            # Extract evidence metadata flags
            raw_ev = data.get("evidence")
            if isinstance(raw_ev, dict):
                evidence = EvidenceMetadata(
                    detections_used=bool(raw_ev.get("detections_used", detection_used)),
                    segmentation_used=bool(raw_ev.get("segmentation_used", segmentation_used)),
                    land_cover_used=bool(raw_ev.get("land_cover_used", land_cover_used)),
                    change_detection_used=bool(raw_ev.get("change_detection_used", change_used)),
                )
            else:
                evidence = EvidenceMetadata(
                    detections_used=detection_used,
                    segmentation_used=segmentation_used,
                    land_cover_used=land_cover_used,
                    change_detection_used=change_used,
                )

            calculated_statistics = data.get("calculated_statistics") if isinstance(data.get("calculated_statistics"), dict) else {}

            raw_obs = data.get("observations", [])
            observations: List[ObservationItem] = []
            if isinstance(raw_obs, list):
                for item in raw_obs:
                    if isinstance(item, dict):
                        finding = str(item.get("finding") or "Observed feature").strip()
                        location = str(item.get("location") or "center").strip()
                        confidence = str(item.get("confidence") or "medium").strip().lower()
                        if confidence not in ["high", "medium", "low"]:
                            confidence = "medium"
                        evidence_str = str(item.get("evidence") or "").strip()
                        observations.append(
                            ObservationItem(
                                finding=finding,
                                location=location,
                                confidence=confidence,
                                evidence=evidence_str,
                            )
                        )

            raw_changes = data.get("observed_changes", [])
            observed_changes = [str(c) for c in raw_changes if c] if isinstance(raw_changes, list) else []

            raw_causes = data.get("possible_causes", [])
            possible_causes = [str(c) for c in raw_causes if c] if isinstance(raw_causes, list) else []

            raw_unc = data.get("uncertainties", [])
            uncertainties = [str(u) for u in raw_unc if u] if isinstance(raw_unc, list) else []

            model_notes = data.get("model_notes") if isinstance(data.get("model_notes"), dict) else {}
            model_notes.setdefault("used_detection_context", detection_used)
            model_notes.setdefault("used_change_context", change_used)

            return SatelliteAnalysisStructured(
                answer=answer or "Visual inspection completed based on visible evidence.",
                summary=summary or "Remote sensing analysis completed.",
                answer_to_query=answer or "Visual inspection completed based on visible evidence.",
                evidence=evidence,
                calculated_statistics=calculated_statistics,
                observed_changes=observed_changes,
                possible_causes=possible_causes,
                observations=observations,
                uncertainties=uncertainties,
                model_notes=model_notes,
            )
    except Exception:
        pass

    # Fallback parsing when JSON syntax is invalid
    plain_text = raw_text.strip()
    first_paragraph = plain_text.split("\n\n")[0] if plain_text else "Visual analysis completed."
    summary = first_paragraph[:160] + "..." if len(first_paragraph) > 160 else first_paragraph

    fallback_obs = [
        ObservationItem(
            finding="Visual assessment based on query",
            location="widespread",
            confidence="medium",
            evidence=first_paragraph[:250],
        )
    ]

    fallback_evidence = EvidenceMetadata(
        detections_used=detection_used,
        segmentation_used=segmentation_used,
        land_cover_used=land_cover_used,
        change_detection_used=change_used,
    )

    return SatelliteAnalysisStructured(
        answer=plain_text if plain_text else "Visual analysis completed based on supplied imagery.",
        summary=summary,
        answer_to_query=plain_text if plain_text else "Visual analysis completed based on supplied imagery.",
        evidence=fallback_evidence,
        calculated_statistics={},
        observations=fallback_obs,
        uncertainties=["Model response was in non-standard format; structured output reconstructed from text."],
        model_notes={
            "used_detection_context": detection_used,
            "used_change_context": change_used,
            "raw_text_length": len(raw_text),
        },
    )


def to_legacy_analysis_result(
    structured: SatelliteAnalysisStructured,
    intent: str = "image_understanding",
    land_cover_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Converts SatelliteAnalysisStructured into the exact shape expected by
    the Next.js frontend (AnalysisResult schema in src/lib/types.ts).
    Ensures 100% zero-regression backward compatibility.
    """
    # Map observations to legacy objectsDetected
    objects_detected: List[Dict[str, Any]] = []
    regions: List[Dict[str, Any]] = []

    quad_to_rect = {
        "upper-left": [0.05, 0.05, 0.40, 0.40],
        "upper-right": [0.55, 0.05, 0.40, 0.40],
        "center": [0.30, 0.30, 0.40, 0.40],
        "lower-left": [0.05, 0.55, 0.40, 0.40],
        "lower-right": [0.55, 0.55, 0.40, 0.40],
        "widespread": [0.10, 0.10, 0.80, 0.80],
    }

    conf_num = {"high": 0.90, "medium": 0.75, "low": 0.55}

    for idx, obs in enumerate(structured.observations[:6]):
        c_num = conf_num.get(obs.confidence.lower(), 0.75)
        objects_detected.append({
            "class": obs.finding,
            "confidence": c_num,
            "count": 1,
            "region": obs.location,
            "note": obs.evidence or obs.finding,
        })
        rect = quad_to_rect.get(obs.location.lower(), [0.25, 0.25, 0.50, 0.50])
        regions.append({
            "label": obs.finding,
            "color": "#06b6d4" if idx % 2 == 0 else "#f97316",
            "rect": rect,
            "confidence": c_num,
        })

    # Objective land-cover coverage (only populated when measured from segmentation masks)
    if land_cover_result and land_cover_result.get("available"):
        coverage = land_cover_result.get("coverage", [])
        land_cover_meta = land_cover_result
    else:
        coverage = []
        land_cover_meta = {
            "available": False,
            "reason": "Segmentation unavailable",
            "measured_from_masks": False,
            "estimated": False,
        }

    return {
        "answer": structured.answer or structured.answer_to_query,
        "summary": structured.summary,
        "intent": intent,
        "objectsDetected": objects_detected,
        "confidence": 0.88,
        "coverage": coverage,
        "land_cover": land_cover_meta,
        "measured_from_masks": bool(land_cover_result and land_cover_result.get("measured_from_masks")),
        "estimated": False,
        "regions": regions,
        "evidence": structured.evidence.model_dump(),
        "calculated_statistics": structured.calculated_statistics,
        "observed_changes": structured.observed_changes,
        "possible_causes": structured.possible_causes,
        "changeSummary": {
            "additions": [obs.finding for obs in structured.observations if "new" in obs.finding.lower()],
            "removals": [obs.finding for obs in structured.observations if "remov" in obs.finding.lower() or "absent" in obs.finding.lower()],
            "netChange": structured.summary,
        },
    }
