"""
Unit and Integration Tests for Grounded Multimodal Vision Reasoning (Gemini & GLM).
Verifies:
1. Compact vision context builder (counts, spatial bbox distributions, land-cover percentages).
2. Authoritative grounding rules in prompts (prefer machine counts, use bbox distribution, flood exposure).
3. Structured output response parser with EvidenceMetadata and calculated_statistics.
4. Mocked Gemini and GLM provider execution across counting, localization, and flood exposure queries.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from services.vision.context_builder import build_vision_context
from services.vision.prompt_builder import (
    build_satellite_analysis_prompt,
    SYSTEM_INSTRUCTION,
    JSON_OUTPUT_SPEC,
)
from services.vision.response_parser import (
    ObservationItem,
    SatelliteAnalysisStructured,
    EvidenceMetadata,
    parse_structured_response,
    to_legacy_analysis_result,
)
from services.vision.gemini_provider import GeminiVisionProvider
from services.vision.glm_provider import GLMVisionProvider
from services.vision.vision_service import VisionService


# =====================================================================
# 1. COMPACT CONTEXT BUILDER TESTS
# =====================================================================

def test_build_vision_context_with_detections_and_spatial_distribution():
    """
    Verifies that build_vision_context produces compact summaries like:
    - 14 buildings, mostly left side
    - 3 vehicles, near center-left
    - 1 water region, crossing center-right
    without dumping raw JSON coordinates.
    """
    img_w, img_h = 1000.0, 1000.0

    # 14 buildings on the left side (x < 400)
    buildings = [
        {"label": "building", "box": [50 + (i * 20), 100 + (i * 40), 90 + (i * 20), 150 + (i * 40)]}
        for i in range(14)
    ]
    # 3 vehicles near center-left (x between 250 and 450)
    vehicles = [
        {"label": "vehicle", "box": [300, 500, 340, 530]},
        {"label": "vehicle", "box": [320, 540, 360, 570]},
        {"label": "vehicle", "box": [340, 520, 380, 550]},
    ]
    # 1 wide water body crossing center-right
    water = [
        {"label": "water", "box": [450, 100, 950, 800]}
    ]

    all_detections = buildings + vehicles + water

    ctx = build_vision_context(
        detections={"count": len(all_detections), "width": img_w, "height": img_h, "detections": all_detections},
        image_size=(1000, 1000),
    )

    assert ctx["evidence_flags"]["detections_used"] is True
    assert ctx["object_counts"]["building"] == 14
    assert ctx["object_counts"]["vehicle"] == 3
    assert ctx["object_counts"]["water"] == 1

    summary = ctx["summary_text"]
    assert "14 buildings, mostly left side" in summary
    assert "3 vehicles, near center-left" in summary
    assert "1 water region, crossing center-right" in summary

    # Ensure no raw coordinate arrays in summary
    assert "[50, 100, 90, 150]" not in summary
    assert "MACHINE-GENERATED SATELLITE EVIDENCE" in summary


def test_build_vision_context_with_land_cover_percentages():
    """
    Verifies that objective land-cover statistics are formatted cleanly:
    - vegetation: 44.8%
    - built-up: 31.2%
    - water: 18.3%
    """
    land_cover_data = {
        "available": True,
        "measured_from_masks": True,
        "land_cover": {
            "vegetation": {"pixels": 44800, "percentage": 44.8},
            "built_up": {"pixels": 31200, "percentage": 31.2},
            "water": {"pixels": 18300, "percentage": 18.3},
            "other": {"pixels": 5700, "percentage": 5.7},
        },
    }

    ctx = build_vision_context(land_cover=land_cover_data)

    assert ctx["evidence_flags"]["land_cover_used"] is True
    summary = ctx["summary_text"]
    assert "Land cover:" in summary
    assert "- vegetation: 44.8%" in summary
    assert "- built-up: 31.2%" in summary
    assert "- water: 18.3%" in summary


def test_build_vision_context_empty():
    """Verifies that empty/None inputs return safe default dictionaries without error."""
    ctx = build_vision_context(None, None, None, None)
    assert ctx["summary_text"] == ""
    assert ctx["evidence_flags"]["detections_used"] is False
    assert ctx["evidence_flags"]["segmentation_used"] is False
    assert ctx["evidence_flags"]["land_cover_used"] is False
    assert ctx["evidence_flags"]["change_detection_used"] is False


# =====================================================================
# 2. PROMPT GROUNDING RULES TESTS
# =====================================================================

def test_prompt_contains_grounding_rules():
    """Verifies that SYSTEM_INSTRUCTION contains the strict grounding directives."""
    assert "PRIMARY GROUNDING RULES" in SYSTEM_INSTRUCTION
    assert "Do NOT independently invent objects or counts" in SYSTEM_INSTRUCTION
    assert "PREFER the machine detection count" in SYSTEM_INSTRUCTION
    assert "USE the bounding box and mask spatial distribution" in SYSTEM_INSTRUCTION
    assert "possible exposure" in SYSTEM_INSTRUCTION
    assert "Strictly separate:" in SYSTEM_INSTRUCTION
    assert "observed objects" in SYSTEM_INSTRUCTION
    assert "calculated statistics" in SYSTEM_INSTRUCTION
    assert "model interpretation" in SYSTEM_INSTRUCTION


def test_build_satellite_analysis_prompt_integrates_machine_evidence():
    """Verifies that build_satellite_analysis_prompt includes machine evidence summary."""
    detections = [
        {"label": "building", "box": [50, 50, 100, 100]},
        {"label": "building", "box": [70, 70, 120, 120]},
    ]
    land_cover = {
        "available": True,
        "measured_from_masks": True,
        "coverage": [
            {"class": "built-up", "coverage": 0.40},
            {"class": "vegetation", "coverage": 0.60},
        ],
    }

    sys_prompt, user_prompt = build_satellite_analysis_prompt(
        user_query="How many buildings are visible and what is the land cover?",
        detection_context={"detections": detections, "count": 2},
        land_cover=land_cover,
    )

    assert "MACHINE-GENERATED SATELLITE EVIDENCE" in user_prompt
    assert "2 buildings" in user_prompt
    assert "built-up: 40.0%" in user_prompt
    assert "vegetation: 60.0%" in user_prompt
    assert "Treat detection and segmentation results as machine-generated evidence" in user_prompt


# =====================================================================
# 3. RESPONSE PARSER WITH EVIDENCE METADATA TESTS
# =====================================================================

def test_structured_response_parsing_with_evidence():
    """Verifies parsing of structured responses containing evidence and calculated_statistics."""
    mock_json = """
    {
      "answer": "Grounding DINO detected 14 buildings primarily on the left side of the scene.",
      "summary": "14 buildings confirmed in the western quadrant.",
      "evidence": {
        "detections_used": true,
        "segmentation_used": true,
        "land_cover_used": true
      },
      "calculated_statistics": {
        "building_count": 14,
        "built_up_coverage": "31.2%"
      },
      "observations": [
        {
          "finding": "14 Buildings cluster",
          "location": "left side",
          "confidence": "high",
          "evidence": "High-reflectance residential structures aligned along access roads"
        }
      ],
      "uncertainties": [
        "Faint structures under dense tree canopy may be uncounted"
      ]
    }
    """
    structured = parse_structured_response(mock_json, query="How many buildings?")
    assert structured.answer.startswith("Grounding DINO detected 14 buildings")
    assert structured.summary == "14 buildings confirmed in the western quadrant."
    assert structured.evidence.detections_used is True
    assert structured.evidence.segmentation_used is True
    assert structured.evidence.land_cover_used is True
    assert structured.calculated_statistics["building_count"] == 14
    assert len(structured.observations) == 1
    assert structured.observations[0].location == "left side"

    # Legacy mapping test
    legacy = to_legacy_analysis_result(structured)
    assert legacy["answer"] == structured.answer
    assert legacy["evidence"]["detections_used"] is True
    assert legacy["calculated_statistics"]["building_count"] == 14


# =====================================================================
# 4. MOCKED PROVIDER INTEGRATION TESTS
# =====================================================================

@pytest.mark.asyncio
async def test_mocked_gemini_counting_query():
    """
    Test: 'How many buildings?'
    Grounding DINO detected 14 buildings.
    Verifies that Gemini uses the machine-detected count instead of guessing.
    """
    provider = GeminiVisionProvider(api_key="mock_test_key")

    mock_gemini_reply = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "answer": "Grounding DINO detected exactly 14 buildings across the image, concentrated predominantly on the left side.",
                        "summary": "14 buildings identified via machine detection.",
                        "evidence": {
                            "detections_used": True,
                            "segmentation_used": False,
                            "land_cover_used": False
                        },
                        "calculated_statistics": {"building_count": 14},
                        "observations": [{
                            "finding": "14 buildings in residential cluster",
                            "location": "left side",
                            "confidence": "high",
                            "evidence": "Regular rectangular rooflines matching detector bounding boxes"
                        }],
                        "uncertainties": []
                    })
                }]
            }
        }]
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_gemini_reply

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        detections = [
            {"label": "building", "box": [50 + (i * 20), 50, 90 + (i * 20), 90]}
            for i in range(14)
        ]
        result = await provider.analyze(
            image_bytes=b"fake_image_bytes",
            mime_type="image/jpeg",
            user_query="How many buildings?",
            detection_context={"detections": detections, "count": 14, "width": 1000, "height": 1000},
        )

        assert "14 buildings" in result.answer
        assert result.evidence.detections_used is True
        assert result.calculated_statistics.get("building_count") == 14


@pytest.mark.asyncio
async def test_mocked_glm_localization_query():
    """
    Test: 'Where are the buildings?'
    Detections are on the left side.
    Verifies that GLM uses the bbox distribution ('mostly left side').
    """
    provider = GLMVisionProvider(api_key="mock_glm_key")

    mock_glm_reply = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "answer": "The detected buildings are located mostly on the left side of the scene, forming a compact residential strip.",
                    "summary": "Buildings concentrated on the western flank.",
                    "evidence": {
                        "detections_used": True,
                        "segmentation_used": False,
                        "land_cover_used": False
                    },
                    "calculated_statistics": {"spatial_distribution": "mostly left side"},
                    "observations": [{
                        "finding": "Linear settlement alignment",
                        "location": "left side",
                        "confidence": "high",
                        "evidence": "Aligned parallel to the western boundary track"
                    }],
                    "uncertainties": []
                })
            }
        }]
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_glm_reply

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        detections = [
            {"label": "building", "box": [50 + (i * 20), 50, 90 + (i * 20), 90]}
            for i in range(10)
        ]
        result = await provider.analyze(
            image_bytes=b"fake_image_bytes",
            mime_type="image/jpeg",
            user_query="Where are the buildings?",
            detection_context={"detections": detections, "count": 10, "width": 1000, "height": 1000},
        )

        assert "left side" in result.answer.lower()
        assert result.evidence.detections_used is True


@pytest.mark.asyncio
async def test_mocked_flood_danger_query():
    """
    Test: 'What is the flood danger?'
    Uses water mask + nearby structures.
    Verifies that conclusions are phrased as possible exposure, NOT guaranteed flood risk.
    """
    provider = GeminiVisionProvider(api_key="mock_test_key")

    mock_reply = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "answer": "Structures situated near the lower-right perimeter show potential exposure due to proximity to the delineated water body; this indicates possible flood exposure during high-water events rather than confirmed inundation.",
                        "summary": "Potential exposure noted for structures adjacent to the water channel.",
                        "evidence": {
                            "detections_used": True,
                            "segmentation_used": True,
                            "land_cover_used": True
                        },
                        "calculated_statistics": {"water_coverage": "28.5%"},
                        "observations": [{
                            "finding": "Structures adjacent to drainage channel",
                            "location": "lower-right",
                            "confidence": "medium",
                            "evidence": "Dwellings situated within 50 meters of the water mask boundary"
                        }],
                        "uncertainties": [
                            "Terrain elevation profiles and tidal variations are unverified from single 2D optical imagery"
                        ]
                    })
                }]
            }
        }]
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_reply

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        result = await provider.analyze(
            image_bytes=b"fake_image_bytes",
            mime_type="image/jpeg",
            user_query="What is the flood danger?",
            detection_context={"detections": [{"label": "house", "box": [700, 700, 750, 750]}], "count": 1},
            segmentation_summary={"segmentation_available": True, "segmented_count": 2, "classes_segmented": ["water", "house"]},
            land_cover={"available": True, "measured_from_masks": True, "coverage": [{"class": "water", "coverage": 0.285}]},
        )

        # Ensure phrased as possible exposure, not guaranteed flood risk
        assert "possible" in result.answer.lower() or "potential" in result.answer.lower()
        assert "exposure" in result.answer.lower()
        assert result.evidence.detections_used is True
        assert result.evidence.segmentation_used is True
        assert result.evidence.land_cover_used is True
        assert len(result.uncertainties) > 0
