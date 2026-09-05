"""
Unit and Integration Test Suite for SatQuery Enhanced Multimodal Vision Pipeline.
Validates prompt construction, context formatting, robust JSON recovery, image preprocessing,
and fallback logic without using real API tokens.
"""

import io
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from PIL import Image

from services.vision.prompt_builder import (
    build_satellite_analysis_prompt,
    format_detection_context,
    format_change_context,
    ANALYSIS_MODE_GUIDELINES,
)
from services.vision.response_parser import (
    ObservationItem,
    SatelliteAnalysisStructured,
    parse_structured_response,
    to_legacy_analysis_result,
)
from services.vision.image_processor import (
    preprocess_image,
    generate_tiles,
    decode_data_url,
    encode_image_base64,
)
from services.vision.base_provider import (
    VisionProviderAuthError,
    VisionProviderRateLimitError,
    VisionProviderError,
)
from services.vision.vision_service import VisionService


# =========================================================
# 1. Prompt Builder Tests
# =========================================================

def test_prompt_builder_contains_system_rules():
    sys_prompt, user_prompt = build_satellite_analysis_prompt(
        user_query="Identify all storage tanks",
        analysis_mode="infrastructure",
    )
    assert "expert remote-sensing" in sys_prompt.lower()
    assert "anti-hallucination" in sys_prompt.lower() or "never claim exact" in sys_prompt.lower()
    assert "Identify all storage tanks" in user_prompt
    assert "INFRASTRUCTURE" in user_prompt


def test_prompt_builder_all_modes():
    modes = ["general", "objects", "changes", "urban", "agriculture", "water", "infrastructure", "disaster", "landcover", "custom"]
    for mode in modes:
        sys_prompt, user_prompt = build_satellite_analysis_prompt(
            user_query="Test query",
            analysis_mode=mode,
        )
        assert len(sys_prompt) > 200
        assert len(user_prompt) > 200
        if mode in ANALYSIS_MODE_GUIDELINES:
            guideline = ANALYSIS_MODE_GUIDELINES[mode]
            assert guideline[:30] in user_prompt


def test_prompt_builder_two_image_comparative():
    sys_prompt, user_prompt = build_satellite_analysis_prompt(
        user_query="What changed between these dates?",
        analysis_mode="changes",
        has_second_image=True,
    )
    assert "BEFORE" in user_prompt
    assert "AFTER" in user_prompt
    assert "IMAGE PAIR COMPARISON" in user_prompt


# =========================================================
# 2. Context Formatter Tests
# =========================================================

def test_detection_context_formatting():
    detections = [
        {"label": "building", "confidence": 0.92, "box": [50, 60, 120, 150]},
        {"label": "building", "confidence": 0.88, "box": [70, 80, 140, 170]},
        {"label": "vessel", "confidence": 0.95, "box": [400, 450, 500, 560]},
    ]
    formatted = format_detection_context({"count": 3, "width": 640, "height": 640, "detections": detections})
    assert formatted is not None
    assert "Total candidate detections: 3" in formatted
    assert "building" in formatted
    assert "vessel" in formatted
    assert "Grounding DINO" in formatted
    assert "supporting machine-generated hints" in formatted


def test_detection_context_empty():
    assert format_detection_context(None) is None
    assert format_detection_context({}) is None
    res = format_detection_context({"count": 0, "detections": []})
    assert "0 candidate objects" in res


def test_change_context_formatting():
    change_meta = {
        "changePercentage": 8.4,
        "timeDifference": "2 years, 3 months",
        "summary": {
            "newCount": 4,
            "removedCount": 1,
            "modifiedCount": 2,
        },
    }
    formatted = format_change_context(change_meta)
    assert formatted is not None
    assert "8.4%" in formatted
    assert "4 newly appeared" in formatted
    assert "1 absent/removed" in formatted
    assert "2 years, 3 months" in formatted


# =========================================================
# 3. Response Parser Tests
# =========================================================

def test_response_parser_raw_json(mock_gemini_json_response: str):
    parsed = parse_structured_response(mock_gemini_json_response, query="Find berths")
    assert isinstance(parsed, SatelliteAnalysisStructured)
    assert "port infrastructure" in parsed.summary
    assert len(parsed.observations) == 2
    assert parsed.observations[0].location == "center"
    assert parsed.observations[0].confidence == "high"
    assert len(parsed.uncertainties) == 1


def test_response_parser_markdown_wrapped_json(mock_gemini_json_response: str):
    wrapped = f"Here is the satellite analysis:\n```json\n{mock_gemini_json_response}\n```\nHope this helps!"
    parsed = parse_structured_response(wrapped, query="Find berths")
    assert isinstance(parsed, SatelliteAnalysisStructured)
    assert "port infrastructure" in parsed.summary
    assert len(parsed.observations) == 2


def test_response_parser_satquery_wrapped_json(mock_gemini_json_response: str):
    wrapped = f"Observations below:\n```satquery\n{mock_gemini_json_response}\n```"
    parsed = parse_structured_response(wrapped, query="Find berths")
    assert len(parsed.observations) == 2


def test_response_parser_corrupted_fallback():
    broken_output = "The satellite scene shows extensive agricultural fields with visible irrigation pivot circles in the upper-right."
    parsed = parse_structured_response(broken_output, query="Analyze agriculture")
    assert isinstance(parsed, SatelliteAnalysisStructured)
    assert "agricultural fields" in parsed.summary
    assert len(parsed.observations) >= 1
    assert len(parsed.uncertainties) >= 1


def test_legacy_analysis_result_mapping(mock_gemini_json_response: str):
    parsed = parse_structured_response(mock_gemini_json_response, query="Find berths")
    legacy = to_legacy_analysis_result(parsed, intent="building_detection")
    assert "answer" in legacy
    assert "objectsDetected" in legacy
    assert "coverage" in legacy
    assert "regions" in legacy
    assert len(legacy["objectsDetected"]) == 2
    assert legacy["objectsDetected"][0]["region"] == "center"


# =========================================================
# 4. Image Preprocessor Tests
# =========================================================

def test_image_preprocessing_within_bounds(sample_test_image_bytes: bytes):
    out_bytes, mime, (w, h) = preprocess_image(sample_test_image_bytes, max_dimension=3072)
    assert mime == "image/jpeg"
    assert w == 400
    assert h == 300
    assert len(out_bytes) > 0


def test_image_preprocessing_oversized_downscaling():
    huge_img = Image.new("RGB", (4000, 2000), color=(100, 150, 200))
    buf = io.BytesIO()
    huge_img.save(buf, format="JPEG")

    out_bytes, mime, (w, h) = preprocess_image(buf.getvalue(), max_dimension=2000)
    assert w == 2000
    assert h == 1000


def test_image_tiling_2x2(sample_test_image_bytes: bytes):
    tiles = generate_tiles(sample_test_image_bytes, grid=(2, 2))
    assert len(tiles) == 4
    labels = [t["label"] for t in tiles]
    assert "top-left" in labels
    assert "top-right" in labels
    assert "bottom-left" in labels
    assert "bottom-right" in labels
    for t in tiles:
        assert len(t["bytes"]) > 0


def test_decode_and_encode_data_url(sample_test_image_bytes: bytes):
    url = encode_image_base64(sample_test_image_bytes, mime_type="image/jpeg")
    assert url.startswith("data:image/jpeg;base64,")
    decoded_bytes, mime = decode_data_url(url)
    assert mime == "image/jpeg"
    assert len(decoded_bytes) == len(sample_test_image_bytes)


# =========================================================
# 5. Fallback & Orchestrator Tests
# =========================================================

@pytest.mark.asyncio
async def test_fallback_gemini_fails_glm_succeeds(sample_data_url: str, mock_gemini_json_response: str):
    service = VisionService()
    parsed_mock = parse_structured_response(mock_gemini_json_response, query="test")

    # Mock Gemini failing with 503 Service Unavailable
    service.gemini.analyze = AsyncMock(side_effect=VisionProviderError("Gemini service unavailable", status_code=503, provider="gemini"))
    # Mock GLM succeeding
    service.glm.analyze = AsyncMock(return_value=parsed_mock)

    result = await service.analyze_image(
        image_data=sample_data_url,
        user_query="Find vessels",
        provider="gemini",  # requested gemini first
    )

    assert result["success"] is True
    assert result["provider_used"] == "glm"
    assert result["fallback_used"] is True
    assert service.gemini.analyze.called
    assert service.glm.analyze.called


@pytest.mark.asyncio
async def test_fallback_glm_fails_gemini_succeeds(sample_data_url: str, mock_gemini_json_response: str):
    service = VisionService()
    parsed_mock = parse_structured_response(mock_gemini_json_response, query="test")

    # Mock GLM failing with 429
    service.glm.analyze = AsyncMock(side_effect=VisionProviderRateLimitError("Rate limit", status_code=429, provider="glm"))
    # Mock Gemini succeeding
    service.gemini.analyze = AsyncMock(return_value=parsed_mock)

    result = await service.analyze_image(
        image_data=sample_data_url,
        user_query="Find vessels",
        provider="glm",  # requested glm first
    )

    assert result["success"] is True
    assert result["provider_used"] == "gemini"
    assert result["fallback_used"] is True


@pytest.mark.asyncio
async def test_ensemble_mode_cross_verification(sample_data_url: str):
    service = VisionService()

    gemini_out = SatelliteAnalysisStructured(
        summary="Gemini summary",
        answer_to_query="Gemini answer",
        observations=[
            ObservationItem(finding="Runway", location="center", confidence="high", evidence="Long paved surface"),
            ObservationItem(finding="Aircraft hangar", location="upper-right", confidence="medium", evidence="Large square roof"),
        ],
    )
    glm_out = SatelliteAnalysisStructured(
        summary="GLM summary",
        answer_to_query="GLM answer",
        observations=[
            ObservationItem(finding="Runway", location="center", confidence="high", evidence="High-reflectance tarmac corridor"),
            ObservationItem(finding="Fuel depot", location="lower-left", confidence="low", evidence="Circular tanks"),
        ],
    )

    service.gemini.analyze = AsyncMock(return_value=gemini_out)
    service.glm.analyze = AsyncMock(return_value=glm_out)

    result = await service.analyze_image(
        image_data=sample_data_url,
        user_query="Identify airport features",
        provider="ensemble",
    )

    assert result["success"] is True
    assert "ensemble" in result["provider_used"]
    assert len(result["consensus_findings"]) >= 1
    assert result["consensus_findings"][0]["finding"] == "Runway"
    assert len(result["provider_disagreements"]) >= 1


def test_fastapi_analyze_endpoint(sample_data_url: str, mock_gemini_json_response: str):
    from fastapi.testclient import TestClient
    from main import app, vision_service

    parsed_mock = parse_structured_response(mock_gemini_json_response, query="Find berths")
    with patch.object(vision_service.gemini, "analyze", AsyncMock(return_value=parsed_mock)):
        client = TestClient(app)
        res = client.post(
            "/analyze",
            json={
                "imageDataUrl": sample_data_url,
                "user_query": "Find berths and port structures",
                "provider": "gemini",
                "analysis_mode": "infrastructure",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["provider"] == "gemini"
        assert "observations" in data
        assert len(data["observations"]) == 2
        assert "analysis" in data  # backward-compatible legacy key
        assert "answer" in data["analysis"]
