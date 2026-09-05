"""
Pytest configuration and shared test fixtures for SatQuery Vision Pipeline.
"""

import io
import base64
import pytest
from PIL import Image


@pytest.fixture
def sample_test_image_bytes() -> bytes:
    """Generates a small test JPEG image in memory."""
    img = Image.new("RGB", (400, 300), color=(70, 130, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_data_url(sample_test_image_bytes: bytes) -> str:
    """Generates a base64 data URL from sample test image."""
    b64 = base64.b64encode(sample_test_image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@pytest.fixture
def mock_gemini_json_response() -> str:
    """Realistic satellite analysis JSON as expected from Gemini."""
    return """{
      "summary": "Satellite scene shows dense commercial port infrastructure and deep water channels.",
      "answer_to_query": "Multiple vessel berths and warehouse structures are visible across the central waterfront.",
      "observations": [
        {
          "finding": "Berthing pier structures",
          "location": "center",
          "confidence": "high",
          "evidence": "Linear concrete finger piers extending into navigable water with visible shadow casting"
        },
        {
          "finding": "Industrial storage buildings",
          "location": "upper-right",
          "confidence": "high",
          "evidence": "Regular rectangular high-albedo roofs arranged along primary access routes"
        }
      ],
      "uncertainties": [
        "Small vessel outlines near outer channel cannot be confirmed due to 10m spatial resolution limit."
      ],
      "model_notes": {
        "used_detection_context": true,
        "used_change_context": false
      }
    }"""
