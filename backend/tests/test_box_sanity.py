"""
Giant-box / false-fallback-box regression tests.

Covers:
1. Giant low-confidence boxes are rejected (giant_low_confidence).
2. Giant border-hugging boxes are rejected (giant_border_hugging).
3. Valid discrete object boxes are accepted (building regression guard).
4. Large high-confidence boxes that do not hug borders are accepted.
5. Zero-area and non-numeric boxes are rejected.
6. filter_and_format_detections applies the sanity gate and tags source /
   geometry_quality metadata.
7. LLM/VLM text can never become bbox geometry ("Visual assessment based on
   query" policy).
8. Zero detections produce empty geometry (no synthetic fallback box).
"""

import pytest

from services.detection.vocabulary import (
    count_edge_touches,
    filter_and_format_detections,
    passes_box_sanity,
)
from services.detection.config import (
    LARGE_BOX_MIN_CONFIDENCE,
    MAX_BOX_AREA_RATIO,
    MAX_EDGE_TOUCHES_FOR_LARGE_BOX,
)
from services.vision.response_parser import (
    ObservationItem,
    SatelliteAnalysisStructured,
    parse_structured_response,
    to_legacy_analysis_result,
)


# =====================================================================
# 1-5. Giant-box sanity rule
# =====================================================================

class TestGiantBoxSanity:
    IMG_W, IMG_H = 1024, 1024

    def test_normal_building_box_accepted(self):
        """Small high-confidence building box must always pass."""
        ok, quality, reason = passes_box_sanity(
            [100, 100, 200, 200], 0.93, self.IMG_W, self.IMG_H, "building"
        )
        assert ok is True
        assert quality == "good"
        assert reason is None

    def test_giant_low_confidence_box_rejected(self):
        """Box covering ~73% of image with 0.75 confidence must be rejected."""
        box = [51, 51, 911, 911]
        ok, quality, reason = passes_box_sanity(
            box, 0.75, self.IMG_W, self.IMG_H, "water body"
        )
        assert ok is False
        assert quality == "rejected"
        assert reason is not None and reason.startswith("giant_low_confidence")

    def test_giant_border_hugging_box_rejected(self):
        """Box covering most of the image and touching all 4 borders is rejected
        even with high confidence (classic full-frame fallback artifact)."""
        box = [2, 2, 1022, 1022]
        ok, quality, reason = passes_box_sanity(
            box, 0.95, self.IMG_W, self.IMG_H, "water body"
        )
        assert ok is False
        assert quality == "rejected"
        assert reason is not None and reason.startswith("giant_border_hugging")

    def test_large_high_confidence_interior_box_accepted(self):
        """A genuinely large object (e.g. big reservoir) that is high-confidence
        and interior may pass the combined escape checks."""
        box = [200, 200, 1000, 1000]
        ok, quality, reason = passes_box_sanity(
            box, 0.9, self.IMG_W, self.IMG_H, "lake"
        )
        assert ok is True
        assert quality == "good"

    def test_zero_area_box_rejected(self):
        ok, quality, reason = passes_box_sanity(
            [100, 100, 100, 200], 0.9, self.IMG_W, self.IMG_H, "building"
        )
        assert ok is False
        assert quality == "rejected"
        assert reason == "zero_area"

    def test_non_numeric_box_rejected(self):
        ok, quality, _ = passes_box_sanity(
            ["a", 1, 2, 3], 0.9, self.IMG_W, self.IMG_H, "building"
        )
        assert ok is False
        assert quality == "rejected"

    def test_config_thresholds_are_central(self):
        assert 0 < MAX_BOX_AREA_RATIO < 1
        assert 0 < LARGE_BOX_MIN_CONFIDENCE <= 1
        assert MAX_EDGE_TOUCHES_FOR_LARGE_BOX >= 0

    def test_count_edge_touches(self):
        assert count_edge_touches([2, 2, 1022, 1022], self.IMG_W, self.IMG_H) == 4
        assert count_edge_touches([0, 0, 300, 300], self.IMG_W, self.IMG_H) == 2
        assert count_edge_touches([300, 300, 700, 700], self.IMG_W, self.IMG_H) == 0


# =====================================================================
# 6. Sanity gate inside filter_and_format_detections
# =====================================================================

class TestFilterPipelineSanity:
    def test_giant_detection_filtered_with_metadata(self):
        raw = [
            # Giant low-confidence artifact -> rejected by sanity gate
            {"label": "water body", "score": 0.72, "box": [10, 10, 1010, 1010]},
            # Normal compact building -> kept
            {"label": "building", "score": 0.9, "box": [100, 100, 200, 200]},
        ]
        out = filter_and_format_detections(raw, width=1024, height=1024)
        labels = [d["label"] for d in out]
        assert "building" in labels
        assert "water body" not in labels
        for d in out:
            assert d["source"] == "grounding_dino"
            assert d["geometry_quality"] == "good"

    def test_no_detections_returns_empty_no_fallback(self):
        out = filter_and_format_detections([], width=1024, height=1024)
        assert out == []


# =====================================================================
# 7-8. LLM/VLM text can never become geometry
# =====================================================================

FALLBACK_TEXT = (
    "The image shows a large coastal water body in the west and dense "
    "vegetation in the east. Urban density is moderate."
)


class TestNoSyntheticGeometryFromLLM:
    def _structured(self) -> SatelliteAnalysisStructured:
        return SatelliteAnalysisStructured(
            summary="Coastal scene",
            answer_to_query="Coastal water and vegetation detected visually.",
            observations=[
                ObservationItem(
                    finding="Large water body",
                    location="widespread",
                    confidence="medium",
                    evidence="Blue region covering west",
                ),
                ObservationItem(
                    finding="Dense vegetation",
                    location="upper-right",
                    confidence="high",
                    evidence="Green canopy",
                ),
            ],
        )

    def test_visual_assessment_never_becomes_region(self):
        """'Visual assessment based on query' (fallback observation) must never
        produce a spatial overlay region."""
        structured = parse_structured_response(
            FALLBACK_TEXT, query="identify water body"
        )
        findings = [o.finding for o in structured.observations]
        assert "Visual assessment based on query" in findings

        legacy = to_legacy_analysis_result(structured)
        assert legacy["regions"] == []

    def test_llm_observations_produce_no_rects_or_fake_confidence(self):
        legacy = to_legacy_analysis_result(self._structured())
        assert legacy["regions"] == []
        assert legacy["objectsDetected"] == []
        assert legacy["confidence"] == 0.0

    def test_non_json_model_response_still_yields_text_analysis(self):
        """AI analysis must still work (answer preserved) when the model
        returns plain text and zero boxes exist."""
        structured = parse_structured_response(FALLBACK_TEXT, query="analyze this image")
        legacy = to_legacy_analysis_result(structured)
        assert legacy["answer"]
        assert "coastal water" in legacy["answer"].lower()
        assert legacy["regions"] == []

    def test_land_cover_passthrough_unaffected(self):
        real_lc = {
            "available": True,
            "measured_from_masks": True,
            "coverage": [
                {"class": "water", "coverage": 0.39, "color": "#06b6d4"},
                {"class": "vegetation", "coverage": 0.51, "color": "#10b981"},
                {"class": "built-up", "coverage": 0.10, "color": "#ef4444"},
            ],
        }
        legacy = to_legacy_analysis_result(self._structured(), land_cover_result=real_lc)
        # Mask-measured land-cover statistics are real CV output -> kept.
        assert legacy["measured_from_masks"] is True
        assert legacy["coverage"] == real_lc["coverage"]
        # But still no synthetic bboxes.
        assert legacy["regions"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
