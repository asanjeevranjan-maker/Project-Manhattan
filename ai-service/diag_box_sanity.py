"""End-to-end validation of the false-giant-box fixes (run with ai-service venv).

Exercises the EXACT import path the running FastAPI service uses:
  ai-service/grounding_dino.py -> backend/services/detection/vocabulary.py
No model is loaded — pure pipeline-logic validation.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "backend"))

from dino_vocabulary import filter_and_format_detections, passes_box_sanity
from services.vision.response_parser import (
    parse_structured_response,
    to_legacy_analysis_result,
)

print("=" * 60)
print("TEST 1: Giant-box sanity rule")
print("=" * 60)
raw = [
    {"label": "water body", "score": 0.72, "box": [10, 10, 1010, 1010]},  # giant, low conf
    {"label": "water body", "score": 0.95, "box": [2, 2, 1022, 1022]},  # giant, hugs borders
    {"label": "vegetation", "score": 0.68, "box": [0, 0, 1024, 800]},  # giant, low conf
    {"label": "building", "score": 0.93, "box": [100, 100, 200, 200]},  # legit
    {"label": "building", "score": 0.88, "box": [400, 400, 520, 560]},  # legit
]
out = filter_and_format_detections(raw, width=1024, height=1024)
print(f"Input: {len(raw)} raw -> {len(out)} detections")
for d in out:
    print(
        f"  KEPT  {d['label']:<12} conf={d['confidence']:.2f} "
        f"box={d['box']} source={d['source']} quality={d['geometry_quality']}"
    )
labels = [d["label"] for d in out]
assert "building" in labels, "legit buildings lost!"
assert "water body" not in labels and "vegetation" not in labels, "giant boxes survived!"
assert all(d["source"] == "grounding_dino" for d in out)
assert all(d["geometry_quality"] == "good" for d in out)
print("PASS: giant boxes rejected, buildings kept, metadata attached\n")

print("=" * 60)
print("TEST 2: Zero detections -> no fallback box")
print("=" * 60)
out0 = filter_and_format_detections([], width=1024, height=1024)
assert out0 == []
print("PASS: empty input returns empty list (no synthetic box)\n")

print("=" * 60)
print("TEST 3: Large valid interior box still accepted")
print("=" * 60)
ok, q, reason = passes_box_sanity([200, 200, 1000, 1000], 0.92, 1024, 1024, "lake")
print(f"accepted={ok} quality={q} reason={reason}")
assert ok is True and q == "good"
print("PASS\n")

print("=" * 60)
print("TEST 4: VLM text never becomes geometry")
print("=" * 60)
text = "The image shows a large water body in the west and vegetation in the east."
structured = parse_structured_response(text, query="identify water body")
findings = [o.finding for o in structured.observations]
print(f"Fallback observation created: {findings}")
assert "Visual assessment based on query" in findings
legacy = to_legacy_analysis_result(structured)
print(
    f"regions={legacy['regions']} objectsDetected={legacy['objectsDetected']} "
    f"confidence={legacy['confidence']}"
)
assert legacy["regions"] == []
assert legacy["objectsDetected"] == []
assert legacy["confidence"] == 0.0
assert legacy["answer"]  # AI text preserved separately
print("PASS: text stayed text; no bbox, no fabricated confidence\n")

print("=" * 60)
print("TEST 5: Direct backend import path (as grounding_dino.py uses)")
print("=" * 60)
from services.detection.vocabulary import (  # noqa: E402
    filter_and_format_detections as ffd_bknd,
)

out_bk = ffd_bknd(
    [{"label": "bridge", "score": 0.99, "box": [200, 200, 400, 400]}],
    width=1024,
    height=1024,
)
print(f"bridge box -> {len(out_bk)} kept (quality={out_bk[0]['geometry_quality']})")
assert len(out_bk) == 1 and out_bk[0]["geometry_quality"] == "good"
print("PASS\n")

print("ALL END-TO-END VALIDATIONS PASSED")
