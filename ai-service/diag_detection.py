"""Temporary diagnostic: run the full detect_objects pipeline on dense urban imagery."""
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "backend"))

from PIL import Image
from grounding_dino import detect_objects

IMG = HERE.parent / "public" / "samples" / "urban.jpg"
image = Image.open(IMG).convert("RGB")
print(f"Image: {IMG.name} {image.size}", flush=True)

t0 = time.time()
dets, meta = detect_objects(
    image=image,
    prompt="Detect buildings in this image.",
    enable_segmentation=False,   # measure detection quality first
    return_tiling_metadata=True,
)
t1 = time.time()

tiles = meta  # detect_objects returns the tiling metadata dict directly
print(f"\nTiling: enabled={tiles.get('enabled')} tiles={tiles.get('tile_count')} tile_size={tiles.get('tile_size')}")
for t in tiles.get("tiles", []):
    print(f"  {t['tile_id']}: {t['detections_count']} dets")
print(f"Dedup: {meta.get('deduplication')}")
print(f"Verification: {meta.get('verification')}")
print(f"\nTotal time (no SAM2): {t1 - t0:.1f}s")
print(f"Final detections: {len(dets)}")
for d in dets:
    v = d.get("verification", {})
    print(f"  {d['id']:8s} {d['label']:10s} score={d['score']:.3f} box={d['box']} "
          f"vscore={v.get('score')} alt={v.get('best_alternative')}")
print("\nDone.")
