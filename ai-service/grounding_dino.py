import torch

from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


MODEL_ID = "IDEA-Research/grounding-dino-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
print("Loading model:", MODEL_ID)


processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID
).to(device)

model.eval()


# ------------------------------------------------
# Tuning constants
# ------------------------------------------------

# Minimum score to keep a detection at all.
# Raised from 0.24 to filter out weak hits.
BOX_THRESHOLD = 0.35

# Minimum text-alignment score.
# Raised from 0.20 to reduce hallucination.
TEXT_THRESHOLD = 0.28

# Hard post-filter: drop anything below this
# even if it passed the model thresholds.
MIN_CONFIDENCE = 0.35

# Maximum fraction of a tile that a single box
# may cover (reduced from 0.18 to 0.12).
MAX_AREA_RATIO = 0.12

# NMS IoU threshold for duplicate removal
# (tightened from 0.35 to 0.25).
NMS_IOU_THRESHOLD = 0.25


# ------------------------------------------------
# Class families for geometry validation
# ------------------------------------------------

# Labels that should produce long thin boxes
# (roads, bridges, runways).
ELONGATED_LABELS = {
    "road", "street", "highway", "pathway",
    "paved road", "roadway", "path",
    "bridge", "overpass", "viaduct", "flyover",
    "runway",
}

# Labels that represent point-like compact objects
# (vehicles, cars, trucks).
COMPACT_LABELS = {
    "car", "vehicle", "truck", "bus", "van",
    "lorry", "automobile",
}

# Labels that represent large elongated watercraft.
SHIP_LABELS = {
    "ship", "vessel", "cargo ship", "tanker",
    "boat", "freighter", "ferry",
}

# Labels that represent compact rooftop structures.
BUILDING_LABELS = {
    "building", "structure", "rooftop",
    "warehouse", "factory", "house",
    "residential building",
}

# Labels that represent aircraft.
AIRCRAFT_LABELS = {
    "aircraft", "airplane", "jet", "plane",
    "helicopter",
}


def _classify_label(label: str) -> str:
    """Return the geometry family of a detected label."""
    lower = label.lower().strip()
    for word in ELONGATED_LABELS:
        if word in lower:
            return "elongated"
    for word in COMPACT_LABELS:
        if word in lower:
            return "compact"
    for word in SHIP_LABELS:
        if word in lower:
            return "ship"
    for word in BUILDING_LABELS:
        if word in lower:
            return "building"
    for word in AIRCRAFT_LABELS:
        if word in lower:
            return "aircraft"
    return "generic"


# ------------------------------------------------
# IOU
# ------------------------------------------------

def box_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0, x2 - x1)
    intersection_h = max(0, y2 - y1)
    intersection = intersection_w * intersection_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


# ------------------------------------------------
# NMS-style duplicate removal
# ------------------------------------------------

def remove_duplicates(
    detections,
    iou_threshold=NMS_IOU_THRESHOLD,
):
    """
    Sort by confidence (highest first), then greedily keep
    detections that do not heavily overlap with already-kept
    boxes of the same label.
    """
    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    kept = []

    for detection in detections:

        duplicate = False

        for existing in kept:

            # Only compare same-class boxes
            if (
                detection["label"].lower()
                !=
                existing["label"].lower()
            ):
                continue

            iou = box_iou(
                detection["box"],
                existing["box"],
            )

            if iou > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept


# ------------------------------------------------
# Generic geometry guard
# ------------------------------------------------

def valid_box(
    box,
    image_width,
    image_height,
):
    """
    Reject boxes that are clearly noise regardless
    of the detected class.
    """
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return False

    box_area = width * height
    image_area = image_width * image_height

    area_ratio = box_area / image_area

    # Reject extremely tiny detections
    if width < 10 or height < 10:
        return False

    # Reject boxes covering too large a fraction
    # of the tile (reduced from 0.18 to 0.12)
    if area_ratio > MAX_AREA_RATIO:
        return False

    # Reject unusually wide boxes
    if width > image_width * 0.60:
        return False

    # Reject unusually tall boxes
    if height > image_height * 0.80:
        return False

    return True


# ------------------------------------------------
# Class-specific geometry guard
# ------------------------------------------------

def class_valid_box(
    box,
    label: str,
    image_width,
    image_height,
):
    """
    Additional per-class checks that catch the most
    common false-positive patterns in satellite imagery.

    Returns False to discard a detection that passes
    the generic filter but violates class-specific
    shape expectations.
    """
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return False

    # Aspect ratio: width / height
    # > 1  → wider than tall
    # < 1  → taller than wide
    aspect = width / height if height > 0 else 0

    box_area = width * height
    image_area = image_width * image_height
    area_ratio = box_area / image_area

    family = _classify_label(label)

    # --------------------------------------------------
    # Roads / Bridges / Runways
    # Must be elongated (aspect >= 1.8 OR <= 0.55)
    # A ship-shaped box in water would be boxy or
    # moderately elongated — that is rejected here.
    # --------------------------------------------------
    if family == "elongated":

        is_horizontal = aspect >= 1.8
        is_vertical = aspect <= 0.55

        if not (is_horizontal or is_vertical):
            print(
                f"  [FILTER] Rejected {label!r}: "
                f"aspect={aspect:.2f} not elongated enough "
                f"for a road/bridge"
            )
            return False

        # Roads should not be huge blobs
        if area_ratio > 0.10:
            print(
                f"  [FILTER] Rejected {label!r}: "
                f"area_ratio={area_ratio:.3f} too large"
            )
            return False

    # --------------------------------------------------
    # Vehicles / Cars / Trucks
    # Must be small and compact.
    # --------------------------------------------------
    elif family == "compact":

        if area_ratio > 0.025:
            print(
                f"  [FILTER] Rejected {label!r}: "
                f"area_ratio={area_ratio:.3f} too large for a vehicle"
            )
            return False

        # Not wildly elongated (not a road misdetected as car)
        if aspect > 4.0 or aspect < 0.25:
            print(
                f"  [FILTER] Rejected {label!r}: "
                f"aspect={aspect:.2f} too extreme for a vehicle"
            )
            return False

    # --------------------------------------------------
    # Ships / Vessels
    # Typically elongated (bow to stern).
    # Not overly small, not absurdly large.
    # --------------------------------------------------
    elif family == "ship":

        # Ships take up a reasonable patch of a tile
        if area_ratio < 0.003:
            print(
                f"  [FILTER] Rejected {label!r}: "
                f"area_ratio={area_ratio:.4f} too small for a ship"
            )
            return False

        if area_ratio > MAX_AREA_RATIO:
            print(
                f"  [FILTER] Rejected {label!r}: "
                f"area_ratio={area_ratio:.3f} too large for a ship"
            )
            return False

    # --------------------------------------------------
    # Buildings
    # Should be roughly square or compact,
    # not extremely elongated.
    # --------------------------------------------------
    elif family == "building":

        if aspect > 5.0 or aspect < 0.2:
            print(
                f"  [FILTER] Rejected {label!r}: "
                f"aspect={aspect:.2f} too extreme for a building"
            )
            return False

    # --------------------------------------------------
    # Aircraft
    # Moderate size, somewhat elongated.
    # --------------------------------------------------
    elif family == "aircraft":

        if area_ratio > 0.06:
            print(
                f"  [FILTER] Rejected {label!r}: "
                f"area_ratio={area_ratio:.3f} too large for aircraft"
            )
            return False

    return True


# ------------------------------------------------
# Grounding DINO on one tile
# ------------------------------------------------

def run_model_on_image(
    image: Image.Image,
    prompt: str,
    threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
):
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = (
        processor
        .post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[
                image.size[::-1]
            ],
        )
    )

    result = results[0]

    labels = result.get(
        "text_labels",
        result.get("labels", []),
    )

    detections = []

    for box, score, label in zip(
        result["boxes"],
        result["scores"],
        labels,
    ):
        confidence = float(score)

        # Hard confidence floor — discard weak hits
        if confidence < MIN_CONFIDENCE:
            continue

        box_list = [
            float(v)
            for v in box.tolist()
        ]

        # Generic geometry check
        if not valid_box(
            box_list,
            image.width,
            image.height,
        ):
            continue

        # Class-specific geometry check
        if not class_valid_box(
            box_list,
            str(label),
            image.width,
            image.height,
        ):
            continue

        detections.append(
            {
                "label": str(label),
                "confidence": confidence,
                "box": box_list,
            }
        )

    return detections


# ------------------------------------------------
# Create overlapping tiles
# ------------------------------------------------

def create_tiles(
    image,
    tile_size=512,
    overlap=96,
):
    """
    Split image into overlapping tiles.
    Overlap reduced to 96 px (from 128) to
    reduce redundant area while still ensuring
    objects near tile boundaries are captured.
    """
    width, height = image.size
    step = tile_size - overlap

    tiles = []

    y = 0

    while y < height:

        x = 0

        bottom = min(y + tile_size, height)
        top = max(0, bottom - tile_size)

        while x < width:

            right = min(x + tile_size, width)
            left = max(0, right - tile_size)

            crop = image.crop((left, top, right, bottom))

            tiles.append((crop, left, top))

            if right >= width:
                break

            x += step

        if bottom >= height:
            break

        y += step

    return tiles


# ------------------------------------------------
# Main detector
# ------------------------------------------------

def detect_objects(
    image: Image.Image,
    prompt: str,
):
    image = image.convert("RGB")

    full_width, full_height = image.size

    print(
        f"\nImage size: {full_width}x{full_height}"
    )
    print(
        f"Prompt    : {prompt!r}"
    )
    print(
        f"Thresholds: box={BOX_THRESHOLD}, "
        f"text={TEXT_THRESHOLD}, "
        f"min_conf={MIN_CONFIDENCE}"
    )

    tiles = create_tiles(
        image,
        tile_size=512,
        overlap=96,
    )

    print(f"Tiles     : {len(tiles)}")

    all_detections = []

    for index, (
        tile,
        offset_x,
        offset_y,
    ) in enumerate(tiles):

        print(
            f"\n[Tile {index + 1}/{len(tiles)}] "
            f"offset=({offset_x},{offset_y})"
        )

        tile_detections = run_model_on_image(
            tile,
            prompt,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        print(
            f"  Raw detections: {len(tile_detections)}"
        )

        for detection in tile_detections:

            x1, y1, x2, y2 = detection["box"]

            detection["box"] = [
                x1 + offset_x,
                y1 + offset_y,
                x2 + offset_x,
                y2 + offset_y,
            ]

            all_detections.append(detection)

    print(
        f"\nBefore NMS: {len(all_detections)}"
    )

    final_detections = remove_duplicates(
        all_detections,
        iou_threshold=NMS_IOU_THRESHOLD,
    )

    print(
        f"After NMS : {len(final_detections)}"
    )

    return final_detections