"""
Satellite and Remote-Sensing Vision Prompt Builder
Shared prompt generation architecture for Gemini and GLM providers.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
from .context_builder import build_vision_context


SYSTEM_INSTRUCTION = """You are an expert remote-sensing and geospatial image analyst.
You analyze satellite and aerial imagery using only visible evidence and any machine-generated metadata supplied to you.
Your task is to provide technically useful, reliable observations while strictly avoiding unsupported claims.

PRIMARY GROUNDING RULES:
1. Treat detection and segmentation results as machine-generated evidence.
   - Use them directly when answering the user's question.
   - Do NOT independently invent objects or counts that conflict with the detection pipeline.
   - For count queries (e.g. "How many buildings?"): PREFER the machine detection count over visual guesswork.
   - For location queries (e.g. "Where are the buildings?"): USE the bounding box and mask spatial distribution.
   - If visual inspection disagrees with supplied detections, explicitly report that uncertainty instead of overriding without evidence.
2. For flood danger, hazard, or risk inquiries:
   - Use the water mask extent, its spatial proximity to detected structures or built-up land cover, and any supplied elevation/terrain metadata.
   - Phrase conclusions strictly as possible exposure or proximity (e.g. "structures in the lower-right are in close proximity to the delineated water body, indicating potential flood exposure during high-water events"), NOT as guaranteed flood risk or confirmed disaster.
3. Strictly separate:
   - observed objects (what is visually delineated and detected)
   - calculated statistics (counts, percentages, and spatial metrics)
   - model interpretation (analytical context and domain assessment)
4. Treat the image as remote-sensing / Earth observation imagery, not a ground-level photograph.
5. For bi-temporal change analysis:
   - Strictly quote machine-calculated land-cover percentage point deltas (e.g. '+5.2 percentage points') and object change counts. NEVER invent contradictory numbers or percentages.
   - You MUST explicitly distinguish:
     a. OBSERVED CHANGE: Factual physical differences confirmed by detection, masks, or radar backscatter (e.g. 'Water coverage increased by 5.2 percentage points, concentrated primarily in the lower-right quadrant').
     b. POSSIBLE CAUSE: Plausible analytical hypotheses or potential drivers (e.g. 'Likely caused by seasonal river surge or heavy precipitation runoff; ground hydrological data required for confirmation').
   - When SAR radar backscatter shifts are present, reference them as cloud-penetrating structural/inundation evidence.
6. Never claim exact geographic coordinates, object identity, building ownership, event cause, historical date, physical distance, area, or scale unless explicitly provided in metadata.
7. If image resolution is insufficient to identify small objects, explicitly report that limitation.
8. Use standard spatial localization terms (left side, right side, center-left, center-right, upper-left, upper-right, center, lower-left, lower-right, widespread).
9. Prioritize evidence directly answering the user's question.
10. Keep answers concise, structured, evidence-grounded, and technically informative."""


ANALYSIS_MODE_GUIDELINES: Dict[str, str] = {
    "general": """ANALYSIS MODE: GENERAL LAND-COVER & SCENE UNDERSTANDING
- Identify dominant land-cover classes (built-up, vegetation, water, bare soil, transport).
- Note major spatial patterns, development intensity, and distinctive terrain features.
- Evaluate overall visual texture, spectral reflectance contrasts, and surface distribution.""",

    "objects": """ANALYSIS MODE: OBJECT DETECTION & ENUMERATION
- Focus on discrete, localized structures or features matching the query (e.g. buildings, vessels, aircraft, storage tanks, bridges).
- Describe spatial clustering, approximate density, and spatial distribution across image quadrants.
- Categorize confidence based on clarity of geometric shapes, roof outlines, and contrast against background.""",

    "changes": """ANALYSIS MODE: BI-TEMPORAL MULTIMODAL CHANGE DETECTION
- Focus primarily on differences between Image 1 (BEFORE/T1) and Image 2 (AFTER/T2).
- Quote supplied machine deltas for land-cover transitions (water expansion/reduction, urban growth, vegetation loss).
- Report object dynamics (appeared, disappeared, modified, persisted) with spatial quadrants.
- Cite SAR radar scattering changes where optical imagery is cloudy or ambiguous.
- MANDATORY: Distinguish OBSERVED CHANGE (what the imagery shows) from POSSIBLE CAUSE (hypothesized explanation).""",

    "urban": """ANALYSIS MODE: URBAN & SETTLEMENT PATTERNS
- Analyze building density, roof outlines, and spatial layout (organic vs planned grid).
- Identify road connectivity, intersections, parking or paved areas, and impervious surface ratio.
- Note transitions between high-density built-up zones, commercial/industrial structures, and open or residential space.""",

    "agriculture": """ANALYSIS MODE: AGRICULTURAL & RURAL MONITORING
- Identify field boundaries, crop parcels, rectangular cultivation plots, and irrigation channels.
- Note variations in vegetation vigour, fallow/bare fields, freshly tilled soil, and tree stands.
- Look for visible patterns of water access, access tracks, and rural farm infrastructure.""",

    "water": """ANALYSIS MODE: HYDROLOGY & WATER EXTENT
- Delineate water bodies (rivers, lakes, reservoirs, ponds, canals, coastal waters).
- Observe turbidity, sediment plumes, algae bloom indications, or water level fluctuations along shorelines.
- Distinguish permanent deep water from shallow wetlands, drainage channels, or tidal mudflats.""",

    "infrastructure": """ANALYSIS MODE: TRANSPORTATION & INFRASTRUCTURE
- Map linear features: highways, primary roads, rail lines, runways, bridges, powerline corridors.
- Assess connectivity, road surface condition hints, bridge crossings over water/terrain, and junction complexity.
- Inspect large industrial facilities, port docks, logistic depots, and energy installations.""",

    "disaster": """ANALYSIS MODE: DISASTER & DAMAGE ASSESSMENT (HIGH CAUTION)
- Look for visible anomalies: standing water outside normal channels, structural collapse, debris accumulation, ground scouring, burn scars, or blocked roads.
- Distinguish normal seasonal water from anomalous inundation.
- Never claim disaster severity or causality without verifiable visual markers; clearly express uncertainties.""",

    "landcover": """ANALYSIS MODE: LAND-COVER CLASSIFICATION
- Categorize visible surfaces into standard classes: built-up, tree canopy / forest, cropland / pasture, barren soil / rock, surface water, road / paved.
- Describe relative dominance qualitatively (e.g. predominantly forested with scattered clearings).
- Avoid fake exact percentages unless supported by external segmentation masks.""",

    "custom": """ANALYSIS MODE: CUSTOM FOCUSED INQUIRY
- Strictly answer the user query based on visible remote-sensing evidence in the image.""",
}


JSON_OUTPUT_SPEC = """
OUTPUT FORMAT:
You must return a valid JSON object matching this exact schema:
{
  "answer": "Detailed, direct answer to the user query referencing specific image evidence and detector counts",
  "summary": "Short direct answer summarizing the findings in 1-2 concise sentences",
  "answer_to_query": "Detailed, direct answer to the user query (mirror of answer)",
  "evidence": {
    "detections_used": true,
    "segmentation_used": true,
    "land_cover_used": true
  },
  "calculated_statistics": {
    "object_counts": {},
    "land_cover_percentages": {}
  },
  "observed_changes": [
    "Confirmed physical changes between T1 and T2 quoting exact percentages or counts"
  ],
  "possible_causes": [
    "Plausible hypotheses explaining the observed changes (clearly distinguished from observed facts)"
  ],
  "observations": [
    {
      "finding": "Concise title of observed feature or pattern",
      "location": "upper-left | upper-right | center | lower-left | lower-right | widespread | left side | right side",
      "confidence": "high | medium | low",
      "evidence": "Specific visible features (e.g., high-reflectance rectangular roofs with visible shadow orientation, dark linear channel with sediment plume)"
    }
  ],
  "uncertainties": [
    "Limitations regarding sensor resolution, cloud shadows, or discrepancies between visual appearance and detector hints"
  ],
  "model_notes": {
    "used_detection_context": false,
    "used_change_context": false
  }
}
Return raw JSON ONLY. Do not prepend conversational filler. Ensure valid JSON syntax."""


def format_detection_context(detection_context: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Summarizes machine-generated Grounding DINO detection results into a spatial summary.
    Avoids dumping overwhelming raw pixel coordinates.
    """
    if not detection_context:
        return None

    raw_detections = detection_context.get("detections", [])
    if not raw_detections and not isinstance(raw_detections, list):
        return None

    total_count = detection_context.get("count", len(raw_detections))
    img_w = detection_context.get("width", 640) or 640
    img_h = detection_context.get("height", 640) or 640

    if not raw_detections:
        return f"Grounding DINO detector ran but found 0 candidate objects matching the prompt."

    # Group by label and compute spatial distribution
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for d in raw_detections:
        lbl = str(d.get("label", "object")).strip()
        by_label.setdefault(lbl, []).append(d)

    lines: List[str] = [
        f"MACHINE-GENERATED DETECTION HINTS (Grounding DINO):",
        f"Total candidate detections: {total_count}"
    ]

    for lbl, items in by_label.items():
        confidences = [float(i.get("confidence", 0.0)) for i in items if i.get("confidence") is not None]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.85

        # Spatial quadrant estimation
        quadrants: Dict[str, int] = {"upper-left": 0, "upper-right": 0, "lower-left": 0, "lower-right": 0, "center": 0}
        for i in items:
            box = i.get("box")
            if box and len(box) >= 4:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                # Check center region (25% to 75%)
                if 0.3 * img_w <= cx <= 0.7 * img_w and 0.3 * img_h <= cy <= 0.7 * img_h:
                    quadrants["center"] += 1
                elif cx < img_w / 2 and cy < img_h / 2:
                    quadrants["upper-left"] += 1
                elif cx >= img_w / 2 and cy < img_h / 2:
                    quadrants["upper-right"] += 1
                elif cx < img_w / 2 and cy >= img_h / 2:
                    quadrants["lower-left"] += 1
                else:
                    quadrants["lower-right"] += 1

        top_quads = [q for q, cnt in sorted(quadrants.items(), key=lambda x: x[1], reverse=True) if cnt > 0]
        loc_desc = ", ".join(top_quads[:2]) if top_quads else "distributed across scene"

        lines.append(f"- {len(items)} '{lbl}' candidates (avg confidence {avg_conf:.2f}), located predominantly in {loc_desc}")

    lines.append("INSTRUCTION: Use these detections as supporting machine-generated hints. Independently inspect the image and do not blindly repeat detections if visual evidence is inconsistent.")
    return "\n".join(lines)


def format_change_context(change_context: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Summarizes bi-temporal radiometric change detection metadata.
    Instructs model to explain WHAT changed without hallucinating causes.
    """
    if not change_context:
        return None

    lines: List[str] = ["MACHINE-GENERATED BI-TEMPORAL CHANGE CONTEXT:"]

    change_pct = change_context.get("changePercentage") or change_context.get("change_percentage")
    if change_pct is not None:
        lines.append(f"- Estimated surface radiometric change: {change_pct}% of the compared area")

    summary = change_context.get("summary")
    if summary and isinstance(summary, dict):
        new_cnt = summary.get("newCount", summary.get("new_count", 0))
        rem_cnt = summary.get("removedCount", summary.get("removed_count", 0))
        mod_cnt = summary.get("modifiedCount", summary.get("modified_count", 0))
        lines.append(f"- Categorized object changes: {new_cnt} newly appeared, {rem_cnt} absent/removed, {mod_cnt} altered")

    time_diff = change_context.get("timeDifference") or change_context.get("time_difference")
    if time_diff:
        lines.append(f"- Temporal baseline interval: {time_diff}")

    lines.append("INSTRUCTION: Explain WHAT visible features or surfaces appear altered between the timestamps. Do not invent ungrounded real-world causes.")
    return "\n".join(lines)


def build_satellite_analysis_prompt(
    user_query: str,
    detection_context: Optional[Dict[str, Any]] = None,
    change_context: Optional[Dict[str, Any]] = None,
    image_metadata: Optional[Dict[str, Any]] = None,
    analysis_mode: str = "general",
    has_second_image: bool = False,
    spatial_tile_label: Optional[str] = None,
    segmentation_summary: Optional[Dict[str, Any]] = None,
    land_cover: Optional[Dict[str, Any]] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[str, str]:
    """
    Builds separated (system_instruction, user_content) prompt components.
    Safely embeds user query and grounded machine evidence to prevent prompt injection.
    """
    mode_key = (analysis_mode or "general").lower().strip()
    mode_instructions = ANALYSIS_MODE_GUIDELINES.get(mode_key, ANALYSIS_MODE_GUIDELINES["general"])

    user_sections: List[str] = []

    # 1. Tile Context (if tiled mode)
    if spatial_tile_label:
        user_sections.append(f"IMAGE SUB-TILE FOCUS: You are analyzing the [{spatial_tile_label.upper()}] quadrant of the overall satellite scene.")

    # 2. Comparative Bi-Temporal Context
    if has_second_image:
        user_sections.append(
            "IMAGE PAIR COMPARISON:\n"
            "- Image 1 represents the BEFORE / Historical reference state.\n"
            "- Image 2 represents the AFTER / Latest observation state.\n"
            "- Compare corresponding spatial areas carefully. Disregard sensor variations and transient cloud shadows unless relevant."
        )

    # 3. Metadata Context
    if image_metadata and isinstance(image_metadata, dict):
        meta_items = [f"  - {k}: {v}" for k, v in image_metadata.items() if v is not None]
        if meta_items:
            user_sections.append("IMAGE METADATA:\n" + "\n".join(meta_items))

    # 4. Machine Evidence Context (Detections, Segmentation, Land-Cover, Changes)
    if detection_context or segmentation_summary or land_cover or change_context:
        v_ctx = build_vision_context(
            detections=detection_context,
            segmentation_summary=segmentation_summary,
            land_cover=land_cover,
            change_detection=change_context,
            image_size=image_size,
        )
        if v_ctx.get("summary_text"):
            user_sections.append(v_ctx["summary_text"])
    elif detection_context:
        dino_formatted = format_detection_context(detection_context)
        if dino_formatted:
            user_sections.append(dino_formatted)

    # 5. Mode-specific focus
    user_sections.append(mode_instructions)

    # 6. User Query (Strictly Isolated)
    clean_query = user_query.strip() if user_query else "Provide a comprehensive remote-sensing assessment of this imagery."
    user_sections.append(
        "--------------------------------------------------\n"
        f"USER ANALYTICAL QUERY:\n\"{clean_query}\"\n"
        "--------------------------------------------------\n"
        "Analyze the imagery to specifically answer this query using visible remote-sensing evidence."
    )

    # 7. Schema reminder
    user_sections.append(JSON_OUTPUT_SPEC)

    system_prompt = SYSTEM_INSTRUCTION
    user_prompt = "\n\n".join(user_sections)

    return system_prompt, user_prompt
