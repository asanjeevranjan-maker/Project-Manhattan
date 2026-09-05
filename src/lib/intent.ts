// Intent classification — lightweight deterministic router.
//
// This module:
// 1. Determines what the user is asking for.
// 2. Builds a specialized prompt for Gemini / GLM.
// 3. Forces the VLM to distinguish visual evidence from
//    measurements that require computer-vision processing.
//
// Future pipeline:
//
// Gemini / GLM
//      ↓
// Visual reasoning
//      ↓
// Grounding DINO
//      ↓
// SAM 2
//      ↓
// Spectral / SAR
//      ↓
// Evidence Fusion
//      ↓
// Final measurements

import type { Intent } from './types';

interface Rule {
  intent: Intent;
  keywords: string[];
}

// ---------------------------------------------------------
// Intent rules
// ---------------------------------------------------------

const RULES: Rule[] = [
  // Flood must be checked before general water detection.
  {
    intent: 'flood_detection',
    keywords: [
      'flood',
      'flooding',
      'flooded',
      'inundat',
      'deluge',
      'overflow',
      'inundation',
    ],
  },

  // Water
  {
    intent: 'water_detection',
    keywords: [
      'water',
      'river',
      'lake',
      'reservoir',
      'pond',
      'sea',
      'ocean',
      'stream',
      'wetland',
    ],
  },

  // Buildings / urban
  {
    intent: 'building_detection',
    keywords: [
      'building',
      'buildings',
      'urban',
      'structure',
      'structures',
      'house',
      'houses',
      'settlement',
      'city',
      'town',
      'construction',
      'infrastructure',
    ],
  },

  // Vegetation
  {
    intent: 'vegetation_segmentation',
    keywords: [
      'forest',
      'vegetation',
      'tree',
      'trees',
      'canopy',
      'jungle',
      'grassland',
      'green area',
      'plantation',
      'deforest',
      'deforestation',
      'crop',
      'crops',
      'agriculture',
    ],
  },

  // Roads
  {
    intent: 'road_detection',
    keywords: [
      'road',
      'roads',
      'highway',
      'street',
      'expressway',
      'pathway',
      'lane',
      'track',
      'route',
    ],
  },

  // Change detection
  {
    intent: 'change_detection',
    keywords: [
      'change',
      'changed',
      'changes',
      'compare',
      'comparison',
      'before',
      'after',
      'difference',
      'differences',
      'new',
      'lost',
      'loss',
      'gain',
      'removed',
      'added',
    ],
  },

  // Land cover
  {
    intent: 'land_cover',
    keywords: [
      'land cover',
      'land use',
      'landcover',
      'landuse',
      'land type',
      'classify',
      'classification',
    ],
  },
];

// ---------------------------------------------------------
// Intent labels
// ---------------------------------------------------------

const INTENT_LABEL: Record<Intent, string> = {
  water_detection: 'Water Body Detection',
  building_detection: 'Building / Urban Detection',
  vegetation_segmentation: 'Vegetation Segmentation',
  change_detection: 'Change Detection',
  road_detection: 'Road Detection',
  ship_detection: 'Ship / Vessel Detection',
  aircraft_detection: 'Aircraft Detection',
  vehicle_detection: 'Vehicle / Car Detection',
  bridge_detection: 'Bridge Detection',
  image_understanding: 'Image Understanding',
  flood_detection: 'Flood Detection',
  land_cover: 'Land Cover Classification',
  other: 'General Analysis',
};

// ---------------------------------------------------------
// Classify intent
// ---------------------------------------------------------

export function classifyIntent(
  query: string
): Intent {
  const q = query
    .toLowerCase()
    .trim();

  if (!q) {
    return 'other';
  }

  for (const rule of RULES) {
    for (const keyword of rule.keywords) {
      if (q.includes(keyword)) {
        return rule.intent;
      }
    }
  }

  return 'image_understanding';
}

// ---------------------------------------------------------
// Human-readable intent label
// ---------------------------------------------------------

export function intentLabel(
  intent: Intent
): string {
  return (
    INTENT_LABEL[intent] ??
    'General Analysis'
  );
}

// ---------------------------------------------------------
// Common VLM instructions
// ---------------------------------------------------------

const COMMON_RULES = `
IMPORTANT VISUAL ANALYSIS RULES

1. Analyze ONLY what is actually visible in the image.

2. Never invent objects, locations, measurements, dates,
   coordinates, areas, or percentages.

3. Do not confuse model confidence with physical measurement
   accuracy.

4. If visual evidence is weak or ambiguous, explicitly say so.

5. Distinguish permanent features from temporary phenomena
   whenever possible.

6. Do not claim that a region is definitely flooded merely
   because it contains blue/gray water.

7. Do not calculate flood area, physical area, distance,
   water depth, or exact percentages from visual inspection
   alone.

8. Numerical area measurements must eventually come from
   segmentation, geospatial information, or other measurable
   data sources.

9. Bounding boxes generated by the VLM are APPROXIMATE.
   Never present them as exact geographic boundaries.

10. Do not fabricate geographic coordinates.

11. Every important conclusion should have visual evidence.

12. If the image resolution, cloud cover, image quality,
    viewing angle, or missing information prevents a reliable
    conclusion, report the limitation.

13. Be conservative. A false positive is worse than saying
    "insufficient visual evidence."

14. Do not assume that a satellite image contains information
    that is not visible.

15. Return the structured JSON block exactly as requested.
`;

// ---------------------------------------------------------
// Structured output schema
// ---------------------------------------------------------

const JSON_SCHEMA = `
STRUCTURED OUTPUT

After your human-readable answer, append exactly ONE fenced
JSON block using:

\`\`\`satquery
{
  "objects_detected": [],
  "confidence": 0.0,
  "coverage": [],
  "regions": []
}
\`\`\`

The JSON must be valid JSON.

Do not add comments.

Do not add trailing commas.

Do not put markdown inside the JSON.

Do not put text after the closing JSON brace.

---------------------------------------------------------
OBJECTS
---------------------------------------------------------

"objects_detected" is an array.

Each object should have:

{
  "class": "water",
  "confidence": 0.90,
  "count": 1,
  "region": "center",
  "note": "Visible river channel"
}

Valid region values:

north
south
east
west
center
widespread

Only include objects that are visually supported.

---------------------------------------------------------
CONFIDENCE
---------------------------------------------------------

"confidence" must be between 0 and 1.

It represents the quality and strength of the visual evidence.

It does NOT represent certainty about measurements.

---------------------------------------------------------
COVERAGE
---------------------------------------------------------

"coverage" contains visually estimated land-cover classes.

Example:

{
  "class": "vegetation",
  "coverage": 0.42,
  "color": "#10b981"
}

IMPORTANT:

Coverage values are VISUAL ESTIMATES only.

Do not create precise measurements such as 12.37% unless
the supplied data actually supports that measurement.

For flood detection specifically, DO NOT use "coverage"
to invent a flood percentage.

Actual flood coverage will later come from segmentation.

Allowed colors:

water       #06b6d4
flood       #22d3ee
urban       #ef4444
building    #f97316
vegetation  #10b981
forest      #10b981
agriculture #eab308
road        #a8a29e
cloud       #e5e7eb
snow        #e5e7eb
bare        #d97706

---------------------------------------------------------
REGIONS
---------------------------------------------------------

"regions" contains approximate visual regions.

Format:

{
  "label": "Possible flooded area",
  "color": "#22d3ee",
  "rect": [0.40, 0.20, 0.25, 0.30],
  "confidence": 0.85
}

Coordinates are normalized:

x = 0..1
y = 0..1
w = 0..1
h = 0..1

Origin is the top-left.

Only provide a region when there is meaningful visual
evidence.

Maximum 5 regions.

Remember: these are approximate image regions,
NOT exact geographic boundaries.
`;

// ---------------------------------------------------------
// Intent-specific prompt
// ---------------------------------------------------------

export function buildPromptForIntent(
  intent: Intent,
  userQuery: string,
  hasSecondImage: boolean
): string {

  const base = `
You are SatQuery AI, a remote-sensing
vision-language assistant.

Your job is to analyze satellite or remote-sensing
imagery and answer the user's question using
evidence visible in the supplied image.

You are a VISUAL REASONING component.

You are NOT a replacement for:
- object detection models
- segmentation models
- spectral analysis
- SAR analysis
- GIS measurement systems

Therefore, separate visual interpretation from
precise physical measurements.
`;

  let intentGuidance = '';

  // -------------------------------------------------------
  // Water
  // -------------------------------------------------------

  switch (intent) {

    case 'water_detection':

      intentGuidance = `
TASK: WATER BODY DETECTION

Identify visible water bodies such as:

- rivers
- lakes
- reservoirs
- ponds
- sea
- ocean
- streams
- wetlands

For each important water body:

- describe its approximate image location
- describe its visual characteristics
- provide confidence
- mention turbidity, sediment, algae, or unusual
  appearance only when visually supported

Do not claim exact water area.
Do not invent geographic coordinates.
`;

      break;

    // -----------------------------------------------------
    // Flood
    // -----------------------------------------------------

    case 'flood_detection':

      intentGuidance = `
TASK: FLOOD / INUNDATION DETECTION

This is a high-caution task.

Determine whether the image provides visual evidence
consistent with flooding or inundation.

IMPORTANT:

Permanent water ≠ flooding.

A river, lake, ocean, reservoir, or pond should NOT
automatically be classified as flooded.

Look for evidence such as:

- water outside an apparent normal water channel
- water covering normally dry-looking land
- apparent river overflow
- inundation of agricultural fields
- inundation near buildings
- inundated roads
- unusual water extent
- sediment-laden water
- debris associated with water flow
- water occupying areas that appear normally terrestrial

For each suspected flooded region:

- describe the region
- provide approximate image location
- explain the visual evidence
- classify severity as low, medium, or high ONLY when
  the visual evidence supports it
- provide confidence from 0 to 1
- state whether it appears to be permanent water
  or potentially inundated land

VERY IMPORTANT:

Do NOT invent flood coverage percentages.

Do NOT calculate square kilometers.

Do NOT claim an exact flood boundary.

Do NOT infer flood depth.

Do NOT claim a flood event solely from water color.

If flooding cannot be reliably determined from the image,
say:

"Insufficient visual evidence to confirm flooding."

The future segmentation pipeline will calculate the
actual flood mask and area.
`;

      break;

    // -----------------------------------------------------
    // Buildings
    // -----------------------------------------------------

    case 'building_detection':

      intentGuidance = `
TASK: BUILDING / URBAN DETECTION

Identify:

- individual visible buildings
- building clusters
- settlements
- dense urban areas
- sparse development
- industrial-looking structures
- residential-looking areas

Describe approximate image locations.

Do not claim an exact building count unless individual
buildings are clearly distinguishable.

Do not invent building coordinates.

Use approximate regions only.
`;

      break;

    // -----------------------------------------------------
    // Vegetation
    // -----------------------------------------------------

    case 'vegetation_segmentation':

      intentGuidance = `
TASK: VEGETATION ANALYSIS

Identify visually distinguishable:

- forest
- dense vegetation
- grassland
- cropland
- plantations
- cleared areas

Describe:

- vegetation density
- approximate image location
- visible patterns
- possible clearing or deforestation

Do not identify a specific plant species unless it is
visually obvious and relevant.

Do not invent precise vegetation percentages.
`;

      break;

    // -----------------------------------------------------
    // Roads
    // -----------------------------------------------------

    case 'road_detection':

      intentGuidance = `
TASK: ROAD DETECTION

Identify visible:

- highways
- roads
- streets
- tracks
- paths
- major junctions

Describe:

- approximate location
- connectivity
- road density qualitatively
- whether roads appear continuous or interrupted

Do not invent road lengths or geographic coordinates.
`;
      
      break;

    // -----------------------------------------------------
    // Change detection
    // -----------------------------------------------------

    case 'change_detection':

      intentGuidance = `
TASK: CHANGE DETECTION

Two images may be provided.

Image 1 = BEFORE
Image 2 = AFTER

Compare only corresponding visible areas.

Look for:

- new construction
- demolished structures
- vegetation loss
- vegetation growth
- water expansion
- water reduction
- road changes
- land-use changes
- flood/inundation changes
- major landscape changes

Only report a change when the visual evidence supports it.

Do not confuse different lighting, shadows, clouds,
seasonal appearance, image quality, or viewing angle
with real physical change.

Do not invent numerical change percentages.

For changeSummary use:

{
  "additions": [],
  "removals": [],
  "netChange": ""
}
`;

      break;

    // -----------------------------------------------------
    // Land cover
    // -----------------------------------------------------

    case 'land_cover':

      intentGuidance = `
TASK: LAND COVER CLASSIFICATION

Identify visually distinguishable land-cover classes:

- water
- urban
- vegetation
- forest
- agriculture
- bare land
- roads
- clouds
- snow

Describe the dominant classes.

Coverage values should be treated as approximate visual
estimates unless actual measured data is supplied.

Do not fabricate precise percentages.
`;
      
      break;

    // -----------------------------------------------------
    // General understanding
    // -----------------------------------------------------

    case 'image_understanding':
    default:

      intentGuidance = `
TASK: GENERAL SATELLITE IMAGE UNDERSTANDING

Describe the major visible features.

Identify:

- water
- vegetation
- settlements
- roads
- agricultural areas
- terrain
- unusual features

Focus on observations that are visually defensible.

Do not invent geographic information.
`;
      
      break;
  }

  // -------------------------------------------------------
  // Second image instructions
  // -------------------------------------------------------

  const secondImageInstruction =
    hasSecondImage
      ? `

TWO-IMAGE MODE

Two images have been supplied.

Treat:

Image 1 = BEFORE
Image 2 = AFTER

Compare corresponding locations carefully.

Do not assume that differences are real-world changes
when they could be caused by:

- cloud cover
- shadows
- different illumination
- seasonal variation
- image registration
- image quality
- sensor differences
`
      : '';

  // -------------------------------------------------------
  // Final prompt
  // -------------------------------------------------------

  return `
${base}

${intentGuidance}

${COMMON_RULES}

${JSON_SCHEMA}

${secondImageInstruction}

USER QUERY:
${userQuery}
`.trim();
}
