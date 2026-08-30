// Intent classification — lightweight, deterministic rule-based router
// Maps a free-text user query to one of the predefined remote-sensing analysis intents
// defined in the PRD (Section 22). Used as a fast pre-filter before the VLM is invoked
// so the VLM prompt can be specialized per intent.

import type { Intent } from './types';

interface Rule {
  intent: Intent;
  // Matched case-insensitively against the normalized query
  keywords: string[];
}

const RULES: Rule[] = [
  // Flood detection — high priority, check before general water
  { intent: 'flood_detection', keywords: ['flood', 'flooding', 'flooded', 'inundat', 'deluge', 'overflow'] },
  // Water bodies
  { intent: 'water_detection', keywords: ['water', 'river', 'lake', 'reservoir', 'pond', 'sea', 'ocean', 'stream', 'wetland'] },
  // Buildings / urban
  { intent: 'building_detection', keywords: ['building', 'buildings', 'urban', 'structure', 'structures', 'house', 'houses', 'settlement', 'city', 'town', 'construction', 'infrastructure'] },
  // Vegetation / forest
  { intent: 'vegetation_segmentation', keywords: ['forest', 'vegetation', 'tree', 'trees', 'canopy', 'jungle', 'grassland', 'green area', 'plantation', 'deforest'] },
  // Roads
  { intent: 'road_detection', keywords: ['road', 'roads', 'highway', 'street', 'expressway', 'pathway', 'lane', 'track', 'route'] },
  // Change detection
  { intent: 'change_detection', keywords: ['change', 'changed', 'changes', 'compare', 'comparison', 'before', 'after', 'difference', 'differences', 'new', 'lost', 'loss', 'gain'] },
  // Land cover
  { intent: 'land_cover', keywords: ['land cover', 'land use', 'landcover', 'landuse', 'land type', 'classify', 'classification'] },
];

const INTENT_LABEL: Record<Intent, string> = {
  water_detection: 'Water Body Detection',
  building_detection: 'Building / Urban Detection',
  vegetation_segmentation: 'Vegetation Segmentation',
  change_detection: 'Change Detection',
  road_detection: 'Road Detection',
  image_understanding: 'Image Understanding',
  flood_detection: 'Flood Detection',
  land_cover: 'Land Cover Classification',
  other: 'General Analysis',
};

export function classifyIntent(query: string): Intent {
  const q = query.toLowerCase().trim();
  if (!q) return 'other';

  // Each rule's first keyword match wins, in priority order
  for (const rule of RULES) {
    for (const kw of rule.keywords) {
      if (q.includes(kw)) {
        return rule.intent;
      }
    }
  }
  // Default — generic image understanding
  return 'image_understanding';
}

export function intentLabel(intent: Intent): string {
  return INTENT_LABEL[intent] ?? 'General Analysis';
}

// Build a VLM-optimized instruction string for a given intent.
// This makes the model's answer structured & actionable for visualization.
export function buildPromptForIntent(intent: Intent, userQuery: string, hasSecondImage: boolean): string {
  const base = `You are SatQuery AI, a remote-sensing vision-language assistant. Analyze the satellite/remote sensing image(s) and answer the user's query precisely. Be specific, quantitative where possible, and ground every claim in what is visible in the image.`;

  const commonRules = `
Respond in MINIMAL markdown. Use short bullet points. No long paragraphs.

CRITICAL — STRUCTURED OUTPUT:
After your human-readable answer, you MUST append a fenced JSON block in EXACTLY this format so the UI can render overlays and statistics:

\`\`\`satquery
{
  "objects_detected": [
    {"class": "water", "confidence": 0.94, "count": 3, "region": "center", "note": "Three lakes clustered centrally"}
  ],
  "confidence": 0.91,
  "coverage": [
    {"class": "water", "coverage": 0.12, "color": "#06b6d4"},
    {"class": "urban", "coverage": 0.34, "color": "#ef4444"},
    {"class": "vegetation", "coverage": 0.42, "color": "#10b981"}
  ],
  "regions": [
    {"label": "Water Body", "color": "#06b6d4", "rect": [0.12, 0.45, 0.30, 0.25], "confidence": 0.93}
  ]
}
\`\`\`

Rules for the JSON block:
- "objects_detected": list of detected classes with confidence 0..1. Use "region" of: north|south|east|west|center|widespread.
- "confidence": overall analysis confidence 0..1.
- "coverage": ONLY include classes that are clearly visible. Each entry: class name, coverage fraction 0..1, and a hex color (use #06b6d4 for water/flood, #ef4444 for urban/buildings, #10b981 for forest/vegetation, #eab308 for agriculture, #a8a29e for roads, #e5e7eb for clouds/snow, #d97706 for bare soil).
- "regions": 0..5 rectangular overlays. rect is [x, y, w, h] with all values NORMALIZED 0..1 (top-left origin). Only include regions you are confident about. If unsure, omit "regions".
- The JSON must be valid. No trailing commas. No comments. No text after the closing brace.
- If the image is not a satellite/remote-sensing image, still answer but set confidence low and skip coverage/regions.`;

  let intentGuidance = '';
  switch (intent) {
    case 'water_detection':
      intentGuidance = `
TASK: Water Body Detection
- Identify all visible water bodies: rivers, lakes, reservoirs, ponds, sea, oceans.
- Estimate total water coverage as a fraction of the image.
- For each distinct water body, provide a bounding region.
- Mention water quality indicators if visible (turbidity, sediment, algae blooms).`;
      break;
    case 'flood_detection':
      intentGuidance = `
TASK: Flood Detection
- Identify flooded / inundated areas. Distinguish flood water from permanent water bodies if possible.
- Estimate flood extent as a fraction of the image.
- Highlight affected regions (villages, fields, infrastructure) if visible.
- Mention indicators like sediment-laden water, overflow channels.`;
      break;
    case 'building_detection':
      intentGuidance = `
TASK: Building / Urban Detection
- Identify built-up areas, individual buildings, urban clusters.
- Estimate urban coverage and approximate building count if feasible.
- Note distinguishing features: dense urban vs sparse settlement, industrial vs residential.
- Provide bounding regions for major clusters.`;
      break;
    case 'vegetation_segmentation':
      intentGuidance = `
TASK: Vegetation Segmentation
- Identify forest, grassland, plantations, cropland.
- Estimate vegetation coverage as a fraction.
- Note vegetation density and type indicators.
- Highlight deforestation or clearing if visible.`;
      break;
    case 'road_detection':
      intentGuidance = `
TASK: Road Detection
- Identify road networks, highways, tracks.
- Estimate road density (km per square km if feasible, else low/medium/high).
- Note road connectivity and major junctions.`;
      break;
    case 'change_detection':
      intentGuidance = `
TASK: Change Detection
- You are given TWO satellite images of the same area at different times (Image 1 = before, Image 2 = after).
- Identify what changed: new construction, deforestation, water level changes, urban expansion, etc.
- Quantify changes (areas, percentages, counts) where possible.
- In the JSON, set "changeSummary" with three optional arrays: "additions", "removals", "netChange".
- Provide regions highlighting the most significant changes.`;
      break;
    case 'land_cover':
      intentGuidance = `
TASK: Land Cover Classification
- Classify the entire image into land cover types: water, urban, forest, agriculture, bare, cloud, snow.
- Provide coverage fractions for each class.
- Provide regions for the dominant classes.`;
      break;
    case 'image_understanding':
    default:
      intentGuidance = `
TASK: General Image Understanding
- Describe what is visible in the satellite image.
- Identify major geographical features and land cover types.
- Provide coverage estimates and regions for any clearly identifiable classes.`;
      break;
  }

  const second = hasSecondImage
    ? `\n\nNOTE: The user has provided TWO images for change detection. Treat the first image as "before" and the second as "after". Compare them in your answer.`
    : '';

  return `${base}\n${intentGuidance}\n${commonRules}\n\nUSER QUERY: ${userQuery}${second}`;
}
