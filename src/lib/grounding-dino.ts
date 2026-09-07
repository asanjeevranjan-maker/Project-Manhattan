// Grounding DINO client — open-vocabulary object detection via the
// Hugging Face Inference API (IDEA-Research/grounding-dino-tiny).
//
// Grounding DINO runs ALONGSIDE the VLM, not instead of it:
//   - The VLM (GLM / Gemini) provides textual analysis.
//   - Grounding DINO provides precise, detector-grade bounding boxes.
// Cross-model agreement on the same objects increases overall confidence.
//
// Pipeline position:
//   user query → intent classification → extractGroundingQueries()
//   → callGroundingDINO() → normalized detections → NMS → UI overlay
//
// Notes:
//   - The HF free tier is rate limited (~1000 requests/day).
//   - The model may be "cold" (not loaded); the `x-wait-for-model: true`
//     request header makes HF load it before responding.
//   - First call after a cold start can take 20-30s, hence a 60s timeout.

import type { GroundingDinoDetection, Intent } from './types';

// ---------------------------------------------------------
// Model + endpoint constants
// ---------------------------------------------------------

export const GDINO_MODEL_ID = 'IDEA-Research/grounding-dino-tiny';

const HF_INFERENCE_URL = `https://api-inference.huggingface.co/models/${GDINO_MODEL_ID}`;

const GDINO_TIMEOUT_MS = 60_000;

const MAX_DETECTIONS = 30;

// Confidence floor. Grounding DINO on satellite imagery produces many
// low-confidence false positives; below this they are visual noise.
const MIN_CONFIDENCE = 0.2;

// ---------------------------------------------------------
// Raw Hugging Face response shape
// ---------------------------------------------------------

export interface HfDetection {
  score: number; // 0..1 confidence
  label: string; // detected label
  box: {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
  }; // pixel coordinates
}

// ---------------------------------------------------------
// Intent → Grounding DINO text keywords.
// Detection-style intents benefit from GDINO; general scene
// understanding queries get an empty list and are skipped.
// ---------------------------------------------------------

const INTENT_KEYWORDS: Record<Intent, string[]> = {
  flood_detection: ['water', 'flood', 'river'],
  water_detection: ['water', 'river', 'lake'],
  building_detection: ['building', 'house', 'roof'],
  vegetation_segmentation: ['forest', 'tree', 'vegetation'],
  change_detection: ['building', 'water', 'forest'],
  road_detection: ['road', 'highway'],
  ship_detection: ['ship', 'boat', 'vessel'],
  aircraft_detection: ['airplane', 'aircraft'],
  vehicle_detection: ['car', 'truck'],
  bridge_detection: ['bridge'],
  land_cover: ['field', 'forest', 'water', 'building'],
  image_understanding: [],
  other: [],
};

/** Intents where detector boxes are actually useful. */
const GDINO_ELIGIBLE_INTENTS = new Set<Intent>([
  'water_detection',
  'flood_detection',
  'building_detection',
  'vegetation_segmentation',
  'road_detection',
  'ship_detection',
  'aircraft_detection',
  'vehicle_detection',
  'bridge_detection',
  'land_cover',
  'change_detection',
]);

export function isGroundingDinoIntent(intent: Intent): boolean {
  return GDINO_ELIGIBLE_INTENTS.has(intent);
}

// ---------------------------------------------------------
// Stopword removal for query → noun-phrase extraction
// ---------------------------------------------------------

const STOPWORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to',
  'from', 'by', 'with', 'for', 'is', 'are', 'was', 'were', 'be', 'been',
  'this', 'that', 'these', 'those', 'it', 'its', 'there', 'here',
  'i', 'you', 'we', 'they', 'he', 'she', 'me', 'us', 'them',
  'my', 'our', 'your', 'their', 'can', 'could', 'would', 'should',
  'do', 'does', 'did', 'please', 'image', 'photo', 'picture', 'satellite',
  'show', 'shows', 'showing', 'find', 'detect', 'locate', 'identify',
  'many', 'much', 'how', 'what', 'where', 'which', 'any', 'some',
  'all', 'tell', 'give', 'about', 'into', 'over', 'under',
  'near', 'around', 'using', 'use', 'analysis', 'analyze',
]);

function normalizeWord(word: string): string {
  return word.toLowerCase().replace(/[^a-z0-9-]/g, '');
}

/**
 * Extracts 1-3 short noun phrases from a natural-language query for use
 * as Grounding DINO text prompts.
 *
 * Strategy:
 *  1. Tokenize the query, drop stopwords / punctuation.
 *  2. Build 1-2 word phrases from consecutive content words.
 *  3. Merge in intent-specific keywords ("water", "building", "forest"…)
 *     so GDINO always gets detector-friendly vocabulary.
 *  4. De-duplicate, cap at 3.
 *
 * Returns [] for general scene-understanding queries — GDINO should be
 * skipped entirely for those (it would only return random boxes).
 */
export function extractGroundingQueries(
  query: string,
  intent: Intent,
): string[] {
  if (!isGroundingDinoIntent(intent)) {
    return [];
  }

  // 1. Tokenize & strip stopwords.
  const contentWords = query
    .toLowerCase()
    .split(/\s+/)
    .map(normalizeWord)
    .filter((w) => w.length > 1 && !STOPWORDS.has(w));

  // 2. Candidate phrases: single content words + consecutive bigrams.
  const phrases: string[] = [];

  for (const word of contentWords) {
    phrases.push(word);
  }

  for (let i = 0; i < contentWords.length - 1; i++) {
    phrases.push(`${contentWords[i]} ${contentWords[i + 1]}`);
  }

  // 3. Intent keywords fill the remainder and act as fallback.
  const intentKeywords = INTENT_KEYWORDS[intent] ?? [];

  const combined = [...phrases, ...intentKeywords];

  // 4. De-duplicate (exact phrase), keep order & cap at 3.
  const seen = new Set<string>();
  const result: string[] = [];

  for (const phrase of combined) {
    const key = phrase.trim().toLowerCase();

    if (!key || seen.has(key)) continue;

    seen.add(key);
    result.push(phrase.trim());

    if (result.length >= 3) break;
  }

  return result;
}

// ---------------------------------------------------------
// Prompt formatting
// ---------------------------------------------------------

/**
 * Grounding DINO expects queries as space-separated phrases, each
 * terminated by a period: "water . flood . building ."
 */
export function buildGroundingPromptText(queries: string[]): string {
  return queries.map((q) => `${q.trim()} .`).join(' ');
}

// ---------------------------------------------------------
// Image decoding helpers
// ---------------------------------------------------------

function dataUrlToBuffer(dataUrl: string): Buffer {
  const commaIndex = dataUrl.indexOf(',');

  if (!dataUrl.startsWith('data:') || commaIndex === -1) {
    throw new Error('Invalid data URL for Grounding DINO.');
  }

  const base64 = dataUrl.slice(commaIndex + 1);

  return Buffer.from(base64, 'base64');
}

async function resolveImageBuffer(
  imageDataUrl: string,
): Promise<Buffer> {
  if (imageDataUrl.startsWith('data:')) {
    return dataUrlToBuffer(imageDataUrl);
  }

  if (
    imageDataUrl.startsWith('http://') ||
    imageDataUrl.startsWith('https://')
  ) {
    const res = await fetch(imageDataUrl);

    if (!res.ok) {
      throw new Error(
        `Failed to fetch image for Grounding DINO (HTTP ${res.status}).`,
      );
    }

    return Buffer.from(await res.arrayBuffer());
  }

  throw new Error(
    'imageDataUrl must be a data: URL or an http(s) URL.',
  );
}

// ---------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/** Intersection-over-Union between two [x, y, w, h] normalized boxes. */
export function iou(
  a: [number, number, number, number],
  b: [number, number, number, number],
): number {
  const ax2 = a[0] + a[2];
  const ay2 = a[1] + a[3];
  const bx2 = b[0] + b[2];
  const by2 = b[1] + b[3];

  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a[1], b[1]));

  const intersection = ix * iy;

  const areaA = Math.max(0, a[2]) * Math.max(0, a[3]);
  const areaB = Math.max(0, b[2]) * Math.max(0, b[3]);

  const union = areaA + areaB - intersection;

  return union <= 0 ? 0 : intersection / union;
}

/**
 * Greedy Non-Maximum Suppression on normalized detections.
 * Keeps highest-confidence boxes first, discards boxes overlapping
 * an already-kept box by more than `iouThreshold`.
 */
export function nmsDetections(
  detections: GroundingDinoDetection[],
  iouThreshold: number,
): GroundingDinoDetection[] {
  const sorted = [...detections].sort(
    (a, b) => b.confidence - a.confidence,
  );

  const kept: GroundingDinoDetection[] = [];

  for (const candidate of sorted) {
    const overlaps = kept.some(
      (k) => iou(k.rect, candidate.rect) > iouThreshold,
    );

    if (!overlaps) {
      kept.push(candidate);
    }
  }

  return kept;
}

function deduplicateByIou(
  detections: GroundingDinoDetection[],
): GroundingDinoDetection[] {
  // Keep the highest-scoring box of any pair with IoU > 0.7
  // (duplicates arise when multiple query phrases hit the same object).
  const kept: GroundingDinoDetection[] = [];

  for (const detection of [...detections].sort(
    (a, b) => b.confidence - a.confidence,
  )) {
    const duplicate = kept.some(
      (k) => iou(k.rect, detection.rect) > 0.7,
    );

    if (!duplicate) {
      kept.push(detection);
    }
  }

  return kept;
}

// ---------------------------------------------------------
// Main API call
// ---------------------------------------------------------

/**
 * Calls the Hugging Face Inference API with the image and returns
 * normalized Grounding DINO detections.
 *
 * @param imageDataUrl  data: URL (or http(s) URL) of the image
 * @param queries       1-3 short noun phrases, e.g. ["water", "flood"]
 * @param hfToken       Hugging Face access token (Read scope is enough)
 */
export async function callGroundingDINO(
  imageDataUrl: string,
  queries: string[],
  hfToken: string,
): Promise<GroundingDinoDetection[]> {
  if (!hfToken) {
    throw new Error('Missing Hugging Face token for Grounding DINO.');
  }

  if (queries.length === 0) {
    return [];
  }

  const imageBuffer = await resolveImageBuffer(imageDataUrl);

  // Grounding DINO text queries: lowercase, space-separated phrases,
  // each terminated by a period — "water . flood . building ."
  // They are passed as the `text_prompt` query parameter while the
  // raw image bytes form the request body.
  const response = await fetch(
    `${HF_INFERENCE_URL}?text_prompt=${encodeURIComponent(
      buildGroundingPromptText(queries),
    )}`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${hfToken}`,
        'Content-Type': 'image/jpeg',
        // Makes HF load the model if it is cold (first call can take 20-30s).
        'x-wait-for-model': 'true',
      },
      body: new Uint8Array(imageBuffer),
      signal: AbortSignal.timeout(GDINO_TIMEOUT_MS),
    },
  );

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');

    throw new Error(
      `Grounding DINO request failed (HTTP ${response.status}): ${
        errorText.slice(0, 300) || response.statusText
      }`,
    );
  }

  const hfDetections =
    (await response.json()) as HfDetection[];

  if (!Array.isArray(hfDetections)) {
    throw new Error(
      'Grounding DINO returned an unexpected response format.',
    );
  }

  // Image dimensions for pixel → [0..1] normalization.
  const sharp = (await import('sharp')).default;
  const metadata = await sharp(imageBuffer).metadata();

  const width = metadata.width ?? 0;
  const height = metadata.height ?? 0;

  if (width <= 0 || height <= 0) {
    throw new Error(
      'Could not determine image dimensions for Grounding DINO.',
    );
  }

  const normalized: GroundingDinoDetection[] = [];

  for (const det of hfDetections) {
    const {
      score,
      label,
      box: { xmin, ymin, xmax, ymax },
    } = det;

    if (
      typeof score !== 'number' ||
      typeof xmin !== 'number' ||
      typeof ymin !== 'number' ||
      typeof xmax !== 'number' ||
      typeof ymax !== 'number'
    ) {
      continue;
    }

    if (score < MIN_CONFIDENCE) {
      continue;
    }

    const x = clamp01(xmin / width);
    const y = clamp01(ymin / height);
    const w = clamp01(xmax / width) - x;
    const h = clamp01(ymax / height) - y;

    // Discard degenerate boxes.
    if (w <= 0.001 || h <= 0.001) {
      continue;
    }

    normalized.push({
      label: typeof label === 'string' && label ? label : 'object',
      confidence: clamp01(score),
      rect: [x, y, w, h],
    });
  }

  // Deduplicate overlapping boxes (IoU > 0.7), then cap at 30.
  return deduplicateByIou(normalized).slice(0, MAX_DETECTIONS);
}


