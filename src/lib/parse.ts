// Parse the VLM response into a structured AnalysisResult.
// The VLM is prompted to append a ```satquery JSON block to its answer;
// this module extracts and validates that block.

import type { AnalysisResult, ClassCoverage, DetectedObject, Intent } from './types';

interface RawSatQuery {
  objects_detected?: Array<Record<string, unknown>>;
  confidence?: number;
  coverage?: Array<Record<string, unknown>>;
  regions?: Array<Record<string, unknown>>;
  changeSummary?: {
    additions?: string[];
    removals?: string[];
    netChange?: string;
  };
}

const FENCE_PATTERN = /```satquery\s*([\s\S]*?)```/i;

function asNumber(v: unknown, fallback = 0): number {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string') {
    const n = parseFloat(v);
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

function asString(v: unknown, fallback = ''): string {
  if (typeof v === 'string') return v;
  if (typeof v === 'number') return String(v);
  return fallback;
}

function clamp01(n: number): number {
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

function normalizeRect(rect: unknown): [number, number, number, number] | null {
  if (!Array.isArray(rect) || rect.length < 4) return null;
  const nums = rect.slice(0, 4).map((v) => asNumber(v, 0));
  // Clamp each to [0, 1]
  const [x, y, w, h] = nums.map(clamp01);
  // Reject degenerate rectangles
  if (w <= 0.01 || h <= 0.01) return null;
  return [x, y, w, h];
}

export function parseAnalysis(
  rawAnswer: string,
  intent: Intent
): AnalysisResult {
  // Default empty result
  const empty: AnalysisResult = {
    answer: rawAnswer.trim(),
    intent,
    objectsDetected: [],
    confidence: 0.5,
    coverage: [],
  };

  const match = rawAnswer.match(FENCE_PATTERN);
  if (!match) {
    // No structured block — return the raw text answer with no overlays
    return empty;
  }

  let parsed: RawSatQuery;
  try {
    parsed = JSON.parse(match[1].trim()) as RawSatQuery;
  } catch {
    // Malformed JSON — fall back to raw answer
    return empty;
  }

  const objectsDetected: DetectedObject[] = (parsed.objects_detected ?? []).map((o) => ({
    class: asString(o.class, 'unknown'),
    confidence: clamp01(asNumber(o.confidence, 0.5)),
    count: typeof o.count === 'number' ? o.count : undefined,
    region: (asString(o.region, 'center') as DetectedObject['region']) ?? 'center',
    note: asString(o.note, '') || undefined,
  }));

  const coverage: ClassCoverage[] = (parsed.coverage ?? []).map((c) => ({
    class: asString(c.class, 'unknown'),
    coverage: clamp01(asNumber(c.coverage, 0)),
    color: asString(c.color, '#8b5cf6'),
  }));

  const regions = (parsed.regions ?? [])
    .map((r) => {
      const rect = normalizeRect(r.rect);
      if (!rect) return null;
      return {
        label: asString(r.label, asString(r.class, 'Region')),
        color: asString(r.color, '#8b5cf6'),
        rect,
        confidence: clamp01(asNumber(r.confidence, 0.8)),
      };
    })
    .filter((r): r is NonNullable<typeof r> => r !== null);

  // Strip the JSON block from the human-readable answer
  const answerWithoutJson = rawAnswer.replace(FENCE_PATTERN, '').trim();

  return {
    answer: answerWithoutJson || empty.answer,
    intent,
    objectsDetected,
    confidence: clamp01(asNumber(parsed.confidence, 0.7)),
    coverage,
    regions,
    changeSummary: parsed.changeSummary,
  };
}
