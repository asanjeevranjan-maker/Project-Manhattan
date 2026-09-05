// Parse the VLM response into a structured AnalysisResult.
//
// The VLM is expected to return a ```satquery JSON block.
// This parser validates and normalizes the response before
// sending it to the frontend.
//
// IMPORTANT:
// Flood percentage / area should NOT be invented by the VLM.
// Later, Grounding DINO + SAM 2 + Spectral/SAR will provide
// real measurements. This parser only accepts measurements
// supplied by the model/pipeline.

import type {
  AnalysisResult,
  ClassCoverage,
  DetectedObject,
  Intent,
  FloodAnalysis,
  EvidenceSource,
  DetectionBox,
  SegmentationResult,
} from './types';

interface RawSatQuery {
  objects_detected?: Array<Record<string, unknown>>;
  confidence?: number;

  coverage?: Array<Record<string, unknown>>;

  regions?: Array<Record<string, unknown>>;

  flood?: Record<string, unknown>;

  changeSummary?: {
    additions?: string[];
    removals?: string[];
    netChange?: string;
  };
}

const FENCE_PATTERN = /```satquery\s*([\s\S]*?)```/i;

// ---------------------------------------------------------
// Basic conversion helpers
// ---------------------------------------------------------

function asNumber(
  value: unknown,
  fallback = 0
): number {
  if (
    typeof value === 'number' &&
    Number.isFinite(value)
  ) {
    return value;
  }

  if (typeof value === 'string') {
    const number = parseFloat(value);

    if (Number.isFinite(number)) {
      return number;
    }
  }

  return fallback;
}

function asString(
  value: unknown,
  fallback = ''
): string {
  if (typeof value === 'string') {
    return value;
  }

  if (typeof value === 'number') {
    return String(value);
  }

  return fallback;
}

function asBoolean(
  value: unknown,
  fallback = false
): boolean {
  if (typeof value === 'boolean') {
    return value;
  }

  if (typeof value === 'string') {
    const normalized = value
      .trim()
      .toLowerCase();

    if (normalized === 'true') return true;
    if (normalized === 'false') return false;
  }

  return fallback;
}

function asStringArray(
  value: unknown
): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter(
      (item): item is string =>
        typeof item === 'string'
    )
    .map((item) => item.trim())
    .filter(Boolean);
}

// ---------------------------------------------------------
// Number normalization
// ---------------------------------------------------------

function clamp01(number: number): number {
  if (number < 0) return 0;
  if (number > 1) return 1;

  return number;
}

// ---------------------------------------------------------
// Rectangle normalization
// ---------------------------------------------------------

function normalizeRect(
  rect: unknown
): [number, number, number, number] | null {
  if (
    !Array.isArray(rect) ||
    rect.length < 4
  ) {
    return null;
  }

  const numbers = rect
    .slice(0, 4)
    .map((value) =>
      asNumber(value, 0)
    );

  const [x, y, w, h] =
    numbers.map(clamp01);

  // Reject extremely small rectangles
  if (
    w <= 0.01 ||
    h <= 0.01
  ) {
    return null;
  }

  return [x, y, w, h];
}

// ---------------------------------------------------------
// Detection box
// ---------------------------------------------------------

function parseDetectionBox(
  value: unknown
): DetectionBox | undefined {
  if (
    !value ||
    typeof value !== 'object'
  ) {
    return undefined;
  }

  const detection =
    value as Record<string, unknown>;

  const box = detection.box;

  if (!Array.isArray(box) || box.length < 4) {
    return undefined;
  }

  const normalizedBox = normalizeRect(box);

  if (!normalizedBox) {
    return undefined;
  }

  const source =
    detection.source ===
    'grounding_dino'
      ? 'grounding_dino'
      : 'llm';

  return {
    label: asString(
      detection.label,
      'detected region'
    ),

    confidence: clamp01(
      asNumber(
        detection.confidence,
        0
      )
    ),

    box: normalizedBox,

    source,
  };
}

// ---------------------------------------------------------
// Segmentation
// ---------------------------------------------------------

function parseSegmentation(
  value: unknown
): SegmentationResult | undefined {
  if (
    !value ||
    typeof value !== 'object'
  ) {
    return undefined;
  }

  const segmentation =
    value as Record<string, unknown>;

  const sourceValue =
    asString(
      segmentation.source,
      'none'
    );

  let source:
    | 'sam2'
    | 'spectral'
    | 'sar'
    | 'none' = 'none';

  if (
    sourceValue === 'sam2' ||
    sourceValue === 'spectral' ||
    sourceValue === 'sar'
  ) {
    source = sourceValue;
  }

  const maskPixels =
    segmentation.maskPixels !== undefined
      ? asNumber(
          segmentation.maskPixels,
          0
        )
      : undefined;

  const totalPixels =
    segmentation.totalPixels !== undefined
      ? asNumber(
          segmentation.totalPixels,
          0
        )
      : undefined;

  const coverage =
    segmentation.coverage !== undefined
      ? clamp01(
          asNumber(
            segmentation.coverage,
            0
          )
        )
      : undefined;

  return {
    available: asBoolean(
      segmentation.available,
      source !== 'none'
    ),

    maskPixels,

    totalPixels,

    coverage,

    source,
  };
}

// ---------------------------------------------------------
// Evidence sources
// ---------------------------------------------------------

function parseEvidenceSources(
  value: unknown
): EvidenceSource[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const validSources: EvidenceSource['source'][] =
    [
      'llm',
      'grounding_dino',
      'sam2',
      'spectral',
      'sar',
    ];

  return value
    .filter(
      (item): item is Record<string, unknown> =>
        Boolean(item) &&
        typeof item === 'object'
    )
    .map((item) => {
      const sourceValue =
        asString(
          item.source,
          'llm'
        );

      const source =
        validSources.includes(
          sourceValue as EvidenceSource['source']
        )
          ? (sourceValue as EvidenceSource['source'])
          : 'llm';

      return {
        source,

        confidence: clamp01(
          asNumber(
            item.confidence,
            0
          )
        ),

        available: asBoolean(
          item.available,
          true
        ),

        notes: asStringArray(
          item.notes
        ),
      };
    });
}

// ---------------------------------------------------------
// Flood region parser
// ---------------------------------------------------------

function parseFloodRegions(
  value: unknown
): FloodAnalysis['regions'] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter(
      (item): item is Record<string, unknown> =>
        Boolean(item) &&
        typeof item === 'object'
    )
    .map((region) => {
      const evidence =
        asStringArray(
          region.evidence
        );

      return {
        label: asString(
          region.label,
          asString(
            region.class,
            'Possible flooded region'
          )
        ),

        confidence: clamp01(
          asNumber(
            region.confidence,
            0
          )
        ),

        location:
          asString(
            region.location,
            ''
          ) || undefined,

        evidence,

        severity:
          region.severity === 'low' ||
          region.severity === 'medium' ||
          region.severity === 'high'
            ? region.severity
            : undefined,

        permanentWater:
          region.permanentWater !== undefined
            ? asBoolean(
                region.permanentWater
              )
            : undefined,

        inundatedLand:
          region.inundatedLand !== undefined
            ? asBoolean(
                region.inundatedLand
              )
            : undefined,

        detection:
          parseDetectionBox(
            region.detection
          ),

        segmentation:
          parseSegmentation(
            region.segmentation
          ),
      };
    });
}

// ---------------------------------------------------------
// Flood analysis parser
// ---------------------------------------------------------

function parseFloodAnalysis(
  value: unknown
): FloodAnalysis | undefined {
  if (
    !value ||
    typeof value !== 'object'
  ) {
    return undefined;
  }

  const flood =
    value as Record<string, unknown>;

  const coverage =
    flood.coverage !== undefined
      ? clamp01(
          asNumber(
            flood.coverage,
            0
          )
        )
      : undefined;

  const areaKm2 =
    flood.areaKm2 !== undefined
      ? asNumber(
          flood.areaKm2,
          0
        )
      : undefined;

  return {
    detected: asBoolean(
      flood.detected,
      false
    ),

    confidence: clamp01(
      asNumber(
        flood.confidence,
        0
      )
    ),

    coverage,

    areaKm2,

    regions:
      parseFloodRegions(
        flood.regions
      ),

    evidenceSources:
      parseEvidenceSources(
        flood.evidenceSources
      ),

    limitations:
      asStringArray(
        flood.limitations
      ),
  };
}

// ---------------------------------------------------------
// Main parser
// ---------------------------------------------------------

export function parseAnalysis(
  rawAnswer: string,
  intent: Intent
): AnalysisResult {

  // -------------------------------------------------------
  // Default result
  // -------------------------------------------------------

  const empty: AnalysisResult = {
    answer: rawAnswer.trim(),

    intent,

    objectsDetected: [],

    confidence: 0.5,

    coverage: [],
  };

  // -------------------------------------------------------
  // Find structured JSON block
  // -------------------------------------------------------

  const match =
    rawAnswer.match(
      FENCE_PATTERN
    );

  if (!match) {
    console.warn(
      '[parseAnalysis] No satquery JSON block found.'
    );

    return empty;
  }

  // -------------------------------------------------------
  // Parse JSON
  // -------------------------------------------------------

  let parsed: RawSatQuery;

  try {
    parsed =
      JSON.parse(
        match[1].trim()
      ) as RawSatQuery;
  } catch (error) {
    console.error(
      '[parseAnalysis] Invalid JSON:',
      error
    );

    return empty;
  }

  // -------------------------------------------------------
  // Objects
  // -------------------------------------------------------

  const objectsDetected:
    DetectedObject[] =
    (parsed.objects_detected ?? [])
      .filter(
        (item) =>
          Boolean(item) &&
          typeof item === 'object'
      )
      .map((object) => ({
        class: asString(
          object.class,
          'unknown'
        ),

        confidence: clamp01(
          asNumber(
            object.confidence,
            0.5
          )
        ),

        count:
          typeof object.count === 'number'
            ? object.count
            : undefined,

        region: (() => {
  const region = asString(
    object.region,
    'center'
  );

  const validRegions: DetectedObject['region'][] = [
    'north',
    'south',
    'east',
    'west',
    'center',
    'widespread',
  ];

  return validRegions.includes(
    region as DetectedObject['region']
  )
    ? (region as DetectedObject['region'])
    : 'center';
})(),
        note:
          asString(
            object.note,
            ''
          ) || undefined,
      }));

  // -------------------------------------------------------
  // Land-cover coverage
  // -------------------------------------------------------

  const coverage:
    ClassCoverage[] =
    (parsed.coverage ?? [])
      .filter(
        (item) =>
          Boolean(item) &&
          typeof item === 'object'
      )
      .map((item) => ({
        class: asString(
          item.class,
          'unknown'
        ),

        coverage: clamp01(
          asNumber(
            item.coverage,
            0
          )
        ),

        color: asString(
          item.color,
          '#8b5cf6'
        ),
      }));

  // -------------------------------------------------------
  // Existing visualization regions
  // -------------------------------------------------------

  const regions =
    (parsed.regions ?? [])
      .map((region) => {
        const rect =
          normalizeRect(
            region.rect
          );

        if (!rect) {
          return null;
        }

        return {
          label: asString(
            region.label,
            asString(
              region.class,
              'Region'
            )
          ),

          color: asString(
            region.color,
            '#8b5cf6'
          ),

          rect,

          confidence: clamp01(
            asNumber(
              region.confidence,
              0.8
            )
          ),
        };
      })
      .filter(
        (
          region
        ): region is NonNullable<
          typeof region
        > =>
          region !== null
      );

  // -------------------------------------------------------
  // Flood analysis
  // -------------------------------------------------------

  const flood =
    parseFloodAnalysis(
      parsed.flood
    );

  // -------------------------------------------------------
  // Remove JSON block from visible answer
  // -------------------------------------------------------

  const answerWithoutJson =
    rawAnswer
      .replace(
        FENCE_PATTERN,
        ''
      )
      .trim();

  // -------------------------------------------------------
  // Final result
  // -------------------------------------------------------

  return {
    answer:
      answerWithoutJson ||
      empty.answer,

    intent,

    objectsDetected,

    confidence:
      clamp01(
        asNumber(
          parsed.confidence,
          0.7
        )
      ),

    coverage,

    regions,

    flood,

    changeSummary:
      parsed.changeSummary,
  };
}