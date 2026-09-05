// Shared types for SatQuery AI
// SIH26167 — ISRO Smart India Hackathon

export type Intent =
  | "water_detection"
  | "building_detection"
  | "vegetation_segmentation"
  | "change_detection"
  | "road_detection"
  | "ship_detection"
  | "aircraft_detection"
  | "vehicle_detection"
  | "bridge_detection"
  | "image_understanding"
  | "flood_detection"
  | "land_cover"
  | "other";

export interface DetectedObject {
  class: string;
  confidence: number; // 0..1
  count?: number;

  region?:
    | "north"
    | "south"
    | "east"
    | "west"
    | "center"
    | "widespread";

  note?: string;
}

export interface ClassCoverage {
  class: string;
  coverage: number; // 0..1
  color: string; // hex color for visualization
}


// ---------------------------------------------------------
// Evidence from different analysis systems
// ---------------------------------------------------------

export interface EvidenceSource {
  source:
    | "llm"
    | "grounding_dino"
    | "sam2"
    | "spectral"
    | "sar";

  confidence: number; // 0..1

  available: boolean;

  notes?: string[];
}


// ---------------------------------------------------------
// Bounding box returned by object detection
// Coordinates are normalized to 0..1
// ---------------------------------------------------------

export interface DetectionBox {
  label: string;

  confidence: number;

  // x, y, width, height
  box: [
    number,
    number,
    number,
    number
  ];

  source:
    | "grounding_dino"
    | "llm";
}


// ---------------------------------------------------------
// Segmentation information
// ---------------------------------------------------------

export interface SegmentationResult {
  available: boolean;

  // Number of pixels belonging to the detected region
  maskPixels?: number;

  // Total valid pixels in the image
  totalPixels?: number;

  // Calculated coverage from the segmentation mask
  coverage?: number;

  source:
    | "sam2"
    | "spectral"
    | "sar"
    | "none";
}


// ---------------------------------------------------------
// Flood-specific analysis
// ---------------------------------------------------------

export interface FloodAnalysis {
  detected: boolean;

  confidence: number;

  // IMPORTANT:
  // This should eventually be calculated from a real mask,
  // not guessed by Gemini/GLM.
  coverage?: number;

  areaKm2?: number;

  regions: Array<{
    label: string;

    confidence: number;

    location?: string;

    evidence: string[];

    severity?:
      | "low"
      | "medium"
      | "high";

    permanentWater?: boolean;

    inundatedLand?: boolean;

    detection?: DetectionBox;

    segmentation?: SegmentationResult;
  }>;

  evidenceSources?: EvidenceSource[];

  limitations: string[];
}


// ---------------------------------------------------------
// Main analysis result
// ---------------------------------------------------------

export interface AnalysisResult {
  answer: string;

  intent: Intent;

  objectsDetected: DetectedObject[];

  confidence: number;

  coverage: ClassCoverage[];

  // Detailed flood analysis
  // Used primarily for flood_detection intent
  flood?: FloodAnalysis;

  // Visualization regions
  regions?: Array<{
    label: string;

    color: string;

    // x, y, width, height
    // normalized to 0..1
    rect: [
      number,
      number,
      number,
      number
    ];

    confidence: number;
  }>;

  // Change detection
  changeSummary?: {
    additions?: string[];

    removals?: string[];

    netChange?: string;
  };
}


// ---------------------------------------------------------
// Chat
// ---------------------------------------------------------

export interface ChatMessage {
  id: string;

  role:
    | "user"
    | "assistant";

  content: string;

  // Optional structured analysis
  // attached to assistant messages
  analysis?: AnalysisResult;

  // ISO timestamp
  createdAt: string;

  // Whether this message is currently being generated
  pending?: boolean;

  // Optional error message
  error?: string;
}


// ---------------------------------------------------------
// Uploaded satellite image
// ---------------------------------------------------------

export interface UploadedImage {
  id: string;

  filename: string;

  mimeType: string;

  size: number;

  dataUrl: string;

  // For change detection
  secondDataUrl?: string;

  // Optional location label
  location?: string;
}


// ---------------------------------------------------------
// Sample images
// ---------------------------------------------------------

export interface SampleImage {
  id: string;

  title: string;

  description: string;

  url: string;

  category:
    | "flood"
    | "urban"
    | "forest"
    | "agriculture"
    | "wildfire"
    | "coastal";

  location: string;
}


// ---------------------------------------------------------
// Stable class colours
// ---------------------------------------------------------

export const CLASS_COLORS: Record<
  string,
  string
> = {
  water: "#06b6d4",

  river: "#0891b2",

  lake: "#0e7490",

  flood: "#22d3ee",

  urban: "#ef4444",

  building: "#f97316",

  buildings: "#f97316",

  road: "#a8a29e",

  roads: "#a8a29e",

  ship: "#8b5cf6",

  ships: "#8b5cf6",

  vessel: "#8b5cf6",

  aircraft: "#ec4899",

  airplane: "#ec4899",

  plane: "#ec4899",

  vehicle: "#eab308",

  vehicles: "#eab308",

  car: "#eab308",

  cars: "#eab308",

  bridge: "#3b82f6",

  bridges: "#3b82f6",

  forest: "#10b981",

  vegetation: "#22c55e",

  agriculture: "#eab308",

  agricultural: "#eab308",

  cloud: "#e5e7eb",

  clouds: "#e5e7eb",

  bare: "#d97706",

  snow: "#f1f5f9",
};


// ---------------------------------------------------------
// Get colour for a detected class
// ---------------------------------------------------------

export function colorForClass(
  cls: string,
): string {

  const key =
    cls
      .toLowerCase()
      .trim();


  for (
    const k
    of Object.keys(
      CLASS_COLORS,
    )
  ) {

    if (
      key.includes(k)
    ) {
      return CLASS_COLORS[k];
    }

  }


  // Default violet for unknown classes
  return "#8b5cf6";
}


// ---------------------------------------------------------
// Real-Time / Bi-Temporal Comparison Types
// ---------------------------------------------------------

export interface AOIBounds {
  north: number;
  south: number;
  east: number;
  west: number;
}

export interface TemporalSceneMeta {
  sceneId: string;
  provider: string;
  satellite: string;
  acquisitionDate: string;
  acquisitionTime?: string | null;
  cloudCoverage?: number | null;
  resolution: string;
  dataFreshnessDays?: number | null;
  previewUrl?: string | null;
  extraProperties?: Record<string, unknown>;
}

export type TemporalChangeType = 'new' | 'removed' | 'modified' | 'unchanged';

export interface TemporalChangeItem {
  id: string;
  type: TemporalChangeType;
  label: string;
  confidence: number;
  boxT1?: [number, number, number, number] | null;
  boxT2?: [number, number, number, number] | null;
  currentBox?: [number, number, number, number];
  latitude?: number | null;
  longitude?: number | null;
  details: string;
  historicalDate: string;
  latestDate: string;
  metrics?: {
    iou?: number;
    areaRatio?: number;
    confidenceT1?: number;
    confidenceT2?: number;
  };
}

export interface TemporalSummary {
  totalBefore: number;
  totalLatest: number;
  newCount: number;
  removedCount: number;
  modifiedCount: number;
  unchangedCount: number;
  totalChanges: number;
}

export interface PixelChangeResult {
  changePercentage: number;
  overlayDataUrl: string;
  changedPixels: number;
  totalPixels: number;
}

export interface TemporalComparisonResult {
  success: boolean;
  aoi?: AOIBounds | null;
  prompt: string;
  timeDifference: string;
  latestImageryAge: string;
  registration: {
    quality: number;
    warning?: string | null;
    transformation: string;
  };
  historical: TemporalSceneMeta;
  latest: TemporalSceneMeta;
  summary: TemporalSummary;
  changes: TemporalChangeItem[];
  pixelChange?: PixelChangeResult | null;
  images: {
    t1DataUrl: string;
    t2DataUrl: string;
  };
}