// Shared types for SatQuery AI
// SIH26167 — ISRO Smart India Hackathon

export type Intent =
  | "water_detection"
  | "building_detection"
  | "vegetation_segmentation"
  | "change_detection"
  | "road_detection"
  | "image_understanding"
  | "flood_detection"
  | "land_cover"
  | "other";

export interface DetectedObject {
  class: string;
  confidence: number; // 0..1
  count?: number;
  region?: "north" | "south" | "east" | "west" | "center" | "widespread";
  note?: string;
}

export interface ClassCoverage {
  class: string;
  coverage: number; // 0..1
  color: string; // hex color for visualization
}

export interface AnalysisResult {
  answer: string;
  intent: Intent;
  objectsDetected: DetectedObject[];
  confidence: number;
  coverage: ClassCoverage[];
  // Optional: bounding boxes / region masks described in plain text for canvas rendering
  // Each region is described as a relative rectangle [x, y, w, h] (0..1) with a label.
  regions?: Array<{
    label: string;
    color: string;
    rect: [number, number, number, number]; // x, y, w, h (normalized)
    confidence: number;
  }>;
  // For change detection: textual summary of changes
  changeSummary?: {
    additions?: string[];
    removals?: string[];
    netChange?: string;
  };
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  // Optional structured analysis attached to assistant messages
  analysis?: AnalysisResult;
  // ISO timestamp
  createdAt: string;
  // Whether this message is currently being generated
  pending?: boolean;
  // Optional error message
  error?: string;
}

export interface UploadedImage {
  id: string;
  filename: string;
  mimeType: string;
  size: number;
  dataUrl: string;
  // For change detection — second image
  secondDataUrl?: string;
  // Optional location label
  location?: string;
}

export interface SampleImage {
  id: string;
  title: string;
  description: string;
  url: string;
  category: "flood" | "urban" | "forest" | "agriculture" | "wildfire" | "coastal";
  location: string;
}

// Color palette for class overlays — kept stable for visual consistency
export const CLASS_COLORS: Record<string, string> = {
  water: "#06b6d4",
  river: "#0891b2",
  lake: "#0e7490",
  flood: "#22d3ee",
  urban: "#ef4444",
  building: "#f97316",
  buildings: "#f97316",
  road: "#a8a29e",
  roads: "#a8a29e",
  forest: "#10b981",
  vegetation: "#22c55e",
  agriculture: "#eab308",
  agricultural: "#eab308",
  cloud: "#e5e7eb",
  clouds: "#e5e7eb",
  bare: "#d97706",
  snow: "#f1f5f9",
};

export function colorForClass(cls: string): string {
  const key = cls.toLowerCase().trim();
  for (const k of Object.keys(CLASS_COLORS)) {
    if (key.includes(k)) return CLASS_COLORS[k];
  }
  return "#8b5cf6"; // default violet for unknown classes
}
