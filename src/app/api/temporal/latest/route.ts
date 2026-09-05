// POST /api/temporal/latest
//
// Proxies Real-Time Bi-Temporal Comparison to the Python AI service when running,
// or seamlessly falls back to Cloud-Native Satellite Imagery (Esri Wayback WMTS),
// Gemini Vision, and Sharp-powered pixel-by-pixel change detection.

import { NextRequest, NextResponse } from "next/server";
import { runCloudTemporalComparison } from "@/lib/cloud-temporal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";

export async function POST(req: NextRequest) {
  let aoi = {
    north: 12.98,
    south: 12.96,
    east: 77.61,
    west: 77.59,
  };
  let prompt = "building";
  let historicalMode = "date";
  let historicalDate = "2024-01-15";
  let historicalFileBase64: string | undefined = undefined;
  let formDataToSend: FormData | null = null;

  try {
    const contentType = req.headers.get("content-type") || "";

    if (contentType.includes("multipart/form-data")) {
      const reqFormData = await req.formData();
      formDataToSend = new FormData();

      for (const [key, val] of reqFormData.entries()) {
        formDataToSend.append(key, val);
      }

      const n = reqFormData.get("aoi_north");
      const s = reqFormData.get("aoi_south");
      const e = reqFormData.get("aoi_east");
      const w = reqFormData.get("aoi_west");
      if (n && s && e && w) {
        aoi = { north: Number(n), south: Number(s), east: Number(e), west: Number(w) };
      }
      const p = reqFormData.get("prompt");
      if (p) prompt = String(p);
      const hm = reqFormData.get("historical_mode");
      if (hm) historicalMode = String(hm);
      const hd = reqFormData.get("historical_date");
      if (hd) historicalDate = String(hd);
      const hf = reqFormData.get("historical_file");
      if (hf && typeof (hf as any).arrayBuffer === "function") {
        try {
          const buf = Buffer.from(await (hf as File).arrayBuffer());
          historicalFileBase64 = `data:${(hf as File).type || "image/jpeg"};base64,${buf.toString("base64")}`;
        } catch {
          // Ignore upload parsing error
        }
      }
    } else {
      const jsonBody = await req.json();
      formDataToSend = new FormData();

      if (jsonBody.aoi) aoi = jsonBody.aoi;
      if (jsonBody.prompt) prompt = jsonBody.prompt;
      if (jsonBody.historical_mode || jsonBody.historicalMode) {
        historicalMode = jsonBody.historical_mode || jsonBody.historicalMode;
      }
      if (jsonBody.historical_date || jsonBody.historicalDate) {
        historicalDate = jsonBody.historical_date || jsonBody.historicalDate;
      }
      if (jsonBody.historical_file || jsonBody.historicalFile) {
        historicalFileBase64 = jsonBody.historical_file || jsonBody.historicalFile;
      }

      for (const [key, val] of Object.entries(jsonBody)) {
        if (val !== undefined && val !== null) {
          if (typeof val === "object" && key === "aoi") {
            const aoiObj = val as { north: number; south: number; east: number; west: number };
            formDataToSend.append("aoi_north", String(aoiObj.north));
            formDataToSend.append("aoi_south", String(aoiObj.south));
            formDataToSend.append("aoi_east", String(aoiObj.east));
            formDataToSend.append("aoi_west", String(aoiObj.west));
          } else {
            formDataToSend.append(key, String(val));
          }
        }
      }
    }

    // 1. Check if the Python AI service backend is alive with a fast health check probe
    let backendOnline = false;
    try {
      const healthCheck = await fetch(`${AI_SERVICE_URL}/health`, {
        signal: AbortSignal.timeout(1500),
      });
      backendOnline = healthCheck.ok;
    } catch {
      backendOnline = false;
    }

    // 2. If the backend is running, route directly to it with ample processing time
    if (backendOnline && formDataToSend) {
      try {
        console.log("[/api/temporal/latest] Python backend is online! Routing comparison to AI service...");
        const aiRes = await fetch(`${AI_SERVICE_URL}/temporal/latest`, {
          method: "POST",
          body: formDataToSend,
          signal: AbortSignal.timeout(180000), // 3 minutes for Grounding DINO + full OpenCV
        });

        if (aiRes.ok) {
          const data = await aiRes.json();
          return NextResponse.json(data);
        } else {
          console.warn(`[/api/temporal/latest] Python service returned HTTP ${aiRes.status}`);
        }
      } catch (backendErr) {
        console.warn("[/api/temporal/latest] Python backend execution timed out or failed:", backendErr);
      }
    }

    // 3. Cloud-native fallback: Esri Wayback satellite retrieval, Gemini Vision & Sharp pixel-by-pixel change mask
    console.log("[/api/temporal/latest] Using cloud-native satellite comparison engine with Sharp pixel changes...");
    const cloudResult = await runCloudTemporalComparison({
      aoi,
      prompt,
      historicalMode,
      historicalDate,
      historicalFileBase64,
    });

    return NextResponse.json(cloudResult);
  } catch (error) {
    console.error("[/api/temporal/latest fatal error]:", error);
    try {
      const fallbackResult = await runCloudTemporalComparison({
        aoi,
        prompt,
        historicalMode,
        historicalDate,
        historicalFileBase64,
      });
      return NextResponse.json(fallbackResult);
    } catch {
      const msg = error instanceof Error ? error.message : "Temporal service unavailable";
      return NextResponse.json(
        {
          success: false,
          error: "Temporal comparison failed",
          details: msg,
        },
        { status: 500 }
      );
    }
  }
}
