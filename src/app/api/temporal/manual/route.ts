// POST /api/temporal/manual
//
// Proxies Manual Two-Image Bi-Temporal Comparison to the Python AI service when available,
// or falls back to Cloud-Native Gemini Vision and Sharp pixel-by-pixel change comparison.

import { NextRequest, NextResponse } from "next/server";
import { runCloudManualComparison } from "@/lib/cloud-temporal";
import type { AOIBounds } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";

export async function POST(req: NextRequest) {
  let prompt = "building";
  let dateT1 = "Time 1 (Reference)";
  let dateT2 = "Time 2 (Recent)";
  let t1DataUrl = "";
  let t2DataUrl = "";
  let aoi: AOIBounds | null = null;
  let formDataToSend: FormData | null = null;

  try {
    const rawFormData = await req.formData();
    formDataToSend = new FormData();

    for (const [key, val] of rawFormData.entries()) {
      formDataToSend.append(key, val);
    }

    const p = rawFormData.get("prompt");
    if (p) prompt = String(p);

    const d1 = rawFormData.get("date_t1");
    if (d1) dateT1 = String(d1);

    const d2 = rawFormData.get("date_t2");
    if (d2) dateT2 = String(d2);

    const aoiStr = rawFormData.get("aoi_json");
    if (aoiStr) {
      try {
        aoi = JSON.parse(String(aoiStr));
      } catch {
        // ignore
      }
    }

    const f1 = rawFormData.get("file_t1");
    if (f1 && typeof (f1 as any).arrayBuffer === "function") {
      const buf1 = Buffer.from(await (f1 as File).arrayBuffer());
      t1DataUrl = `data:${(f1 as File).type || "image/jpeg"};base64,${buf1.toString("base64")}`;
    }

    const f2 = rawFormData.get("file_t2");
    if (f2 && typeof (f2 as any).arrayBuffer === "function") {
      const buf2 = Buffer.from(await (f2 as File).arrayBuffer());
      t2DataUrl = `data:${(f2 as File).type || "image/jpeg"};base64,${buf2.toString("base64")}`;
    }

    // 1. Health check probe for Python AI service
    let backendOnline = false;
    try {
      const healthCheck = await fetch(`${AI_SERVICE_URL}/health`, {
        signal: AbortSignal.timeout(1500),
      });
      backendOnline = healthCheck.ok;
    } catch {
      backendOnline = false;
    }

    // 2. If Python backend is online, forward manual comparison
    if (backendOnline && formDataToSend) {
      try {
        console.log("[/api/temporal/manual] Routing manual comparison to Python AI service...");
        const aiRes = await fetch(`${AI_SERVICE_URL}/temporal/manual`, {
          method: "POST",
          body: formDataToSend,
          signal: AbortSignal.timeout(180000),
        });

        if (aiRes.ok) {
          const data = await aiRes.json();
          return NextResponse.json(data);
        }
      } catch (backendErr) {
        console.warn("[/api/temporal/manual] Python AI service execution failed:", backendErr);
      }
    }

    // 3. Cloud fallback with Sharp pixel-by-pixel change detection
    if (t1DataUrl && t2DataUrl) {
      console.log("[/api/temporal/manual] Using cloud-native manual comparison with Sharp pixel changes...");
      const result = await runCloudManualComparison({
        t1DataUrl,
        t2DataUrl,
        prompt,
        dateT1,
        dateT2,
        aoi,
      });
      return NextResponse.json(result);
    }

    return NextResponse.json(
      {
        success: false,
        error: "Missing image data for manual comparison",
      },
      { status: 400 }
    );
  } catch (error) {
    console.error("[/api/temporal/manual error]:", error);
    if (t1DataUrl && t2DataUrl) {
      try {
        const result = await runCloudManualComparison({
          t1DataUrl,
          t2DataUrl,
          prompt,
          dateT1,
          dateT2,
          aoi,
        });
        return NextResponse.json(result);
      } catch {
        // continue to error response
      }
    }
    const msg = error instanceof Error ? error.message : "Manual comparison failed";
    return NextResponse.json(
      {
        success: false,
        error: "Manual comparison failed",
        details: msg,
      },
      { status: 500 }
    );
  }
}
