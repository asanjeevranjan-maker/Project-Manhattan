// POST /api/fusion/optical-sar
//
// Third analysis mode: Optical + SAR Fusion. Proxies to the Python AI
// service (/fusion/optical-sar). When the backend is offline we return a
// structured error — fusion is NEVER faked on the Node side.

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const f1 = formData.get("file1");
    const f2 = formData.get("file2");
    if (!f1 || !f2 || typeof (f1 as any).arrayBuffer !== "function" || typeof (f2 as any).arrayBuffer !== "function") {
      return NextResponse.json(
        {
          task: "optical_sar_fusion",
          status: "INSUFFICIENT_EVIDENCE",
          error: "Two image files are required: file1 and file2 (one optical, one SAR — order does not matter).",
        },
        { status: 400 }
      );
    }

    // 1. Health probe
    let backendOnline = false;
    try {
      const health = await fetch(`${AI_SERVICE_URL}/health`, { signal: AbortSignal.timeout(1500) });
      backendOnline = health.ok;
    } catch {
      backendOnline = false;
    }
    if (!backendOnline) {
      return NextResponse.json(
        {
          task: "optical_sar_fusion",
          status: "INSUFFICIENT_EVIDENCE",
          error:
            "The Optical + SAR fusion engine (Python AI service) is not running. Start it with ai-service\\venv\\Scripts\\python.exe backend\\main.py and try again. Fusion results are never simulated.",
        },
        { status: 503 }
      );
    }

    // 2. Forward to the Python fusion service (input order preserved — the
    //    backend detects modalities and normalizes slots itself).
    const forward = new FormData();
    forward.append("file1", f1);
    forward.append("file2", f2);
    const p = formData.get("prompt");
    if (p) forward.append("prompt", String(p));

    const aiRes = await fetch(`${AI_SERVICE_URL}/fusion/optical-sar`, {
      method: "POST",
      body: forward,
      signal: AbortSignal.timeout(240000), // DINO/SAM localization can be slow
    });

    const data = await aiRes.json().catch(() => null);
    if (!aiRes.ok || !data) {
      return NextResponse.json(
        data || {
          task: "optical_sar_fusion",
          status: "INSUFFICIENT_EVIDENCE",
          error: `Fusion service returned HTTP ${aiRes.status}.`,
        },
        { status: aiRes.status }
      );
    }
    return NextResponse.json(data);
  } catch (error) {
    console.error("[/api/fusion/optical-sar error]:", error);
    return NextResponse.json(
      {
        task: "optical_sar_fusion",
        status: "INSUFFICIENT_EVIDENCE",
        error: "Optical + SAR fusion request failed.",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
