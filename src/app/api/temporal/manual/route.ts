// POST /api/temporal/manual
//
// Proxies Manual Two-Image Bi-Temporal Comparison to the Python AI service.

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();

    const aiRes = await fetch(`${AI_SERVICE_URL}/temporal/manual`, {
      method: "POST",
      body: formData,
    });

    if (!aiRes.ok) {
      let errDetail = `HTTP ${aiRes.status}`;
      try {
        const errJson = await aiRes.json();
        if (errJson.error || errJson.details) {
          errDetail = errJson.details || errJson.error;
        }
      } catch {
        // fallback
      }
      return NextResponse.json(
        {
          success: false,
          error: "Manual comparison failed",
          details: errDetail,
        },
        { status: aiRes.status }
      );
    }

    const data = await aiRes.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("[/api/temporal/manual error]:", error);
    const msg = error instanceof Error ? error.message : "Manual comparison failed";
    return NextResponse.json(
      {
        success: false,
        error: "Failed to connect to temporal comparison engine",
        details: `${msg}. Please ensure the AI service backend is running on port 8000.`,
      },
      { status: 500 }
    );
  }
}

