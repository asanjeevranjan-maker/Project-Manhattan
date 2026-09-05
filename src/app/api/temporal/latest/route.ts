// POST /api/temporal/latest
//
// Proxies Real-Time Bi-Temporal Comparison to the Python AI service.
// Supports both Historical Date retrieval and Historical Image Upload.

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";

export async function POST(req: NextRequest) {
  try {
    const contentType = req.headers.get("content-type") || "";
    let formDataToSend: FormData;

    if (contentType.includes("multipart/form-data")) {
      formDataToSend = await req.formData();
    } else {
      const jsonBody = await req.json();
      formDataToSend = new FormData();
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

    const aiRes = await fetch(`${AI_SERVICE_URL}/temporal/latest`, {
      method: "POST",
      body: formDataToSend,
    });

    if (!aiRes.ok) {
      let errDetail = `HTTP ${aiRes.status}`;
      try {
        const errJson = await aiRes.json();
        if (errJson.error || errJson.details) {
          errDetail = errJson.details || errJson.error;
        }
      } catch {
        // use fallback text
      }
      return NextResponse.json(
        {
          success: false,
          error: "Temporal comparison failed",
          details: errDetail,
        },
        { status: aiRes.status }
      );
    }

    const data = await aiRes.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("[/api/temporal/latest error]:", error);
    const msg = error instanceof Error ? error.message : "Temporal service unavailable";
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

