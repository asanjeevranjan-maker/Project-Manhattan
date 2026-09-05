import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const { imageDataUrl, prompt } = body;

    if (!imageDataUrl) {
      return NextResponse.json(
        { error: "imageDataUrl is required" },
        { status: 400 }
      );
    }

    if (!prompt) {
      return NextResponse.json(
        { error: "prompt is required" },
        { status: 400 }
      );
    }

    let blob: Blob;

    if (imageDataUrl.startsWith("data:")) {
      const commaIdx = imageDataUrl.indexOf(",");
      const mimeMatch = imageDataUrl.match(/data:([^;]+);base64,/);
      const mime = mimeMatch ? mimeMatch[1] : "image/jpeg";
      const base64Data = commaIdx >= 0 ? imageDataUrl.substring(commaIdx + 1) : imageDataUrl;
      const buffer = Buffer.from(base64Data, "base64");
      blob = new Blob([buffer], { type: mime });
    } else {
      const imageResponse = await fetch(imageDataUrl);
      blob = await imageResponse.blob();
    }

    const formData = new FormData();

    formData.append(
      "file",
      blob,
      "satellite-image.jpg"
    );

    formData.append(
      "prompt",
      prompt
    );

    const aiServiceUrl = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";
    const response = await fetch(
      `${aiServiceUrl}/detect`,
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      const errorText = await response.text();

      return NextResponse.json(
        {
          error: "Grounding DINO request failed",
          details: errorText,
        },
        {
          status: response.status,
        }
      );
    }

    const data = await response.json();

    return NextResponse.json(data);

  } catch (error) {
    console.error("Detection API error:", error);

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Detection failed",
      },
      {
        status: 500,
      }
    );
  }
}