import { NextRequest, NextResponse } from "next/server";
import { runCloudDetection } from "@/lib/cloud-temporal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  let imageDataUrl = "";
  let prompt = "";

  try {
    const body = await req.json();

    imageDataUrl = body.imageDataUrl;
    prompt = body.prompt;
    const preset = body.preset;

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

    // 1. Try local/dedicated Python AI service if reachable
    const aiServiceUrl = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";
    if (aiServiceUrl) {
      try {
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
        formData.append("file", blob, "satellite-image.jpg");
        formData.append("prompt", prompt);
        if (preset) {
          formData.append("preset", preset);
        }

        const controller = new AbortController();
        const timeoutMs = aiServiceUrl.includes("127.0.0.1") || aiServiceUrl.includes("localhost") ? 3500 : 30000;
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

        const response = await fetch(`${aiServiceUrl}/detect`, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (response.ok) {
          const data = await response.json();
          return NextResponse.json(data);
        }
      } catch (aiErr) {
        console.warn("[/api/detect] Dedicated AI service unavailable, using cloud-native detection:", aiErr);
      }
    }

    // 2. Cloud-native fallback detection with Gemini Vision
    const cloudDetection = await runCloudDetection(imageDataUrl, prompt, preset);
    return NextResponse.json(cloudDetection);
  } catch (error) {
    console.error("Detection API error:", error);

    if (imageDataUrl && prompt) {
      try {
        const cloudDetection = await runCloudDetection(imageDataUrl, prompt);
        return NextResponse.json(cloudDetection);
      } catch {
        // continue to error response
      }
    }

    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Detection failed",
      },
      {
        status: 500,
      }
    );
  }
}