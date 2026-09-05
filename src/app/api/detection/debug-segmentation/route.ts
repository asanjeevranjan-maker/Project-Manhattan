import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const aiServiceUrl = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";
    const contentType = req.headers.get("content-type") || "";

    let formData: FormData;
    if (contentType.includes("multipart/form-data")) {
      formData = await req.formData();
    } else {
      const body = await req.json();
      formData = new FormData();
      const imageDataUrl = body.imageDataUrl || body.image_data;
      if (!imageDataUrl) {
        return NextResponse.json({ error: "imageDataUrl is required" }, { status: 400 });
      }
      const commaIdx = imageDataUrl.indexOf(",");
      const base64Data = commaIdx >= 0 ? imageDataUrl.substring(commaIdx + 1) : imageDataUrl;
      const buffer = Buffer.from(base64Data, "base64");
      const blob = new Blob([buffer], { type: "image/jpeg" });
      formData.append("file", blob, "image.jpg");
      formData.append("box", typeof body.box === "string" ? body.box : JSON.stringify(body.box));
      formData.append("label", body.label || "ship");
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000);

    const res = await fetch(`${aiServiceUrl}/detect/debug-segmentation`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err) {
    return NextResponse.json(
      { error: "Debug segmentation proxy failed", details: String(err) },
      { status: 500 }
    );
  }
}

