import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";

export async function POST(request: NextRequest) {
  try {
    const { imageDataUrl } = await request.json();

    if (!imageDataUrl) {
      return NextResponse.json(
        { error: "Image is required" },
        { status: 400 }
      );
    }

    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return NextResponse.json(
        { error: "GEMINI_API_KEY is missing" },
        { status: 500 }
      );
    }

    const ai = new GoogleGenAI({
      apiKey,
    });

    const model = process.env.GEMINI_MODEL || "gemini-3.6-flash";
    const response = await ai.models.generateContent({
      model,
      contents: [
        {
          role: "user",
          parts: [
            {
              inlineData: {
                mimeType: "image/jpeg",
                data: imageDataUrl.split(",")[1],
              },
            },
            {
              text: `
Analyze this satellite image.

Determine whether there are visible signs of flooding.

Focus on:
1. Water bodies
2. Possible inundation of normally dry land
3. River overflow
4. Roads or urban areas affected by water
5. Distinguishing permanent water from possible flooding

Do not invent measurements or coordinates.

Explain what visual evidence supports your conclusion.
              `,
            },
          ],
        },
      ],
    });

    return NextResponse.json({
      success: true,
      result: response.text,
    });

  } catch (error) {
    console.error("Gemini error:", error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}