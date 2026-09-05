// POST /api/analyze
//
// SatQuery AI multimodal analysis endpoint
//
// Supports:
//
// 1. GLM-4.6V-Flash
// 2. Gemini
// 3. GLM -> Gemini fallback
// 4. Grounding DINO detection context
//
// New pipeline:
//
// Grounding DINO
//      ↓
// structured detections
//      ↓
// /api/analyze
//      ↓
// GLM / Gemini
//      ↓
// contextual explanation
//      ↓
// parseAnalysis()
//      ↓
// UI

import {
  NextRequest,
  NextResponse,
} from 'next/server';

import {
  classifyIntent,
  buildPromptForIntent,
  intentLabel,
} from '@/lib/intent';

import { parseAnalysis } from '@/lib/parse';

import type {
  AnalysisResult,
} from '@/lib/types';


// =========================================================
// NEXT.JS CONFIGURATION
// =========================================================

export const runtime = 'nodejs';

export const dynamic =
  'force-dynamic';

export const maxDuration = 300;


// =========================================================
// GROUNDING DINO TYPES
// =========================================================

interface DetectionContextItem {
  label: string;

  confidence: number;

  // Grounding DINO:
  // [x1, y1, x2, y2]
  box: [
    number,
    number,
    number,
    number
  ];
}


interface DetectionContext {
  count: number;

  width: number;

  height: number;

  detections:
    DetectionContextItem[];
}


// =========================================================
// REQUEST TYPE
// =========================================================

interface AnalyzeRequest {

  imageDataUrl: string;

  secondImageDataUrl?: string;

  query: string;

  model?:
    | 'glm'
    | 'gemini';


  history?: Array<{
    role:
      | 'user'
      | 'assistant';

    content: string;
  }>;


  /*
   * NEW
   *
   * Optional detections produced by
   * Grounding DINO before calling
   * GLM/Gemini.
   */

  detectionContext?:
    DetectionContext;
}


// =========================================================
// GLM VISION TYPES
// =========================================================

interface VisionContentItem {

  type:
    | 'text'
    | 'image_url';


  text?: string;


  image_url?: {
    url: string;
  };
}


interface VisionMessage {

  role:
    | 'system'
    | 'user'
    | 'assistant';


  content:
    | string
    | VisionContentItem[];
}


// =========================================================
// INTERNAL MODEL RESPONSE
// =========================================================

interface ModelResponse {

  rawAnswer: string;

  modelUsed:
    | 'glm'
    | 'gemini';
}


// =========================================================
// CUSTOM API ERROR
// =========================================================

class ModelApiError
  extends Error {

  status: number;

  code?:
    | string
    | number;


  constructor(
    message: string,
    status: number,
    code?: string | number,
  ) {

    super(message);

    this.name =
      'ModelApiError';

    this.status =
      status;

    this.code =
      code;
  }
}


// =========================================================
// BUILD GROUNDING DINO CONTEXT
// =========================================================

function buildDetectionContextBlock(
  detectionContext?:
    DetectionContext,
): string {

  if (
    !detectionContext ||
    !Array.isArray(
      detectionContext.detections,
    )
  ) {

    return '';

  }


  const detections =
    detectionContext.detections;


  /*
   * Grounding DINO was called but
   * returned zero detections.
   */

  if (
    detections.length === 0
  ) {

    return `

GROUNDING DINO DETECTION EVIDENCE

A dedicated Grounding DINO object detector was run on this image.

Detector result:
- Matching objects detected: 0

IMPORTANT INSTRUCTIONS:
- Treat the detector result as structured evidence.
- Do not invent object detections that are not supported by the detector.
- You may still describe general visual/geographic context if it is clearly visible.
- Clearly distinguish visual interpretation from detector-confirmed objects.
`;

  }


  // -------------------------------------------------------
  // Group detections by class
  // -------------------------------------------------------

  const grouped =
    new Map<
      string,
      {
        count: number;
        confidences: number[];
      }
    >();


  for (
    const detection
    of detections
  ) {

    const label =
      String(
        detection.label,
      )
        .toLowerCase()
        .trim();


    const existing =
      grouped.get(
        label,
      );


    if (existing) {

      existing.count += 1;

      existing.confidences.push(
        detection.confidence,
      );

    } else {

      grouped.set(
        label,
        {
          count: 1,

          confidences: [
            detection.confidence,
          ],
        },
      );

    }
  }


  // -------------------------------------------------------
  // Class summary
  // -------------------------------------------------------

  const summary =
    Array.from(
      grouped.entries(),
    )
      .map(
        ([label, info]) => {

          const average =
            info.confidences.reduce(
              (
                total,
                value,
              ) =>
                total + value,
              0,
            ) /
            info.confidences.length;


          return (
            `- ${label}: ` +
            `${info.count} detected, ` +
            `average detector confidence ` +
            `${Math.round(
              average * 100,
            )}%`
          );
        },
      )
      .join('\n');


  // -------------------------------------------------------
  // Individual detections
  //
  // Keep a limit so prompts don't become enormous
  // on dense scenes.
  // -------------------------------------------------------

  const MAX_DETECTIONS_IN_PROMPT =
    80;


  const individual =
    detections
      .slice(
        0,
        MAX_DETECTIONS_IN_PROMPT,
      )
      .map(
        (
          detection,
          index,
        ) => {

          const [
            x1,
            y1,
            x2,
            y2,
          ] =
            detection.box;


          const confidence =
            Math.round(
              detection.confidence *
                100,
            );


          /*
           * Convert center to normalized
           * coordinates so the VLM can reason
           * about approximate position.
           */

          const centerX =
            detectionContext.width > 0

              ? (
                  (
                    x1 +
                    x2
                  ) /
                  2
                ) /
                detectionContext.width

              : 0;


          const centerY =
            detectionContext.height > 0

              ? (
                  (
                    y1 +
                    y2
                  ) /
                  2
                ) /
                detectionContext.height

              : 0;


          const horizontal =
            centerX < 0.33
              ? 'left'
              : centerX > 0.66
                ? 'right'
                : 'center';


          const vertical =
            centerY < 0.33
              ? 'upper'
              : centerY > 0.66
                ? 'lower'
                : 'middle';


          return (
            `${index + 1}. ` +
            `${detection.label} | ` +
            `confidence ${confidence}% | ` +
            `approximate image position: ` +
            `${vertical}-${horizontal} | ` +
            `box: [` +
            `${Math.round(x1)}, ` +
            `${Math.round(y1)}, ` +
            `${Math.round(x2)}, ` +
            `${Math.round(y2)}]`
          );
        },
      )
      .join('\n');


  const truncatedNotice =
    detections.length >
    MAX_DETECTIONS_IN_PROMPT

      ? `\nNote: only the first ${MAX_DETECTIONS_IN_PROMPT} individual detections are listed because the detector returned ${detections.length} objects.`

      : '';


  return `

============================================================
GROUNDING DINO DETECTION EVIDENCE
============================================================

A dedicated Grounding DINO object detector has already analyzed the image.

Image size:
${detectionContext.width} x ${detectionContext.height} pixels

Total detector objects:
${detectionContext.count}

DETECTION SUMMARY:

${summary}

INDIVIDUAL DETECTIONS:

${individual}

${truncatedNotice}

IMPORTANT REASONING RULES:

1. Grounding DINO is the primary source of truth for object COUNT and bounding-box LOCATION.

2. Do NOT invent additional detector-confirmed objects.

3. Do NOT change the detector count simply because you visually think there may be more or fewer objects.

4. You may use the image itself to provide contextual interpretation, such as:
   - surrounding land use
   - port activity
   - urban density
   - proximity to roads/water
   - spatial arrangement
   - possible geographic context

5. Clearly distinguish:
   - detector-confirmed facts
   from
   - your visual interpretation.

6. Detector confidence is NOT a probability that your entire explanation is correct.

7. If the detector result seems uncertain or incomplete, mention that limitation instead of inventing detections.

8. Base your human-readable explanation on BOTH:
   - the structured Grounding DINO detections
   - the visible image context.

============================================================
`;
}


// =========================================================
// GLM
// =========================================================

async function callGLM(
  messages:
    VisionMessage[],
  apiKey: string,
): Promise<ModelResponse> {

  console.log(
    '[GLM] Sending request to GLM-4.6V-Flash...',
  );


  let response:
    Response;


  try {

    response =
      await fetch(
        'https://api.z.ai/api/paas/v4/chat/completions',
        {
          method:
            'POST',

          headers: {
            'Content-Type':
              'application/json',

            Authorization:
              `Bearer ${apiKey}`,
          },

          body:
            JSON.stringify({
              model:
                'glm-4.6v-flash',

              messages,

              temperature:
                0.1,

              max_tokens:
                1800,
            }),
        },
      );

  } catch (error) {

    console.error(
      '[GLM] Network error:',
      error,
    );


    throw new ModelApiError(
      `GLM network request failed: ${
        error instanceof Error
          ? error.message
          : String(error)
      }`,
      503,
    );
  }


  // -------------------------------------------------------
  // Read response once
  // -------------------------------------------------------

  let data:
    any;


  try {

    data =
      await response.json();

  } catch {

    throw new ModelApiError(
      'GLM returned an invalid JSON response.',
      response.status ||
        502,
    );
  }


  console.log(
    '[GLM] HTTP status:',
    response.status,
  );


  // -------------------------------------------------------
  // API error
  // -------------------------------------------------------

  if (
    !response.ok
  ) {

    const message =
      data?.error?.message ||
      data?.message ||
      'GLM API request failed.';


    const code =
      data?.error?.code ||
      data?.code;


    console.error(
      '[GLM] API error:',
      {
        status:
          response.status,

        code,

        message,
      },
    );


    throw new ModelApiError(
      message,
      response.status,
      code,
    );
  }


  // -------------------------------------------------------
  // Extract answer
  // -------------------------------------------------------

  const rawAnswer =
    data
      ?.choices
      ?.[0]
      ?.message
      ?.content;


  if (
    typeof rawAnswer !==
      'string' ||
    rawAnswer
      .trim()
      .length === 0
  ) {

    console.error(
      '[GLM] Empty response:',
      data,
    );


    throw new ModelApiError(
      'GLM returned no usable analysis response.',
      502,
    );
  }


  console.log(
    '[GLM] Analysis received successfully.',
  );


  return {
    rawAnswer,

    modelUsed:
      'glm',
  };
}


// =========================================================
// GEMINI IMAGE CONVERSION
// =========================================================

async function imageToGeminiPart(
  imageUrl: string,
) {

  // -------------------------------------------------------
  // Base64 data URL
  // -------------------------------------------------------

  if (
    imageUrl.startsWith(
      'data:',
    )
  ) {

    const match =
      imageUrl.match(
        /^data:([^;]+);base64,(.+)$/,
      );


    if (!match) {

      throw new Error(
        'Invalid base64 image data URL.',
      );
    }


    const mimeType =
      match[1];


    const base64Data =
      match[2];


    return {
      inline_data: {

        mime_type:
          mimeType,

        data:
          base64Data,
      },
    };
  }


  // -------------------------------------------------------
  // HTTP / HTTPS image
  // -------------------------------------------------------

  if (
    imageUrl.startsWith(
      'http://',
    ) ||
    imageUrl.startsWith(
      'https://',
    )
  ) {

    const imageResponse =
      await fetch(
        imageUrl,
      );


    if (
      !imageResponse.ok
    ) {

      throw new Error(
        `Unable to download image. HTTP ${imageResponse.status}`,
      );
    }


    const mimeType =
      imageResponse.headers.get(
        'content-type',
      ) ||
      'image/jpeg';


    const buffer =
      await imageResponse.arrayBuffer();


    const base64Data =
      Buffer
        .from(
          buffer,
        )
        .toString(
          'base64',
        );


    return {
      inline_data: {

        mime_type:
          mimeType,

        data:
          base64Data,
      },
    };
  }


  throw new Error(
    'Image must be a data URL or HTTP/HTTPS URL.',
  );
}


// =========================================================
// GEMINI
// =========================================================

async function callGemini(

  imageDataUrl:
    string,

  secondImageDataUrl:
    string |
    undefined,

  prompt:
    string,

  apiKey:
    string,

): Promise<ModelResponse> {

  const model =
    process.env
      .GEMINI_MODEL ||
    'gemini-3.6-flash';


  console.log(
    `[GEMINI] Sending request to ${model}...`,
  );


  // -------------------------------------------------------
  // Build parts
  // -------------------------------------------------------

  const parts:
    Array<
      Record<
        string,
        unknown
      >
    > = [];


  // Prompt
  parts.push({
    text:
      prompt,
  });


  // Primary image
  parts.push(
    await imageToGeminiPart(
      imageDataUrl,
    ),
  );


  // -------------------------------------------------------
  // Optional second image
  // -------------------------------------------------------

  if (
    secondImageDataUrl
  ) {

    parts.push({
      text:
        'The next image is the AFTER image. ' +
        'The previous image was the BEFORE image. ' +
        'Compare corresponding areas carefully.',
    });


    parts.push(
      await imageToGeminiPart(
        secondImageDataUrl,
      ),
    );
  }


  // -------------------------------------------------------
  // Endpoint
  // -------------------------------------------------------

  const endpoint =
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;


  let response:
    Response;


  try {

    response =
      await fetch(
        endpoint,
        {
          method:
            'POST',

          headers: {
            'Content-Type':
              'application/json',
          },

          body:
            JSON.stringify({
              contents: [
                {
                  role:
                    'user',

                  parts,
                },
              ],

              generationConfig: {

                temperature:
                  0.1,

                maxOutputTokens:
                  1800,
              },
            }),
        },
      );

  } catch (error) {

    throw new ModelApiError(
      `Gemini network request failed: ${
        error instanceof Error
          ? error.message
          : String(error)
      }`,
      503,
    );
  }


  // -------------------------------------------------------
  // Parse response
  // -------------------------------------------------------

  let data:
    any;


  try {

    data =
      await response.json();

  } catch {

    throw new ModelApiError(
      'Gemini returned an invalid response.',
      response.status ||
        502,
    );
  }


  console.log(
    '[GEMINI] HTTP status:',
    response.status,
  );


  // -------------------------------------------------------
  // API error
  // -------------------------------------------------------

  if (
    !response.ok
  ) {

    const message =
      data?.error?.message ||
      'Gemini API request failed.';


    const code =
      data?.error?.status ||
      data?.error?.code;


    console.error(
      '[GEMINI] API error:',
      {
        status:
          response.status,

        code,

        message,
      },
    );


    throw new ModelApiError(
      message,
      response.status,
      code,
    );
  }


  // -------------------------------------------------------
  // Extract response
  // -------------------------------------------------------

  const candidates =
    data?.candidates;


  const rawAnswer =
    candidates
      ?.[0]
      ?.content
      ?.parts
      ?.map(
        (
          part: {
            text?: string;
          },
        ) =>
          part?.text || '',
      )
      .join('') ||
    '';


  if (
    typeof rawAnswer !==
      'string' ||
    rawAnswer
      .trim()
      .length === 0
  ) {

    console.error(
      '[GEMINI] Empty response:',
      data,
    );


    throw new ModelApiError(
      'Gemini returned an empty response.',
      502,
    );
  }


  console.log(
    `[GEMINI] ${model} analysis received successfully.`,
  );


  return {
    rawAnswer,

    modelUsed:
      'gemini',
  };
}


// =========================================================
// POST /api/analyze
// =========================================================

export async function POST(
  req:
    NextRequest,
) {

  // -------------------------------------------------------
  // 1. Read request
  // -------------------------------------------------------

  let body:
    AnalyzeRequest;


  try {

    body =
      (
        await req.json()
      ) as AnalyzeRequest;

  } catch {

    return NextResponse.json(
      {
        error:
          'Invalid JSON request body.',
      },
      {
        status:
          400,
      },
    );
  }


  // -------------------------------------------------------
  // 2. Extract fields
  // -------------------------------------------------------

  const {
    imageDataUrl,

    secondImageDataUrl,

    query,

    model =
      'glm',

    history,

    // NEW
    detectionContext,

  } = body;

  // -------------------------------------------------------
  // Check if Python FastAPI vision service is running
  // -------------------------------------------------------
  const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";
  let isBackendAlive = false;
  try {
    const probe = await fetch(`${AI_SERVICE_URL}/health`, {
      signal: AbortSignal.timeout(1500),
    });
    isBackendAlive = probe.ok;
  } catch {
    isBackendAlive = false;
  }

  if (isBackendAlive) {
    try {
      console.log("[/api/analyze] Forwarding analysis to Python FastAPI vision service...");
      const pyRes = await fetch(`${AI_SERVICE_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(60000),
      });
      if (pyRes.ok) {
        const pyData = await pyRes.json();
        return NextResponse.json(pyData);
      }
    } catch (err) {
      console.warn("[/api/analyze] Python backend request failed, continuing to TypeScript engine:", err);
    }
  }

  // -------------------------------------------------------
  // Debug
  // -------------------------------------------------------

  if (
    detectionContext
  ) {

    console.log(
      '[ANALYZE] Grounding DINO context received:',
      {
        count:
          detectionContext.count,

        detections:
          detectionContext
            .detections
            ?.length ||
          0,

        width:
          detectionContext.width,

        height:
          detectionContext.height,
      },
    );

  } else {

    console.log(
      '[ANALYZE] No Grounding DINO context received.',
    );
  }


  // -------------------------------------------------------
  // 3. Validate model
  // -------------------------------------------------------

  if (
    model !==
      'glm' &&
    model !==
      'gemini'
  ) {

    return NextResponse.json(
      {
        error:
          'Invalid model. Use "glm" or "gemini".',
      },
      {
        status:
          400,
      },
    );
  }


  // -------------------------------------------------------
  // 4. Validate image
  // -------------------------------------------------------

  if (
    !imageDataUrl ||
    typeof imageDataUrl !==
      'string'
  ) {

    return NextResponse.json(
      {
        error:
          'imageDataUrl is required.',
      },
      {
        status:
          400,
      },
    );
  }


  // -------------------------------------------------------
  // 5. Validate query
  // -------------------------------------------------------

  if (
    !query ||
    typeof query !==
      'string' ||
    query
      .trim()
      .length === 0
  ) {

    return NextResponse.json(
      {
        error:
          'query is required.',
      },
      {
        status:
          400,
      },
    );
  }


  // -------------------------------------------------------
  // 6. Validate primary image URL
  // -------------------------------------------------------

  const isDataUrl =
    imageDataUrl
      .startsWith(
        'data:',
      );


  const isHttpUrl =
    imageDataUrl
      .startsWith(
        'http://',
      ) ||
    imageDataUrl
      .startsWith(
        'https://',
      );


  if (
    !isDataUrl &&
    !isHttpUrl
  ) {

    return NextResponse.json(
      {
        error:
          'imageDataUrl must be a data: URL or an http(s) URL.',
      },
      {
        status:
          400,
      },
    );
  }


  // -------------------------------------------------------
  // 7. Validate second image
  // -------------------------------------------------------

  if (
    secondImageDataUrl
  ) {

    const secondIsDataUrl =
      secondImageDataUrl
        .startsWith(
          'data:',
        );


    const secondIsHttpUrl =
      secondImageDataUrl
        .startsWith(
          'http://',
        ) ||
      secondImageDataUrl
        .startsWith(
          'https://',
        );


    if (
      !secondIsDataUrl &&
      !secondIsHttpUrl
    ) {

      return NextResponse.json(
        {
          error:
            'secondImageDataUrl must be a data: URL or an http(s) URL.',
        },
        {
          status:
            400,
        },
      );
    }
  }


  // -------------------------------------------------------
  // 8. Validate detection context if supplied
  // -------------------------------------------------------

  if (
    detectionContext
  ) {

    if (
      typeof detectionContext.count !==
        'number' ||
      !Array.isArray(
        detectionContext.detections,
      )
    ) {

      return NextResponse.json(
        {
          error:
            'Invalid detectionContext.',
        },
        {
          status:
            400,
        },
      );
    }
  }


  // -------------------------------------------------------
  // 9. Determine intent
  // -------------------------------------------------------

  const intent =
    classifyIntent(
      query,
    );


  // -------------------------------------------------------
  // 10. Build normal SatQuery prompt
  // -------------------------------------------------------

  const basePrompt =
    buildPromptForIntent(
      intent,

      query,

      Boolean(
        secondImageDataUrl,
      ),
    );


  // -------------------------------------------------------
  // 11. Add Grounding DINO evidence
  // -------------------------------------------------------

  const detectionBlock =
    buildDetectionContextBlock(
      detectionContext,
    );


  const enhancedPrompt =
    detectionBlock

      ? `${basePrompt}

${detectionBlock}

FINAL TASK:

Answer the user's original question using the required SatQuery structured response format.

Use Grounding DINO for detector-confirmed object counts and locations.

Use your vision capability to provide useful context and interpretation around those detector results.

Original user question:
"${query}"
`

      : basePrompt;


  // -------------------------------------------------------
  // Debug
  // -------------------------------------------------------

  console.log(
    '[ANALYZE] Using detection context:',
    Boolean(
      detectionBlock,
    ),
  );


  // -------------------------------------------------------
  // 12. Build GLM multimodal content
  // -------------------------------------------------------

  const content:
    VisionContentItem[] = [
      {
        type:
          'text',

        text:
          enhancedPrompt,
      },
    ];


  // Primary image
  content.push({
    type:
      'image_url',

    image_url: {
      url:
        imageDataUrl,
    },
  });


  // -------------------------------------------------------
  // Optional second image
  // -------------------------------------------------------

  if (
    secondImageDataUrl
  ) {

    content.push({
      type:
        'text',

      text:
        'The next image is the AFTER image. ' +
        'The previous image was the BEFORE image.',
    });


    content.push({
      type:
        'image_url',

      image_url: {
        url:
          secondImageDataUrl,
      },
    });
  }


  // -------------------------------------------------------
  // 13. Build conversation history
  // -------------------------------------------------------

  const messages:
    VisionMessage[] = [];


  if (
    history &&
    Array.isArray(
      history,
    ) &&
    history.length > 0
  ) {

    const recent =
      history.slice(
        -6,
      );


    for (
      const message
      of recent
    ) {

      if (
        message.role ===
          'user' ||
        message.role ===
          'assistant'
      ) {

        messages.push({
          role:
            message.role,

          content:
            message.content,
        });
      }
    }
  }


  // -------------------------------------------------------
  // Current multimodal request
  // -------------------------------------------------------

  messages.push({
    role:
      'user',

    content,
  });


  // -------------------------------------------------------
  // 14. API keys
  // -------------------------------------------------------

  const zaiKey =
    process.env
      .ZAI_API_KEY;


  const geminiKey =
    process.env
      .GEMINI_API_KEY;


  // -------------------------------------------------------
  // 15. Validate selected model key
  // -------------------------------------------------------

  if (
    model ===
      'glm' &&
    !zaiKey
  ) {

    return NextResponse.json(
      {
        error:
          'ZAI_API_KEY is not configured in .env.',
      },
      {
        status:
          500,
      },
    );
  }


  if (
    model ===
      'gemini' &&
    !geminiKey
  ) {

    return NextResponse.json(
      {
        error:
          'GEMINI_API_KEY is not configured in .env.',
      },
      {
        status:
          500,
      },
    );
  }


  // -------------------------------------------------------
  // 16. Call model
  // -------------------------------------------------------

  try {

    let result:
      ModelResponse;


    let fallbackUsed =
      false;


    // =====================================================
    // USER SELECTED GLM
    // =====================================================

    if (
      model ===
      'glm'
    ) {

      try {

        result =
          await callGLM(
            messages,
            zaiKey!,
          );

      } catch (error) {

        const status =
          error instanceof
            ModelApiError

            ? error.status

            : 0;


        const shouldFallback =
          status === 429 ||
          status === 500 ||
          status === 502 ||
          status === 503 ||
          status === 504;


        if (
          !shouldFallback
        ) {

          throw error;
        }


        console.warn(
          `[ANALYZE] GLM unavailable (${status}).`,
        );


        if (
          !geminiKey
        ) {

          return NextResponse.json(
            {
              error:
                'GLM is currently unavailable and GEMINI_API_KEY is not configured for fallback.',

              code:
                error instanceof
                  ModelApiError

                  ? error.code

                  : undefined,

              retryable:
                true,
            },
            {
              status:
                status ||
                503,
            },
          );
        }


        // -----------------------------------------------
        // GLM -> GEMINI FALLBACK
        // -----------------------------------------------

        console.log(
          '[ANALYZE] Falling back from GLM → Gemini.',
        );


        /*
         * IMPORTANT:
         *
         * Use enhancedPrompt here,
         * not basePrompt.
         *
         * This ensures Gemini also receives
         * Grounding DINO detections.
         */

        result =
          await callGemini(
            imageDataUrl,

            secondImageDataUrl,

            enhancedPrompt,

            geminiKey,
          );


        fallbackUsed =
          true;
      }

    }


    // =====================================================
    // USER SELECTED GEMINI
    // =====================================================

    else {

      /*
       * Gemini also gets
       * Grounding DINO context.
       */

      result =
        await callGemini(
          imageDataUrl,

          secondImageDataUrl,

          enhancedPrompt,

          geminiKey!,
        );
    }


    // -------------------------------------------------------
    // 17. Parse AI response
    // -------------------------------------------------------

    const analysis:
      AnalysisResult =
        parseAnalysis(
          result.rawAnswer,
          intent,
        );


    // -------------------------------------------------------
    // 18. Return result
    // -------------------------------------------------------

    return NextResponse.json({

      analysis,


      rawAnswer:
        result.rawAnswer,


      intent,


      intentLabel:
        intentLabel(
          intent,
        ),


      modelUsed:
        result.modelUsed,


      fallbackUsed,


      /*
       * Useful while we're testing
       * DINO -> VLM integration.
       */

      usedDetectionContext:
        Boolean(
          detectionBlock,
        ),
    });


  } catch (error) {

    // -------------------------------------------------------
    // Final error handling
    // -------------------------------------------------------

    const message =
      error instanceof
        Error

        ? error.message

        : 'Unknown error';


    const status =
      error instanceof
        ModelApiError

        ? error.status

        : 500;


    const code =
      error instanceof
        ModelApiError

        ? error.code

        : undefined;


    console.error(
      '[/api/analyze] Final error:',
      {
        message,
        status,
        code,
      },
    );


    // -------------------------------------------------------
    // Rate limit
    // -------------------------------------------------------

    if (
      status ===
      429
    ) {

      return NextResponse.json(
        {
          error:
            'The selected AI service is currently rate-limited. Please try again in a moment.',

          code,

          retryable:
            true,
        },
        {
          status:
            429,
        },
      );
    }


    // -------------------------------------------------------
    // Authentication
    // -------------------------------------------------------

    if (
      status ===
        401 ||
      status ===
        403
    ) {

      return NextResponse.json(
        {
          error:
            'API authentication failed. Check your API key and model permissions.',

          code,

          retryable:
            false,
        },
        {
          status,
        },
      );
    }


    // -------------------------------------------------------
    // Bad request
    // -------------------------------------------------------

    if (
      status ===
      400
    ) {

      return NextResponse.json(
        {
          error:
            message ||
            'The AI service rejected the request.',

          code,

          retryable:
            false,
        },
        {
          status:
            400,
        },
      );
    }


    // -------------------------------------------------------
    // Request too large
    // -------------------------------------------------------

    if (
      status ===
      413
    ) {

      return NextResponse.json(
        {
          error:
            'The image or request is too large. Try a smaller or compressed image.',

          code,

          retryable:
            false,
        },
        {
          status:
            413,
        },
      );
    }


    // -------------------------------------------------------
    // Model/server failure
    // -------------------------------------------------------

    if (
      status >= 500 &&
      status <= 599
    ) {

      return NextResponse.json(
        {
          error:
            'The AI service is temporarily unavailable. Please try again.',

          code,

          retryable:
            true,
        },
        {
          status,
        },
      );
    }


    // -------------------------------------------------------
    // Generic
    // -------------------------------------------------------

    return NextResponse.json(
      {
        error:
          `Failed to analyze image: ${message}`,

        code,

        retryable:
          false,
      },
      {
        status:
          500,
      },
    );
  }
}