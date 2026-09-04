// POST /api/analyze
// Accepts:
// {
//   imageDataUrl: string,
//   secondImageDataUrl?: string,
//   query: string,
//   history?: Array<{ role: 'user' | 'assistant'; content: string }>
// }
//
// Returns:
// {
//   analysis: AnalysisResult,
//   rawAnswer: string,
//   intent: Intent
// }
//
// Uses the Z.ai API with GLM-4.6V-Flash for image analysis.

import { NextRequest, NextResponse } from 'next/server';
import {
  classifyIntent,
  buildPromptForIntent,
  intentLabel,
} from '@/lib/intent';
import { parseAnalysis } from '@/lib/parse';
import type { AnalysisResult } from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 300;

interface AnalyzeRequest {
  imageDataUrl: string;
  secondImageDataUrl?: string;
  query: string;

  // User-selected AI model
  model?: 'glm' | 'gemini';

  history?: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
}

interface VisionContentItem {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: {
    url: string;
  };
}

interface VisionMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | VisionContentItem[];
}

export async function POST(req: NextRequest) {
  // -----------------------------------------
  // 1. Read request body
  // -----------------------------------------

  let body: AnalyzeRequest;

  try {
    body = (await req.json()) as AnalyzeRequest;
  } catch {
    return NextResponse.json(
      { error: 'Invalid JSON body' },
      { status: 400 }
    );
  }

const {
  imageDataUrl,
  secondImageDataUrl,
  query,
  model = 'glm',
  history,
} = body;

  // -----------------------------------------
  // 2. Validate image
  // -----------------------------------------

  if (!imageDataUrl || typeof imageDataUrl !== 'string') {
    return NextResponse.json(
      {
        error:
          'imageDataUrl is required (a base64 data URL)',
      },
      { status: 400 }
    );
  }

  // -----------------------------------------
  // 3. Validate query
  // -----------------------------------------

  if (
    !query ||
    typeof query !== 'string' ||
    query.trim().length === 0
  ) {
    return NextResponse.json(
      { error: 'query is required' },
      { status: 400 }
    );
  }

  // -----------------------------------------
  // 4. Validate image URL
  // -----------------------------------------

  const isDataUrl = imageDataUrl.startsWith('data:');

  const isHttpUrl =
    imageDataUrl.startsWith('http://') ||
    imageDataUrl.startsWith('https://');

  if (!isDataUrl && !isHttpUrl) {
    return NextResponse.json(
      {
        error:
          'imageDataUrl must be a data: URL or an https:// URL',
      },
      { status: 400 }
    );
  }

  // -----------------------------------------
  // 5. Classify user intent
  // -----------------------------------------

  const intent = classifyIntent(query);

  const prompt = buildPromptForIntent(
    intent,
    query,
    Boolean(secondImageDataUrl)
  );

  // -----------------------------------------
  // 6. Build vision content
  // -----------------------------------------

  const content: VisionContentItem[] = [
    {
      type: 'text',
      text: prompt,
    },
  ];

  // Primary image
  content.push({
    type: 'image_url',
    image_url: {
      url: imageDataUrl,
    },
  });

  // -----------------------------------------
  // 7. Add second image if provided
  // -----------------------------------------

  if (secondImageDataUrl) {
    const isSecondDataUrl =
      secondImageDataUrl.startsWith('data:');

    const isSecondHttpUrl =
      secondImageDataUrl.startsWith('http://') ||
      secondImageDataUrl.startsWith('https://');

    if (!isSecondDataUrl && !isSecondHttpUrl) {
      return NextResponse.json(
        {
          error:
            'secondImageDataUrl must be a data: URL or an https:// URL',
        },
        { status: 400 }
      );
    }

    content.push({
      type: 'text',
      text:
        'The next image is the AFTER image (later date). ' +
        'The previous image was the BEFORE image (earlier date).',
    });

    content.push({
      type: 'image_url',
      image_url: {
        url: secondImageDataUrl,
      },
    });
  }

  // -----------------------------------------
  // 8. Build messages
  // -----------------------------------------

  const messages: VisionMessage[] = [];

  if (
    history &&
    Array.isArray(history) &&
    history.length > 0
  ) {
    // Keep only the last 6 messages
    const recent = history.slice(-6);

    for (const message of recent) {
      if (
        message.role === 'user' ||
        message.role === 'assistant'
      ) {
        messages.push({
          role: message.role,
          content: message.content,
        });
      }
    }
  }

  // Current message must be last
  messages.push({
    role: 'user',
    content,
  });

  // -----------------------------------------
  // 9. Check API key
  // -----------------------------------------

  const apiKey = process.env.ZAI_API_KEY;

  if (!apiKey) {
    console.error('ZAI_API_KEY is not configured.');

    return NextResponse.json(
      {
        error:
          'Z.ai API key is not configured. Please add ZAI_API_KEY to .env.',
      },
      { status: 500 }
    );
  }

  // -----------------------------------------
  // 10. Call Z.ai API
  // -----------------------------------------

  try {
    const response = await fetch(
      'https://api.z.ai/api/paas/v4/chat/completions',
      {
        method: 'POST',

        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },

        body: JSON.stringify({
          model: 'glm-4.6v-flash',
          messages,
          temperature: 0.2,
          max_tokens: 1800,
        }),
      }
    );

    // -----------------------------------------
    // 11. Read Z.ai response
    // -----------------------------------------

    const completion = await response.json();

    console.log(
      '========== Z.AI RESPONSE =========='
    );

    console.log(
      JSON.stringify(completion, null, 2)
    );

    console.log(
      '==================================='
    );

    // -----------------------------------------
    // 12. Handle API errors
    // -----------------------------------------

    // -----------------------------------------
// 12. Handle API errors
// -----------------------------------------

if (!response.ok) {
  const status = response.status;

  const apiError =
    completion?.error?.message ||
    completion?.message ||
    'Z.ai API request failed';

  const apiCode =
    completion?.error?.code ||
    completion?.code ||
    null;

  console.error('========== Z.AI API ERROR ==========');
  console.error('HTTP STATUS:', status);
  console.error('API CODE:', apiCode);
  console.error('API MESSAGE:', apiError);
  console.error(
    'FULL RESPONSE:',
    JSON.stringify(completion, null, 2)
  );
  console.error('====================================');

  // -----------------------------------------
  // 429 - Rate limit / overloaded
  // -----------------------------------------

  if (status === 429) {
    return NextResponse.json(
      {
        error:
          'Z.ai rate limit reached. Please wait a moment before trying again.',
        code: apiCode,
        status: 429,
        retryable: true,
      },
      {
        status: 429,
      }
    );
  }

  // -----------------------------------------
  // 401 - Invalid API key
  // -----------------------------------------

  if (status === 401) {
    return NextResponse.json(
      {
        error:
          'Z.ai API key is invalid or unauthorized.',
        code: apiCode,
        status: 401,
        retryable: false,
      },
      {
        status: 401,
      }
    );
  }

  // -----------------------------------------
  // 403 - Permission problem
  // -----------------------------------------

  if (status === 403) {
    return NextResponse.json(
      {
        error:
          'Z.ai API access was denied. Check your account, API key, or model permissions.',
        code: apiCode,
        status: 403,
        retryable: false,
      },
      {
        status: 403,
      }
    );
  }

  // -----------------------------------------
  // 400 - Bad request
  // -----------------------------------------

  if (status === 400) {
    return NextResponse.json(
      {
        error:
          'Z.ai rejected the request. Check the model, image format, prompt, or request parameters.',
        code: apiCode,
        status: 400,
        details: apiError,
        retryable: false,
      },
      {
        status: 400,
      }
    );
  }

  // -----------------------------------------
  // 413 - Request too large
  // -----------------------------------------

  if (status === 413) {
    return NextResponse.json(
      {
        error:
          'The image or request is too large for Z.ai. Try using a smaller or compressed image.',
        code: apiCode,
        status: 413,
        retryable: false,
      },
      {
        status: 413,
      }
    );
  }

  // -----------------------------------------
  // 500-599 - Z.ai server error
  // -----------------------------------------

  if (status >= 500 && status <= 599) {
    return NextResponse.json(
      {
        error:
          'Z.ai is currently experiencing a server-side problem. Please try again later.',
        code: apiCode,
        status,
        retryable: true,
      },
      {
        status,
      }
    );
  }

  // -----------------------------------------
  // Other errors
  // -----------------------------------------

  return NextResponse.json(
    {
      error: apiError,
      code: apiCode,
      status,
      retryable: false,
    },
    {
      status,
    }
  );
}

    // -----------------------------------------
    // 13. Extract model response
    // -----------------------------------------

    const rawAnswer =
      completion?.choices?.[0]?.message?.content ?? '';

    if (
      typeof rawAnswer !== 'string' ||
      rawAnswer.trim().length === 0
    ) {
      console.error(
        'Z.ai returned an empty response:',
        JSON.stringify(completion, null, 2)
      );

      return NextResponse.json(
        {
          error:
            'The vision model returned an empty response.',
        },
        { status: 502 }
      );
    }

    // -----------------------------------------
    // 14. Parse analysis
    // -----------------------------------------

    const analysis: AnalysisResult =
      parseAnalysis(rawAnswer, intent);

    // -----------------------------------------
    // 15. Return result to frontend
    // -----------------------------------------

    return NextResponse.json({
      analysis,
      rawAnswer,
      intent,
      intentLabel: intentLabel(intent),
    });
  } catch (err) {
    const message =
      err instanceof Error
        ? err.message
        : 'Unknown error';

    console.error(
      '[/api/analyze] Z.ai API call failed:',
      message
    );

    return NextResponse.json(
      {
        error: `Failed to analyze image: ${message}`,
      },
      { status: 500 }
    );
  }
}