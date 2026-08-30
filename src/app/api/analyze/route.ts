// POST /api/analyze
// Accepts: { imageDataUrl: string, secondImageDataUrl?: string, query: string, history?: Array<{role, content}> }
// Returns: { analysis: AnalysisResult, rawAnswer: string, intent: Intent }
//
// Uses the z-ai-web-dev-sdk VLM (chat.completions.createVision) to analyze the
// satellite image against the user's natural-language query. The VLM is given
// a task-specific prompt depending on the detected intent (see lib/intent.ts).

import { NextRequest, NextResponse } from 'next/server';
import ZAI from 'z-ai-web-dev-sdk';
import { classifyIntent, buildPromptForIntent, intentLabel } from '@/lib/intent';
import { parseAnalysis } from '@/lib/parse';
import type { AnalysisResult } from '@/lib/types';

export const runtime = 'nodejs';
// Disable static optimization — this is a dynamic API
export const dynamic = 'force-dynamic';
// Allow long-running VLM calls (up to ~5 minutes)
export const maxDuration = 300;

interface AnalyzeRequest {
  imageDataUrl: string;
  secondImageDataUrl?: string;
  query: string;
  history?: Array<{ role: 'user' | 'assistant'; content: string }>;
}

interface VisionContentItem {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: { url: string };
}

export async function POST(req: NextRequest) {
  let body: AnalyzeRequest;
  try {
    body = (await req.json()) as AnalyzeRequest;
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const { imageDataUrl, secondImageDataUrl, query, history } = body;

  if (!imageDataUrl || typeof imageDataUrl !== 'string') {
    return NextResponse.json(
      { error: 'imageDataUrl is required (a base64 data URL)' },
      { status: 400 }
    );
  }
  if (!query || typeof query !== 'string' || query.trim().length === 0) {
    return NextResponse.json({ error: 'query is required' }, { status: 400 });
  }

  // Basic data URL sanity check — accept either a data URL or an http(s) URL
  const isDataUrl = imageDataUrl.startsWith('data:');
  const isHttpUrl = imageDataUrl.startsWith('http://') || imageDataUrl.startsWith('https://');
  if (!isDataUrl && !isHttpUrl) {
    return NextResponse.json(
      { error: 'imageDataUrl must be a data: URL or an https:// URL' },
      { status: 400 }
    );
  }

  const intent = classifyIntent(query);
  const prompt = buildPromptForIntent(intent, query, Boolean(secondImageDataUrl));

  // Build the VLM message content
  const content: VisionContentItem[] = [{ type: 'text', text: prompt }];

  // Add the primary image
  content.push({ type: 'image_url', image_url: { url: imageDataUrl } });

  // For change detection, add the second image with a label
  if (secondImageDataUrl) {
    content.push({
      type: 'text',
      text: 'The next image is the AFTER image (later date). The previous image was the BEFORE image (earlier date).',
    });
    content.push({ type: 'image_url', image_url: { url: secondImageDataUrl } });
  }

  // Build messages: optional history first, then the new user message with images
  type VisionMessage = {
    role: 'system' | 'user' | 'assistant';
    content: string | VisionContentItem[];
  };

  const messages: VisionMessage[] = [];

  if (history && Array.isArray(history) && history.length > 0) {
    // Include up to the last 6 messages to preserve context without bloating the prompt
    const recent = history.slice(-6);
    for (const m of recent) {
      if (m.role === 'user' || m.role === 'assistant') {
        messages.push({ role: m.role, content: m.content });
      }
    }
  }

  // The current message — must come last
  messages.push({ role: 'user', content });

  try {
    const zai = await ZAI.create();
    const completion = await zai.chat.completions.createVision({
      messages,
      thinking: { type: 'disabled' },
      // Lower temperature for more deterministic, structured outputs
      temperature: 0.2,
      max_tokens: 1800,
    });

    const rawAnswer: string =
      (completion?.choices?.[0]?.message?.content as string | undefined) ?? '';

    if (!rawAnswer) {
      return NextResponse.json(
        { error: 'The vision model returned an empty response. Please try again.' },
        { status: 502 }
      );
    }

    const analysis: AnalysisResult = parseAnalysis(rawAnswer, intent);

    return NextResponse.json({
      analysis,
      rawAnswer,
      intent,
      intentLabel: intentLabel(intent),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[/api/analyze] VLM call failed:', message);
    return NextResponse.json(
      {
        error: `Failed to analyze image: ${message}`,
      },
      { status: 500 }
    );
  }
}
