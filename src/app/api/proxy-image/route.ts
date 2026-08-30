// GET /api/proxy-image?url=<remote-image-url>
// Fetches a remote image and re-serves it. Used to bypass CORS restrictions
// when loading sample satellite images hosted on external CDNs (e.g. NASA EO).
//
// Returns the raw image bytes with the correct Content-Type.

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 30;

const ALLOWED_HOST_SUFFIXES = [
  'nasa.gov',
  'usgs.gov',
  'earthobservatory.nasa.gov',
  'eoimages.gsfc.nasa.gov',
  'sentinel-copernicus.eu',
  'copernicus.eu',
  'unsplash.com',
  'images.unsplash.com',
];

function isAllowedHost(url: URL): boolean {
  const host = url.hostname.toLowerCase();
  return ALLOWED_HOST_SUFFIXES.some((s) => host === s || host.endsWith('.' + s));
}

export async function GET(req: NextRequest) {
  const urlParam = req.nextUrl.searchParams.get('url');
  if (!urlParam) {
    return NextResponse.json({ error: 'Missing url parameter' }, { status: 400 });
  }

  let target: URL;
  try {
    target = new URL(urlParam);
  } catch {
    return NextResponse.json({ error: 'Invalid url parameter' }, { status: 400 });
  }

  if (target.protocol !== 'https:') {
    return NextResponse.json({ error: 'Only HTTPS URLs are allowed' }, { status: 400 });
  }

  if (!isAllowedHost(target)) {
    return NextResponse.json(
      { error: `Host not allowed: ${target.hostname}` },
      { status: 403 }
    );
  }

  try {
    const upstream = await fetch(target.toString(), {
      // Allow long-poll-ish fetches
      signal: AbortSignal.timeout(20_000),
      headers: {
        // Some image servers require a UA
        'User-Agent': 'SatQuery-AI/1.0 (ISRO-SIH26167-prototype)',
        Accept: 'image/*,*/*;q=0.8',
      },
    });

    if (!upstream.ok) {
      return NextResponse.json(
        { error: `Upstream returned ${upstream.status}` },
        { status: 502 }
      );
    }

    const contentType = upstream.headers.get('content-type') ?? 'image/jpeg';
    const arrayBuf = await upstream.arrayBuffer();
    const buffer = new Uint8Array(arrayBuf);

    return new NextResponse(buffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600, immutable',
        'Content-Length': String(buffer.byteLength),
      },
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    console.error('[/api/proxy-image] failed:', msg);
    return NextResponse.json(
      { error: `Failed to fetch remote image: ${msg}` },
      { status: 502 }
    );
  }
}
