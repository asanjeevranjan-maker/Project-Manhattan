// GET /api/temporal/geocode?q=...
//
// Geocodes location search queries (e.g. "Mumbai, India")
// into latitude, longitude, and default Area of Interest bounding box.

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const q = searchParams.get("q")?.trim();

  if (!q) {
    return NextResponse.json({ error: "Search query 'q' is required" }, { status: 400 });
  }

  try {
    const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
      q
    )}&limit=5&addressdetails=1`;

    const res = await fetch(url, {
      headers: {
        "User-Agent": "SatQuery-AI/1.0 (ISRO SIH26167 Earth Observation Assistant)",
        Accept: "application/json",
      },
    });

    if (!res.ok) {
      throw new Error(`Nominatim returned status ${res.status}`);
    }

    const data = (await res.json()) as Array<{
      display_name: string;
      lat: string;
      lon: string;
      boundingbox?: [string, string, string, string]; // [south, north, west, east]
    }>;

    const results = data.map((item) => {
      const lat = parseFloat(item.lat);
      const lon = parseFloat(item.lon);

      // Create a focused default AOI span (~0.04° lat/lon, approx 4-5 km across)
      const span = 0.02;
      let north = lat + span;
      let south = lat - span;
      let east = lon + span;
      let west = lon - span;

      // If boundingbox provided and reasonable span, use it clamped
      if (item.boundingbox && item.boundingbox.length === 4) {
        const bSouth = parseFloat(item.boundingbox[0]);
        const bNorth = parseFloat(item.boundingbox[1]);
        const bWest = parseFloat(item.boundingbox[2]);
        const bEast = parseFloat(item.boundingbox[3]);

        const bLatSpan = Math.abs(bNorth - bSouth);
        const bLonSpan = Math.abs(bEast - bWest);

        if (bLatSpan <= 0.15 && bLonSpan <= 0.15) {
          north = bNorth;
          south = bSouth;
          west = bWest;
          east = bEast;
        }
      }

      return {
        displayName: item.display_name,
        latitude: lat,
        longitude: lon,
        aoi: {
          north: Math.round(north * 10000) / 10000,
          south: Math.round(south * 10000) / 10000,
          east: Math.round(east * 10000) / 10000,
          west: Math.round(west * 10000) / 10000,
        },
      };
    });

    return NextResponse.json({ results });
  } catch (err) {
    console.error("[Geocode Error]:", err);
    return NextResponse.json(
      {
        error: "Geocoding service unavailable",
        details: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}

