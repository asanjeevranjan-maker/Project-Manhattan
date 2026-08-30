// GET /api/samples
// Returns a curated list of sample satellite images that users can load
// into the workspace with a single click. The images are pre-generated and
// served from /samples/* as static assets — this keeps the demo fully
// self-contained and avoids any dependency on external CDNs.

import { NextResponse } from 'next/server';
import type { SampleImage } from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-static';

const SAMPLES: SampleImage[] = [
  {
    id: 'sample-flood',
    title: 'River Flood Extent',
    description:
      'Aerial view of a river overflowing its banks, flooding surrounding fields and villages. Best for flood detection and water extent queries.',
    url: '/samples/flood.jpg',
    category: 'flood',
    location: 'River floodplain',
  },
  {
    id: 'sample-urban',
    title: 'Dense Urban Area',
    description:
      'Top-down satellite view of a city with grid streets, dense buildings, parks, and a river. Best for building & urban detection.',
    url: '/samples/urban.jpg',
    category: 'urban',
    location: 'Metropolitan area',
  },
  {
    id: 'sample-forest',
    title: 'Tropical Forest with Deforestation',
    description:
      'Rainforest with fishbone-pattern deforestation along access roads. Best for vegetation segmentation queries.',
    url: '/samples/forest.jpg',
    category: 'forest',
    location: 'Tropical forest',
  },
  {
    id: 'sample-agriculture',
    title: 'Center-Pivot Agriculture',
    description:
      'Arid landscape with circular center-pivot irrigation fields. Best for agriculture and land-use classification.',
    url: '/samples/agriculture.jpg',
    category: 'agriculture',
    location: 'Arid agricultural region',
  },
  {
    id: 'sample-wildfire',
    title: 'Wildfire Burn Scars',
    description:
      'Forest with dark burn scars from recent wildfires and visible smoke plumes. Best for fire detection and damage assessment.',
    url: '/samples/wildfire.jpg',
    category: 'wildfire',
    location: 'Forest wildfire area',
  },
  {
    id: 'sample-coastal',
    title: 'Coastal Waters and Reefs',
    description:
      'Turquoise shallow waters with coral reefs, white sand beaches, and small islands. Best for water body detection.',
    url: '/samples/coastal.jpg',
    category: 'coastal',
    location: 'Tropical coastline',
  },
];

export async function GET() {
  return NextResponse.json({ samples: SAMPLES });
}
