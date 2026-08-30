// Generate sample satellite-style images for the SatQuery AI demo.
// We use the z-ai-web-dev-sdk image generation API to create realistic
// satellite-style imagery for each demo category (flood, urban, forest,
// agriculture, wildfire, coastal). The generated images are saved to
// /home/z/my-project/public/samples/ and served as static assets.
//
// Usage: bun /home/z/my-project/scripts/gen-samples.ts

import ZAI from 'z-ai-web-dev-sdk';
import { writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const OUT_DIR = '/home/z/my-project/public/samples';

const SAMPLES = [
  {
    id: 'flood',
    filename: 'flood.jpg',
    title: 'River Flood Extent',
    prompt:
      'Satellite imagery view of a major river flooding its banks, with brown sediment-laden floodwater spreading over agricultural fields and villages. Aerial top-down view, true color, high detail, NASA Landsat style. Some clouds at the edges.',
  },
  {
    id: 'urban',
    filename: 'urban.jpg',
    title: 'Dense Urban Area',
    prompt:
      'Satellite imagery of a dense urban city with grid-like street networks, tall buildings casting shadows, parks, and a river running through. Aerial top-down view, true color, high detail, NASA Landsat style.',
  },
  {
    id: 'forest',
    filename: 'forest.jpg',
    title: 'Tropical Forest with Deforestation',
    prompt:
      'Satellite imagery of dense tropical rainforest with visible fishbone-pattern deforestation along roads. Some cleared patches showing bare red-brown soil. Aerial top-down view, true color, high detail, NASA Landsat style.',
  },
  {
    id: 'agriculture',
    filename: 'agriculture.jpg',
    title: 'Center-Pivot Agriculture',
    prompt:
      'Satellite imagery of arid landscape with circular center-pivot irrigation fields in green and brown. A regular grid of circular cropland patches. Aerial top-down view, true color, high detail, NASA Landsat style.',
  },
  {
    id: 'wildfire',
    filename: 'wildfire.jpg',
    title: 'Wildfire Burn Scars',
    prompt:
      'Satellite imagery of forest with visible dark burn scars from recent wildfires, with some smoke plumes. Mix of green unburnt forest and dark charred areas. Aerial top-down view, true color, high detail, NASA Landsat style.',
  },
  {
    id: 'coastal',
    filename: 'coastal.jpg',
    title: 'Coastal Waters and Reefs',
    prompt:
      'Satellite imagery of tropical coastal waters with turquoise shallow water, coral reefs visible beneath the surface, white sand beaches, and a few small islands. Aerial top-down view, true color, high detail, NASA Landsat style.',
  },
];

async function main() {
  if (!existsSync(OUT_DIR)) {
    await mkdir(OUT_DIR, { recursive: true });
  }

  const zai = await ZAI.create();

  for (const s of SAMPLES) {
    const outPath = path.join(OUT_DIR, s.filename);
    console.log(`Generating ${s.id} → ${outPath}`);
    try {
      const response = await zai.images.generations.create({
        prompt: s.prompt,
        size: '1024x1024',
      });
      const base64: string | undefined = response?.data?.[0]?.base64;
      if (!base64) {
        throw new Error('No base64 returned from image generation');
      }
      const buffer = Buffer.from(base64, 'base64');
      await writeFile(outPath, buffer);
      console.log(`  ✓ saved (${(buffer.length / 1024).toFixed(1)} KB)`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  ✗ failed: ${msg}`);
    }
  }

  console.log('Done.');
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
