'use client';

import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Eye, EyeOff, Image as ImageIcon, Loader2, X, FileWarning } from 'lucide-react';
import { useSatQueryStore } from '@/store/satquery';
import { ImageUploader } from './image-uploader';
import type { AnalysisResult } from '@/lib/types';

/**
 * ImageViewer renders the uploaded satellite image and overlays
 * detected regions (bounding boxes) returned by the VLM analysis.
 *
 * Uses a relative-positioned container with absolutely-positioned overlay
 * divs whose coordinates come from the normalized `rect` field. This avoids
 * the cost & complexity of HTML5 canvas drawing while still being pixel-accurate
 * at any rendered size.
 */
export function ImageViewer() {
  const activeImage = useSatQueryStore((s) => s.activeImage);
  const analysis = useSatQueryStore((s) => s.latestAnalysis);
  const showOverlay = useSatQueryStore((s) => s.showOverlay);
  const setShowOverlay = useSatQueryStore((s) => s.setShowOverlay);
  const isAnalyzing = useSatQueryStore((s) => s.isAnalyzing);
  const reset = useSatQueryStore((s) => s.reset);
  const [loadedImageId, setLoadedImageId] = useState<string | null>(null);
  const [errorImageId, setErrorImageId] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const imgLoaded = !!activeImage && loadedImageId === activeImage.id;
  const imgError = !!activeImage && errorImageId === activeImage.id;

  if (!activeImage) {
    return (
      <div className="flex h-full min-h-[400px] flex-col items-center justify-center gap-3 rounded-xl border border-dashed bg-muted/30 p-8 text-center">
        <div className="flex size-16 items-center justify-center rounded-full bg-primary/10 text-primary">
          <ImageIcon className="size-7" />
        </div>
        <div className="space-y-1">
          <p className="text-base font-semibold">No image selected</p>
          <p className="text-sm text-muted-foreground">
            Upload a satellite image or pick a sample to start analyzing.
          </p>
        </div>
      </div>
    );
  }

  const regions = analysis?.regions ?? [];
  const hasRegions = regions.length > 0;

  return (
    <div className="flex h-full min-h-[400px] flex-col gap-3">
      {/* Image toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 flex-col gap-0.5">
          <p className="truncate text-sm font-semibold" title={activeImage.filename}>
            {activeImage.filename}
          </p>
          {activeImage.location && (
            <p className="text-xs text-muted-foreground">{activeImage.location}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <ImageUploader variant="compact" />
          {hasRegions && (
            <Button
              size="sm"
              variant={showOverlay ? 'secondary' : 'outline'}
              onClick={() => setShowOverlay(!showOverlay)}
              className="gap-1.5"
            >
              {showOverlay ? (
                <>
                  <EyeOff className="size-3.5" /> Hide overlay
                </>
              ) : (
                <>
                  <Eye className="size-3.5" /> Show overlay
                </>
              )}
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={reset} className="gap-1.5">
            <X className="size-3.5" /> Clear
          </Button>
        </div>
      </div>

      {/* Image canvas */}
      <div className="relative flex-1 overflow-hidden rounded-xl border bg-[repeating-conic-gradient(hsl(0_0%_88%)_0%_25%,hsl(0_0%_94%)_0%_50%)] bg-[length:20px_20px] dark:bg-[repeating-conic-gradient(hsl(0_0%_22%)_0%_25%,hsl(0_0%_18%)_0%_50%)]">
        {imgError ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center bg-card/95">
            <div className="flex size-12 items-center justify-center rounded-full bg-destructive/10 text-destructive">
              <FileWarning className="size-6" />
            </div>
            <div className="space-y-1 max-w-sm">
              <p className="text-sm font-semibold">Unable to display this image preview</p>
              <p className="text-xs text-muted-foreground">
                Your browser couldn't render this file directly (common for raw TIFF or uncompressed rasters). You can still query it with AI or upload another image.
              </p>
            </div>
            <ImageUploader variant="compact" />
          </div>
        ) : (
          <>
            {!imgLoaded && (
              <div className="absolute inset-0 flex items-center justify-center bg-background/60">
                <Loader2 className="size-6 animate-spin text-muted-foreground" />
              </div>
            )}
            <img
              ref={imgRef}
              src={activeImage.dataUrl}
              alt={activeImage.filename}
              onLoad={() => setLoadedImageId(activeImage.id)}
              onError={() => setErrorImageId(activeImage.id)}
              className={cn(
                'h-full w-full object-contain transition-opacity duration-300',
                imgLoaded ? 'opacity-100' : 'opacity-0'
              )}
            />

            {/* Overlay layer — same bounding box as the image */}
            {imgLoaded && showOverlay && hasRegions && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                <div className="relative h-full w-full">
                  {regions.map((region, idx) => (
                    <RegionBox key={`${idx}-${region.label}`} region={region} index={idx} />
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* Live "analyzing" badge */}
        {isAnalyzing && (
          <div className="absolute right-3 top-3 flex items-center gap-2 rounded-full bg-background/90 px-3 py-1.5 text-xs font-medium shadow-sm backdrop-blur">
            <span className="size-2 rounded-full bg-primary glow-pulse" />
            Analyzing…
          </div>
        )}
      </div>

      {/* Coverage legend */}
      {analysis && analysis.coverage.length > 0 && (
        <CoverageLegend analysis={analysis} />
      )}
    </div>
  );
}

function RegionBox({
  region,
  index,
}: {
  region: NonNullable<AnalysisResult['regions']>[number];
  index: number;
}) {
  const [x, y, w, h] = region.rect;
  const [showLabel, setShowLabel] = useState(true);
  return (
    <div
      className="group absolute transition-opacity"
      style={{
        left: `${x * 100}%`,
        top: `${y * 100}%`,
        width: `${w * 100}%`,
        height: `${h * 100}%`,
      }}
      onMouseEnter={() => setShowLabel(true)}
    >
      <div
        className="size-full rounded-md border-2"
        style={{
          borderColor: region.color,
          backgroundColor: `${region.color}25`,
          boxShadow: `0 0 0 1px ${region.color}40, 0 0 12px ${region.color}30`,
        }}
      />
      {showLabel && (
        <div
          className="absolute -top-6 left-0 flex items-center gap-1.5 whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-semibold text-white shadow-sm"
          style={{ backgroundColor: region.color }}
        >
          <span>{region.label}</span>
          <span className="opacity-80">{Math.round(region.confidence * 100)}%</span>
        </div>
      )}
    </div>
  );
}

function CoverageLegend({ analysis }: { analysis: AnalysisResult }) {
  return (
    <div className="rounded-lg border bg-card/60 p-3">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Land Cover Coverage
        </p>
        <span className="text-xs text-muted-foreground">
          Overall confidence: <span className="font-semibold text-foreground">{Math.round(analysis.confidence * 100)}%</span>
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        {analysis.coverage.map((c, i) => (
          <div key={`${c.class}-${i}`} className="flex items-center gap-2">
            <span
              className="size-3 shrink-0 rounded-sm"
              style={{ backgroundColor: c.color }}
            />
            <span className="w-24 shrink-0 truncate text-xs font-medium capitalize">
              {c.class}
            </span>
            <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${Math.max(2, c.coverage * 100)}%`,
                  backgroundColor: c.color,
                }}
              />
            </div>
            <span className="w-12 shrink-0 text-right text-xs font-semibold tabular-nums">
              {(c.coverage * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
