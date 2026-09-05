'use client';

import { Button } from '@/components/ui/button';
import { ArrowRight, MessageSquare, Satellite, ScanEye, BarChart3, GitCompareArrows } from 'lucide-react';
import { ImageUploader } from './image-uploader';
import { SampleImages } from './sample-images';
import { useSatQueryStore } from '@/store/satquery';

interface Props {
  onLaunch: () => void;
  onLaunchBiTemporal?: () => void;
}

export function Hero({ onLaunch, onLaunchBiTemporal }: Props) {
  const activeImage = useSatQueryStore((s) => s.activeImage);

  return (
    <section className="starfield-bg relative overflow-hidden">
      {/* Top fade */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-background to-transparent" />
      {/* Bottom fade */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-background to-transparent" />

      <div className="relative mx-auto max-w-7xl px-4 py-16 sm:px-6 sm:py-24 lg:px-8">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          {/* Left — headline + CTA */}
          <div className="flex flex-col gap-6">
            <div className="inline-flex w-fit items-center gap-2 rounded-full border bg-card/60 px-3 py-1 text-xs font-medium backdrop-blur">
              <span className="size-1.5 rounded-full bg-primary glow-pulse" />
              ISRO · SIH 26167 · Space Technology
            </div>
            <h1 className="text-4xl font-bold leading-tight tracking-tight sm:text-5xl lg:text-6xl">
              <span className="gradient-text">Ask Your Satellite Images</span>
              <br />
              <span className="text-foreground">Anything.</span>
            </h1>
            <p className="max-w-xl text-base text-muted-foreground sm:text-lg">
              SatQuery AI is an interactive vision-language assistant that analyzes
              multimodal remote sensing imagery through natural language queries —
              returning visual highlights, detected objects, and explainable insights
              without any GIS expertise required.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <Button size="lg" onClick={onLaunch} className="gap-2">
                <Satellite className="size-4" />
                Single Image Q&A
                <ArrowRight className="size-4" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={onLaunchBiTemporal || onLaunch}
                className="gap-2"
              >
                <GitCompareArrows className="size-4 text-primary" />
                Bi-Temporal Comparison
              </Button>
            </div>

            {/* Quick stats */}
            <div className="mt-4 grid grid-cols-3 gap-4 border-t pt-4">
              <Stat value="< 20s" label="Response time" />
              <Stat value="8+" label="Land cover classes" />
              <Stat value="100%" label="Conversational" />
            </div>
          </div>

          {/* Right — upload / sample images card */}
          <div className="rounded-2xl border bg-card/80 p-5 shadow-xl backdrop-blur">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold">Try SatQuery AI now</p>
                <p className="text-xs text-muted-foreground">
                  Upload a satellite image or pick a sample to begin.
                </p>
              </div>
              {activeImage && (
                <Button size="sm" variant="secondary" onClick={onLaunch} className="gap-1.5">
                  Open Workspace <ArrowRight className="size-3.5" />
                </Button>
              )}
            </div>
            <ImageUploader onSuccess={onLaunch} />
            <div className="my-4 flex items-center gap-3">
              <div className="h-px flex-1 bg-border" />
              <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                Or pick a sample
              </span>
              <div className="h-px flex-1 bg-border" />
            </div>
            <SampleImages onSelect={onLaunch} />

            {/* Quick example queries preview */}
            <div className="mt-4 rounded-lg border border-dashed bg-muted/30 p-3">
              <p className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                <MessageSquare className="size-3" /> Example queries you can ask
              </p>
              <ul className="space-y-1 text-xs">
                <li className="flex items-start gap-1.5">
                  <ScanEye className="mt-0.5 size-3 shrink-0 text-primary" />
                  <span>“Identify water bodies in this image.”</span>
                </li>
                <li className="flex items-start gap-1.5">
                  <BarChart3 className="mt-0.5 size-3 shrink-0 text-primary" />
                  <span>“How much forest area is visible?”</span>
                </li>
                <li className="flex items-start gap-1.5">
                  <ScanEye className="mt-0.5 size-3 shrink-0 text-primary" />
                  <span>“Detect urban areas and buildings.”</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <p className="text-2xl font-bold tracking-tight text-foreground">{value}</p>
      <p className="text-xs text-muted-foreground">{label}</p>
    </div>
  );
}
