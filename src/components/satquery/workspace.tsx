'use client';

import { useEffect, useRef } from 'react';
import { useSatQueryStore } from '@/store/satquery';
import { ImageViewer } from './image-viewer';
import { ChatPanel } from './chat-panel';
import { AnalysisPanel } from './analysis-panel';
import { ImageUploader } from './image-uploader';
import { SampleImages } from './sample-images';
import { Button } from '@/components/ui/button';
import { ChevronLeft, Sparkles } from 'lucide-react';
import { SatQueryWordmark } from './logo';

interface Props {
  onExit: () => void;
}

export function Workspace({ onExit }: Props) {
  const activeImage = useSatQueryStore((s) => s.activeImage);
  const latestAnalysis = useSatQueryStore((s) => s.latestAnalysis);
  const workspaceRef = useRef<HTMLDivElement>(null);

  // Scroll workspace into view on mount
  useEffect(() => {
    if (workspaceRef.current) {
      workspaceRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, []);

  return (
    <div ref={workspaceRef} className="min-h-screen bg-background">
      {/* Top bar */}
      <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b bg-background/80 px-3 backdrop-blur sm:px-4">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={onExit}
            className="gap-1.5"
          >
            <ChevronLeft className="size-4" /> Back
          </Button>
          <div className="hidden h-5 w-px bg-border sm:block" />
          <SatQueryWordmark className="hidden sm:flex" />
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {activeImage ? (
            <>
              <span className="hidden sm:inline">Workspace</span>
              <span className="size-1.5 rounded-full bg-emerald-500" />
              <span className="font-medium text-foreground">
                {latestAnalysis ? 'Analysis ready' : 'Ready'}
              </span>
            </>
          ) : (
            <span className="flex items-center gap-1.5">
              <Sparkles className="size-3" />
              Awaiting image
            </span>
          )}
        </div>
      </header>

      <div className="mx-auto max-w-[1600px] p-3 sm:p-4">
        {!activeImage ? (
          // Onboarding screen when no image is loaded
          <div className="mx-auto max-w-3xl space-y-6 py-8">
            <div className="space-y-2 text-center">
              <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
                Start by adding a satellite image
              </h1>
              <p className="text-sm text-muted-foreground">
                Upload your own image or pick a sample below. Once an image is loaded,
                you can ask any natural-language question about it.
              </p>
            </div>
            <ImageUploader />
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="h-px flex-1 bg-border" />
                <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                  Sample satellite images
                </span>
                <div className="h-px flex-1 bg-border" />
              </div>
              <SampleImages />
            </div>
          </div>
        ) : (
          // Main 3-panel workspace layout
          <div className="grid gap-3 lg:grid-cols-12 lg:gap-4">
            {/* Left — image viewer (largest panel) */}
            <div className="lg:col-span-7 xl:col-span-7">
              <div className="rounded-xl border bg-card p-3">
                <ImageViewer />
              </div>
            </div>

            {/* Middle — chat (always visible) */}
            <div className="lg:col-span-5 xl:col-span-3">
              <div className="flex h-full min-h-[600px] flex-col overflow-hidden rounded-xl border bg-card">
                <ChatPanel />
              </div>
            </div>

            {/* Right — structured analysis panel */}
            <div className="lg:col-span-12 xl:col-span-2">
              <div className="rounded-xl border bg-card">
                <div className="border-b px-3 py-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Analysis Results
                  </p>
                </div>
                <AnalysisPanel />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
