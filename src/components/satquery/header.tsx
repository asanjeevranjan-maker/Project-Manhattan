'use client';

import { Button } from '@/components/ui/button';
import { Github, Rocket, Sparkles, GitCompareArrows } from 'lucide-react';
import { SatQueryWordmark } from './logo';

interface Props {
  onLaunch: () => void;
  onLaunchBiTemporal?: () => void;
  hasImage: boolean;
}

export function Header({ onLaunch, onLaunchBiTemporal, hasImage }: Props) {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-3 px-4 sm:px-6">
        <SatQueryWordmark />
        <nav className="hidden items-center gap-1 md:flex">
          <a
            href="#how-it-works"
            className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            How it works
          </a>
          <a
            href="#features"
            className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            Features
          </a>
          <button
            type="button"
            onClick={onLaunchBiTemporal || onLaunch}
            className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium text-primary transition-colors hover:bg-primary/10"
          >
            <GitCompareArrows className="size-3.5" /> Bi-Temporal
          </button>
          <a
            href="#use-cases"
            className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            Use Cases
          </a>
          <a
            href="#tech-stack"
            className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            Tech
          </a>
        </nav>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="hidden sm:inline-flex"
            onClick={() => window.open('https://www.sih.gov.in/', '_blank', 'noopener')}
          >
            <Github className="size-4" /> SIH
          </Button>
          <Button size="sm" onClick={onLaunch} className="gap-1.5">
            {hasImage ? (
              <>
                <Sparkles className="size-4" /> Open Workspace
              </>
            ) : (
              <>
                <Rocket className="size-4" /> Launch App
              </>
            )}
          </Button>
        </div>
      </div>
    </header>
  );
}
