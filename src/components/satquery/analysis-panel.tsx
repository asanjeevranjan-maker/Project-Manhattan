'use client';

import { cn } from '@/lib/utils';
import { useSatQueryStore } from '@/store/satquery';
import {
  Activity,
  Layers,
  MapPin,
  Percent,
  Target,
  TrendingUp,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
} from 'lucide-react';
import type { AnalysisResult, DetectedObject } from '@/lib/types';

export function AnalysisPanel() {
  const analysis = useSatQueryStore((s) => s.latestAnalysis);
  const activeImage = useSatQueryStore((s) => s.activeImage);

  if (!analysis) {
    return (
      <div className="flex h-full min-h-[300px] flex-col items-center justify-center gap-2 p-6 text-center">
        <div className="flex size-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
          <Activity className="size-5" />
        </div>
        <p className="text-sm font-medium">No analysis yet</p>
        <p className="text-xs text-muted-foreground">
          {activeImage
            ? 'Ask the AI a question to see structured results here.'
            : 'Upload an image and ask a question to see results.'}
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-3 p-3">
      {/* Confidence card */}
      <div className="rounded-lg border bg-gradient-to-br from-primary/5 to-transparent p-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <Target className="size-3.5" />
            Analysis Confidence
          </div>
          <span className="text-2xl font-bold tabular-nums">
            {Math.round(analysis.confidence * 100)}
            <span className="text-base text-muted-foreground">%</span>
          </span>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-all duration-500"
            style={{ width: `${Math.max(3, analysis.confidence * 100)}%` }}
          />
        </div>
        <div className="mt-2 flex items-center justify-between text-[10px] text-muted-foreground">
          <span className="rounded bg-secondary/60 px-1.5 py-0.5 font-medium uppercase tracking-wide">
            {analysis.intent.replace(/_/g, ' ')}
          </span>
          {activeImage?.location && (
            <span className="flex items-center gap-1">
              <MapPin className="size-3" />
              {activeImage.location}
            </span>
          )}
        </div>
      </div>

      {/* Detected objects */}
      {analysis.objectsDetected.length > 0 && (
        <Section icon={<Layers className="size-3.5" />} title="Detected Objects">
          <div className="space-y-1.5">
            {analysis.objectsDetected.map((o, i) => (
              <DetectedObjectRow key={`${o.class}-${i}`} obj={o} />
            ))}
          </div>
        </Section>
      )}

      {/* Coverage breakdown */}
      {analysis.coverage.length > 0 && (
        <Section icon={<Percent className="size-3.5" />} title="Land Cover Coverage">
          <div className="space-y-1.5">
            {analysis.coverage.map((c, i) => (
              <div
                key={`${c.class}-${i}`}
                className="flex items-center gap-2 text-xs"
              >
                <span
                  className="size-2.5 shrink-0 rounded-sm"
                  style={{ backgroundColor: c.color }}
                />
                <span className="w-20 shrink-0 truncate capitalize font-medium">
                  {c.class}
                </span>
                <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.max(2, c.coverage * 100)}%`,
                      backgroundColor: c.color,
                    }}
                  />
                </div>
                <span className="w-12 shrink-0 text-right font-semibold tabular-nums">
                  {(c.coverage * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Change detection summary */}
      {analysis.changeSummary && (
        <Section icon={<TrendingUp className="size-3.5" />} title="Change Summary">
          <div className="space-y-2 text-xs">
            {analysis.changeSummary.additions && analysis.changeSummary.additions.length > 0 && (
              <ChangeList
                title="Additions"
                items={analysis.changeSummary.additions}
                icon={<ArrowUpRight className="size-3 text-emerald-500" />}
                color="text-emerald-600 dark:text-emerald-400"
              />
            )}
            {analysis.changeSummary.removals && analysis.changeSummary.removals.length > 0 && (
              <ChangeList
                title="Removals"
                items={analysis.changeSummary.removals}
                icon={<ArrowDownRight className="size-3 text-red-500" />}
                color="text-red-600 dark:text-red-400"
              />
            )}
            {analysis.changeSummary.netChange && (
              <div className="flex items-start gap-2 rounded-md bg-muted/50 p-2">
                <Minus className="size-3 mt-0.5 shrink-0 text-muted-foreground" />
                <p>{analysis.changeSummary.netChange}</p>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Highlighted regions */}
      {analysis.regions && analysis.regions.length > 0 && (
        <Section icon={<MapPin className="size-3.5" />} title="Highlighted Regions">
          <div className="flex flex-wrap gap-1.5">
            {analysis.regions.map((r, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium"
                style={{ borderColor: r.color, color: r.color }}
              >
                <span
                  className="size-2 rounded-full"
                  style={{ backgroundColor: r.color }}
                />
                {r.label} · {Math.round(r.confidence * 100)}%
              </span>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

function Section({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border bg-card/60 p-3">
      <div className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        {icon}
        {title}
      </div>
      {children}
    </div>
  );
}

function DetectedObjectRow({ obj }: { obj: DetectedObject }) {
  return (
    <div className="flex items-center justify-between gap-2 text-xs">
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="size-2 shrink-0 rounded-full bg-primary/60" />
        <span className="truncate capitalize font-medium">{obj.class}</span>
        {typeof obj.count === 'number' && (
          <span className="shrink-0 text-muted-foreground">×{obj.count}</span>
        )}
        {obj.region && (
          <span className="shrink-0 rounded bg-secondary/60 px-1 py-0.5 text-[9px] uppercase tracking-wide text-muted-foreground">
            {obj.region}
          </span>
        )}
      </div>
      <div className="flex items-center gap-1.5">
        <div className="relative h-1 w-12 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary"
            style={{ width: `${obj.confidence * 100}%` }}
          />
        </div>
        <span className="w-8 text-right text-[10px] font-semibold tabular-nums text-muted-foreground">
          {Math.round(obj.confidence * 100)}%
        </span>
      </div>
    </div>
  );
}

function ChangeList({
  title,
  items,
  icon,
  color,
}: {
  title: string;
  items: string[];
  icon: React.ReactNode;
  color: string;
}) {
  return (
    <div>
      <p className={cn('mb-1 flex items-center gap-1 font-semibold', color)}>
        {icon}
        {title}
      </p>
      <ul className="ml-4 list-disc space-y-0.5 marker:text-muted-foreground">
        {items.map((it, i) => (
          <li key={i}>{it}</li>
        ))}
      </ul>
    </div>
  );
}
