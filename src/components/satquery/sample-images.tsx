'use client';

import { useEffect, useState } from 'react';
import { Loader2, ImageIcon, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { remoteImageToDataUrl, localImageToDataUrl, shortId } from '@/lib/client-utils';
import type { SampleImage, UploadedImage } from '@/lib/types';
import { useSatQueryStore } from '@/store/satquery';

const CATEGORY_LABEL: Record<SampleImage['category'], string> = {
  flood: 'Flood',
  urban: 'Urban',
  forest: 'Forest',
  agriculture: 'Agriculture',
  wildfire: 'Wildfire',
  coastal: 'Coastal',
};

const CATEGORY_COLOR: Record<SampleImage['category'], string> = {
  flood: 'bg-cyan-500/15 text-cyan-700 dark:text-cyan-300',
  urban: 'bg-red-500/15 text-red-700 dark:text-red-300',
  forest: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  agriculture: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  wildfire: 'bg-orange-500/15 text-orange-700 dark:text-orange-300',
  coastal: 'bg-teal-500/15 text-teal-700 dark:text-teal-300',
};

export function SampleImages() {
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const { toast } = useToast();
  const setActiveImage = useSatQueryStore((s) => s.setActiveImage);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const res = await fetch('/api/samples');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as { samples: SampleImage[] };
        if (mounted) {
          setSamples(data.samples);
          setLoading(false);
        }
      } catch (err) {
        if (mounted) {
          setLoading(false);
          console.error('Failed to load samples:', err);
        }
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  async function loadSample(s: SampleImage) {
    setLoadingId(s.id);
    try {
      // Local /samples/* paths can be fetched directly via the proxy-image
      // endpoint (which handles relative paths just fine and returns a blob)
      // OR — since they're same-origin — we can fetch them directly.
      const isLocal = s.url.startsWith('/');
      const dataUrl = isLocal
        ? await localImageToDataUrl(s.url)
        : await remoteImageToDataUrl(s.url);
      const img: UploadedImage = {
        id: shortId('sample-'),
        filename: `${s.id}.jpg`,
        mimeType: 'image/jpeg',
        size: 0,
        dataUrl,
        location: s.location,
      };
      setActiveImage(img);
      toast({
        title: 'Sample loaded',
        description: `${s.title} — ${s.location}`,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load sample';
      toast({
        variant: 'destructive',
        title: 'Could not load sample',
        description: msg,
      });
    } finally {
      setLoadingId(null);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading sample images…
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {samples.map((s) => (
        <button
          key={s.id}
          type="button"
          onClick={() => loadSample(s)}
          disabled={loadingId !== null}
          className="group relative flex flex-col overflow-hidden rounded-lg border bg-card text-left transition-all hover:border-primary/50 hover:shadow-md disabled:opacity-50"
        >
          <div className="relative aspect-[4/3] overflow-hidden bg-muted">
            <img
              src={s.url.startsWith('/') ? s.url : `/api/proxy-image?url=${encodeURIComponent(s.url)}`}
              alt={s.title}
              loading="lazy"
              className="size-full object-cover transition-transform duration-300 group-hover:scale-105"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
            <span
              className={cn(
                'absolute left-2 top-2 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide backdrop-blur-sm',
                CATEGORY_COLOR[s.category]
              )}
            >
              {CATEGORY_LABEL[s.category]}
            </span>
            {loadingId === s.id && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <Loader2 className="size-5 animate-spin text-white" />
              </div>
            )}
          </div>
          <div className="flex flex-1 flex-col gap-0.5 p-2.5">
            <p className="text-xs font-semibold leading-tight">{s.title}</p>
            <p className="text-[10px] text-muted-foreground">{s.location}</p>
          </div>
        </button>
      ))}
    </div>
  );
}

export function SampleImagesTrigger({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'inline-flex items-center gap-1.5 rounded-md bg-secondary/60 px-2.5 py-1 text-xs font-medium text-secondary-foreground',
        className
      )}
    >
      <ImageIcon className="size-3.5" />
      Sample images
      <ExternalLink className="size-3 opacity-50" />
    </div>
  );
}
