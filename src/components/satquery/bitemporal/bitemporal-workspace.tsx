'use client';

import { useState, useCallback, useRef } from 'react';
import type {
  AOIBounds,
  TemporalComparisonResult,
  TemporalChangeItem,
} from '@/lib/types';
import { AOIMap, PRESET_LOCATIONS, computeAoiFromSpan } from './aoi-map';
import { ChangeMap } from './change-map';
import { ImageSlider } from './image-slider';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { useToast } from '@/hooks/use-toast';
import { fileToDataUrl } from '@/lib/client-utils';
import {
  Sparkles,
  GitCompareArrows,
  Upload,
  Calendar,
  CloudSun,
  Search,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  MapPin,
  TrendingUp,
  Layers,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Plus,
  SlidersHorizontal,
  Maximize2,
  Clock,
  ShieldCheck,
  RefreshCw,
  Image as ImageIcon,
  X,
} from 'lucide-react';

const LOADING_STAGES = [
  'Finding satellite imagery for AOI…',
  'Latest scene found — verifying cloud cover…',
  'Downloading & cropping imagery to exact AOI…',
  'Aligning temporal images (geo-registration)…',
  'Detecting objects with Grounding DINO…',
  'Spatiotemporal matching & geolocation…',
  'Generating pixel change mask & change map…',
];

export function BiTemporalWorkspace() {
  const { toast } = useToast();

  // Mode: 'latest' (Live/Latest Satellite) vs 'manual' (Manual 2-image upload)
  const [bitemporalMode, setBitemporalMode] = useState<'latest' | 'manual'>('latest');

  // Location & AOI
  const [aoi, setAoi] = useState<AOIBounds>(PRESET_LOCATIONS[0].aoi);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearchingLoc, setIsSearchingLoc] = useState(false);

  // Time 1 settings
  const [historicalMode, setHistoricalMode] = useState<'date' | 'upload'>('date');
  const [historicalDate, setHistoricalDate] = useState('2024-01-15');
  const [historicalFile, setHistoricalFile] = useState<File | null>(null);
  const [historicalFilePreview, setHistoricalFilePreview] = useState<string | null>(null);

  // Time 2 settings
  const [maxCloudCover, setMaxCloudCover] = useState(20);
  const [searchDays, setSearchDays] = useState(30);

  // Manual Mode files
  const [manualFileT1, setManualFileT1] = useState<File | null>(null);
  const [manualPreviewT1, setManualPreviewT1] = useState<string | null>(null);
  const [manualFileT2, setManualFileT2] = useState<File | null>(null);
  const [manualPreviewT2, setManualPreviewT2] = useState<string | null>(null);
  const [manualDateT1, setManualDateT1] = useState('2024-01-01');
  const [manualDateT2, setManualDateT2] = useState('2024-09-01');

  const handleHistoricalFile = async (f: File | null) => {
    setHistoricalFile(f);
    if (f) {
      try {
        const url = await fileToDataUrl(f);
        setHistoricalFilePreview(url);
      } catch {
        setHistoricalFilePreview(null);
      }
    } else {
      setHistoricalFilePreview(null);
    }
  };

  const handleManualFileT1 = async (f: File | null) => {
    setManualFileT1(f);
    if (f) {
      try {
        const url = await fileToDataUrl(f);
        setManualPreviewT1(url);
      } catch {
        setManualPreviewT1(null);
      }
    } else {
      setManualPreviewT1(null);
    }
  };

  const handleManualFileT2 = async (f: File | null) => {
    setManualFileT2(f);
    if (f) {
      try {
        const url = await fileToDataUrl(f);
        setManualPreviewT2(url);
      } catch {
        setManualPreviewT2(null);
      }
    } else {
      setManualPreviewT2(null);
    }
  };

  // Object Detection Prompt
  const [prompt, setPrompt] = useState('ship . cargo ship . vessel . boat .');

  // Comparison State
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStageIdx, setLoadingStageIdx] = useState(0);
  const [result, setResult] = useState<TemporalComparisonResult | null>(null);
  const [selectedChangeId, setSelectedChangeId] = useState<string | null>(null);

  // Active View Tab for results: 'slider' | 'side' | 'overlay' | 'map'
  const [resultViewMode, setResultViewMode] = useState<'slider' | 'side' | 'overlay' | 'map'>('slider');

  // Location geocode search handler
  const handleLocationSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearchingLoc(true);
    try {
      const res = await fetch(`/api/temporal/geocode?q=${encodeURIComponent(searchQuery.trim())}`);
      const data = await res.json();
      if (data.results && data.results.length > 0) {
        const first = data.results[0];
        setAoi(first.aoi);
        toast({
          title: 'Location found',
          description: `Centered on ${first.displayName.split(',').slice(0, 3).join(',')}`,
        });
      } else {
        toast({
          variant: 'destructive',
          title: 'Location not found',
          description: 'Try searching for a city, harbor, or entering lat/long manually.',
        });
      }
    } catch {
      toast({
        variant: 'destructive',
        title: 'Search failed',
        description: 'Could not reach geocoding service.',
      });
    } finally {
      setIsSearchingLoc(false);
    }
  };

  // Run Latest Satellite Comparison
  const runLatestComparison = async () => {
    if (historicalMode === 'upload' && !historicalFile) {
      toast({
        variant: 'destructive',
        title: 'Reference image required',
        description: 'Please upload a historical reference image file for Time 1.',
      });
      return;
    }

    setIsLoading(true);
    setLoadingStageIdx(0);
    setResult(null);

    // Simulate realistic loading steps for UX
    const interval = setInterval(() => {
      setLoadingStageIdx((prev) => (prev < LOADING_STAGES.length - 1 ? prev + 1 : prev));
    }, 2200);

    try {
      const formData = new FormData();
      formData.append('aoi_north', String(aoi.north));
      formData.append('aoi_south', String(aoi.south));
      formData.append('aoi_east', String(aoi.east));
      formData.append('aoi_west', String(aoi.west));
      formData.append('prompt', prompt.trim() || 'building');
      formData.append('historical_mode', historicalMode);
      formData.append('historical_date', historicalDate);
      formData.append('max_cloud_cover', String(maxCloudCover));
      formData.append('search_days', String(searchDays));
      formData.append('enable_pixel_change', 'true');

      if (historicalMode === 'upload' && historicalFile) {
        formData.append('historical_file', historicalFile);
      }

      const res = await fetch('/api/temporal/latest', {
        method: 'POST',
        body: formData,
      });

      clearInterval(interval);

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.details || errJson.error || `HTTP ${res.status}`);
      }

      const data: TemporalComparisonResult = await res.json();
      setResult(data);
      toast({
        title: 'Bi-Temporal Analysis Complete',
        description: `Identified ${data.summary.totalChanges} changes (${data.summary.newCount} new, ${data.summary.removedCount} removed, ${data.summary.modifiedCount} modified)`,
      });
    } catch (err) {
      clearInterval(interval);
      const msg = err instanceof Error ? err.message : 'Comparison failed';
      toast({
        variant: 'destructive',
        title: 'Comparison Failed',
        description: msg,
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Run Manual Upload Comparison
  const runManualComparison = async () => {
    if (!manualFileT1 || !manualFileT2) {
      toast({
        variant: 'destructive',
        title: 'Images required',
        description: 'Please upload both Time 1 and Time 2 satellite images.',
      });
      return;
    }

    setIsLoading(true);
    setLoadingStageIdx(0);
    setResult(null);

    const interval = setInterval(() => {
      setLoadingStageIdx((prev) => (prev < LOADING_STAGES.length - 1 ? prev + 1 : prev));
    }, 2000);

    try {
      const formData = new FormData();
      formData.append('file_t1', manualFileT1);
      formData.append('file_t2', manualFileT2);
      formData.append('prompt', prompt.trim() || 'building');
      formData.append('date_t1', manualDateT1);
      formData.append('date_t2', manualDateT2);
      formData.append('enable_pixel_change', 'true');

      const res = await fetch('/api/temporal/manual', {
        method: 'POST',
        body: formData,
      });

      clearInterval(interval);

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.details || errJson.error || `HTTP ${res.status}`);
      }

      const data: TemporalComparisonResult = await res.json();
      setResult(data);
      toast({
        title: 'Manual Comparison Complete',
        description: `Found ${data.summary.totalChanges} changes between uploaded scenes.`,
      });
    } catch (err) {
      clearInterval(interval);
      const msg = err instanceof Error ? err.message : 'Manual comparison failed';
      toast({
        variant: 'destructive',
        title: 'Comparison Failed',
        description: msg,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner & Mode Selector */}
      <div className="flex flex-col gap-4 rounded-2xl border bg-card p-4 sm:p-5 shadow-sm">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="flex size-7 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <GitCompareArrows className="size-4" />
              </span>
              <h1 className="text-xl font-bold tracking-tight sm:text-2xl">
                Real-Time Bi-Temporal Analysis
              </h1>
            </div>
            <p className="mt-1 text-xs text-muted-foreground sm:text-sm">
              Compare historical satellite imagery with the latest available observation of the same location to automatically identify newly appeared, removed, or modified objects.
            </p>
          </div>

          {/* Mode Switcher Pill */}
          <div className="flex items-center rounded-lg border bg-muted/50 p-1 shrink-0">
            <button
              type="button"
              onClick={() => setBitemporalMode('latest')}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition ${
                bitemporalMode === 'latest'
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Sparkles className="size-3.5" /> Latest Satellite
            </button>
            <button
              type="button"
              onClick={() => setBitemporalMode('manual')}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition ${
                bitemporalMode === 'manual'
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Upload className="size-3.5" /> Manual Upload
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2 rounded-md bg-muted/40 px-3 py-1.5 text-[11px] text-muted-foreground">
          <ShieldCheck className="size-3.5 text-emerald-500 shrink-0" />
          <span>
            Satellite imagery availability depends on satellite revisit schedules, cloud cover, and data-provider latency.
          </span>
        </div>
      </div>

      {/* Input Configuration Grid */}
      <div className="grid gap-5 lg:grid-cols-12">
        {/* Left Column: Location & AOI Selection (or Manual File T1) */}
        <div className="lg:col-span-7 space-y-4">
          {bitemporalMode === 'latest' ? (
            <div className="rounded-xl border bg-card p-4 space-y-3.5">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground flex items-center gap-1.5">
                  <MapPin className="size-3.5 text-primary" /> 1. Area of Interest (AOI)
                </p>
                <span className="text-[11px] text-muted-foreground">
                  Center: {((aoi.north + aoi.south) / 2).toFixed(3)}°, {((aoi.east + aoi.west) / 2).toFixed(3)}°
                </span>
              </div>

              {/* Location Search Bar */}
              <form onSubmit={handleLocationSearch} className="flex gap-2">
                <div className="relative flex-1">
                  <Search className="absolute left-2.5 top-2.5 size-3.5 text-muted-foreground" />
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search city, port, or region (e.g. Mumbai, India)…"
                    className="h-9 pl-8 text-xs"
                  />
                </div>
                <Button type="submit" size="sm" disabled={isSearchingLoc} className="h-9 gap-1.5 text-xs">
                  {isSearchingLoc ? <Loader2 className="size-3.5 animate-spin" /> : <Search className="size-3.5" />}
                  Locate
                </Button>
              </form>

              {/* Variable AOI Span Controller */}
              {(() => {
                const centerLat = (aoi.north + aoi.south) / 2;
                const centerLng = (aoi.east + aoi.west) / 2;
                const latSpan = Math.abs(aoi.north - aoi.south);
                const lonSpan = Math.abs(aoi.east - aoi.west);
                const cosLat = Math.cos((centerLat * Math.PI) / 180);
                const currentSpanKm = Math.max(0.5, Math.round(Math.max(latSpan * 111, lonSpan * 111 * (Math.abs(cosLat) > 0.05 ? cosLat : 1.0)) * 10) / 10);

                const handleSetSpan = (targetKm: number) => {
                  const updated = computeAoiFromSpan(centerLat, centerLng, targetKm);
                  setAoi(updated);
                };

                return (
                  <div className="rounded-lg border bg-muted/25 p-2.5 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        <SlidersHorizontal className="size-3.5 text-primary" />
                        <span className="text-xs font-semibold text-foreground">Variable AOI Span:</span>
                        <span className="font-mono text-xs font-bold text-primary">~{currentSpanKm} km</span>
                      </div>
                      <span className="text-[11px] text-muted-foreground">
                        Footprint: ~{(currentSpanKm * currentSpanKm).toFixed(1)} km²
                      </span>
                    </div>

                    {/* Slider with Stepper Buttons */}
                    <div className="flex items-center gap-2.5">
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        className="size-7 rounded-md shrink-0"
                        onClick={() => handleSetSpan(Math.max(0.5, currentSpanKm - 1))}
                        title="Decrease span by 1 km"
                      >
                        <Minus className="size-3" />
                      </Button>
                      <Slider
                        value={[currentSpanKm]}
                        min={0.5}
                        max={30}
                        step={0.5}
                        onValueChange={([val]) => handleSetSpan(val)}
                        className="flex-1 py-1"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        className="size-7 rounded-md shrink-0"
                        onClick={() => handleSetSpan(Math.min(50, currentSpanKm + 1))}
                        title="Increase span by 1 km"
                      >
                        <Plus className="size-3" />
                      </Button>
                    </div>

                    {/* Quick Span Chips */}
                    <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                      <span className="text-[10px] uppercase font-semibold text-muted-foreground mr-1">Quick Spans:</span>
                      {[
                        { label: '1 km', desc: 'Facility', km: 1 },
                        { label: '2 km', desc: 'Terminal', km: 2 },
                        { label: '4 km', desc: 'District', km: 4 },
                        { label: '8 km', desc: 'Ward', km: 8 },
                        { label: '15 km', desc: 'Metro', km: 15 },
                        { label: '25 km', desc: 'Region', km: 25 },
                      ].map((item) => (
                        <button
                          key={item.km}
                          type="button"
                          onClick={() => handleSetSpan(item.km)}
                          className={`rounded-md border px-2 py-0.5 text-[11px] font-medium transition-all ${
                            Math.abs(currentSpanKm - item.km) <= (item.km <= 2 ? 0.4 : 1)
                              ? 'border-primary bg-primary/15 text-primary font-semibold shadow-xs'
                              : 'border-border/60 bg-background/60 text-muted-foreground hover:border-primary/40 hover:bg-muted hover:text-foreground'
                          }`}
                        >
                          <span className="font-semibold">{item.label}</span>{' '}
                          <span className="text-[9px] opacity-75">({item.desc})</span>
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })()}

              {/* Interactive AOI Map */}
              <AOIMap
                aoi={aoi}
                onChangeAoi={setAoi}
                onSelectPreset={(p) => {
                  setPrompt(p.defaultPrompt);
                  setHistoricalDate(p.defaultHistoricalDate);
                  toast({
                    title: `Preset loaded: ${p.name.split(',')[0]}`,
                    description: `Prompt: "${p.defaultPrompt.split('.')[0].trim()}" · Reference: ${p.defaultHistoricalDate}`,
                  });
                }}
                className="h-[280px]"
              />

              {/* Coordinates details with editable inputs */}
              <div className="space-y-1 pt-1">
                <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                  <span className="font-medium text-foreground">Bounding Box Coordinates (Degrees):</span>
                  <span>Click map to reposition center</span>
                </div>
                <div className="grid grid-cols-4 gap-2 text-[11px]">
                  <div className="rounded border bg-muted/20 p-1.5 space-y-1">
                    <span className="text-muted-foreground text-[10px] block font-medium">North (Lat Max)</span>
                    <Input
                      type="number"
                      step="0.001"
                      value={aoi.north}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) setAoi((prev) => ({ ...prev, north: val }));
                      }}
                      className="h-7 px-1.5 text-xs font-mono font-semibold bg-background"
                    />
                  </div>
                  <div className="rounded border bg-muted/20 p-1.5 space-y-1">
                    <span className="text-muted-foreground text-[10px] block font-medium">South (Lat Min)</span>
                    <Input
                      type="number"
                      step="0.001"
                      value={aoi.south}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) setAoi((prev) => ({ ...prev, south: val }));
                      }}
                      className="h-7 px-1.5 text-xs font-mono font-semibold bg-background"
                    />
                  </div>
                  <div className="rounded border bg-muted/20 p-1.5 space-y-1">
                    <span className="text-muted-foreground text-[10px] block font-medium">East (Lon Max)</span>
                    <Input
                      type="number"
                      step="0.001"
                      value={aoi.east}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) setAoi((prev) => ({ ...prev, east: val }));
                      }}
                      className="h-7 px-1.5 text-xs font-mono font-semibold bg-background"
                    />
                  </div>
                  <div className="rounded border bg-muted/20 p-1.5 space-y-1">
                    <span className="text-muted-foreground text-[10px] block font-medium">West (Lon Min)</span>
                    <Input
                      type="number"
                      step="0.001"
                      value={aoi.west}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) setAoi((prev) => ({ ...prev, west: val }));
                      }}
                      className="h-7 px-1.5 text-xs font-mono font-semibold bg-background"
                    />
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Manual Mode: Upload Time 1 & Time 2 */
            <div className="rounded-xl border bg-card p-4 space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground flex items-center gap-1.5">
                  <Upload className="size-3.5 text-primary" /> Upload Image Pair (Before & After)
                </p>
                <span className="text-[11px] text-muted-foreground">PNG, JPG, TIFF, WEBP</span>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                {/* Time 1 File */}
                <div className="rounded-xl border bg-muted/20 p-3 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="rounded bg-secondary/80 px-2 py-0.5 text-xs font-semibold">
                      Time 1 (Historical)
                    </span>
                    {manualFileT1 && (
                      <button
                        type="button"
                        onClick={() => handleManualFileT1(null)}
                        className="text-muted-foreground hover:text-destructive text-xs flex items-center gap-1"
                        title="Remove file"
                      >
                        <X className="size-3.5" /> Remove
                      </button>
                    )}
                  </div>

                  {manualPreviewT1 ? (
                    <div className="relative h-36 w-full overflow-hidden rounded-lg border bg-black/40">
                      <img
                        src={manualPreviewT1}
                        alt="Time 1 preview"
                        className="size-full object-contain"
                      />
                      <div className="absolute bottom-1 inset-x-1 rounded bg-black/70 px-2 py-0.5 text-[10px] text-white truncate backdrop-blur">
                        {manualFileT1?.name}
                      </div>
                    </div>
                  ) : (
                    <label
                      onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                      onDrop={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        const file = e.dataTransfer.files?.[0];
                        if (file) handleManualFileT1(file);
                      }}
                      className="flex h-36 cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-border bg-card/60 p-4 text-center transition hover:border-primary/60 hover:bg-accent/40"
                    >
                      <Upload className="size-6 text-muted-foreground" />
                      <div className="space-y-0.5">
                        <span className="text-xs font-semibold text-primary">Upload Time 1 Image</span>
                        <p className="text-[10px] text-muted-foreground">Click or drop reference image</p>
                      </div>
                      <input
                        type="file"
                        accept="image/*,.tif,.tiff"
                        onChange={(e) => {
                          handleManualFileT1(e.target.files?.[0] || null);
                          e.target.value = '';
                        }}
                        className="hidden"
                      />
                    </label>
                  )}

                  <div className="space-y-1">
                    <label className="text-[10px] text-muted-foreground font-medium">Acquisition Date (optional)</label>
                    <Input
                      value={manualDateT1}
                      onChange={(e) => setManualDateT1(e.target.value)}
                      placeholder="YYYY-MM-DD"
                      className="h-8 text-xs font-mono"
                    />
                  </div>
                </div>

                {/* Time 2 File */}
                <div className="rounded-xl border bg-muted/20 p-3 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="rounded bg-primary/20 text-primary px-2 py-0.5 text-xs font-semibold">
                      Time 2 (Recent)
                    </span>
                    {manualFileT2 && (
                      <button
                        type="button"
                        onClick={() => handleManualFileT2(null)}
                        className="text-muted-foreground hover:text-destructive text-xs flex items-center gap-1"
                        title="Remove file"
                      >
                        <X className="size-3.5" /> Remove
                      </button>
                    )}
                  </div>

                  {manualPreviewT2 ? (
                    <div className="relative h-36 w-full overflow-hidden rounded-lg border bg-black/40">
                      <img
                        src={manualPreviewT2}
                        alt="Time 2 preview"
                        className="size-full object-contain"
                      />
                      <div className="absolute bottom-1 inset-x-1 rounded bg-black/70 px-2 py-0.5 text-[10px] text-white truncate backdrop-blur">
                        {manualFileT2?.name}
                      </div>
                    </div>
                  ) : (
                    <label
                      onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                      onDrop={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        const file = e.dataTransfer.files?.[0];
                        if (file) handleManualFileT2(file);
                      }}
                      className="flex h-36 cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-border bg-card/60 p-4 text-center transition hover:border-primary/60 hover:bg-accent/40"
                    >
                      <Upload className="size-6 text-primary" />
                      <div className="space-y-0.5">
                        <span className="text-xs font-semibold text-primary">Upload Time 2 Image</span>
                        <p className="text-[10px] text-muted-foreground">Click or drop recent image</p>
                      </div>
                      <input
                        type="file"
                        accept="image/*,.tif,.tiff"
                        onChange={(e) => {
                          handleManualFileT2(e.target.files?.[0] || null);
                          e.target.value = '';
                        }}
                        className="hidden"
                      />
                    </label>
                  )}

                  <div className="space-y-1">
                    <label className="text-[10px] text-muted-foreground font-medium">Acquisition Date (optional)</label>
                    <Input
                      value={manualDateT2}
                      onChange={(e) => setManualDateT2(e.target.value)}
                      placeholder="YYYY-MM-DD"
                      className="h-8 text-xs font-mono"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Time 1 / Time 2 Configuration & Prompt */}
        <div className="lg:col-span-5 space-y-4">
          {bitemporalMode === 'latest' && (
            <>
              {/* Time 1: Historical Setting */}
              <div className="rounded-xl border bg-card p-4 space-y-2.5">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground flex items-center gap-1.5">
                    <Calendar className="size-3.5 text-primary" /> 2. Historical Reference (Time 1)
                  </p>
                  <div className="flex rounded border bg-muted/40 p-0.5 text-[10px]">
                    <button
                      type="button"
                      onClick={() => setHistoricalMode('date')}
                      className={`px-2 py-0.5 rounded font-medium ${
                        historicalMode === 'date' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground'
                      }`}
                    >
                      By Date
                    </button>
                    <button
                      type="button"
                      onClick={() => setHistoricalMode('upload')}
                      className={`px-2 py-0.5 rounded font-medium ${
                        historicalMode === 'upload' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground'
                      }`}
                    >
                      Upload Image
                    </button>
                  </div>
                </div>

                {historicalMode === 'date' ? (
                  <div className="space-y-1">
                    <label className="text-[11px] text-muted-foreground">Historical Date (±14 days window)</label>
                    <Input
                      type="date"
                      value={historicalDate}
                      onChange={(e) => setHistoricalDate(e.target.value)}
                      className="h-9 text-xs font-mono"
                    />
                  </div>
                ) : (
                  <div className="space-y-2">
                    {historicalFilePreview ? (
                      <div className="flex items-center gap-3 rounded-lg border bg-muted/30 p-2">
                        <img
                          src={historicalFilePreview}
                          alt="Historical reference preview"
                          className="size-12 rounded object-cover border"
                        />
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-xs font-semibold">{historicalFile?.name}</p>
                          <p className="text-[10px] text-muted-foreground">
                            {historicalFile ? `${(historicalFile.size / 1024).toFixed(0)} KB` : ''} · Ready for comparison
                          </p>
                        </div>
                        <button
                          type="button"
                          onClick={() => handleHistoricalFile(null)}
                          className="rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                          title="Remove file"
                        >
                          <X className="size-4" />
                        </button>
                      </div>
                    ) : (
                      <label
                        onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                        onDrop={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          const file = e.dataTransfer.files?.[0];
                          if (file) handleHistoricalFile(file);
                        }}
                        className="flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-lg border-2 border-dashed border-border bg-muted/20 p-4 text-center transition hover:border-primary/60 hover:bg-accent/40"
                      >
                        <Upload className="size-5 text-muted-foreground" />
                        <span className="text-xs font-semibold text-primary">Upload Reference Satellite Image</span>
                        <p className="text-[10px] text-muted-foreground">PNG, JPG, TIFF up to 15MB</p>
                        <input
                          type="file"
                          accept="image/*,.tif,.tiff"
                          onChange={(e) => {
                            handleHistoricalFile(e.target.files?.[0] || null);
                            e.target.value = '';
                          }}
                          className="hidden"
                        />
                      </label>
                    )}
                  </div>
                )}
              </div>

              {/* Time 2: Latest Satellite Configuration */}
              <div className="rounded-xl border bg-card p-4 space-y-2.5">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground flex items-center gap-1.5">
                    <CloudSun className="size-3.5 text-emerald-500" /> 3. Latest Available Satellite (Time 2)
                  </p>
                  <span className="rounded bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 px-1.5 py-0.5 text-[10px] font-semibold">
                    Auto-Search
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <label className="text-[11px] text-muted-foreground">Max Cloud Cover</label>
                    <div className="flex items-center gap-2 mt-1">
                      <input
                        type="range"
                        min="5"
                        max="50"
                        step="5"
                        value={maxCloudCover}
                        onChange={(e) => setMaxCloudCover(parseInt(e.target.value))}
                        className="h-2 flex-1 accent-primary"
                      />
                      <span className="w-8 font-mono text-[11px] font-semibold">{maxCloudCover}%</span>
                    </div>
                  </div>
                  <div>
                    <label className="text-[11px] text-muted-foreground">Search Window</label>
                    <div className="flex items-center gap-2 mt-1">
                      <input
                        type="range"
                        min="7"
                        max="60"
                        step="7"
                        value={searchDays}
                        onChange={(e) => setSearchDays(parseInt(e.target.value))}
                        className="h-2 flex-1 accent-primary"
                      />
                      <span className="w-10 font-mono text-[11px] font-semibold">{searchDays}d</span>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Object Detection Prompt */}
          <div className="rounded-xl border bg-card p-4 space-y-2.5">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground flex items-center gap-1.5">
              <Sparkles className="size-3.5 text-primary" /> Target Object Prompt
            </p>
            <Input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g. ship, building, road, vehicle…"
              className="h-9 text-xs"
            />
            {/* Quick chips */}
            <div className="flex flex-wrap gap-1 pt-1">
              {[
                { label: 'Ships', val: 'ship . vessel . cargo ship . boat .' },
                { label: 'Buildings', val: 'building . structure . rooftop .' },
                { label: 'Roads', val: 'road . street . highway . pathway .' },
                { label: 'Vehicles', val: 'vehicle . car . truck . bus .' },
                { label: 'Bridges', val: 'bridge . overpass . viaduct .' },
              ].map((chip) => (
                <button
                  key={chip.label}
                  type="button"
                  onClick={() => setPrompt(chip.val)}
                  className="rounded-full border bg-muted/40 px-2 py-0.5 text-[10px] font-medium transition hover:border-primary/50 hover:bg-primary/10"
                >
                  {chip.label}
                </button>
              ))}
            </div>
          </div>

          {/* Action Button */}
          <Button
            size="lg"
            disabled={isLoading}
            onClick={bitemporalMode === 'latest' ? runLatestComparison : runManualComparison}
            className="w-full gap-2 text-sm font-semibold shadow-md"
          >
            {isLoading ? (
              <>
                <Loader2 className="size-4 animate-spin" /> Analyzing Imagery…
              </>
            ) : bitemporalMode === 'latest' ? (
              <>
                <GitCompareArrows className="size-4" /> Compare With Latest Satellite
              </>
            ) : (
              <>
                <GitCompareArrows className="size-4" /> Compare Uploaded Images
              </>
            )}
          </Button>

          {/* Multi-Stage Loading Progress */}
          {isLoading && (
            <div className="rounded-xl border bg-card/90 p-4 space-y-3 shadow-inner">
              <div className="flex items-center gap-2 text-xs font-semibold text-primary">
                <Loader2 className="size-4 animate-spin" />
                <span>Processing Bi-Temporal Pipeline</span>
              </div>
              <p className="text-xs font-medium text-foreground">
                {LOADING_STAGES[loadingStageIdx]}
              </p>
              <div className="space-y-1">
                {LOADING_STAGES.map((stage, idx) => (
                  <div
                    key={stage}
                    className={`flex items-center gap-2 text-[11px] ${
                      idx < loadingStageIdx
                        ? 'text-emerald-600 dark:text-emerald-400 font-medium'
                        : idx === loadingStageIdx
                        ? 'text-primary font-semibold'
                        : 'text-muted-foreground/60'
                    }`}
                  >
                    {idx < loadingStageIdx ? (
                      <CheckCircle2 className="size-3 shrink-0" />
                    ) : idx === loadingStageIdx ? (
                      <span className="size-2 rounded-full bg-primary glow-pulse shrink-0 ml-0.5 mr-0.5" />
                    ) : (
                      <span className="size-1.5 rounded-full bg-muted shrink-0 ml-1 mr-0.5" />
                    )}
                    <span>{stage}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Comparison Results Section */}
      {result && (
        <div className="space-y-6 pt-4 border-t">
          {/* Top Metadata Header Banner */}
          <div className="rounded-2xl border bg-gradient-to-r from-card via-card to-primary/5 p-4 sm:p-5 shadow-sm">
            <div className="grid gap-4 md:grid-cols-3">
              {/* Historical Scene Card */}
              <div className="rounded-xl border bg-card/80 p-3 space-y-1">
                <span className="inline-flex items-center gap-1 rounded bg-secondary/80 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                  Time 1 · Historical
                </span>
                <p className="text-base font-bold">{result.historical.acquisitionDate}</p>
                <div className="text-[11px] text-muted-foreground space-y-0.5">
                  <p>Provider: {result.historical.provider}</p>
                  {result.historical.cloudCoverage !== null && result.historical.cloudCoverage !== undefined && (
                    <p>Cloud: {result.historical.cloudCoverage}%</p>
                  )}
                  <p>Resolution: {result.historical.resolution}</p>
                </div>
              </div>

              {/* Center VS & Freshness Card */}
              <div className="flex flex-col items-center justify-center text-center p-2">
                <div className="flex items-center gap-2">
                  <span className="size-2 rounded-full bg-primary" />
                  <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    Temporal Span
                  </span>
                  <span className="size-2 rounded-full bg-primary" />
                </div>
                <p className="text-xl font-bold tracking-tight text-primary mt-1">
                  {result.timeDifference}
                </p>
                <span className="mt-1 inline-flex items-center gap-1 rounded-full border bg-background px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                  <Clock className="size-3 text-emerald-500" /> Latest imagery age: {result.latestImageryAge}
                </span>
              </div>

              {/* Latest Available Scene Card */}
              <div className="rounded-xl border border-primary/40 bg-primary/5 p-3 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="inline-flex items-center gap-1 rounded bg-primary text-primary-foreground px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider">
                    Time 2 · Latest Available
                  </span>
                  <span className="size-2 rounded-full bg-emerald-500 glow-pulse" />
                </div>
                <p className="text-base font-bold text-foreground">{result.latest.acquisitionDate}</p>
                <div className="text-[11px] text-muted-foreground space-y-0.5">
                  <p>Provider: {result.latest.provider}</p>
                  {result.latest.cloudCoverage !== null && result.latest.cloudCoverage !== undefined && (
                    <p>Cloud Coverage: <span className="font-semibold text-foreground">{result.latest.cloudCoverage}%</span></p>
                  )}
                  <p>Resolution: {result.latest.resolution}</p>
                </div>
              </div>
            </div>

            {/* Registration Quality Warning if applicable */}
            {result.registration.warning && (
              <div className="mt-3 flex items-center gap-2 rounded-lg bg-amber-500/10 border border-amber-500/20 px-3 py-1.5 text-xs text-amber-700 dark:text-amber-300">
                <AlertTriangle className="size-3.5 shrink-0" />
                <span>{result.registration.warning} (Alignment Quality: {Math.round(result.registration.quality * 100)}%)</span>
              </div>
            )}
          </div>

          {/* Metrics Summary Strip */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            <div className="rounded-xl border bg-card p-3 text-center">
              <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Before</span>
              <p className="text-xl font-bold">{result.summary.totalBefore}</p>
            </div>
            <div className="rounded-xl border bg-card p-3 text-center">
              <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Latest</span>
              <p className="text-xl font-bold">{result.summary.totalLatest}</p>
            </div>
            <div className="rounded-xl border bg-emerald-500/10 border-emerald-500/30 p-3 text-center">
              <span className="text-[10px] uppercase tracking-wide text-emerald-700 dark:text-emerald-300 font-semibold">New</span>
              <p className="text-xl font-bold text-emerald-600 dark:text-emerald-400">+{result.summary.newCount}</p>
            </div>
            <div className="rounded-xl border bg-red-500/10 border-red-500/30 p-3 text-center">
              <span className="text-[10px] uppercase tracking-wide text-red-700 dark:text-red-300 font-semibold">Removed</span>
              <p className="text-xl font-bold text-red-600 dark:text-red-400">-{result.summary.removedCount}</p>
            </div>
            <div className="rounded-xl border bg-amber-500/10 border-amber-500/30 p-3 text-center">
              <span className="text-[10px] uppercase tracking-wide text-amber-700 dark:text-amber-300 font-semibold">Modified</span>
              <p className="text-xl font-bold text-amber-600 dark:text-amber-400">{result.summary.modifiedCount}</p>
            </div>
            <div className="rounded-xl border bg-card p-3 text-center">
              <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Unchanged</span>
              <p className="text-xl font-bold">{result.summary.unchangedCount}</p>
            </div>
          </div>

          {/* Main Visual Comparison Suite & Change List Grid */}
          <div className="grid gap-6 lg:grid-cols-12">
            {/* Left: Interactive Visual Views */}
            <div className="lg:col-span-8 space-y-3">
              {/* View Mode Switcher */}
              <div className="flex items-center justify-between border-b pb-2">
                <div className="flex items-center gap-1 rounded-lg border bg-muted/40 p-1">
                  <button
                    type="button"
                    onClick={() => setResultViewMode('slider')}
                    className={`rounded-md px-3 py-1 text-xs font-semibold transition ${
                      resultViewMode === 'slider' ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground'
                    }`}
                  >
                    Split Slider
                  </button>
                  <button
                    type="button"
                    onClick={() => setResultViewMode('side')}
                    className={`rounded-md px-3 py-1 text-xs font-semibold transition ${
                      resultViewMode === 'side' ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground'
                    }`}
                  >
                    Side-by-Side
                  </button>
                  {result.pixelChange && (
                    <button
                      type="button"
                      onClick={() => setResultViewMode('overlay')}
                      className={`rounded-md px-3 py-1 text-xs font-semibold transition ${
                        resultViewMode === 'overlay' ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground'
                      }`}
                    >
                      Change Overlay ({result.pixelChange.changePercentage}%)
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => setResultViewMode('map')}
                    className={`rounded-md px-3 py-1 text-xs font-semibold transition ${
                      resultViewMode === 'map' ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground'
                    }`}
                  >
                    Change Map
                  </button>
                </div>
              </div>

              {/* Object detection (Grounding DINO + SAM2) stage summary */}
              {result.objectDetection && (
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 font-semibold text-emerald-700 dark:text-emerald-300">
                    <Layers className="size-3" /> Grounding DINO + SAM
                  </span>
                  <span className="text-muted-foreground">
                    T1: {result.objectDetection.t1.detectionsCount} objects · T2: {result.objectDetection.t2.detectionsCount} objects
                  </span>
                  {(result.objectDetection.t1.segmentationAvailable ||
                    result.objectDetection.t2.segmentationAvailable) ? (
                    <span className="rounded-full bg-primary/15 px-2 py-0.5 font-medium text-primary">
                      SAM masks: {Math.max(result.objectDetection.t1.segmentedCount, result.objectDetection.t2.segmentedCount)} segmented
                    </span>
                  ) : (
                    <span className="rounded-full bg-amber-500/15 px-2 py-0.5 font-medium text-amber-700 dark:text-amber-300">
                      SAM unavailable — box-only matching
                    </span>
                  )}
                  {(result.objectDetection.t1.verificationAvailable ||
                    result.objectDetection.t2.verificationAvailable) && (
                    <span className="rounded-full bg-sky-500/15 px-2 py-0.5 font-medium text-sky-700 dark:text-sky-300">
                      SigLIP verified: {result.objectDetection.t1.verifiedCount + result.objectDetection.t2.verifiedCount}
                    </span>
                  )}
                </div>
              )}

              {/* View 1: Split Slider */}
              {resultViewMode === 'slider' && (
                <ImageSlider
                  t1Url={result.images.t1DataUrl}
                  t2Url={result.images.t2DataUrl}
                  t1Label={result.historical.acquisitionDate}
                  t2Label={result.latest.acquisitionDate}
                  changes={result.changes}
                />
              )}

              {/* View 2: Side-by-Side */}
              {resultViewMode === 'side' && (
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-1.5">
                    <span className="rounded bg-secondary/80 px-2 py-0.5 text-xs font-semibold">
                      Historical: {result.historical.acquisitionDate}
                    </span>
                    <div className="relative h-[380px] overflow-hidden rounded-xl border bg-black">
                      <img
                        src={result.images.t1DataUrl}
                        alt="Historical"
                        className="size-full object-contain"
                      />
                      {/* GDINO change boxes on Time 1 */}
                      <div className="pointer-events-none absolute inset-0">
                        {result.changes
                          .filter((c) => c.boxT1)
                          .map((c) => {
                            const [x1, y1, x2, y2] = c.boxT1!;
                            const color =
                              c.type === 'removed' ? '#ef4444' : c.type === 'modified' ? '#f59e0b' : '#64748b';
                            return (
                              <div
                                key={`side-t1-${c.id}`}
                                className="absolute border-2"
                                style={{
                                  left: `${(x1 / 640) * 100}%`,
                                  top: `${(y1 / 640) * 100}%`,
                                  width: `${((x2 - x1) / 640) * 100}%`,
                                  height: `${((y2 - y1) / 640) * 100}%`,
                                  borderColor: color,
                                  backgroundColor: `${color}25`,
                                }}
                              >
                                <span
                                  className="absolute -top-5 left-0 whitespace-nowrap rounded px-1 text-[9px] font-bold text-white shadow"
                                  style={{ backgroundColor: color }}
                                >
                                  {c.type === 'removed' ? 'REMOVED' : c.type === 'modified' ? 'MOD' : 'OK'} · {c.label}
                                </span>
                              </div>
                            );
                          })}
                      </div>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <span className="rounded bg-primary text-primary-foreground px-2 py-0.5 text-xs font-semibold">
                      Latest: {result.latest.acquisitionDate}
                    </span>
                    <div className="relative h-[380px] overflow-hidden rounded-xl border bg-black">
                      <img
                        src={result.images.t2DataUrl}
                        alt="Latest"
                        className="size-full object-contain"
                      />
                      {/* GDINO change boxes on Time 2 */}
                      <div className="pointer-events-none absolute inset-0">
                        {result.changes
                          .filter((c) => c.boxT2)
                          .map((c) => {
                            const [x1, y1, x2, y2] = c.boxT2!;
                            const color =
                              c.type === 'new' ? '#10b981' : c.type === 'modified' ? '#f59e0b' : '#64748b';
                            return (
                              <div
                                key={`side-t2-${c.id}`}
                                className="absolute border-2"
                                style={{
                                  left: `${(x1 / 640) * 100}%`,
                                  top: `${(y1 / 640) * 100}%`,
                                  width: `${((x2 - x1) / 640) * 100}%`,
                                  height: `${((y2 - y1) / 640) * 100}%`,
                                  borderColor: color,
                                  backgroundColor: `${color}25`,
                                }}
                              >
                                <span
                                  className="absolute -top-5 left-0 whitespace-nowrap rounded px-1 text-[9px] font-bold text-white shadow"
                                  style={{ backgroundColor: color }}
                                >
                                  {c.type === 'new' ? 'NEW' : c.type === 'modified' ? 'MOD' : 'OK'} · {c.label}
                                </span>
                              </div>
                            );
                          })}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* View 3: Change Overlay */}
              {resultViewMode === 'overlay' && result.pixelChange && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>Radiometric pixel changes highlighted in red/amber</span>
                    <span className="font-semibold text-foreground">
                      {result.pixelChange.changePercentage}% surface modified
                    </span>
                  </div>
                  <div className="relative h-[440px] overflow-hidden rounded-xl border bg-black">
                    <img
                      src={result.images.t2DataUrl}
                      alt="Latest base"
                      className="absolute inset-0 size-full object-contain"
                    />
                    <img
                      src={result.pixelChange.overlayDataUrl}
                      alt="Change overlay"
                      className="absolute inset-0 size-full object-contain pointer-events-none"
                    />
                  </div>
                </div>
              )}

              {/* View 4: Change Map */}
              {resultViewMode === 'map' && (
                <ChangeMap
                  aoi={result.aoi}
                  changes={result.changes}
                  selectedChangeId={selectedChangeId}
                  onSelectChange={setSelectedChangeId}
                  className="h-[460px]"
                />
              )}
            </div>

            {/* Right: Detected Changes List */}
            <div className="lg:col-span-4 space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground flex items-center gap-1.5">
                  <Layers className="size-3.5 text-primary" /> Detected Changes ({result.changes.length})
                </p>
                <span className="text-[11px] text-muted-foreground">Click to focus</span>
              </div>

              <div className="satquery-scroll max-h-[500px] overflow-y-auto space-y-2 pr-1">
                {result.changes.map((change) => {
                  const isSelected = change.id === selectedChangeId;
                  return (
                    <div
                      key={change.id}
                      onClick={() => setSelectedChangeId(change.id)}
                      className={`cursor-pointer rounded-lg border p-2.5 text-xs transition-all ${
                        isSelected
                          ? 'border-primary bg-primary/10 shadow-sm'
                          : 'bg-card hover:border-primary/40'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span
                          className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide ${
                            change.type === 'new'
                              ? 'bg-emerald-500/20 text-emerald-700 dark:text-emerald-300'
                              : change.type === 'removed'
                              ? 'bg-red-500/20 text-red-700 dark:text-red-300'
                              : change.type === 'modified'
                              ? 'bg-amber-500/20 text-amber-700 dark:text-amber-300'
                              : 'bg-muted text-muted-foreground'
                          }`}
                        >
                          {change.type === 'new' ? (
                            <ArrowUpRight className="size-3" />
                          ) : change.type === 'removed' ? (
                            <ArrowDownRight className="size-3" />
                          ) : (
                            <Minus className="size-3" />
                          )}
                          {change.type}
                        </span>
                        <span className="font-semibold text-foreground">
                          {Math.round(change.confidence * 100)}%
                        </span>
                      </div>

                      <p className="mt-1 font-semibold text-foreground capitalize">
                        {change.label}
                      </p>
                      <p className="text-[11px] text-muted-foreground line-clamp-2 mt-0.5">
                        {change.details}
                      </p>

                      {change.latitude !== null && change.longitude !== null && (
                        <div className="mt-1.5 flex items-center gap-1 text-[10px] text-muted-foreground font-mono">
                          <MapPin className="size-3 text-primary shrink-0" />
                          <span>{change.latitude?.toFixed(4)}°, {change.longitude?.toFixed(4)}°</span>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

