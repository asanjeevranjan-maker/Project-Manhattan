'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { fileToDataUrl } from '@/lib/client-utils';
import {
  Upload,
  X,
  RadioTower,
  Radar,
  Layers3,
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
  AlertTriangle,
  Sparkles,
  LoaderCircle,
  Workflow,
} from 'lucide-react';
interface FusionResult {
  task: 'optical_sar_fusion';
  status: 'VERIFIED' | 'UNCERTAIN' | 'INSUFFICIENT_EVIDENCE';
  status_reason?: string;
  answer?: string;
  input_validation?: {
    optical_detected: boolean;
    sar_detected: boolean;
    pair_compatible: boolean;
    overlap_ratio?: number | null;
    modality_detection?: any;
  };
  optical?: any;
  sar?: any;
  alignment?: any;
  fusion?: any;
  claims?: any[];
  region_claims?: any[];
  reliability_score?: number;
  visualizations?: {
    opticalDataUrl: string;
    sarDataUrl: string;
    fusedOverlayDataUrl: string;
  };
  warnings?: string[];
  trace?: { workflow: string[]; tools: string[]; runtime_ms: Record<string, number> };
  error?: string;
}

const LOADING_STAGES = [
  'Validating images & detecting modalities…',
  'Aligning optical and SAR on a common grid…',
  'Analyzing optical imagery…',
  'Analyzing SAR backscatter…',
  'Fusing cross-modal evidence…',
  'Verifying result…',
];

const EXAMPLE_PROMPTS = [
  'Where are the built-up areas supported by both optical and SAR?',
  'Is there water in this scene?',
  'Analyze vegetation using both sensors.',
];

function DropSlot({
  label,
  hint,
  file,
  preview,
  onSelect,
  onClear,
  icon,
}: {
  label: string;
  hint: string;
  file: File | null;
  preview: string | null;
  onSelect: (f: File) => void;
  onClear: () => void;
  icon: React.ReactNode;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <div className="rounded-xl border bg-card p-4 space-y-2.5">
      <div className="flex items-center gap-2">
        {icon}
        <div>
          <p className="text-sm font-semibold">{label}</p>
          <p className="text-[11px] text-muted-foreground">{hint}</p>
        </div>
      </div>

      {file && preview ? (
        <div className="flex items-center gap-3 rounded-lg border bg-muted/30 p-2">
          <img src={preview} alt={label} className="size-14 rounded object-cover border" />
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-medium">{file.name}</p>
            <p className="text-[10px] text-muted-foreground">{(file.size / 1024).toFixed(0)} KB</p>
          </div>
          <button
            type="button"
            onClick={onClear}
            className="rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
          >
            <X className="size-3.5" />
          </button>
        </div>
      ) : (
        <label
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files?.[0];
            if (f) onSelect(f);
          }}
          onDragOver={(e) => e.preventDefault()}
          className="flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-lg border-2 border-dashed border-border bg-muted/20 p-6 text-center transition hover:border-primary/60 hover:bg-accent/40"
        >
          <Upload className="size-5 text-muted-foreground" />
          <span className="text-xs font-semibold text-primary">Upload {label}</span>
          <span className="text-[10px] text-muted-foreground">Click or drag & drop (JPEG / PNG / GeoTIFF)</span>
          <input
            ref={inputRef}
            type="file"
            accept="image/*,.tif,.tiff"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onSelect(f);
              e.target.value = '';
            }}
          />
        </label>
      )}
    </div>
  );
}

export function FusionWorkspace() {
  const { toast } = useToast();
  const [opticalFile, setOpticalFile] = useState<File | null>(null);
  const [sarFile, setSarFile] = useState<File | null>(null);
  const [opticalPreview, setOpticalPreview] = useState<string | null>(null);
  const [sarPreview, setSarPreview] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [stageIdx, setStageIdx] = useState(0);
  const [result, setResult] = useState<FusionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!loading) return;
    const t = setInterval(() => setStageIdx((i) => Math.min(i + 1, LOADING_STAGES.length - 1)), 6000);
    return () => clearInterval(t);
  }, [loading]);

  const handleSelect = useCallback(async (which: 'optical' | 'sar', f: File) => {
    try {
      const dataUrl = await fileToDataUrl(f);
      if (which === 'optical') {
        setOpticalFile(f);
        setOpticalPreview(dataUrl);
      } else {
        setSarFile(f);
        setSarPreview(dataUrl);
      }
    } catch {
      toast({ title: 'Could not read file', description: 'Please choose a valid image file.' });
    }
  }, [toast]);

  const runFusion = useCallback(async () => {
    if (!opticalFile || !sarFile) {
      toast({ title: 'Images required', description: 'Please upload both an optical image and a SAR image.' });
      return;
    }
    setLoading(true);
    setStageIdx(0);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      // Slots are labelled, but the backend re-detects modalities — order is safe.
      fd.append('file1', opticalFile);
      fd.append('file2', sarFile);
      if (prompt.trim()) fd.append('prompt', prompt.trim());

      const res = await fetch('/api/fusion/optical-sar', { method: 'POST', body: fd });
      const data: FusionResult = await res.json();
      if (data.error) {
        setError(data.error);
        if (data.status) setResult(data);
        toast({ title: 'Fusion could not proceed', description: data.error });
      } else {
        setResult(data);
        toast({
          title: `Fusion ${data.status.toLowerCase()} — reliability ${Math.round((data.reliability_score ?? 0) * 100)}%`,
          description: `Cross-modal agreement: ${data.fusion?.agreement_level ?? 'n/a'}`,
        });
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Fusion request failed';
      setError(msg);
      toast({ title: 'Fusion failed', description: msg });
    } finally {
      setLoading(false);
    }
  }, [opticalFile, sarFile, prompt, toast]);

  return (
    <FusionLayout
      loading={loading}
      stageIdx={stageIdx}
      prompt={prompt}
      setPrompt={setPrompt}
      runFusion={runFusion}
      opticalSlot={
        <DropSlot
          label="OPTICAL IMAGE"
          hint="e.g. Sentinel-2, Landsat, multispectral / RGB imagery"
          file={opticalFile}
          preview={opticalPreview}
          onSelect={(f) => handleSelect('optical', f)}
          onClear={() => { setOpticalFile(null); setOpticalPreview(null); }}
          icon={<Layers3 className="size-5 text-sky-500" />}
        />
      }
      sarSlot={
        <DropSlot
          label="SAR IMAGE"
          hint="e.g. Sentinel-1 SAR, VV/VH radar imagery"
          file={sarFile}
          preview={sarPreview}
          onSelect={(f) => handleSelect('sar', f)}
          onClear={() => { setSarFile(null); setSarPreview(null); }}
          icon={<Radar className="size-5 text-emerald-500" />}
        />
      }
      error={error}
      result={result}
    />
  );
}

function FusionLayout({
  loading,
  stageIdx,
  prompt,
  setPrompt,
  runFusion,
  opticalSlot,
  sarSlot,
  error,
  result,
}: {
  loading: boolean;
  stageIdx: number;
  prompt: string;
  setPrompt: (v: string) => void;
  runFusion: () => void;
  opticalSlot: React.ReactNode;
  sarSlot: React.ReactNode;
  error: string | null;
  result: FusionResult | null;
}) {
  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="rounded-xl border bg-card p-4 space-y-1">
        <div className="flex items-center gap-2">
          <RadioTower className="size-5 text-primary" />
          <h1 className="text-lg font-bold tracking-tight">Optical + SAR Fusion</h1>
        </div>
        <p className="text-xs text-muted-foreground">
          Combine spectral information from optical imagery with structural/backscatter information from SAR.
          Upload one optical and one SAR image of the same location — the engine detects modalities, aligns the
          pair, and fuses measured evidence from both sensors.
        </p>
      </div>

      {/* Uploads: LEFT optical, RIGHT SAR */}
      <div className="grid gap-3 md:grid-cols-2">
        {opticalSlot}
        {sarSlot}
      </div>

      {/* Prompt + run */}
      <div className="rounded-xl border bg-card p-4 space-y-2.5">
        <label className="text-xs font-semibold text-muted-foreground">
          Analysis question (optional — leave empty for a general fusion analysis)
        </label>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder='e.g. "Where are the built-up areas supported by both optical and SAR?"'
          className="w-full rounded-lg border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring"
        />
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLE_PROMPTS.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setPrompt(p)}
              className="rounded-full border bg-muted/40 px-2.5 py-0.5 text-[10px] text-muted-foreground transition hover:border-primary/50 hover:text-foreground"
            >
              {p}
            </button>
          ))}
        </div>
        <Button onClick={runFusion} disabled={loading} className="w-full gap-2">
          {loading ? <LoaderCircle className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
          {loading ? LOADING_STAGES[stageIdx] : 'RUN FUSION ANALYSIS'}
        </Button>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <div>
            <p className="font-semibold">Fusion could not proceed</p>
            <p className="text-xs opacity-90">{error}</p>
          </div>
        </div>
      )}

      {/* Result */}
      {result && !error && <FusionResultView result={result} />}
      {result && error && <FusionResultView result={result} compact />}
    </div>
  );
}

function FusionResultView({ result, compact = false }: { result: FusionResult; compact?: boolean }) {
  if (compact) return null;
  const statusMeta =
    result.status === 'VERIFIED'
      ? { icon: <ShieldCheck className="size-4" />, cls: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300' }
      : result.status === 'UNCERTAIN'
      ? { icon: <ShieldQuestion className="size-4" />, cls: 'bg-amber-500/15 text-amber-700 dark:text-amber-300' }
      : { icon: <ShieldAlert className="size-4" />, cls: 'bg-red-500/15 text-red-700 dark:text-red-300' };

  const features: any[] = result.fusion?.features ?? [];

  return (
    <div className="space-y-4">
      {/* Status cards */}
      <div className="grid gap-3 sm:grid-cols-3">
        <div className={`rounded-xl border p-4 space-y-1 ${statusMeta.cls}`}>
          <div className="flex items-center gap-2">{statusMeta.icon}
            <p className="text-[10px] font-bold uppercase tracking-wider opacity-80">Status</p>
          </div>
          <p className="text-xl font-bold">{result.status}</p>
          {result.status_reason && <p className="text-[10px] opacity-80 line-clamp-3">{result.status_reason}</p>}
        </div>
        <div className="rounded-xl border bg-card p-4 space-y-1">
          <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Reliability</p>
          <p className="text-xl font-bold">{Math.round((result.reliability_score ?? 0) * 100)}%</p>
          <p className="text-[10px] text-muted-foreground">input quality × agreement</p>
        </div>
        <div className="rounded-xl border bg-card p-4 space-y-1">
          <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Cross-modal agreement</p>
          <p className="text-xl font-bold">{result.fusion?.agreement_level ?? 'n/a'}</p>
          <p className="text-[10px] text-muted-foreground">
            {result.alignment ? `${result.alignment.method} · q ${result.alignment.quality?.toFixed?.(2) ?? '—'}` : ''}
          </p>
        </div>
      </div>

      {/* Answer */}
      {result.answer && (
        <div className="rounded-xl border bg-card p-4">
          <p className="mb-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Fused result</p>
          <p className="text-sm leading-relaxed">{result.answer}</p>
        </div>
      )}

      {/* Visual evidence */}
      {result.visualizations && (
        <div className="rounded-xl border bg-card p-4 space-y-2">
          <p className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
            <Layers3 className="size-3" /> Visual evidence
          </p>
          <div className="grid gap-2 sm:grid-cols-3">
            {[['Optical', result.visualizations.opticalDataUrl],
              ['SAR', result.visualizations.sarDataUrl],
              ['Fused (verified regions)', result.visualizations.fusedOverlayDataUrl]].map(([label, url]) => (
              <div key={label as string} className="space-y-1">
                <span className="rounded bg-secondary/70 px-1.5 py-0.5 text-[10px] font-semibold">{label}</span>
                <img src={url as string} alt={label as string} className="h-44 w-full rounded-lg border bg-black object-contain" />
              </div>
            ))}
          </div>
        </div>
      )}
{/* Per-feature evidence */}
      {features.length > 0 && (
        <div className="rounded-xl border bg-card p-4 space-y-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Feature evidence (optical vs SAR)</p>
          <div className="space-y-1.5">
            {features.map((f) => (
              <div key={f.feature} className="flex items-center gap-2 rounded-lg border bg-muted/20 px-3 py-2 text-xs">
                <span className="w-20 font-semibold capitalize">{f.feature}</span>
                <span className="text-muted-foreground">Opt {Math.round(f.optical_support * 100)}%</span>
                <span className="text-muted-foreground">SAR {Math.round(f.sar_support * 100)}%</span>
                <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                  f.agreement_level === 'HIGH' ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-300'
                  : f.agreement_level === 'MODERATE' ? 'bg-amber-500/15 text-amber-600 dark:text-amber-300'
                  : 'bg-red-500/15 text-red-600 dark:text-red-300'}`}>
                  {f.agreement_level}
                </span>
                <span className="ml-auto font-bold">{Math.round(f.fused_confidence * 100)}%</span>
              </div>
            ))}
          </div>
          {result.fusion?.config_note && (
            <p className="text-[10px] text-muted-foreground">{result.fusion.config_note}</p>
          )}
        </div>
      )}

      {/* Claims — every claim cites the evidence that supports it */}
      {(result.claims?.length || result.region_claims?.length) ? (
        <div className="rounded-xl border bg-card p-4 space-y-2">
          <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
            Supported claims (each cites its evidence)
          </p>
          {result.claims?.map((c: any, i: number) => (
            <div key={`c-${i}`} className="rounded-lg border bg-muted/20 p-2.5 text-xs">
              <p>{c.claim}</p>
              <div className="mt-1 flex flex-wrap gap-1.5 text-[10px] text-muted-foreground">
                {Object.entries(c.evidence || {}).map(([k, v]) => (
                  <span key={k} className="rounded bg-secondary/60 px-1.5 py-0.5">
                    <b className="capitalize">{k}:</b> {v as string}
                  </span>
                ))}
                <span className="rounded bg-secondary/60 px-1.5 py-0.5 font-bold">
                  {Math.round((c.confidence ?? 0) * 100)}%
                </span>
              </div>
            </div>
          ))}
          {result.region_claims?.map((c: any, i: number) => (
            <div key={`rc-${i}`} className="rounded-lg border bg-muted/20 p-2.5 text-xs">
              <p>{c.claim}</p>
              <p className="mt-1 text-[10px] font-bold text-muted-foreground">
                {Math.round((c.confidence ?? 0) * 100)}% · {c.agreement_level}
              </p>
            </div>
          ))}
        </div>
      ) : null}

      {/* Warnings */}
      {result.warnings && result.warnings.length > 0 && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-4 space-y-1.5">
          <p className="flex items-center gap-1.5 text-xs font-semibold text-amber-700 dark:text-amber-300">
            <AlertTriangle className="size-3.5" /> Warnings
          </p>
          {result.warnings.map((w, i) => (
            <p key={i} className="text-[11px] text-amber-700/90 dark:text-amber-200/90">• {w}</p>
          ))}
        </div>
      )}

      {/* Execution trace */}
      {result.trace && (
        <details className="rounded-xl border bg-card p-4">
          <summary className="flex cursor-pointer items-center gap-1.5 text-xs font-semibold text-muted-foreground">
            <Workflow className="size-3.5" /> Execution trace ({result.trace.workflow.length} steps · {result.trace.runtime_ms?.total ?? 0} ms)
          </summary>
          <div className="mt-2 space-y-1 text-[11px] text-muted-foreground">
            {result.trace.workflow.map((s, i) => (
              <p key={i}>{i + 1}. {s}</p>
            ))}
            <p className="pt-1 font-semibold">Tools: {result.trace.tools.join(', ')}</p>
          </div>
        </details>
      )}
    </div>
  );
}
