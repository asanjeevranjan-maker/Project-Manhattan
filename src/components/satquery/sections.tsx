'use client';

import {
  Upload,
  MessageSquareText,
  Cpu,
  ScanEye,
  BarChart3,
  Layers,
  GitCompareArrows,
  FileText,
  Satellite,
  Waves,
  Building2,
  Trees,
  Flame,
  Tractor,
  Truck,
  CloudRain,
  ShieldAlert,
  Microscope,
  GraduationCap,
  LandPlot,
} from 'lucide-react';

const STEPS = [
  {
    icon: Upload,
    title: 'Upload or pick a satellite image',
    body: 'Drag-and-drop a PNG, JPG, or TIFF — or pick one of the curated sample images from NASA Earth Observatory covering floods, urban areas, deforestation, agriculture, wildfires, and coastlines.',
  },
  {
    icon: MessageSquareText,
    title: 'Ask in plain English',
    body: 'No GIS jargon needed. Type queries like “identify water bodies”, “detect urban areas”, or “what changed between these two images?” — the AI does the rest.',
  },
  {
    icon: Cpu,
    title: 'VLM understands image + query',
    body: 'A vision-language model (GLM-4V) jointly processes the image and your question, classifies the intent, and selects the right analysis: detection, segmentation, change detection, or general understanding.',
  },
  {
    icon: ScanEye,
    title: 'Visual + textual answer',
    body: 'Get a structured answer with highlighted regions overlaid on the image, detected objects with confidence scores, and a coverage breakdown of land-cover classes.',
  },
];

const FEATURES = [
  {
    icon: MessageSquareText,
    title: 'Natural Language Satellite Query',
    body: 'Ask questions in everyday language. The AI converts your query into a remote sensing task — segmentation, detection, change detection, or general image understanding.',
  },
  {
    icon: Upload,
    title: 'Multi-format Image Upload',
    body: 'Supports PNG, JPG, JPEG, TIFF, GeoTIFF, WEBP and BMP. Drop your file or pick from curated satellite samples — no preprocessing required.',
  },
  {
    icon: Layers,
    title: 'Semantic Segmentation',
    body: 'Instead of just detecting objects, SatQuery AI identifies exact regions: water (cyan), urban (red), forest (green), agriculture (yellow), bare soil (amber) — all overlaid on the original image.',
  },
  {
    icon: GitCompareArrows,
    title: 'Change Detection',
    body: 'Upload two satellite images of the same region from different dates. Ask “what changed?” and get additions, removals, and net change as structured output.',
  },
  {
    icon: BarChart3,
    title: 'Coverage Statistics',
    body: 'Every analysis returns quantitative land-cover breakdowns: water 12%, urban 34%, forest 42%, and so on — ready for reports and dashboards.',
  },
  {
    icon: FileText,
    title: 'Explainable Results',
    body: 'Every claim is grounded in visible image features. Confidence scores, region overlays, and intent labels make every answer auditable.',
  },
];

const USE_CASES = [
  {
    icon: CloudRain,
    title: 'Disaster Management',
    body: 'Rapid flood extent mapping, wildfire burn-scar detection, and landslide change detection during emergencies — without GIS specialists on call.',
  },
  {
    icon: Trees,
    title: 'Forest Monitoring',
    body: 'Track deforestation patterns, vegetation health, and afforestation progress across seasons through conversational queries.',
  },
  {
    icon: Building2,
    title: 'Urban Planning',
    body: 'Identify new construction, monitor urban sprawl, and classify land use changes for municipal planning departments.',
  },
  {
    icon: Tractor,
    title: 'Agriculture',
    body: 'Estimate cropland extent, detect irrigation patterns, and monitor crop health indicators from multispectral imagery.',
  },
  {
    icon: Waves,
    title: 'Water Resources',
    body: 'Map reservoirs, rivers, and wetlands. Track water-body shrinkage or expansion between seasons and years.',
  },
  {
    icon: ShieldAlert,
    title: 'Climate & Environment',
    body: 'Quantify glacier retreat, coastal erosion, and snow-cover changes — turning raw imagery into climate-action insights.',
  },
];

const TARGET_USERS = [
  { icon: Microscope, label: 'Researchers' },
  { icon: LandPlot, label: 'Government agencies' },
  { icon: GraduationCap, label: 'Students' },
  { icon: ShieldAlert, label: 'Disaster teams' },
  { icon: Satellite, label: 'General users' },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <SectionHeading
        kicker="Workflow"
        title="From raw pixels to insights in four steps"
        subtitle="SatQuery AI replaces the traditional six-step GIS workflow (download → open GIS → understand bands → process → analyze → interpret) with a single conversational interaction."
      />
      <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((s, i) => (
          <div
            key={s.title}
            className="relative flex flex-col gap-3 rounded-xl border bg-card p-5 transition-shadow hover:shadow-md"
          >
            <div className="absolute right-4 top-4 text-4xl font-bold tabular-nums text-muted/40">
              0{i + 1}
            </div>
            <div className="flex size-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <s.icon className="size-5" />
            </div>
            <div className="space-y-1.5">
              <h3 className="text-sm font-semibold leading-tight">{s.title}</h3>
              <p className="text-xs text-muted-foreground">{s.body}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export function Features() {
  return (
    <section id="features" className="bg-muted/30 border-y">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <SectionHeading
          kicker="Capabilities"
          title="Everything you need to interrogate satellite imagery"
          subtitle="Built around the SIH26167 MVP scope — natural language interface, vision-language AI, segmentation overlays, change detection, and explainable structured outputs."
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="group flex flex-col gap-3 rounded-xl border bg-card p-5 transition-all hover:border-primary/40 hover:shadow-md"
            >
              <div className="flex size-10 items-center justify-center rounded-lg bg-primary/10 text-primary transition-transform group-hover:scale-110">
                <f.icon className="size-5" />
              </div>
              <div className="space-y-1.5">
                <h3 className="text-sm font-semibold leading-tight">{f.title}</h3>
                <p className="text-xs text-muted-foreground">{f.body}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function UseCases() {
  return (
    <section id="use-cases" className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <SectionHeading
        kicker="Applications"
        title="Built for the people who actually use satellite data"
        subtitle="Remote sensing data is valuable across disaster response, agriculture, urban planning, climate monitoring, water management, and infrastructure development — SatQuery AI makes it accessible to all of them."
      />
      <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {USE_CASES.map((u) => (
          <div
            key={u.title}
            className="flex flex-col gap-3 rounded-xl border bg-card p-5"
          >
            <div className="flex size-10 items-center justify-center rounded-lg bg-secondary text-secondary-foreground">
              <u.icon className="size-5" />
            </div>
            <div className="space-y-1.5">
              <h3 className="text-sm font-semibold leading-tight">{u.title}</h3>
              <p className="text-xs text-muted-foreground">{u.body}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-10 rounded-2xl border bg-gradient-to-br from-primary/5 via-card to-card p-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-primary">
              Designed for
            </p>
            <p className="text-lg font-bold">Five primary user personas</p>
            <p className="text-sm text-muted-foreground">
              From ISRO researchers to disaster-response teams — anyone can analyze satellite imagery without GIS expertise.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {TARGET_USERS.map((u) => (
              <span
                key={u.label}
                className="inline-flex items-center gap-1.5 rounded-full border bg-background px-3 py-1.5 text-xs font-medium"
              >
                <u.icon className="size-3.5 text-primary" />
                {u.label}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

const TECH_STACK = [
  {
    category: 'Frontend',
    items: ['Next.js 16', 'React 19', 'TypeScript 5', 'Tailwind CSS 4', 'shadcn/ui', 'Zustand'],
  },
  {
    category: 'Backend',
    items: ['Next.js API Routes', 'Prisma ORM', 'SQLite', 'Node.js runtime'],
  },
  {
    category: 'AI Layer',
    items: ['GLM-4V Vision-Language Model', 'Intent classification', 'Structured JSON output parsing', 'Context-aware conversations'],
  },
  {
    category: 'Visualization',
    items: ['Region overlays', 'Land-cover legends', 'Confidence scoring', 'Change-detection summaries'],
  },
];

export function TechStack() {
  return (
    <section id="tech-stack" className="bg-muted/30 border-y">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <SectionHeading
          kicker="Architecture"
          title="A modern fullstack AI architecture"
          subtitle="Built on the recommended SIH stack — React + Tailwind on the frontend, a Python-free Next.js API layer, and a vision-language model orchestrating detection, segmentation, and change-detection tasks."
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {TECH_STACK.map((g) => (
            <div key={g.category} className="rounded-xl border bg-card p-5">
              <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-primary">
                {g.category}
              </p>
              <ul className="space-y-2">
                {g.items.map((it) => (
                  <li key={it} className="flex items-center gap-2 text-xs">
                    <span className="size-1.5 rounded-full bg-primary" />
                    {it}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="border-t bg-background">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-8 sm:px-6 lg:px-8 md:flex-row md:items-center md:justify-between">
        <div className="space-y-1">
          <p className="text-sm font-semibold">SatQuery AI</p>
          <p className="text-xs text-muted-foreground">
            Built for ISRO · Smart India Hackathon 2026 · Problem Statement SIH26167
          </p>
        </div>
        <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
          <span>Theme: Space Technology</span>
          <span>·</span>
          <span>Category: Software</span>
          <span>·</span>
          <span>Vision-Language Assistant</span>
        </div>
      </div>
    </footer>
  );
}

function SectionHeading({
  kicker,
  title,
  subtitle,
}: {
  kicker: string;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="max-w-3xl space-y-2">
      <p className="text-xs font-semibold uppercase tracking-wider text-primary">
        {kicker}
      </p>
      <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">{title}</h2>
      {subtitle && (
        <p className="text-sm text-muted-foreground sm:text-base">{subtitle}</p>
      )}
    </div>
  );
}
