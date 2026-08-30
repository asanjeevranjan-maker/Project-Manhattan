// SatQuery AI logo mark — a stylized satellite orbiting a planet.
// Inline SVG so it inherits currentColor and scales crisply.

import { cn } from '@/lib/utils';

export function SatQueryLogo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('size-8', className)}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="sq-grad" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="oklch(0.75 0.15 165)" />
          <stop offset="100%" stopColor="oklch(0.7 0.14 195)" />
        </linearGradient>
      </defs>
      {/* Planet */}
      <circle cx="24" cy="24" r="12" stroke="url(#sq-grad)" strokeWidth="2.5" fill="none" />
      {/* Orbit ring */}
      <ellipse
        cx="24"
        cy="24"
        rx="20"
        ry="9"
        stroke="url(#sq-grad)"
        strokeWidth="1.5"
        fill="none"
        transform="rotate(-30 24 24)"
        opacity="0.6"
      />
      {/* Satellite body */}
      <rect x="34" y="9" width="6" height="6" rx="1" fill="url(#sq-grad)" />
      {/* Solar panels */}
      <rect x="28" y="11" width="4" height="2" fill="url(#sq-grad)" opacity="0.8" />
      <rect x="42" y="11" width="4" height="2" fill="url(#sq-grad)" opacity="0.8" />
      {/* Signal beam */}
      <path
        d="M36 17 L33 22 M38 17 L40 22"
        stroke="url(#sq-grad)"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function SatQueryWordmark({ className }: { className?: string }) {
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <SatQueryLogo className="size-7" />
      <div className="flex flex-col leading-none">
        <span className="font-bold tracking-tight text-base">SatQuery AI</span>
        <span className="text-[10px] text-muted-foreground font-medium tracking-wide">
          ISRO · SIH26167
        </span>
      </div>
    </div>
  );
}
