'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import type { TemporalChangeItem } from '@/lib/types';
import { Button } from '@/components/ui/button';
import { Eye, EyeOff, ChevronsLeftRight } from 'lucide-react';

interface Props {
  t1Url: string;
  t2Url: string;
  t1Label?: string;
  t2Label?: string;
  changes?: TemporalChangeItem[];
  className?: string;
}

export function ImageSlider({
  t1Url,
  t2Url,
  t1Label = 'Historical',
  t2Label = 'Latest Available',
  changes = [],
  className,
}: Props) {
  const [sliderPosition, setSliderPosition] = useState(50); // percentage 0 - 100
  const [isDragging, setIsDragging] = useState(false);
  const [showBoxes, setShowBoxes] = useState(true);
  const [containerWidth, setContainerWidth] = useState(560);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const update = () => {
      if (el) setContainerWidth(el.clientWidth);
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const handleMove = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setSliderPosition(pct);
    },
    []
  );

  const onMouseDown = () => setIsDragging(true);

  useEffect(() => {
    const onMouseUp = () => setIsDragging(false);
    const onMouseMove = (e: MouseEvent) => {
      if (isDragging) handleMove(e.clientX);
    };
    const onTouchMove = (e: TouchEvent) => {
      if (isDragging && e.touches[0]) handleMove(e.touches[0].clientX);
    };

    if (isDragging) {
      window.addEventListener('mouseup', onMouseUp);
      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('touchend', onMouseUp);
      window.addEventListener('touchmove', onTouchMove);
    }
    return () => {
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('touchend', onMouseUp);
      window.removeEventListener('touchmove', onTouchMove);
    };
  }, [isDragging, handleMove]);

  return (
    <div className={`relative flex flex-col gap-2 ${className || ''}`}>
      {/* Controls toolbar */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded bg-secondary/70 px-2 py-0.5 font-semibold">
            {t1Label}
          </span>
          <span className="text-muted-foreground">↔</span>
          <span className="inline-flex items-center gap-1 rounded bg-primary/20 text-primary px-2 py-0.5 font-semibold">
            {t2Label}
          </span>
        </div>
        {changes.length > 0 && (
          <Button
            size="sm"
            variant="ghost"
            className="h-7 gap-1 text-xs"
            onClick={() => setShowBoxes(!showBoxes)}
          >
            {showBoxes ? (
              <>
                <EyeOff className="size-3" /> Hide Bounding Boxes
              </>
            ) : (
              <>
                <Eye className="size-3" /> Show Bounding Boxes
              </>
            )}
          </Button>
        )}
      </div>

      {/* Slider Viewport */}
      <div
        ref={containerRef}
        className="relative mx-auto aspect-square w-full max-w-[560px] select-none overflow-hidden rounded-xl border bg-black/80 cursor-col-resize shadow-md"
        onMouseDown={onMouseDown}
        onTouchStart={onMouseDown}
        onClick={(e) => handleMove(e.clientX)}
      >
        {/* Layer 2 (Latest / Right / Base) */}
        <img
          src={t2Url}
          alt={t2Label}
          className="absolute inset-0 size-full object-cover pointer-events-none"
        />

        {/* Bounding boxes on Time 2 (Latest) */}
        {showBoxes && (
          <div className="absolute inset-0 pointer-events-none">
            {changes
              .filter((c) => c.boxT2 && (c.type === 'new' || c.type === 'modified' || c.type === 'unchanged'))
              .map((c) => {
                const [x1, y1, x2, y2] = c.boxT2!;
                const color = c.type === 'new' ? '#10b981' : c.type === 'modified' ? '#f59e0b' : '#64748b';
                return (
                  <div
                    key={`t2-${c.id}`}
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
                      className="absolute -top-5 left-0 rounded px-1 text-[9px] font-bold text-white shadow"
                      style={{ backgroundColor: color }}
                    >
                      {c.type === 'new' ? 'NEW' : c.type === 'modified' ? 'MOD' : 'OK'} · {c.label}
                    </span>
                  </div>
                );
              })}
          </div>
        )}

        {/* Layer 1 (Historical / Left / Clipped) */}
        <div
          className="absolute inset-y-0 left-0 overflow-hidden border-r-2 border-white shadow-2xl pointer-events-none"
          style={{ width: `${sliderPosition}%` }}
        >
          {/* Inner container sized to match full viewport width so image doesn't squish */}
          <div
            className="relative h-full"
            style={{
              width: `${containerWidth}px`,
            }}
          >
            <img
              src={t1Url}
              alt={t1Label}
              className="absolute inset-0 size-full object-cover pointer-events-none"
            />

            {/* Bounding boxes on Time 1 (Historical) */}
            {showBoxes && (
              <div className="absolute inset-0 pointer-events-none">
                {changes
                  .filter((c) => c.boxT1 && (c.type === 'removed' || c.type === 'unchanged'))
                  .map((c) => {
                    const [x1, y1, x2, y2] = c.boxT1!;
                    const color = c.type === 'removed' ? '#ef4444' : '#64748b';
                    return (
                      <div
                        key={`t1-${c.id}`}
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
                          className="absolute -top-5 left-0 rounded px-1 text-[9px] font-bold text-white shadow"
                          style={{ backgroundColor: color }}
                        >
                          {c.type === 'removed' ? 'REMOVED' : 'OK'} · {c.label}
                        </span>
                      </div>
                    );
                  })}
              </div>
            )}
          </div>
        </div>

        {/* Interactive Drag Handle Divider */}
        <div
          className="absolute inset-y-0 z-20 flex -translate-x-1/2 items-center pointer-events-none"
          style={{ left: `${sliderPosition}%` }}
        >
          <div className="flex size-8 items-center justify-center rounded-full border-2 border-white bg-primary text-primary-foreground shadow-xl">
            <ChevronsLeftRight className="size-4" />
          </div>
        </div>

        {/* Watermark badges */}
        <div className="pointer-events-none absolute bottom-3 left-3 z-10 rounded bg-black/70 px-2 py-0.5 text-[10px] font-semibold text-white backdrop-blur">
          {t1Label}
        </div>
        <div className="pointer-events-none absolute bottom-3 right-3 z-10 rounded bg-black/70 px-2 py-0.5 text-[10px] font-semibold text-white backdrop-blur">
          {t2Label}
        </div>
      </div>
    </div>
  );
}

