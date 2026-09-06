'use client';

import { useEffect, useRef, useState } from 'react';
import type { AOIBounds } from '@/lib/types';
import { Button } from '@/components/ui/button';
import { MapPin, ZoomIn, ZoomOut, Compass, Minus, Plus, Maximize2 } from 'lucide-react';

interface Props {
  aoi: AOIBounds;
  onChangeAoi: (newAoi: AOIBounds) => void;
  onSelectPreset?: (preset: typeof PRESET_LOCATIONS[0]) => void;
  className?: string;
}

export const PRESET_LOCATIONS: Array<{
  name: string;
  category: string;
  aoi: AOIBounds;
  defaultPrompt: string;
  defaultHistoricalDate: string;
}> = [
  {
    name: 'Mumbai JNPT Port, India',
    category: 'Container Ships & Port',
    aoi: { north: 18.960, south: 18.925, east: 72.965, west: 72.930 },
    defaultPrompt: 'ship . cargo ship . container vessel . boat .',
    defaultHistoricalDate: '2024-01-15',
  },
  {
    name: 'Mumbai Urban & Bandra',
    category: 'Urban Infrastructure',
    aoi: { north: 19.075, south: 19.035, east: 72.865, west: 72.825 },
    defaultPrompt: 'building . road . bridge . structure .',
    defaultHistoricalDate: '2023-05-15',
  },
  {
    name: 'Dubai Palm & Coastline',
    category: 'Urban & Coastal',
    aoi: { north: 25.135, south: 25.095, east: 55.155, west: 55.105 },
    defaultPrompt: 'building . structure . rooftop . villa .',
    defaultHistoricalDate: '2023-01-01',
  },
  {
    name: 'Suez Canal Convoy, Egypt',
    category: 'Shipping Corridor',
    aoi: { north: 30.650, south: 30.610, east: 32.365, west: 32.325 },
    defaultPrompt: 'ship . vessel . container ship . tanker .',
    defaultHistoricalDate: '2024-03-01',
  },
  {
    name: 'Valencia Riverbed, Spain',
    category: 'Flood & Infrastructure',
    aoi: { north: 39.495, south: 39.445, east: -0.340, west: -0.400 },
    defaultPrompt: 'bridge . road . building .',
    defaultHistoricalDate: '2024-08-01',
  },
];

/**
 * Calculates new AOIBounds for a given center coordinate and span in kilometers,
 * correctly scaling longitude by latitude cosine.
 */
export function computeAoiFromSpan(centerLat: number, centerLng: number, spanKm: number): AOIBounds {
  const km = Math.max(0.5, Math.min(60, spanKm));
  const latSpanDeg = km / 111;
  const cosLat = Math.cos((centerLat * Math.PI) / 180);
  const lonScale = Math.abs(cosLat) > 0.05 ? cosLat : 1.0;
  const lonSpanDeg = km / (111 * lonScale);

  return {
    north: parseFloat((centerLat + latSpanDeg / 2).toFixed(4)),
    south: parseFloat((centerLat - latSpanDeg / 2).toFixed(4)),
    east: parseFloat((centerLng + lonSpanDeg / 2).toFixed(4)),
    west: parseFloat((centerLng - lonSpanDeg / 2).toFixed(4)),
  };
}

export function AOIMap({ aoi, onChangeAoi, onSelectPreset, className }: Props) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const rectangleLayerRef = useRef<any>(null);
  const [mapReady, setMapReady] = useState(false);

  // Keep a ref to the latest aoi to prevent stale closures in leaflet event handlers
  const aoiRef = useRef(aoi);
  aoiRef.current = aoi;

  // Initialize Leaflet map
  useEffect(() => {
    let isMounted = true;

    async function initMap() {
      if (typeof window === 'undefined' || !mapContainerRef.current) return;
      if (mapInstanceRef.current) return;

      try {
        const L = (await import('leaflet')).default;

        const currentAoi = aoiRef.current;
        const centerLat = (currentAoi.north + currentAoi.south) / 2;
        const centerLng = (currentAoi.east + currentAoi.west) / 2;

        const map = L.map(mapContainerRef.current, {
          center: [centerLat, centerLng],
          zoom: 13,
          zoomControl: false,
          attributionControl: false,
        });

        // Satellite Basemap (Esri World Imagery)
        L.tileLayer(
          'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          { maxZoom: 18 }
        ).addTo(map);

        // Labels overlay
        L.tileLayer(
          'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
          { maxZoom: 18, opacity: 0.8 }
        ).addTo(map);

        // AOI Bounding Box rectangle
        const bounds = L.latLngBounds(
          [currentAoi.south, currentAoi.west],
          [currentAoi.north, currentAoi.east]
        );

        const rect = L.rectangle(bounds, {
          color: '#10b981',
          weight: 2,
          fillColor: '#10b981',
          fillOpacity: 0.18,
          dashArray: '4, 4',
        }).addTo(map);

        rectangleLayerRef.current = rect;
        mapInstanceRef.current = map;

        // Click to center AOI at clicked point while maintaining current span
        map.on('click', (e: any) => {
          const lat = e.latlng.lat;
          const lng = e.latlng.lng;
          const latSpan = Math.abs(aoiRef.current.north - aoiRef.current.south) / 2;
          const lngSpan = Math.abs(aoiRef.current.east - aoiRef.current.west) / 2;

          onChangeAoi({
            north: parseFloat((lat + latSpan).toFixed(4)),
            south: parseFloat((lat - latSpan).toFixed(4)),
            east: parseFloat((lng + lngSpan).toFixed(4)),
            west: parseFloat((lng - lngSpan).toFixed(4)),
          });
        });

        if (isMounted) setMapReady(true);
      } catch (err) {
        console.error('[AOIMap] Failed to load Leaflet:', err);
      }
    }

    initMap();

    return () => {
      isMounted = false;
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update AOI rectangle on prop changes
  useEffect(() => {
    if (!mapInstanceRef.current || !mapReady) return;

    import('leaflet').then(({ default: L }) => {
      const bounds = L.latLngBounds(
        [aoi.south, aoi.west],
        [aoi.north, aoi.east]
      );

      if (rectangleLayerRef.current) {
        rectangleLayerRef.current.setBounds(bounds);
      } else {
        rectangleLayerRef.current = L.rectangle(bounds, {
          color: '#10b981',
          weight: 2,
          fillColor: '#10b981',
          fillOpacity: 0.18,
          dashArray: '4, 4',
        }).addTo(mapInstanceRef.current);
      }

      mapInstanceRef.current.flyToBounds(bounds, { padding: [30, 30], duration: 0.8 });
    });
  }, [aoi.north, aoi.south, aoi.east, aoi.west, mapReady]);

  const latSpan = Math.abs(aoi.north - aoi.south);
  const lonSpan = Math.abs(aoi.east - aoi.west);
  const centerLat = (aoi.north + aoi.south) / 2;
  const centerLng = (aoi.east + aoi.west) / 2;
  const cosLat = Math.cos((centerLat * Math.PI) / 180);
  const approxKm = Math.max(0.5, Math.round(Math.max(latSpan * 111, lonSpan * 111 * (Math.abs(cosLat) > 0.05 ? cosLat : 1.0))));

  const handleAdjustSpan = (targetKm: number) => {
    const newAoi = computeAoiFromSpan(centerLat, centerLng, targetKm);
    onChangeAoi(newAoi);
  };

  return (
    <div className={`relative flex flex-col overflow-hidden rounded-xl border bg-card ${className || 'h-[340px]'}`}>
      {/* Map container */}
      <div ref={mapContainerRef} className="h-full w-full bg-muted/40" />

      {/* Floating interactive overlay badge with variable span controls */}
      <div className="absolute left-3 top-3 z-[400] flex flex-col gap-1.5 rounded-lg border bg-background/95 p-2 text-xs shadow-md backdrop-blur">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="size-2 rounded-full bg-emerald-500 glow-pulse" />
            <span className="font-semibold text-foreground">Area of Interest</span>
          </div>
          <div className="flex items-center gap-1 rounded bg-muted/80 px-1.5 py-0.5 font-mono text-[11px] font-bold text-primary">
            ~{approxKm} km span
          </div>
        </div>

        {/* Quick Stepper & Chips for Variable Span */}
        <div className="flex items-center gap-1 border-t border-border/50 pt-1.5">
          <span className="text-[10px] text-muted-foreground mr-0.5">Size:</span>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-5 rounded hover:bg-muted"
            onClick={() => handleAdjustSpan(Math.max(1, approxKm - 1))}
            title="Decrease span by 1 km"
          >
            <Minus className="size-2.5" />
          </Button>

          {[1, 2, 4, 8, 15, 25].map((km) => (
            <button
              key={km}
              type="button"
              onClick={() => handleAdjustSpan(km)}
              className={`rounded px-1.5 py-0.5 text-[10px] font-medium transition-all ${
                Math.abs(approxKm - km) <= (km <= 2 ? 0.4 : 1)
                  ? 'bg-primary text-primary-foreground font-semibold shadow-xs'
                  : 'bg-muted/60 text-muted-foreground hover:bg-muted hover:text-foreground'
              }`}
            >
              {km}k
            </button>
          ))}

          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-5 rounded hover:bg-muted"
            onClick={() => handleAdjustSpan(Math.min(50, approxKm + 1))}
            title="Increase span by 1 km"
          >
            <Plus className="size-2.5" />
          </Button>
        </div>
      </div>

      {/* Zoom controls */}
      <div className="absolute right-3 top-3 z-[400] flex flex-col gap-1">
        <Button
          size="icon"
          variant="secondary"
          className="size-7 rounded-md bg-background/90 shadow backdrop-blur"
          onClick={() => mapInstanceRef.current?.zoomIn()}
          aria-label="Zoom in"
        >
          <ZoomIn className="size-3.5" />
        </Button>
        <Button
          size="icon"
          variant="secondary"
          className="size-7 rounded-md bg-background/90 shadow backdrop-blur"
          onClick={() => mapInstanceRef.current?.zoomOut()}
          aria-label="Zoom out"
        >
          <ZoomOut className="size-3.5" />
        </Button>
      </div>

      {/* Preset Quick Select Chips */}
      <div className="absolute bottom-2 inset-x-2 z-[400] flex items-center gap-1.5 overflow-x-auto rounded-lg border bg-background/90 p-1.5 shadow backdrop-blur">
        <span className="flex items-center gap-1 pl-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground shrink-0">
          <Compass className="size-3 text-primary" /> Presets:
        </span>
        {PRESET_LOCATIONS.map((preset) => (
          <button
            key={preset.name}
            type="button"
            onClick={() => {
              onChangeAoi(preset.aoi);
              onSelectPreset?.(preset);
            }}
            className="shrink-0 rounded-md border bg-card/70 px-2 py-0.5 text-[11px] font-medium transition hover:border-primary/60 hover:bg-primary/10"
          >
            {preset.name.split(',')[0]}
          </button>
        ))}
      </div>
    </div>
  );
}
