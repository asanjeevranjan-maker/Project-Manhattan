'use client';

import { useEffect, useRef, useState } from 'react';
import type { AOIBounds, TemporalChangeItem } from '@/lib/types';
import { Button } from '@/components/ui/button';
import { ZoomIn, ZoomOut, Layers, MapPin } from 'lucide-react';

interface Props {
  aoi?: AOIBounds | null;
  changes: TemporalChangeItem[];
  selectedChangeId?: string | null;
  onSelectChange?: (id: string) => void;
  className?: string;
}

const TYPE_COLORS: Record<string, string> = {
  new: '#10b981',       // Emerald
  removed: '#ef4444',   // Red
  modified: '#f59e0b',  // Amber
  unchanged: '#64748b', // Slate
};

export function ChangeMap({
  aoi,
  changes,
  selectedChangeId,
  onSelectChange,
  className,
}: Props) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const markersGroupRef = useRef<any>(null);
  const [filterType, setFilterType] = useState<string>('all');
  const [mapReady, setMapReady] = useState(false);

  // Initialize Map
  useEffect(() => {
    let isMounted = true;

    async function init() {
      if (typeof window === 'undefined' || !mapContainerRef.current) return;
      if (mapInstanceRef.current) return;

      try {
        const L = (await import('leaflet')).default;

        const centerLat = aoi ? (aoi.north + aoi.south) / 2 : 18.95;
        const centerLng = aoi ? (aoi.east + aoi.west) / 2 : 72.85;

        const map = L.map(mapContainerRef.current, {
          center: [centerLat, centerLng],
          zoom: 14,
          zoomControl: false,
          attributionControl: false,
        });

        // Satellite tiles
        L.tileLayer(
          'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          { maxZoom: 18 }
        ).addTo(map);

        // Place labels
        L.tileLayer(
          'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
          { maxZoom: 18, opacity: 0.8 }
        ).addTo(map);

        // AOI Bounding Box
        if (aoi) {
          const bounds = L.latLngBounds([aoi.south, aoi.west], [aoi.north, aoi.east]);
          L.rectangle(bounds, {
            color: '#38bdf8',
            weight: 2,
            fillColor: '#38bdf8',
            fillOpacity: 0.08,
            dashArray: '3, 3',
          }).addTo(map);
          map.fitBounds(bounds, { padding: [25, 25] });
        }

        markersGroupRef.current = L.featureGroup().addTo(map);
        mapInstanceRef.current = map;

        if (isMounted) setMapReady(true);
      } catch (err) {
        console.error('[ChangeMap] Leaflet init failed:', err);
      }
    }

    init();

    return () => {
      isMounted = false;
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update Markers when changes, filterType or selectedChangeId change
  useEffect(() => {
    if (!mapInstanceRef.current || !markersGroupRef.current || !mapReady) return;

    import('leaflet').then(({ default: L }) => {
      markersGroupRef.current.clearLayers();

      const filtered = changes.filter(
        (c) => filterType === 'all' || c.type === filterType
      );

      filtered.forEach((change) => {
        if (typeof change.latitude !== 'number' || typeof change.longitude !== 'number') return;

        const isSelected = change.id === selectedChangeId;
        const color = TYPE_COLORS[change.type] || '#8b5cf6';
        const radius = isSelected ? 10 : 7;

        const marker = L.circleMarker([change.latitude, change.longitude], {
          radius,
          color: isSelected ? '#ffffff' : color,
          weight: isSelected ? 3 : 2,
          fillColor: color,
          fillOpacity: isSelected ? 0.95 : 0.8,
        });

        const popupContent = `
          <div style="font-family: sans-serif; font-size: 12px; line-height: 1.4; min-width: 180px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
              <span style="font-weight: 700; text-transform: uppercase; color: ${color}; font-size: 10px; background: ${color}20; padding: 2px 6px; border-radius: 4px;">
                ${change.type}
              </span>
              <span style="font-weight: 600; color: #0f172a;">${Math.round(change.confidence * 100)}%</span>
            </div>
            <div style="font-weight: 600; font-size: 13px; text-transform: capitalize; margin-bottom: 2px;">
              ${change.label}
            </div>
            <div style="color: #64748b; font-size: 11px; margin-bottom: 4px;">
              ${change.details}
            </div>
            <div style="font-size: 10px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 4px;">
              Lat: ${change.latitude.toFixed(5)}, Lon: ${change.longitude.toFixed(5)}
            </div>
          </div>
        `;

        marker.bindPopup(popupContent);

        marker.on('click', () => {
          onSelectChange?.(change.id);
        });

        if (isSelected) {
          marker.openPopup();
          mapInstanceRef.current.panTo([change.latitude, change.longitude]);
        }

        marker.addTo(markersGroupRef.current);
      });
    });
  }, [changes, filterType, selectedChangeId, mapReady]);

  return (
    <div className={`relative flex flex-col overflow-hidden rounded-xl border bg-card ${className || 'h-[440px]'}`}>
      <div ref={mapContainerRef} className="h-full w-full bg-muted/40" />

      {/* Top Filter Chips */}
      <div className="absolute left-3 top-3 z-[400] flex flex-wrap items-center gap-1.5 rounded-lg border bg-background/90 p-1.5 shadow-md backdrop-blur">
        <span className="flex items-center gap-1 px-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          <Layers className="size-3 text-primary" /> Filter:
        </span>
        {['all', 'new', 'removed', 'modified'].map((t) => {
          const count = t === 'all' ? changes.length : changes.filter((c) => c.type === t).length;
          return (
            <button
              key={t}
              type="button"
              onClick={() => setFilterType(t)}
              className={`flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-semibold capitalize transition ${
                filterType === t
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'bg-muted/60 text-muted-foreground hover:bg-muted'
              }`}
            >
              {t}
              <span className="text-[10px] opacity-80">({count})</span>
            </button>
          );
        })}
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

      {/* Legend */}
      <div className="pointer-events-none absolute bottom-3 left-3 z-[400] flex items-center gap-2 rounded-md border bg-background/90 px-2 py-1 text-[10px] font-medium shadow backdrop-blur">
        <span className="flex items-center gap-1">
          <span className="size-2 rounded-full bg-emerald-500" /> New
        </span>
        <span className="flex items-center gap-1">
          <span className="size-2 rounded-full bg-red-500" /> Removed
        </span>
        <span className="flex items-center gap-1">
          <span className="size-2 rounded-full bg-amber-500" /> Modified
        </span>
      </div>
    </div>
  );
}

