'use client';

import { useCallback, useRef, useState } from 'react';
import { Upload, ImagePlus, Loader2, X, FileWarning } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { fileToDataUrl, formatBytes, shortId } from '@/lib/client-utils';
import type { UploadedImage } from '@/lib/types';
import { useSatQueryStore } from '@/store/satquery';

const MAX_FILE_BYTES = 12 * 1024 * 1024;
const ACCEPTED = 'image/png,image/jpeg,image/jpg,image/tiff,image/webp,image/bmp';

interface Props {
  variant?: 'dropzone' | 'compact';
  className?: string;
}

export function ImageUploader({ variant = 'dropzone', className }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();
  const setActiveImage = useSatQueryStore((s) => s.setActiveImage);

  const handleFile = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      setError(null);

      if (!file.type.startsWith('image/')) {
        setError('Please select an image file.');
        toast({
          variant: 'destructive',
          title: 'Unsupported file',
          description: 'Only image files (PNG, JPG, JPEG, TIFF, WEBP, BMP) are supported.',
        });
        return;
      }

      if (file.size > MAX_FILE_BYTES) {
        const msg = `File is ${formatBytes(file.size)} — max allowed is ${formatBytes(MAX_FILE_BYTES)}.`;
        setError(msg);
        toast({
          variant: 'destructive',
          title: 'File too large',
          description: msg,
        });
        return;
      }

      setIsLoading(true);
      try {
        const dataUrl = await fileToDataUrl(file);
        const img: UploadedImage = {
          id: shortId('img-'),
          filename: file.name,
          mimeType: file.type,
          size: file.size,
          dataUrl,
        };
        setActiveImage(img);
        toast({
          title: 'Image loaded',
          description: `${file.name} · ${formatBytes(file.size)}`,
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Failed to read file';
        setError(msg);
        toast({
          variant: 'destructive',
          title: 'Upload failed',
          description: msg,
        });
      } finally {
        setIsLoading(false);
      }
    },
    [setActiveImage, toast]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      handleFile(file);
    },
    [handleFile]
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  if (variant === 'compact') {
    return (
      <>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <Button
          variant="outline"
          size="sm"
          disabled={isLoading}
          onClick={() => inputRef.current?.click()}
          className="gap-2"
        >
          {isLoading ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <ImagePlus className="size-4" />
          )}
          Upload Image
        </Button>
      </>
    );
  }

  return (
    <div className={cn('w-full', className)}>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        disabled={isLoading}
        aria-label="Upload satellite image"
        className={cn(
          'group relative flex w-full flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed bg-card/50 px-6 py-12 text-center transition-all',
          'hover:border-primary/60 hover:bg-accent/40',
          isDragging
            ? 'border-primary bg-primary/5'
            : 'border-border',
          isLoading && 'pointer-events-none opacity-70'
        )}
      >
        <div className="flex size-14 items-center justify-center rounded-full bg-primary/10 text-primary transition-transform group-hover:scale-110">
          {isLoading ? (
            <Loader2 className="size-6 animate-spin" />
          ) : (
            <Upload className="size-6" />
          )}
        </div>
        <div className="space-y-1">
          <p className="text-base font-semibold">
            {isLoading ? 'Loading image…' : 'Drop a satellite image here'}
          </p>
          <p className="text-xs text-muted-foreground">
            or <span className="text-primary font-medium">click to browse</span> · PNG, JPG, TIFF, WEBP up to {formatBytes(MAX_FILE_BYTES)}
          </p>
        </div>
        {error && (
          <div className="mt-2 flex items-center gap-2 rounded-md bg-destructive/10 px-3 py-1.5 text-xs text-destructive">
            <FileWarning className="size-3.5" />
            <span>{error}</span>
            <button
              type="button"
              className="ml-1 hover:opacity-70"
              onClick={(e) => {
                e.stopPropagation();
                setError(null);
              }}
            >
              <X className="size-3" />
            </button>
          </div>
        )}
      </button>
    </div>
  );
}
