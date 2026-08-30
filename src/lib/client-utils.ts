// Client-side utilities for SatQuery AI

/**
 * Read a File as a base64 data URL.
 */
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result === 'string') resolve(result);
      else reject(new Error('Unexpected FileReader result type'));
    };
    reader.onerror = () => reject(reader.error ?? new Error('FileReader failed'));
    reader.readAsDataURL(file);
  });
}

/**
 * Fetch a remote image URL and convert it to a base64 data URL.
 * Used so the VLM can analyze externally-hosted sample images.
 *
 * Goes through the /api/proxy-image endpoint to avoid CORS issues.
 */
export async function remoteImageToDataUrl(url: string): Promise<string> {
  const proxyUrl = `/api/proxy-image?url=${encodeURIComponent(url)}`;
  const res = await fetch(proxyUrl);
  if (!res.ok) {
    throw new Error(`Failed to fetch remote image: ${res.status} ${res.statusText}`);
  }
  const blob = await res.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const r = reader.result;
      if (typeof r === 'string') resolve(r);
      else reject(new Error('Unexpected reader result'));
    };
    reader.onerror = () => reject(reader.error ?? new Error('Reader failed'));
    reader.readAsDataURL(blob);
  });
}

/**
 * Fetch a same-origin image path (e.g. /samples/foo.jpg) and convert
 * it to a base64 data URL. Used for locally-hosted sample images.
 */
export async function localImageToDataUrl(path: string): Promise<string> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to fetch local image: ${res.status}`);
  }
  const blob = await res.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const r = reader.result;
      if (typeof r === 'string') resolve(r);
      else reject(new Error('Unexpected reader result'));
    };
    reader.onerror = () => reject(reader.error ?? new Error('Reader failed'));
    reader.readAsDataURL(blob);
  });
}

/**
 * Format a byte count into a human-readable string.
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(i === 0 ? 0 : 1)} ${sizes[i]}`;
}

/**
 * Generate a short unique id (client-side only).
 */
export function shortId(prefix = ''): string {
  return `${prefix}${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}
