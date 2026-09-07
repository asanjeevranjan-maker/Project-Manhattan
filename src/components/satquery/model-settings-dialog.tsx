'use client';

import { useState } from 'react';

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';

import { Switch } from '@/components/ui/switch';

import { Input } from '@/components/ui/input';

import { Label } from '@/components/ui/label';

import { Button } from '@/components/ui/button';

import {
  Settings2,
  CheckCircle2,
  AlertTriangle,
  ExternalLink,
} from 'lucide-react';

import { useSatQueryStore } from '@/store/satquery';

/**
 * Model Settings dialog.
 *
 * Configures which VLM to use and toggles Grounding DINO
 * (open-vocabulary object detection). Settings are persisted in
 * the browser's localStorage via the Zustand persist middleware —
 * the HF token is sent per-request to /api/analyze and is never
 * stored server-side.
 */
export function ModelSettingsDialog() {
  const modelSettings = useSatQueryStore((s) => s.modelSettings);
  const setModelSettings = useSatQueryStore((s) => s.setModelSettings);

  const [open, setOpen] = useState(false);

  const tokenConfigured = modelSettings.hfToken.trim().length > 0;

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-1.5">
          <Settings2 className="size-4" />
          <span className="hidden sm:inline">Model Settings</span>
        </Button>
      </DialogTrigger>

      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Model Settings</DialogTitle>
          <DialogDescription>
            Configure the vision model and optional Grounding DINO
            detection. Settings are stored locally in your browser.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 pt-2">
          {/* Grounding DINO toggle */}
          <div className="space-y-3 rounded-lg border p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="space-y-0.5">
                <Label htmlFor="gdino-toggle" className="cursor-pointer">
                  Grounding DINO (open-vocabulary detection)
                </Label>
                <p className="text-xs text-muted-foreground">
                  Adds precise bounding boxes alongside the VLM analysis.
                  Uses the Hugging Face free tier (~1000 requests/day).
                </p>
              </div>
              <Switch
                id="gdino-toggle"
                checked={modelSettings.useGroundingDino}
                onCheckedChange={(checked) =>
                  setModelSettings({ useGroundingDino: checked })
                }
              />
            </div>

            {modelSettings.useGroundingDino && (
              <div className="space-y-2 border-t pt-3">
                <Label htmlFor="hf-token">Hugging Face Access Token</Label>

                <Input
                  id="hf-token"
                  type="password"
                  placeholder="hf_…"
                  value={modelSettings.hfToken}
                  onChange={(e) =>
                    setModelSettings({ hfToken: e.target.value })
                  }
                  autoComplete="off"
                />

                {/* Token status */}
                {tokenConfigured ? (
                  <p className="flex items-center gap-1.5 text-xs font-medium text-emerald-600 dark:text-emerald-400">
                    <CheckCircle2 className="size-3.5" />
                    Token configured — Grounding DINO is ready.
                  </p>
                ) : (
                  <p className="flex items-start gap-1.5 text-xs font-medium text-amber-600 dark:text-amber-400">
                    <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                    No token set — Grounding DINO will be skipped during
                    analysis.
                  </p>
                )}

                <a
                  href="https://huggingface.co/settings/tokens"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-primary underline-offset-4 hover:underline"
                >
                  Get a token at huggingface.co/settings/tokens
                  <ExternalLink className="size-3" />
                </a>
                <p className="text-[11px] text-muted-foreground">
                  A token with the <span className="font-medium">Read</span>{' '}
                  scope is sufficient.
                </p>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
