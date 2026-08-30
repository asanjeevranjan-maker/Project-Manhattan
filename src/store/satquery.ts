'use client';

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { ChatMessage, UploadedImage, AnalysisResult } from '@/lib/types';

interface SatQueryState {
  // Currently active image (the one being analyzed)
  activeImage: UploadedImage | null;
  // Chat history for the current image
  messages: ChatMessage[];
  // Latest analysis result (also referenced by the latest assistant message)
  latestAnalysis: AnalysisResult | null;
  // Whether the AI is currently processing a query
  isAnalyzing: boolean;
  // Whether to show overlay on the image
  showOverlay: boolean;
  // View mode for the image viewer
  viewMode: 'original' | 'overlay';

  // Actions
  setActiveImage: (img: UploadedImage | null) => void;
  addMessage: (msg: ChatMessage) => void;
  updateMessage: (id: string, updates: Partial<ChatMessage>) => void;
  setLatestAnalysis: (analysis: AnalysisResult | null) => void;
  setIsAnalyzing: (v: boolean) => void;
  setShowOverlay: (v: boolean) => void;
  setViewMode: (m: 'original' | 'overlay') => void;
  clearChat: () => void;
  reset: () => void;
}

export const useSatQueryStore = create<SatQueryState>()(
  persist(
    (set) => ({
      activeImage: null,
      messages: [],
      latestAnalysis: null,
      isAnalyzing: false,
      showOverlay: true,
      viewMode: 'original',

      setActiveImage: (img) =>
        set((s) => ({
          activeImage: img,
          // Reset chat & analysis when image changes — each image gets a fresh conversation
          messages: img && s.activeImage?.id === img.id ? s.messages : [],
          latestAnalysis: img && s.activeImage?.id === img.id ? s.latestAnalysis : null,
        })),

      addMessage: (msg) =>
        set((s) => ({ messages: [...s.messages, msg] })),

      updateMessage: (id, updates) =>
        set((s) => ({
          messages: s.messages.map((m) => (m.id === id ? { ...m, ...updates } : m)),
        })),

      setLatestAnalysis: (analysis) => set({ latestAnalysis: analysis }),
      setIsAnalyzing: (v) => set({ isAnalyzing: v }),
      setShowOverlay: (v) => set({ showOverlay: v }),
      setViewMode: (m) => set({ viewMode: m }),
      clearChat: () => set({ messages: [], latestAnalysis: null }),
      reset: () =>
        set({
          activeImage: null,
          messages: [],
          latestAnalysis: null,
          isAnalyzing: false,
          showOverlay: true,
          viewMode: 'original',
        }),
    }),
    {
      name: 'satquery-store',
      storage: createJSONStorage(() => localStorage),
      // Persist only the active image (without the heavy data URL is too large to fit; keep it
      // transient by selectively persisting nothing large)
      partialize: (state) => ({
        showOverlay: state.showOverlay,
        viewMode: state.viewMode,
      }),
    }
  )
);
