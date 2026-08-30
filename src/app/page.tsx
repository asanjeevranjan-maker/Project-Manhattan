'use client';

import { useState, useEffect, useCallback } from 'react';
import { Header } from '@/components/satquery/header';
import { Hero } from '@/components/satquery/hero';
import { HowItWorks, Features, UseCases, TechStack, Footer } from '@/components/satquery/sections';
import { Workspace } from '@/components/satquery/workspace';
import { useSatQueryStore } from '@/store/satquery';

type View = 'landing' | 'workspace';

function readInitialView(): View {
  if (typeof window === 'undefined') return 'landing';
  return window.location.hash === '#workspace' ? 'workspace' : 'landing';
}

export default function Home() {
  // Read the URL hash exactly once on first render — no setState-in-effect needed.
  const [view, setView] = useState<View>(readInitialView);
  const activeImage = useSatQueryStore((s) => s.activeImage);

  // Listen for hash changes (browser back/forward) — only fire setState in response
  // to a genuine external event, never synchronously inside the effect body.
  useEffect(() => {
    const onHash = () => {
      setView(window.location.hash === '#workspace' ? 'workspace' : 'landing');
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const openWorkspace = useCallback(() => {
    setView('workspace');
    window.location.hash = 'workspace';
    window.scrollTo({ top: 0, behavior: 'auto' });
  }, []);

  const exitToLanding = useCallback(() => {
    setView('landing');
    // Clear hash without leaving a # in the URL
    history.replaceState(null, '', window.location.pathname + window.location.search);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  if (view === 'workspace') {
    return <Workspace onExit={exitToLanding} />;
  }

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Header onLaunch={openWorkspace} hasImage={Boolean(activeImage)} />
      <main className="flex-1">
        <Hero onLaunch={openWorkspace} />
        <HowItWorks />
        <Features />
        <UseCases />
        <TechStack />
      </main>
      <Footer />
    </div>
  );
}
