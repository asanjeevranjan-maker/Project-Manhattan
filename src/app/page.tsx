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
  return window.location.hash === '#workspace' || window.location.hash === '#bitemporal' || window.location.hash === '#fusion'
    ? 'workspace'
    : 'landing';
}

function readInitialTab(): 'single' | 'bitemporal' | 'fusion' {
  if (typeof window === 'undefined') return 'single';
  if (window.location.hash === '#bitemporal') return 'bitemporal';
  if (window.location.hash === '#fusion') return 'fusion';
  return 'single';
}

export default function Home() {
  const [view, setView] = useState<View>(readInitialView);
  const [activeTab, setActiveTab] = useState<'single' | 'bitemporal' | 'fusion'>(readInitialTab);
  const activeImage = useSatQueryStore((s) => s.activeImage);

  useEffect(() => {
    const onHash = () => {
      const h = window.location.hash;
      if (h === '#bitemporal') {
        setView('workspace');
        setActiveTab('bitemporal');
      } else if (h === '#fusion') {
        setView('workspace');
        setActiveTab('fusion');
      } else if (h === '#workspace') {
        setView('workspace');
        setActiveTab('single');
      } else {
        setView('landing');
      }
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const openWorkspace = useCallback((tab: 'single' | 'bitemporal' | 'fusion' = 'single') => {
    setActiveTab(tab);
    setView('workspace');
    window.location.hash = tab === 'bitemporal' ? 'bitemporal' : tab === 'fusion' ? 'fusion' : 'workspace';
    window.scrollTo({ top: 0, behavior: 'auto' });
  }, []);

  const exitToLanding = useCallback(() => {
    setView('landing');
    history.replaceState(null, '', window.location.pathname + window.location.search);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  if (view === 'workspace') {
    return <Workspace onExit={exitToLanding} initialTab={activeTab} />;
  }

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Header
        onLaunch={() => openWorkspace('single')}
        onLaunchBiTemporal={() => openWorkspace('bitemporal')}
        onLaunchFusion={() => openWorkspace('fusion')}
        hasImage={Boolean(activeImage)}
      />
      <main className="flex-1">
        <Hero
          onLaunch={() => openWorkspace('single')}
          onLaunchBiTemporal={() => openWorkspace('bitemporal')}
          onLaunchFusion={() => openWorkspace('fusion')}
        />
        <HowItWorks />
        <Features />
        <UseCases />
        <TechStack />
      </main>
      <Footer />
    </div>
  );
}
