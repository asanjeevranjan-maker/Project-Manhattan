'use client';

import { useCallback, useRef, useState, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send, Sparkles, Trash2, AlertCircle, User, Bot, Loader2 } from 'lucide-react';
import { useSatQueryStore } from '@/store/satquery';
import { useToast } from '@/hooks/use-toast';
import { shortId } from '@/lib/client-utils';
import type { AnalysisResult, ChatMessage } from '@/lib/types';
import { MarkdownLite } from './markdown-lite';

const SUGGESTED_QUERIES = [
  'Identify water bodies in this image.',
  'How much forest area is visible?',
  'Detect urban areas and buildings.',
  'Estimate the land cover breakdown.',
  'Find any flooded regions.',
  'Describe the major geographic features.',
];

interface AnalyzeResponse {
  analysis: AnalysisResult;
  rawAnswer: string;
  intent: string;
  intentLabel: string;
}

export function ChatPanel() {
  const activeImage = useSatQueryStore((s) => s.activeImage);
  const messages = useSatQueryStore((s) => s.messages);
  const isAnalyzing = useSatQueryStore((s) => s.isAnalyzing);
  const addMessage = useSatQueryStore((s) => s.addMessage);
  const updateMessage = useSatQueryStore((s) => s.updateMessage);
  const setIsAnalyzing = useSatQueryStore((s) => s.setIsAnalyzing);
  const setLatestAnalysis = useSatQueryStore((s) => s.setLatestAnalysis);
  const clearChat = useSatQueryStore((s) => s.clearChat);
  const { toast } = useToast();

  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isAnalyzing]);

  const submitQuery = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      if (!activeImage) {
        toast({
          variant: 'destructive',
          title: 'No image selected',
          description: 'Please upload or select a satellite image first.',
        });
        return;
      }
      if (isAnalyzing) return;

      // Append the user's message
      const userMsg: ChatMessage = {
        id: shortId('u-'),
        role: 'user',
        content: trimmed,
        createdAt: new Date().toISOString(),
      };
      addMessage(userMsg);
      setInput('');
      setIsAnalyzing(true);

      // Append a placeholder assistant message that we'll update when the VLM responds
      const assistantMsgId = shortId('a-');
      addMessage({
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        createdAt: new Date().toISOString(),
        pending: true,
      });

      try {
        // Build conversation history (exclude the placeholder)
        const history = messages
          .filter((m) => !m.pending && !m.error)
          .slice(-6)
          .map((m) => ({ role: m.role, content: m.content }));

        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            imageDataUrl: activeImage.dataUrl,
            secondImageDataUrl: activeImage.secondDataUrl,
            query: trimmed,
            history,
          }),
        });

        if (!res.ok) {
          let errText = `HTTP ${res.status}`;
          try {
            const errJson = (await res.json()) as { error?: string };
            if (errJson.error) errText = errJson.error;
          } catch {
            /* ignore */
          }
          throw new Error(errText);
        }

        const data = (await res.json()) as AnalyzeResponse;
        updateMessage(assistantMsgId, {
          content: data.analysis.answer,
          analysis: data.analysis,
          pending: false,
        });
        setLatestAnalysis(data.analysis);
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Failed to analyze';
        updateMessage(assistantMsgId, {
          content: '',
          pending: false,
          error: msg,
        });
        toast({
          variant: 'destructive',
          title: 'Analysis failed',
          description: msg,
        });
      } finally {
        setIsAnalyzing(false);
      }
    },
    [activeImage, addMessage, updateMessage, setIsAnalyzing, setLatestAnalysis, messages, isAnalyzing, toast]
  );

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Cmd/Ctrl+Enter to submit
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      submitQuery(input);
    }
  };

  const showEmpty = messages.length === 0;

  return (
    <div className="flex h-full min-h-[500px] flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-3 py-2.5">
        <div className="flex items-center gap-2">
          <div className="flex size-7 items-center justify-center rounded-md bg-primary/10 text-primary">
            <Sparkles className="size-4" />
          </div>
          <div className="leading-tight">
            <p className="text-sm font-semibold">AI Assistant</p>
            <p className="text-[11px] text-muted-foreground">Ask questions about the image</p>
          </div>
        </div>
        {messages.length > 0 && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              clearChat();
              toast({ title: 'Conversation cleared' });
            }}
            className="gap-1.5 text-muted-foreground hover:text-foreground"
          >
            <Trash2 className="size-3.5" />
            Clear
          </Button>
        )}
      </div>

      {/* Messages area */}
      <div
        ref={scrollRef}
        className="satquery-scroll flex-1 space-y-4 overflow-y-auto p-3"
      >
        {showEmpty && (
          <EmptyState
            hasImage={Boolean(activeImage)}
            onPick={(q) => submitQuery(q)}
          />
        )}

        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}

        {isAnalyzing && (
          <div className="flex items-center gap-2 pl-9 text-xs text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            SatQuery AI is analyzing the satellite image…
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t p-3">
        <div className="relative">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              activeImage
                ? 'Ask anything about this satellite image…  (⌘/Ctrl+↵ to send)'
                : 'Select an image first to start asking questions…'
            }
            disabled={isAnalyzing || !activeImage}
            rows={2}
            className="resize-none pr-12 text-sm"
          />
          <Button
            type="button"
            size="icon"
            className="absolute bottom-2 right-2 size-8 rounded-md"
            onClick={() => submitQuery(input)}
            disabled={isAnalyzing || !input.trim() || !activeImage}
            aria-label="Send query"
          >
            <Send className="size-4" />
          </Button>
        </div>
        <p className="mt-1.5 text-[10px] text-muted-foreground">
          Press <kbd className="rounded bg-muted px-1 py-0.5 font-mono text-[9px]">⌘/Ctrl + ↵</kbd> to send · Powered by GLM-4V Vision-Language Model
        </p>
      </div>
    </div>
  );
}

function EmptyState({
  hasImage,
  onPick,
}: {
  hasImage: boolean;
  onPick: (q: string) => void;
}) {
  return (
    <div className="flex flex-col items-center gap-4 py-6 text-center">
      <div className="flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Bot className="size-6" />
      </div>
      <div className="space-y-1 px-4">
        <p className="text-sm font-semibold">
          {hasImage ? 'Ready to analyze your image' : 'Welcome to SatQuery AI'}
        </p>
        <p className="text-xs text-muted-foreground">
          {hasImage
            ? 'Try one of the suggested queries below to get started.'
            : 'Upload a satellite image, then ask any question in natural language.'}
        </p>
      </div>
      {hasImage && (
        <div className="grid w-full max-w-sm gap-1.5">
          {SUGGESTED_QUERIES.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => onPick(q)}
              className="group flex items-center gap-2 rounded-md border bg-card/50 px-2.5 py-1.5 text-left text-xs transition-all hover:border-primary/50 hover:bg-accent/50"
            >
              <Sparkles className="size-3 shrink-0 text-primary/70 group-hover:text-primary" />
              <span>{q}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  return (
    <div
      className={cn(
        'flex w-full gap-2 animate-fade-in-up',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      <div
        className={cn(
          'flex size-7 shrink-0 items-center justify-center rounded-full',
          isUser
            ? 'bg-secondary text-secondary-foreground'
            : 'bg-primary/10 text-primary'
        )}
      >
        {isUser ? <User className="size-3.5" /> : <Bot className="size-3.5" />}
      </div>
      <div
        className={cn(
          'max-w-[85%] space-y-2 rounded-lg px-3 py-2 text-sm',
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-card border'
        )}
      >
        {message.pending ? (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            <span className="text-xs italic">Analyzing image…</span>
          </div>
        ) : message.error ? (
          <div className="flex items-start gap-2 text-destructive">
            <AlertCircle className="size-4 shrink-0 translate-y-0.5" />
            <div className="space-y-1">
              <p className="text-xs font-semibold">Analysis failed</p>
              <p className="text-xs">{message.error}</p>
            </div>
          </div>
        ) : (
          <>
            <MarkdownLite text={message.content} />
            {message.analysis && message.analysis.objectsDetected.length > 0 && (
              <ObjectsDetectedBadges analysis={message.analysis} />
            )}
            {message.analysis && (
              <div className="flex flex-wrap items-center gap-2 pt-1 text-[10px] text-muted-foreground">
                <span className="rounded bg-secondary/60 px-1.5 py-0.5 font-medium">
                  Intent: {message.analysis.intent.replace(/_/g, ' ')}
                </span>
                <span className="rounded bg-secondary/60 px-1.5 py-0.5 font-medium">
                  Confidence: {Math.round(message.analysis.confidence * 100)}%
                </span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function ObjectsDetectedBadges({ analysis }: { analysis: AnalysisResult }) {
  return (
    <div className="flex flex-wrap gap-1 pt-1">
      {analysis.objectsDetected.map((o, i) => (
        <span
          key={`${o.class}-${i}`}
          className="inline-flex items-center gap-1 rounded-full bg-secondary/70 px-2 py-0.5 text-[10px] font-medium"
        >
          <span className="capitalize">{o.class}</span>
          {typeof o.count === 'number' && <span className="opacity-70">×{o.count}</span>}
          <span className="text-primary font-semibold">{Math.round(o.confidence * 100)}%</span>
        </span>
      ))}
    </div>
  );
}
