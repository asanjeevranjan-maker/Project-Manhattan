'use client';

import ReactMarkdown from 'react-markdown';

/**
 * Lightweight markdown renderer used for VLM responses.
 * Uses react-markdown with conservative settings — no raw HTML, no unsafe
 * transformations.
 */
export function MarkdownLite({ text }: { text: string }) {
  if (!text) return null;
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none break-words [&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0.5 [&_strong]:font-semibold [&_em]:italic [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs [&_code]:font-mono [&_h1]:text-sm [&_h1]:font-semibold [&_h1]:mt-2 [&_h1]:mb-1 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:mt-2 [&_h2]:mb-1 [&_h3]:text-xs [&_h3]:font-semibold [&_h3]:uppercase [&_h3]:tracking-wide [&_h3]:text-muted-foreground [&_h3]:mt-2 [&_h3]:mb-1 [&_blockquote]:border-l-2 [&_blockquote]:border-primary/40 [&_blockquote]:pl-2 [&_blockquote]:italic [&_blockquote]:text-muted-foreground">
      <ReactMarkdown
        disallowedElements={['script', 'iframe', 'object', 'embed', 'form']}
        unwrapDisallowed
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
