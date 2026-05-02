"use client";

import { ChangeEvent, FormEvent, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Source = {
  title?: string;
  source?: string;
  url?: string;
  score?: number;
  origin?: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
};

type ChatApiResponse = {
  answer: string;
  sources: Source[];
};

type UploadPdfApiResponse = {
  filename: string;
  chunks_indexed: number;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

function sourceHref(source: Source): string | null {
  const value = source.source ?? source.url ?? "";
  if (!value) {
    return null;
  }

  try {
    const parsed = new URL(value);
    return parsed.toString();
  } catch {
    return null;
  }
}

function dedupeSources(sources: Source[]): Source[] {
  const seen = new Set<string>();
  const unique: Source[] = [];

  for (const source of sources) {
    const href = sourceHref(source) ?? "";
    const title = (source.title ?? "").trim().toLowerCase();
    const key = href || title;
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    unique.push(source);
  }

  return unique;
}

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "assistant-welcome",
      role: "assistant",
      text: "Ask me anything. You can also upload a PDF paper and ask questions about its content.",
      // text: "Ask about an ML concept, paper trend, or architecture and I will respond using your indexed ArXiv corpus.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const canSend = useMemo(
    () => input.trim().length > 0 && !isSending,
    [input, isSending],
  );

  const canUpload = useMemo(
    () => !isUploading && !isSending,
    [isUploading, isSending],
  );

  async function handleSend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = input.trim();
    if (!text || isSending) {
      return;
    }

    setError(null);
    setIsSending(true);
    setInput("");

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      text,
    };

    setMessages((previous) => [...previous, userMessage]);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        throw new Error(`Backend request failed with ${response.status}`);
      }

      const payload = (await response.json()) as ChatApiResponse;
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        text: payload.answer,
        sources: dedupeSources(payload.sources ?? []),
      };

      setMessages((previous) => [...previous, assistantMessage]);
    } catch (sendError) {
      const fallbackMessage: Message = {
        id: `assistant-error-${Date.now()}`,
        role: "assistant",
        text: "I could not reach the backend. Confirm FastAPI is running on port 8000 and try again.",
      };
      setMessages((previous) => [...previous, fallbackMessage]);
      setError(
        sendError instanceof Error ? sendError.message : "Unknown error",
      );
    } finally {
      setIsSending(false);
    }
  }

  function handlePickPdf() {
    if (!canUpload) {
      return;
    }
    fileInputRef.current?.click();
  }

  async function handleUploadFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";

    if (!file) {
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a .pdf file.");
      return;
    }

    setError(null);
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Upload failed with ${response.status}`);
      }

      const payload = (await response.json()) as UploadPdfApiResponse;
      const uploadedMessage: Message = {
        id: `assistant-upload-${Date.now()}`,
        role: "assistant",
        text: `Indexed **${payload.filename}** successfully. Added **${payload.chunks_indexed}** chunks to the vector store. You can ask questions about this PDF now.`,
      };
      setMessages((previous) => [...previous, uploadedMessage]);
    } catch (uploadError) {
      setError(
        uploadError instanceof Error ? uploadError.message : "Upload failed",
      );
      const failedMessage: Message = {
        id: `assistant-upload-error-${Date.now()}`,
        role: "assistant",
        text: "I could not upload this PDF. Confirm the backend is running and try again.",
      };
      setMessages((previous) => [...previous, failedMessage]);
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <div className="flex min-h-[70vh] flex-col gap-4 rounded-2xl border border-ink/10 bg-white/80 p-3 sm:p-4">
      <div className="flex-1 space-y-3 overflow-y-auto rounded-xl bg-canvas/60 p-3">
        {messages.map((message, index) => {
          const isUser = message.role === "user";
          return (
            <article
              key={message.id}
              className={`animate-rise rounded-2xl border px-4 py-3 ${
                isUser
                  ? "ml-auto max-w-[90%] border-accent/40 bg-accent/10"
                  : "mr-auto max-w-[95%] border-ink/10 bg-white"
              }`}
              style={{ animationDelay: `${index * 40}ms` }}
            >
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink/50">
                {isUser ? "You" : "Agent"}
              </p>

              {isUser ? (
                <p className="mt-1 whitespace-pre-wrap text-sm leading-relaxed sm:text-base">
                  {message.text}
                </p>
              ) : (
                <div className="mt-1 text-sm leading-relaxed sm:text-base">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => (
                        <p className="mb-3 last:mb-0">{children}</p>
                      ),
                      ol: ({ children }) => (
                        <ol className="mb-3 list-decimal space-y-1 pl-5 last:mb-0">
                          {children}
                        </ol>
                      ),
                      ul: ({ children }) => (
                        <ul className="mb-3 list-disc space-y-1 pl-5 last:mb-0">
                          {children}
                        </ul>
                      ),
                      li: ({ children }) => <li>{children}</li>,
                      strong: ({ children }) => (
                        <strong className="font-semibold">{children}</strong>
                      ),
                      em: ({ children }) => (
                        <em className="italic">{children}</em>
                      ),
                      a: ({ href, children }) => (
                        <a
                          href={href}
                          target="_blank"
                          rel="noreferrer"
                          className="text-accent underline decoration-accent/50 underline-offset-2 transition hover:text-ember"
                        >
                          {children}
                        </a>
                      ),
                      code: ({ children }) => (
                        <code className="rounded bg-ink/10 px-1 py-0.5 text-[0.92em]">
                          {children}
                        </code>
                      ),
                    }}
                  >
                    {message.text}
                  </ReactMarkdown>
                </div>
              )}

              {!!message.sources?.length && (
                <div className="mt-3 border-t border-ink/10 pt-2">
                  <p className="text-xs font-semibold uppercase tracking-[0.15em] text-ember">
                    Sources
                  </p>
                  <ul className="mt-2 space-y-2 text-sm">
                    {message.sources.map((source, sourceIndex) => {
                      const href = sourceHref(source);
                      const title =
                        source.title || href || `Source ${sourceIndex + 1}`;
                      const badge = source.origin
                        ? source.origin.toUpperCase()
                        : null;

                      return (
                        <li
                          key={`${message.id}-source-${sourceIndex}`}
                          className="rounded-xl border border-ink/10 bg-canvas/50 px-3 py-2"
                        >
                          <div className="flex items-center justify-between gap-2">
                            {href ? (
                              <a
                                href={href}
                                target="_blank"
                                rel="noreferrer"
                                className="font-medium text-accent underline decoration-accent/50 underline-offset-2 transition hover:text-ember"
                              >
                                {title}
                              </a>
                            ) : (
                              <span className="font-medium text-ink">
                                {title}
                              </span>
                            )}

                            {badge ? (
                              <span className="rounded-full border border-ink/20 px-2 py-0.5 text-[10px] font-semibold tracking-[0.12em] text-ink/60">
                                {badge}
                              </span>
                            ) : null}
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
            </article>
          );
        })}
      </div>

      <form
        onSubmit={handleSend}
        className="grid gap-2 sm:grid-cols-[1fr_auto]"
      >
        <label htmlFor="chat-input" className="sr-only">
          Message
        </label>
        <input
          id="chat-input"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          disabled={isSending}
          placeholder="Ask about papers..."
          className="w-full rounded-xl border border-ink/20 bg-white px-4 py-3 text-sm outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20 sm:text-base"
        />
        <button
          type="submit"
          disabled={!canSend}
          className="rounded-xl bg-ink px-5 py-3 text-sm font-semibold text-white transition hover:bg-accent disabled:cursor-not-allowed disabled:opacity-60 sm:text-base"
        >
          {isSending ? "Sending..." : "Send"}
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf,.pdf"
          onChange={handleUploadFile}
          className="hidden"
        />
      </form>

      <div className="flex justify-end">
        <button
          type="button"
          onClick={handlePickPdf}
          disabled={!canUpload}
          className="rounded-xl border border-ink/20 bg-white px-4 py-2 text-sm font-semibold text-ink transition hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isUploading ? "Uploading PDF..." : "Upload PDF"}
        </button>
      </div>

      {error ? <p className="text-xs text-ember">{error}</p> : null}
    </div>
  );
}
