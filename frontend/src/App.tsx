import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  Bot,
  ChevronDown,
  FileText,
  LoaderCircle,
  MessageSquareText,
  RefreshCcw,
  Send,
  Trash2,
  Upload,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

type Source = {
  id: string;
  source: string;
  page: number;
  content: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

type DashboardData = {
  indexed_file_count: number;
  chunk_count: number;
  sessions: number;
  messages: number;
};

const defaultDashboard: DashboardData = {
  indexed_file_count: 0,
  chunk_count: 0,
  sessions: 0,
  messages: 0,
};

const API_BASE_URL = import.meta.env.DEV
  ? (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8001")
  : "";

function apiUrl(path: string) {
  /** Builds an API URL that works in both Vite dev mode and built mode. */
  return `${API_BASE_URL}${path}`;
}

async function readJsonOrThrow(response: Response, requestPath: string) {
  /** Parses JSON responses and turns HTTP/API failures into readable UI errors. */
  const contentType = response.headers.get("content-type") ?? "";
  const rawText = await response.text();

  if (!response.ok) {
    let detail = rawText;
    if (contentType.includes("application/json")) {
      try {
        const payload = JSON.parse(rawText) as { detail?: string };
        detail = payload.detail ?? rawText;
      } catch {
        // Fall back to the raw response body.
      }
    }
    throw new Error(detail || `Request failed for ${requestPath}.`);
  }

  if (!contentType.includes("application/json")) {
    if (
      rawText.trim().startsWith("<!doctype") ||
      rawText.trim().startsWith("<html")
    ) {
      throw new Error(
        `Expected JSON from ${requestPath}, but received HTML. Restart the Vite dev server so its proxy config reloads.`,
      );
    }
    throw new Error(
      `Expected JSON from ${requestPath}, but received ${contentType || "unknown content"}.`,
    );
  }

  return JSON.parse(rawText);
}

function getSessionId() {
  /** Reuses the browser session id or creates one so chat history stays tied to this client. */
  const stored = window.localStorage.getItem("invoice-bot-session");
  if (stored) {
    return stored;
  }

  const created =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `session-${Date.now()}`;

  window.localStorage.setItem("invoice-bot-session", created);
  return created;
}

export default function App() {
  /** Main screen that coordinates dashboard loading, uploads, and document chat. */
  const [dashboard, setDashboard] = useState<DashboardData>(defaultDashboard);
  const [files, setFiles] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [uploadQueue, setUploadQueue] = useState<File[]>([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const sessionIdRef = useRef("");
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    sessionIdRef.current = getSessionId();
    void refreshAll();
  }, []);

  async function refreshAll() {
    /** Reloads dashboard stats, indexed files, and chat history together. */
    setIsLoading(true);
    setError("");

    try {
      const [dashboardResponse, filesResponse, historyResponse] =
        await Promise.all([
          fetch(apiUrl("/api/dashboard")),
          fetch(apiUrl("/api/files")),
          fetch(apiUrl(`/api/history/${sessionIdRef.current}`)),
        ]);

      const dashboardData = (await readJsonOrThrow(
        dashboardResponse,
        "/api/dashboard",
      )) as DashboardData;
      const filesData = (await readJsonOrThrow(
        filesResponse,
        "/api/files",
      )) as {
        files: string[];
      };
      const historyData = (await readJsonOrThrow(
        historyResponse,
        `/api/history/${sessionIdRef.current}`,
      )) as {
        messages: ChatMessage[];
      };

      setDashboard(dashboardData);
      setFiles(filesData.files);
      setMessages(historyData.messages ?? []);
    } catch (refreshError) {
      const message =
        refreshError instanceof Error
          ? refreshError.message
          : "Unexpected error while loading data.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleUpload() {
    /** Sends selected PDFs to the backend and refreshes the workspace after indexing finishes. */
    if (!uploadQueue.length) {
      setError("Choose one or more PDF files before indexing.");
      return;
    }

    setIsUploading(true);
    setError("");
    setUploadStatus("");

    try {
      const formData = new FormData();
      uploadQueue.forEach((file) => formData.append("files", file));

      const response = await fetch(apiUrl("/api/upload"), {
        method: "POST",
        body: formData,
      });

      const payload = (await readJsonOrThrow(response, "/api/upload")) as {
        detail?: string;
        message?: string;
      };

      setUploadStatus(payload.message ?? "Files indexed successfully.");
      setUploadQueue([]);
      if (uploadInputRef.current) {
        uploadInputRef.current.value = "";
      }
      await refreshAll();
    } catch (uploadError) {
      const message =
        uploadError instanceof Error ? uploadError.message : "Upload failed.";
      setError(message);
    } finally {
      setIsUploading(false);
    }
  }

  async function handleSubmit(nextQuestion?: string) {
    /** Sends a user question to the backend and appends the grounded answer to chat. */
    const prompt = (nextQuestion ?? question).trim();
    if (!prompt) {
      setError("Enter a question about your uploaded documents.");
      return;
    }

    const userMessage: ChatMessage = { role: "user", content: prompt };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setIsSending(true);
    setError("");

    try {
      const response = await fetch(apiUrl("/api/chat"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: prompt,
          session_id: sessionIdRef.current,
        }),
      });

      const payload = (await readJsonOrThrow(response, "/api/chat")) as {
        answer: string;
        sources?: Source[];
      };

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: payload.answer,
        sources: payload.sources,
      };

      setMessages((current) => [...current, assistantMessage]);
      await refreshDashboardOnly();
    } catch (chatError) {
      const message =
        chatError instanceof Error
          ? chatError.message
          : "Unable to get an answer.";
      setMessages((current) => current.slice(0, -1));
      setError(message);
    } finally {
      setIsSending(false);
    }
  }

  async function refreshDashboardOnly() {
    /** Refreshes just the top-level counters after chat actions without reloading everything. */
    try {
      const response = await fetch(apiUrl("/api/dashboard"));
      const payload = (await readJsonOrThrow(
        response,
        "/api/dashboard",
      )) as DashboardData;
      setDashboard(payload);
    } catch {
      // The chat already succeeded, so silently keep the previous counters.
    }
  }

  async function clearChat() {
    /** Clears the current session history both in the backend and the local UI state. */
    setError("");

    try {
      const response = await fetch(
        apiUrl(`/api/history/${sessionIdRef.current}`),
        {
          method: "DELETE",
        },
      );

      if (!response.ok) {
        throw new Error("Unable to clear chat history.");
      }

      setMessages([]);
      await refreshDashboardOnly();
    } catch (clearError) {
      const message =
        clearError instanceof Error
          ? clearError.message
          : "Unable to clear chat history.";
      setError(message);
    }
  }

  const scrollContainerRef = useRef<HTMLDivElement | null>(null);

  return (
    <main className="grain-overlay h-screen overflow-hidden">
      <div className="mx-auto flex h-screen max-w-7xl flex-col px-4 py-6 sm:px-6 lg:px-8">
        <header className="relative z-20 shrink-0 overflow-hidden rounded-[2rem] border border-white/60 bg-white/70 p-6 shadow-glow backdrop-blur sm:p-8">
          <div className="absolute inset-y-0 right-0 hidden w-1/2 bg-[radial-gradient(circle_at_top,rgba(20,184,166,0.20),transparent_58%)] lg:block" />
          <div className="relative z-10 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <h1 className="font-display text-4xl leading-tight text-slate-900 sm:text-5xl">
                Simple PDF RAG Chat Bot.
              </h1>
              <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-600 sm:text-base">
                Upload PDFs, split them into chunks, store them in the vector
                database, and ask questions from the same FastAPI backend.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 xxl:grid-cols-3">
              <MetricCard
                label="Indexed files"
                value={dashboard.indexed_file_count}
                tone="amber"
              />
              <MetricCard
                label="Chunks"
                value={dashboard.chunk_count}
                tone="teal"
              />
              <MetricCard
                label="Stored messages"
                value={dashboard.messages}
                tone="slate"
              />
            </div>
          </div>
        </header>

        <section
          ref={scrollContainerRef}
          className="mt-6 min-h-0 flex-1 overflow-y-auto pr-1"
        >
          <div className="grid mt-6 gap-6 lg:grid-cols-2 ">
            <div className="border rounded-lg">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5 text-primary" />
                  Index PDFs
                </CardTitle>
                <CardDescription>
                  Upload one or more billing documents and rebuild the vector
                  index in one step.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <label className="block rounded-[1.5rem] border border-dashed border-border bg-secondary/40 p-4 text-sm text-slate-600 ">
                  <span className="mb-2 block font-medium text-slate-900">
                    Choose PDF files
                  </span>
                  <Input
                    className="cursor-pointer"
                    accept=".pdf"
                    multiple
                    ref={uploadInputRef}
                    type="file"
                    onChange={(event) =>
                      setUploadQueue(Array.from(event.target.files ?? []))
                    }
                  />
                </label>

                {!!uploadQueue.length && (
                  <div className="flex flex-wrap gap-2 cursor-pointer">
                    {uploadQueue.map((file) => (
                      <Badge key={file.name} variant="secondary">
                        {file.name}
                      </Badge>
                    ))}
                  </div>
                )}

                <Button
                  className="w-full"
                  disabled={isUploading}
                  onClick={handleUpload}
                >
                  {isUploading ? (
                    <>
                      <LoaderCircle className="h-4 w-4 animate-spin" />
                      Indexing documents
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4" />
                      Upload and index
                    </>
                  )}
                </Button>

                {uploadStatus && (
                  <p className="rounded-2xl bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                    {uploadStatus}
                  </p>
                )}
              </CardContent>
            </div>

            <div className="border rounded-lg">
              <CardHeader className="flex-row items-center justify-between space-y-0">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5 text-primary" />
                    Indexed files
                  </CardTitle>
                  <CardDescription>
                    Current PDFs available for retrieval.
                  </CardDescription>
                </div>
                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() => void refreshAll()}
                >
                  <RefreshCcw className="h-4 w-4" />
                </Button>
              </CardHeader>
              <CardContent className="space-y-3">
                {files.length ? (
                  files.map((file) => (
                    <div
                      key={file}
                      className="flex items-center gap-3 rounded-2xl border border-border/70 bg-white/70 px-4 py-3"
                    >
                      <div className="rounded-2xl bg-secondary p-2 text-secondary-foreground">
                        <FileText className="h-4 w-4" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium text-slate-900">
                          {file}
                        </p>
                      </div>
                    </div>
                  ))
                ) : (
                  <EmptyBlock
                    icon={<FileText className="h-5 w-5" />}
                    title="No indexed PDFs yet"
                    description="Upload at least one invoice or billing PDF to start querying."
                  />
                )}
              </CardContent>
            </div>
          </div>

          <div className="mt-6 border rounded-lg">
            <CardHeader className="border-border/70 bg-white/50">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <MessageSquareText className="h-5 w-5 text-primary" />
                    Document chat
                  </CardTitle>
                  <CardDescription>
                    Ask questions about anything found in the uploaded PDFs.
                  </CardDescription>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="outline">
                    Session: {sessionIdRef.current || "..."}
                  </Badge>
                  <Button
                    disabled={isSending}
                    variant="outline"
                    onClick={clearChat}
                  >
                    <Trash2 className="h-4 w-4" />
                    Clear chat
                  </Button>
                </div>
              </div>

              <div className="p-4 sm:p-6">
                <div className="rounded-[1.75rem] border border-border/80 bg-white/80 p-3 shadow-sm">
                  <Textarea
                    className="min-h-[120px] border-0 bg-transparent p-2 shadow-none focus-visible:ring-0"
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="Ask anything about the uploaded PDFs..."
                    value={question}
                  />
                  <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <p className="text-xs text-slate-500">
                      Tip: ask about names, dates, amounts, pages, or summaries
                      for better results.
                    </p>
                    <Button
                      className="sm:min-w-40"
                      disabled={isSending}
                      onClick={() => void handleSubmit()}
                    >
                      {isSending ? (
                        <>
                          <LoaderCircle className="h-4 w-4 animate-spin" />
                          Asking
                        </>
                      ) : (
                        <>
                          <Send className="h-4 w-4" />
                          Send question
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              </div>
            </CardHeader>
          </div>

          <div className="mt-6 border rounded-lg mb-[60px]">
            <CardContent className="p-0">
              {error && (
                <div className="mx-6 mt-6 flex items-start gap-3 rounded-[1.5rem] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              {(isLoading || messages.length > 0 || isSending) && (
                <>
                  <div className="px-6 py-6">
                    {isLoading ? (
                      <EmptyBlock
                        icon={<LoaderCircle className="h-5 w-5 animate-spin" />}
                        title="Loading workspace"
                        description="Fetching indexed files, dashboard stats, and your session history."
                      />
                    ) : (
                      <div className="space-y-4">
                        {messages.map((message, index) => (
                          <MessageBubble
                            key={`${message.role}-${index}`}
                            message={message}
                          />
                        ))}
                        {isSending && (
                          <div className="flex items-center gap-3 rounded-[1.5rem] bg-secondary/60 px-4 py-3 text-sm text-slate-600">
                            <LoaderCircle className="h-4 w-4 animate-spin" />
                            Searching the indexed documents...
                          </div>
                        )}
                        <div ref={messagesEndRef} />
                      </div>
                    )}
                  </div>

                  <div className="absolute bottom-5 right-5 flex flex-col gap-3">
                    {/* Scroll to Top */}
                    <button
                      type="button"
                      aria-label="Scroll to top"
                      onClick={() =>
                        scrollContainerRef.current?.scrollTo({
                          top: 0,
                          behavior: "smooth",
                        })
                      }
                      className="flex h-12 w-12 items-center justify-center rounded-full border border-slate-300/70 bg-slate-900/85 text-white shadow-lg transition hover:bg-slate-800"
                    >
                      <ArrowUp className="h-6 w-6" />
                    </button>

                    {/* Scroll to Bottom */}
                    <button
                      type="button"
                      aria-label="Scroll to bottom"
                      onClick={() =>
                        messagesEndRef.current?.scrollIntoView({
                          behavior: "smooth",
                          block: "end",
                        })
                      }
                      className="flex h-12 w-12 items-center justify-center rounded-full border border-slate-300/70 bg-slate-900/85 text-white shadow-lg transition hover:bg-slate-800"
                    >
                      <ArrowDown className="h-6 w-6" />
                    </button>
                  </div>
                </>
              )}
            </CardContent>
          </div>
        </section>
      </div>
    </main>
  );
}

function MetricCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "amber" | "teal" | "slate";
}) {
  /** Displays a single dashboard stat card with a small visual tone variant. */
  const toneClassName =
    tone === "amber"
      ? "bg-amber-50 text-amber-700"
      : tone === "teal"
        ? "bg-teal-50 text-teal-700"
        : "bg-slate-100 text-slate-700";

  return (
    <div className="rounded-[1.5rem] border border-white/80 bg-white/85 p-4 shadow-sm backdrop-blur">
      <div
        className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${toneClassName}`}
      >
        {label}
      </div>
      <p className="mt-4 text-3xl font-bold text-slate-900">{value}</p>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  /** Renders one chat message and optionally shows the supporting source snippets. */
  const isAssistant = message.role === "assistant";

  return (
    <div
      className={`rounded-[1.75rem] border px-4 py-4 shadow-sm sm:px-5 ${
        isAssistant
          ? "border-primary/10 bg-white"
          : "border-amber-200 bg-amber-50/70"
      }`}
    >
      <div className="mb-3 flex items-center gap-2">
        <div
          className={`flex h-9 w-9 items-center justify-center rounded-full ${
            isAssistant
              ? "bg-teal-100 text-teal-700"
              : "bg-amber-100 text-amber-700"
          }`}
        >
          {isAssistant ? (
            <Bot className="h-4 w-4" />
          ) : (
            <MessageSquareText className="h-4 w-4" />
          )}
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-900">
            {isAssistant ? "Assistant" : "You"}
          </p>
          <p className="text-xs text-slate-500">
            {isAssistant ? "Grounded answer" : "Question"}
          </p>
        </div>
      </div>

      {isAssistant ? (
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({ children }) => (
              <p className="mb-4 text-sm leading-7 text-slate-700 last:mb-0">
                {children}
              </p>
            ),
            ul: ({ children }) => (
              <ul className="mb-4 list-disc space-y-2 pl-5 text-sm leading-7 text-slate-700 last:mb-0">
                {children}
              </ul>
            ),
            ol: ({ children }) => (
              <ol className="mb-4 list-decimal space-y-2 pl-5 text-sm leading-7 text-slate-700 last:mb-0">
                {children}
              </ol>
            ),
            table: ({ children }) => (
              <div className="mb-4 overflow-x-auto rounded-[1rem] border border-border/70 last:mb-0">
                <table className="min-w-full border-collapse bg-white text-sm">
                  {children}
                </table>
              </div>
            ),
            thead: ({ children }) => (
              <thead className="bg-slate-50 text-slate-900">{children}</thead>
            ),
            tbody: ({ children }) => (
              <tbody className="text-slate-700">{children}</tbody>
            ),
            tr: ({ children }) => (
              <tr className="border-b border-border/60 last:border-b-0">
                {children}
              </tr>
            ),
            th: ({ children }) => (
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.08em]">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="px-4 py-3 align-top leading-6">{children}</td>
            ),
            li: ({ children }) => <li>{children}</li>,
            strong: ({ children }) => (
              <strong className="font-semibold text-slate-900">
                {children}
              </strong>
            ),
            em: ({ children }) => <em className="italic">{children}</em>,
            code: ({ children }) => (
              <code className="rounded bg-slate-100 px-1.5 py-0.5 text-[0.9em] text-slate-800">
                {children}
              </code>
            ),
          }}
        >
          {message.content}
        </ReactMarkdown>
      ) : (
        <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
          {message.content}
        </p>
      )}

      {!!message.sources?.length && (
        <details className="group mt-4 rounded-[1.25rem] border border-border/70 bg-secondary/40 p-4">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                Sources
              </p>
              <Badge variant="outline">{message.sources.length}</Badge>
            </div>
            <ChevronDown className="h-4 w-4 text-slate-500 transition group-open:rotate-180" />
          </summary>

          <div className="mt-4 space-y-3">
            {message.sources.map((source) => (
              <div
                key={source.id}
                className="rounded-[1.25rem] border border-border/60 bg-white/80 p-3"
              >
                <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                  <Badge variant="outline">{source.id}</Badge>
                  <span>{source.source}</span>
                  <span>Page {source.page}</span>
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-700">
                  {source.content}
                </p>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function EmptyBlock({
  icon,
  title,
  description,
}: {
  icon: ReactNode;
  title: string;
  description: string;
}) {
  /** Renders the reusable empty/loading state used across the UI panels. */
  return (
    <div className="flex min-h-[220px] flex-col items-center justify-center rounded-[1.75rem] border border-dashed border-border bg-white/60 px-6 py-8 text-center">
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
      <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
        {description}
      </p>
    </div>
  );
}
