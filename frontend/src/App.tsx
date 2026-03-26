import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowUpRight,
  Bot,
  ChevronDown,
  FileText,
  FileJson,
  LoaderCircle,
  MessageSquareText,
  RefreshCcw,
  Send,
  Sparkles,
  Trash2,
  Upload,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
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
  structured_invoice_count?: number;
};

type StructuredInvoiceSummary = {
  source_file: string;
  json_file: string;
  invoice_number?: string | null;
  issue_date?: string | null;
  currency?: string | null;
  gross_total?: string | null;
};

const defaultDashboard: DashboardData = {
  indexed_file_count: 0,
  chunk_count: 0,
  sessions: 0,
  messages: 0,
  structured_invoice_count: 0,
};

const starterQuestions = [
  "Summarize the uploaded invoice in plain language.",
  "Which pages mention the billing amount or due date?",
  "List the customer details found across the documents.",
];

const API_BASE_URL = import.meta.env.DEV
  ? (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8001")
  : "";

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

async function readJsonOrThrow(response: Response, requestPath: string) {
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
  const [dashboard, setDashboard] = useState<DashboardData>(defaultDashboard);
  const [files, setFiles] = useState<string[]>([]);
  const [invoices, setInvoices] = useState<StructuredInvoiceSummary[]>([]);
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

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, isSending]);

  async function refreshAll() {
    setIsLoading(true);
    setError("");

    try {
      const [
        dashboardResponse,
        filesResponse,
        invoicesResponse,
        historyResponse,
      ] = await Promise.all([
        fetch(apiUrl("/api/dashboard")),
        fetch(apiUrl("/api/files")),
        fetch(apiUrl("/api/invoices")),
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
      const invoicesData = (await readJsonOrThrow(
        invoicesResponse,
        "/api/invoices",
      )) as {
        invoices: StructuredInvoiceSummary[];
      };
      const historyData = (await readJsonOrThrow(
        historyResponse,
        `/api/history/${sessionIdRef.current}`,
      )) as {
        messages: ChatMessage[];
      };

      setDashboard(dashboardData);
      setFiles(filesData.files);
      setInvoices(invoicesData.invoices ?? []);
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

  return (
    <main className="grain-overlay min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-6 sm:px-6 lg:px-8">
        <header className="relative overflow-hidden rounded-[2rem] border border-white/60 bg-white/70 p-6 shadow-glow backdrop-blur sm:p-8">
          <div className="absolute inset-y-0 right-0 hidden w-1/2 bg-[radial-gradient(circle_at_top,rgba(20,184,166,0.20),transparent_58%)] lg:block" />
          <div className="relative z-10 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <h1 className="font-display text-4xl leading-tight text-slate-900 sm:text-5xl">
                Invoice & Billing Query Chat Bot.
              </h1>
              <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-600 sm:text-base">
                Upload PDFs, inspect index health, and chat with grounded
                answers from the same FastAPI backend. The layout adapts cleanly
                from phones to large screens.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 xxl:grid-cols-4">
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
              <MetricCard
                label="JSON records"
                value={dashboard.structured_invoice_count ?? invoices.length}
                tone="teal"
              />
            </div>
          </div>
        </header>

        <section className="mt-6 grid gap-6 lg:items-start lg:grid-cols-[360px_minmax(0,1fr)]">
          <div className="space-y-6">
            <Card>
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
            </Card>

            <Card>
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
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileJson className="h-5 w-5 text-primary" />
                  Extracted JSON
                </CardTitle>
                <CardDescription>
                  Structured invoice records saved on the backend after upload.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {invoices.length ? (
                  invoices.map((invoice) => (
                    <a
                      key={invoice.json_file}
                      className="block rounded-[1.5rem] border border-border/70 bg-white/70 px-4 py-3 transition hover:border-primary/30 hover:bg-accent"
                      href={apiUrl(`/api/invoices/${invoice.json_file}`)}
                      rel="noreferrer"
                      target="_blank"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate text-sm font-semibold text-slate-900">
                            {invoice.invoice_number || invoice.source_file}
                          </p>
                          <p className="mt-1 truncate text-xs text-slate-500">
                            {invoice.source_file}
                          </p>
                          <p className="mt-2 text-xs text-slate-600">
                            Date: {invoice.issue_date || "unknown"} | Total:{" "}
                            {invoice.gross_total || "unknown"}{" "}
                            {invoice.currency || ""}
                          </p>
                        </div>
                        <ArrowUpRight className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                      </div>
                    </a>
                  ))
                ) : (
                  <EmptyBlock
                    icon={<FileJson className="h-5 w-5" />}
                    title="No JSON extracted yet"
                    description="Upload an invoice PDF and the backend will save a structured JSON version automatically."
                  />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  Suggested prompts
                </CardTitle>
                <CardDescription>
                  Start with a quick question and refine from there.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {starterQuestions.map((item) => (
                  <button
                    key={item}
                    className="w-full rounded-[1.5rem] border border-border/70 bg-white/70 px-4 py-3 text-left text-sm text-slate-700 transition hover:border-primary/30 hover:bg-accent disabled:cursor-not-allowed disabled:opacity-60"
                    disabled={isSending}
                    onClick={() => void handleSubmit(item)}
                    type="button"
                  >
                    <span className="flex items-start justify-between gap-3">
                      <span>{item}</span>
                      <ArrowUpRight className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                    </span>
                  </button>
                ))}
              </CardContent>
            </Card>
          </div>

          <Card className="overflow-hidden">
            <CardHeader className="border-b border-border/70 bg-white/50">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <MessageSquareText className="h-5 w-5 text-primary" />
                    Document chat
                  </CardTitle>
                  <CardDescription>
                    Ask questions about invoice totals, line items, customer
                    info, or due dates.
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
            </CardHeader>

            <CardContent className="p-0">
              {error && (
                <div className="mx-6 mt-6 flex items-start gap-3 rounded-[1.5rem] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              <div className="p-4 sm:p-6">
                <div className="rounded-[1.75rem] border border-border/80 bg-white/80 p-3 shadow-sm">
                  <Textarea
                    className="min-h-[120px] border-0 bg-transparent p-2 shadow-none focus-visible:ring-0"
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="Ask about totals, billing periods, invoice IDs, contact details, or anything found in the PDFs..."
                    value={question}
                  />
                  <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <p className="text-xs text-slate-500">
                      Tip: mention a field like invoice number, amount, vendor,
                      page, or date to get a more precise answer.
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

              {(isLoading || messages.length > 0 || isSending) && (
                <>
                  <Separator />

                  <div className="max-h-[52rem] overflow-y-auto px-6 py-6">
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
                </>
              )}
            </CardContent>
          </Card>
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
            li: ({ children }) => <li>{children}</li>,
            strong: ({ children }) => (
              <strong className="font-semibold text-slate-900">{children}</strong>
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
