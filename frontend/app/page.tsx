"use client";

import { ChangeEvent, DragEvent, FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

type Provider = "local" | "openai" | "deepseek" | "claude";

type Citation = {
  chunk_id: string;
  document_id: string;
  document_name: string;
  page_number: number;
  excerpt: string;
  rrf_score: number;
  rerank_score: number;
};

type Thought = {
  stage: string;
  summary: string;
  subqueries?: string[];
  cache_hit?: boolean;
};

type DocumentProgress = {
  id: string;
  original_filename: string;
  status: string;
  page_count: number | null;
  chunk_count: number;
  error_message: string | null;
};

type SummaryCitation = {
  chunk_id: string;
  page_number: number;
};

type SummaryClaim = {
  english: string;
  chinese: string;
  citations: SummaryCitation[];
};

type BilingualSummaryData = {
  english_summary: string;
  chinese_summary: string;
  overview_citations: SummaryCitation[];
  key_findings: SummaryClaim[];
  methods: SummaryClaim[];
  limitations: SummaryClaim[];
  translation_mode: "llm_bilingual" | "local_verification";
};

type BilingualSummaryResponse = {
  document_id: string;
  status: "pending" | "summarizing" | "completed" | "failed";
  task_id: string | null;
  source_chunk_count: number;
  route_provider: Provider;
  route_model: string;
  error_message: string | null;
  summary: BilingualSummaryData | null;
};

type SSEFrame = {
  id: number;
  event: "thought" | "citation" | "delta" | "error";
  data: Thought | Citation | { text: string; cache_hit: boolean } | { message: string };
};

type Institution = {
  id: string;
  name: string;
};

type Paper = {
  title: string;
  authors: string[];
  publication_year: number | null;
  abstract: string | null;
  doi: string | null;
  landing_page_url: string | null;
  open_access_url: string | null;
  open_access_pdf_url: string | null;
  citation_count: number | null;
  sources: string[];
  access_type: "open_access" | "institution_login" | "metadata_only";
  institution_name: string | null;
  institution_access_url: string | null;
};

type PaperSearchResponse = {
  query: string;
  is_doi_lookup: boolean;
  results: Paper[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const citationPattern = /(\[Ref: ([0-9a-fA-F-]{36}), Page (\d+)\])/g;

declare global {
  interface Window {
    deepRagDesktop?: { sessionToken: string };
  }
}

function apiFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  if (typeof window !== "undefined" && window.deepRagDesktop) {
    headers.set("X-Desktop-Session", window.deepRagDesktop.sessionToken);
  }
  return fetch(url, { ...init, headers });
}

export default function EvidenceWorkbench() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isThoughtOpen, setIsThoughtOpen] = useState(true);
  const [streamId, setStreamId] = useState<string | null>(null);
  const lastEventId = useRef(0);
  const [error, setError] = useState<string | null>(null);
  const [provider, setProvider] = useState<Provider>("local");
  const [model, setModel] = useState("");
  const [documentProgress, setDocumentProgress] = useState<DocumentProgress | null>(null);
  const [isDesktop, setIsDesktop] = useState(false);
  const [documents, setDocuments] = useState<DocumentProgress[]>([]);
  const [showDesktopSettings, setShowDesktopSettings] = useState(false);
  const [configuredProviders, setConfiguredProviders] = useState<Record<string, boolean>>({});
  const [keyProvider, setKeyProvider] = useState<"openai" | "deepseek" | "claude">("openai");
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [settingsNotice, setSettingsNotice] = useState<string | null>(null);
  const [isBilingualSummaryEnabled, setIsBilingualSummaryEnabled] = useState(false);
  const [bilingualSummary, setBilingualSummary] = useState<BilingualSummaryResponse | null>(null);
  const [isSummaryPolling, setIsSummaryPolling] = useState(false);
  const [summaryLanguage, setSummaryLanguage] = useState<"both" | "zh" | "en">("both");
  const [isDragging, setIsDragging] = useState(false);
  const [paperQuery, setPaperQuery] = useState("");
  const [institutions, setInstitutions] = useState<Institution[]>([]);
  const [institutionId, setInstitutionId] = useState("");
  const [paperResults, setPaperResults] = useState<Paper[]>([]);
  const [isPaperSearching, setIsPaperSearching] = useState(false);
  const [paperSearchComplete, setPaperSearchComplete] = useState(false);
  const [paperError, setPaperError] = useState<string | null>(null);
  const [importingPdfUrl, setImportingPdfUrl] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const groupedCitations = useMemo(() => {
    return citations.reduce<Record<string, Citation[]>>((groups, citation) => {
      groups[citation.document_name] = [...(groups[citation.document_name] ?? []), citation];
      return groups;
    }, {});
  }, [citations]);

  useEffect(() => {
    void loadInstitutions();
    if (window.deepRagDesktop) {
      setIsDesktop(true);
      void loadDesktopDocuments();
      void loadDesktopSettings();
    }
  }, []);

  async function loadDesktopDocuments() {
    try {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/documents`);
      if (!response.ok) return;
      const saved = (await response.json()) as DocumentProgress[];
      setDocuments(saved);
      if (saved.length > 0) setDocumentProgress(saved[0]);
    } catch {
      setError("本机文档列表暂时无法读取。请重新打开应用。");
    }
  }

  async function loadDesktopSettings() {
    try {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/desktop/settings`);
      if (response.ok) {
        const payload = (await response.json()) as { configured_providers: Record<string, boolean> };
        setConfiguredProviders(payload.configured_providers);
      }
    } catch {
      setSettingsNotice("无法读取模型设置。");
    }
  }

  async function saveDesktopKey(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiKeyInput.trim()) return;
    try {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/desktop/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: keyProvider, api_key: apiKeyInput.trim() }),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as { configured_providers: Record<string, boolean> };
      setConfiguredProviders(payload.configured_providers);
      setApiKeyInput("");
      setSettingsNotice(`${keyProvider} 密钥已保存到 macOS 钥匙串。`);
    } catch {
      setSettingsNotice("密钥保存失败。请确认 macOS 钥匙串已解锁。");
    }
  }

  async function loadInstitutions() {
    try {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/papers/institutions`);
      if (!response.ok) return;
      setInstitutions((await response.json()) as Institution[]);
    } catch {
      // Paper discovery remains useful without a configured institutional connector.
    }
  }

  async function handleUpload(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("请选择 PDF 文件。系统会保留原始页码作为引用依据。");
      return;
    }
    setError(null);
    setBilingualSummary(null);
    setIsBilingualSummaryEnabled(false);
    setDocumentProgress({
      id: "pending",
      original_filename: file.name,
      status: "uploading",
      page_count: null,
      chunk_count: 0,
      error_message: null,
    });
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/documents`, { method: "POST", body });
      if (!response.ok) throw new Error(await response.text());
      const uploaded = (await response.json()) as { id: string; status: string };
      await pollDocument(uploaded.id, file.name);
    } catch {
      setError("上传未完成。请确认后端服务正在运行后再试。");
      setDocumentProgress(null);
    }
  }

  async function pollDocument(id: string, fallbackName: string) {
    for (let attempt = 0; attempt < 90; attempt += 1) {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/documents/${id}`);
      if (!response.ok) throw new Error("Unable to obtain ingestion progress");
      const document = (await response.json()) as DocumentProgress;
      setDocumentProgress({ ...document, original_filename: document.original_filename || fallbackName });
      if (document.status === "completed" || document.status === "failed") {
        if (window.deepRagDesktop) void loadDesktopDocuments();
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }
    setError("文档仍在处理中。稍后刷新可继续查看摄取状态。");
  }

  async function requestBilingualSummary(regenerate: boolean) {
    if (!documentProgress || documentProgress.status !== "completed" || isSummaryPolling) return;
    setError(null);
    setIsSummaryPolling(true);
    try {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/documents/${documentProgress.id}/bilingual-summary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ route: { provider, model: model || null }, regenerate }),
      });
      if (!response.ok) throw new Error(await response.text());
      const created = (await response.json()) as BilingualSummaryResponse;
      setBilingualSummary(created);
      if (created.status === "pending" || created.status === "summarizing") await pollBilingualSummary(documentProgress.id);
    } catch {
      setError("双语总结任务未能启动。请确认后端、任务队列和所选模型配置均可用后重试。");
    } finally {
      setIsSummaryPolling(false);
    }
  }

  async function pollBilingualSummary(documentId: string) {
    for (let attempt = 0; attempt < 120; attempt += 1) {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/documents/${documentId}/bilingual-summary`);
      if (!response.ok) throw new Error("Unable to obtain bilingual summary progress");
      const current = (await response.json()) as BilingualSummaryResponse;
      setBilingualSummary(current);
      if (current.status === "completed" || current.status === "failed") return;
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }
    setError("双语总结仍在生成中。稍后可再次开启此面板查看进度。");
  }

  function onFileInput(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) void handleUpload(file);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void handleUpload(file);
  }

  async function searchPapers(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const query = paperQuery.trim();
    if (query.length < 2 || isPaperSearching) return;
    setPaperError(null);
    setPaperResults([]);
    setPaperSearchComplete(false);
    setIsPaperSearching(true);
    try {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/papers/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 6, institution_id: institutionId || null }),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as PaperSearchResponse;
      setPaperResults(payload.results);
      setPaperSearchComplete(true);
    } catch {
      setPaperError("论文检索暂时不可用。请检查网络后重试，或直接输入 DOI。\n");
    } finally {
      setIsPaperSearching(false);
    }
  }

  async function importOpenAccessPdf(paper: Paper) {
    if (!paper.open_access_pdf_url || importingPdfUrl) return;
    setPaperError(null);
    setError(null);
    setBilingualSummary(null);
    setIsBilingualSummaryEnabled(false);
    setImportingPdfUrl(paper.open_access_pdf_url);
    setDocumentProgress({
      id: "pending",
      original_filename: `${paper.title}.pdf`,
      status: "downloading",
      page_count: null,
      chunk_count: 0,
      error_message: null,
    });
    try {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/papers/import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pdf_url: paper.open_access_pdf_url, title: paper.title }),
      });
      if (!response.ok) throw new Error(await response.text());
      const uploaded = (await response.json()) as { id: string; status: string };
      await pollDocument(uploaded.id, `${paper.title}.pdf`);
    } catch {
      setError("开放 PDF 未能自动导入。可打开开放版本并下载后，再拖拽到导入区。");
      setDocumentProgress(null);
    } finally {
      setImportingPdfUrl(null);
    }
  }

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim() || isStreaming) return;
    setAnswer("");
    setThoughts([]);
    setCitations([]);
    setSelectedCitation(null);
    setError(null);
    setIsStreaming(true);
    lastEventId.current = 0;
    try {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          route: { provider, model: model || null },
          use_semantic_cache: true,
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      const nextStreamId = response.headers.get("X-Stream-ID");
      setStreamId(nextStreamId);
      await consumeSSE(response, applyFrame);
    } catch {
      setError("连接在生成过程中中断。可以使用“续接流”重放已生成内容。");
    } finally {
      setIsStreaming(false);
    }
  }

  async function resumeStream() {
    if (!streamId || isStreaming) return;
    setError(null);
    setIsStreaming(true);
    try {
      const response = await apiFetch(`${API_BASE_URL}/api/v1/chat/stream/${streamId}`, {
        headers: { "Last-Event-ID": String(lastEventId.current) },
      });
      if (!response.ok) throw new Error(await response.text());
      await consumeSSE(response, applyFrame);
    } catch {
      setError("该流已过期，或暂时无法从服务端重放。");
    } finally {
      setIsStreaming(false);
    }
  }

  function applyFrame(frame: SSEFrame) {
    lastEventId.current = frame.id;
    if (frame.event === "thought") {
      setThoughts((current) => [...current, frame.data as Thought]);
    } else if (frame.event === "citation") {
      const citation = frame.data as Citation;
      setCitations((current) => (current.some((item) => item.chunk_id === citation.chunk_id) ? current : [...current, citation]));
    } else if (frame.event === "delta") {
      setAnswer((current) => current + (frame.data as { text: string }).text);
    } else {
      setError((frame.data as { message: string }).message);
    }
  }

  return (
    <main className="mx-auto min-h-screen max-w-[1600px] px-4 py-5 sm:px-8 lg:px-10">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b border-ink/15 pb-5">
        <div className="flex items-center gap-4">
          <div className="grid h-11 w-11 place-items-center border-2 border-ink bg-paper text-lg font-semibold text-ink">DR</div>
          <div>
            <p className="text-xs font-semibold tracking-[0.2em] text-verify">EVIDENCE WORKBENCH</p>
            <h1 className="evidence-serif text-2xl font-semibold tracking-tight text-ink">Deep-RAG 学术文档知识库</h1>
          </div>
        </div>
        <div className="flex items-center gap-3 text-sm text-ink/70">
          {isDesktop && <span className="border border-verify/30 bg-verify/5 px-2 py-1 text-xs font-semibold text-verify">本机版</span>}
          <span className="inline-flex items-center gap-2"><i className="h-2 w-2 rounded-full bg-verify" />引文约束已启用</span>
          {documentProgress?.status === "completed" && <span>{documentProgress.chunk_count} 个可检索片段</span>}
          {isDesktop && <button type="button" onClick={() => setShowDesktopSettings((current) => !current)} className="border border-ink/20 bg-white px-3 py-1.5 text-xs font-semibold text-ink hover:border-verify">模型设置</button>}
        </div>
      </header>

      {isDesktop && showDesktopSettings && <section className="mb-6 border border-ink/20 bg-paper p-5 shadow-ledger" aria-labelledby="desktop-settings-heading">
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3"><div><h2 id="desktop-settings-heading" className="text-sm font-semibold text-ink">模型设置</h2><p className="mt-1 text-xs leading-5 text-ink/60">PDF 和索引保存在这台 Mac。若要使用云端模型，在这里保存对应提供方的密钥。</p></div><button type="button" onClick={() => setShowDesktopSettings(false)} className="text-xs font-semibold text-verify underline underline-offset-4">收起</button></div>
        <form onSubmit={saveDesktopKey} className="flex flex-wrap items-end gap-2">
          <label className="grid gap-1 text-xs font-semibold text-ink/65">提供方<select value={keyProvider} onChange={(event) => { setKeyProvider(event.target.value as typeof keyProvider); setSettingsNotice(null); }} className="border border-ink/20 bg-white px-3 py-2 text-sm text-ink"><option value="openai">OpenAI</option><option value="deepseek">DeepSeek</option><option value="claude">Claude</option></select></label>
          <label className="grid min-w-[220px] flex-1 gap-1 text-xs font-semibold text-ink/65">API 密钥<input type="password" value={apiKeyInput} onChange={(event) => setApiKeyInput(event.target.value)} placeholder={configuredProviders[keyProvider] ? "已保存；输入新密钥可替换" : "输入密钥"} autoComplete="new-password" className="border border-ink/20 bg-white px-3 py-2 text-sm text-ink" /></label>
          <button type="submit" disabled={apiKeyInput.trim().length < 8} className="bg-ink px-4 py-2 text-sm font-semibold text-white disabled:bg-ink/35">保存到钥匙串</button>
        </form>
        {settingsNotice && <p role="status" className="mt-3 text-xs text-verify">{settingsNotice}</p>}
      </section>}

      {isDesktop && <section className="mb-6 border border-ink/20 bg-paper px-5 py-4" aria-labelledby="desktop-library-heading"><div className="mb-3 flex flex-wrap items-baseline justify-between gap-2"><h2 id="desktop-library-heading" className="text-sm font-semibold text-ink">本机文档</h2><span className="text-xs text-ink/55">{documents.length} 篇，关闭应用后仍会保留</span></div>{documents.length === 0 ? <p className="text-sm text-ink/60">尚无文档。可在下方搜索开放论文，或导入自己的 PDF。</p> : <div className="flex flex-wrap gap-2">{documents.map((document) => <button key={document.id} type="button" onClick={() => { setDocumentProgress(document); setBilingualSummary(null); setIsBilingualSummaryEnabled(false); }} className={`max-w-[330px] truncate border px-3 py-2 text-left text-xs ${documentProgress?.id === document.id ? "border-verify bg-verify/10 text-verify" : "border-ink/20 text-ink hover:border-verify"}`} title={document.original_filename}>{document.original_filename}<span className="ml-2 text-ink/45">{document.status === "completed" ? `${document.page_count ?? 0} 页` : document.status}</span></button>)}</div>}</section>}

      <section className="mb-6 border border-ink/20 bg-paper shadow-ledger" aria-labelledby="paper-discovery-heading">
        <div className="grid gap-5 p-5 lg:grid-cols-[minmax(0,1fr)_300px] lg:items-end">
          <div>
            <p className="mb-1 text-sm font-semibold text-ink" id="paper-discovery-heading">论文发现</p>
            <p className="max-w-2xl text-sm leading-6 text-ink/60">输入关键词、论文题目或 DOI。可将开放 PDF 直接入库；学校检索只会跳转至官方页面，不会读取校园账号。</p>
          </div>
          <p className="border-l-2 border-verify/50 pl-3 text-xs leading-5 text-ink/60">DOI 精确解析 · OpenAlex + Crossref 元数据 · 学校访问可选</p>
        </div>
        <form onSubmit={searchPapers} className="grid gap-2 border-t border-ink/15 p-4 sm:grid-cols-[minmax(0,1fr)_auto_auto] sm:p-5">
          <label className="sr-only" htmlFor="paper-query">论文关键词、题目或 DOI</label>
          <input
            id="paper-query"
            value={paperQuery}
            onChange={(event) => setPaperQuery(event.target.value)}
            placeholder="例如：retrieval augmented generation，或 10.1038/s41586-023-06221-2"
            className="min-w-0 border border-ink/20 bg-white px-3 py-2.5 text-sm text-ink placeholder:text-ink/40"
          />
          <select aria-label="可选学校图书馆" value={institutionId} onChange={(event) => setInstitutionId(event.target.value)} className="border border-ink/20 bg-white px-3 py-2.5 text-sm text-ink">
            <option value="">不使用学校图书馆</option>
            {institutions.map((institution) => <option key={institution.id} value={institution.id}>{institution.name}</option>)}
          </select>
          <button type="submit" disabled={paperQuery.trim().length < 2 || isPaperSearching} className="bg-verify px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-ink disabled:cursor-not-allowed disabled:bg-verify/40">
            {isPaperSearching ? "正在检索…" : "搜索论文"}
          </button>
        </form>
        {paperError && <p role="alert" className="border-t border-marker/30 bg-marker/10 px-5 py-3 text-sm text-ink">{paperError}</p>}
        {paperSearchComplete && (
          <div className="border-t border-ink/15" aria-live="polite">
            {paperResults.length === 0 ? (
              <p className="px-5 py-7 text-sm leading-6 text-ink/60">未找到匹配记录。尝试更完整的题名、作者和年份，或直接输入 DOI。</p>
            ) : (
              <ol className="divide-y divide-ink/15">{paperResults.map((paper, index) => <PaperRecord key={`${paper.doi ?? paper.title}-${index}`} paper={paper} onImport={importOpenAccessPdf} isImporting={importingPdfUrl === paper.open_access_pdf_url} />)}</ol>
            )}
          </div>
        )}
      </section>

      <section className="mb-6 grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <form onSubmit={ask} className="border border-ink/20 bg-paper p-4 shadow-ledger sm:p-5">
          <label htmlFor="question" className="mb-2 block text-sm font-semibold text-ink">向证据库提问</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="例如：文档摄取过程中如何保留页码级来源信息？"
            className="min-h-24 w-full resize-y border border-ink/20 bg-white px-3 py-3 text-[15px] leading-6 text-ink placeholder:text-ink/40"
          />
          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-2">
              <select aria-label="模型提供方" value={provider} onChange={(event) => setProvider(event.target.value as Provider)} className="border border-ink/20 bg-white px-2 py-2 text-sm">
                <option value="local">本地验证模式</option>
                <option value="openai">OpenAI</option>
                <option value="deepseek">DeepSeek</option>
                <option value="claude">Claude</option>
              </select>
              <input aria-label="模型名称" value={model} onChange={(event) => setModel(event.target.value)} placeholder="可选：覆盖模型名称" className="w-44 border border-ink/20 bg-white px-2 py-2 text-sm" />
            </div>
            <button type="submit" disabled={!question.trim() || isStreaming} className="bg-ink px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-signal disabled:cursor-not-allowed disabled:bg-ink/35">
              {isStreaming ? "正在核对证据…" : "开始研读"}
            </button>
          </div>
        </form>

        <button
          type="button"
          onClick={() => fileInput.current?.click()}
          onDragEnter={() => setIsDragging(true)}
          onDragLeave={() => setIsDragging(false)}
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
          className={`min-h-40 border-2 border-dashed p-5 text-left transition-colors ${isDragging ? "border-signal bg-signal/5" : "border-ink/25 bg-fog/45 hover:border-verify"}`}
        >
          <span className="mb-5 block text-2xl text-verify">↥</span>
          <span className="block text-sm font-semibold text-ink">导入 PDF</span>
          <span className="mt-1 block text-sm leading-5 text-ink/65">拖放长文档，后台会解析页码、段落与语义切片。</span>
          <input ref={fileInput} type="file" accept="application/pdf,.pdf" onChange={onFileInput} className="hidden" />
        </button>
      </section>

      {documentProgress && <IngestionLedger document={documentProgress} />}
      {documentProgress?.status === "completed" && (
        <BilingualSummaryLedger
          enabled={isBilingualSummaryEnabled}
          onEnabledChange={setIsBilingualSummaryEnabled}
          summary={bilingualSummary}
          isPolling={isSummaryPolling}
          language={summaryLanguage}
          onLanguageChange={setSummaryLanguage}
          onGenerate={() => void requestBilingualSummary(Boolean(bilingualSummary))}
        />
      )}
      {error && <div className="mb-5 border-l-4 border-marker bg-marker/10 px-4 py-3 text-sm text-ink">{error}</div>}

      <section className="grid gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(300px,0.65fr)]">
        <article className="min-h-[570px] border border-ink/20 bg-paper shadow-ledger">
          <div className="flex items-center justify-between border-b border-ink/15 px-5 py-4">
            <div>
              <p className="text-sm font-semibold text-ink">研读记录</p>
              <p className="mt-0.5 text-xs text-ink/55">先展示检索依据，再呈现经引文约束的回答。</p>
            </div>
            {thoughts.some((thought) => thought.cache_hit) && <span className="border border-verify/40 bg-verify/10 px-2 py-1 text-xs font-semibold text-verify">CACHE HIT</span>}
          </div>

          <div className="p-5 sm:p-7">
            {thoughts.length > 0 && (
              <section className="mb-7 border-l-2 border-signal/55 pl-4">
                <button type="button" onClick={() => setIsThoughtOpen((open) => !open)} className="flex w-full items-center justify-between gap-3 text-left">
                  <span className="text-sm font-semibold text-ink">检索过程</span>
                  <span className="text-xs text-ink/55">{isThoughtOpen ? "收起" : "展开"}</span>
                </button>
                {isThoughtOpen && <ol className="mt-3 space-y-3">{thoughts.map((thought, index) => <li key={`${thought.stage}-${index}`} className="text-sm leading-6 text-ink/75"><span className="mr-2 font-semibold text-signal">{String(index + 1).padStart(2, "0")}</span>{thought.summary}{thought.subqueries?.map((query) => <span key={query} className="mt-1 block border-l border-ink/15 pl-3 text-xs text-ink/55">{query}</span>)}</li>)}</ol>}
              </section>
            )}

            {!answer && !isStreaming && <EmptyAnswer />}
            {(answer || isStreaming) && <div className={`evidence-serif max-w-[76ch] whitespace-pre-wrap text-[18px] leading-8 text-ink ${isStreaming ? "live-cursor" : ""}`}><CitationText text={answer} citations={citations} onSelect={setSelectedCitation} /></div>}
          </div>

          {streamId && <div className="border-t border-ink/15 px-5 py-3 text-right"><button type="button" onClick={() => void resumeStream()} disabled={isStreaming} className="text-xs font-semibold text-verify underline decoration-verify/35 underline-offset-4 disabled:text-ink/35">从事件 #{lastEventId.current} 续接流</button></div>}
        </article>

        <aside className="border border-ink/20 bg-fog/35 p-5">
          <div className="mb-5 border-b border-ink/15 pb-4">
            <p className="text-sm font-semibold text-ink">证据分布</p>
            <p className="mt-1 text-xs leading-5 text-ink/55">按原始文件与页码组织；选择引文即可查看对应片段。</p>
          </div>
          {Object.keys(groupedCitations).length === 0 ? <EmptyEvidence /> : <EvidenceMap groups={groupedCitations} selected={selectedCitation} onSelect={setSelectedCitation} />}
          {selectedCitation && <SelectedExcerpt citation={selectedCitation} />}
        </aside>
      </section>
    </main>
  );
}

function IngestionLedger({ document }: { document: DocumentProgress }) {
  const labels: Record<string, string> = { downloading: "正在导入开放 PDF", uploading: "正在上传", pending: "等待任务队列", parsing: "提取页码与段落", chunking: "递归切片", embedding: "写入语义向量", completed: "已可检索", failed: "摄取失败" };
  const complete = document.status === "completed";
  return <section className="mb-5 flex flex-wrap items-center justify-between gap-3 border border-ink/15 bg-paper px-4 py-3 text-sm"><div><span className="font-semibold text-ink">{document.original_filename}</span><span className="ml-3 text-ink/55">{labels[document.status] ?? document.status}</span></div><div className={complete ? "font-semibold text-verify" : "text-ink/60"}>{document.page_count ? `${document.page_count} 页 · ${document.chunk_count} 片段` : "页码信息准备中"}</div></section>;
}

function BilingualSummaryLedger({ enabled, onEnabledChange, summary, isPolling, language, onLanguageChange, onGenerate }: {
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  summary: BilingualSummaryResponse | null;
  isPolling: boolean;
  language: "both" | "zh" | "en";
  onLanguageChange: (language: "both" | "zh" | "en") => void;
  onGenerate: () => void;
}) {
  const inProgress = isPolling || summary?.status === "pending" || summary?.status === "summarizing";
  const data = summary?.summary;
  return <section className="mb-5 border border-ink/20 bg-paper shadow-ledger" aria-labelledby="bilingual-summary-heading">
    <div className="flex flex-wrap items-center justify-between gap-4 p-4 sm:p-5">
      <div>
        <div className="flex items-center gap-3"><h2 id="bilingual-summary-heading" className="text-sm font-semibold text-ink">双语论文总结</h2><span className="border border-verify/35 bg-verify/10 px-2 py-0.5 text-[11px] font-semibold text-verify">OPTIONAL</span></div>
        <p className="mt-1 text-xs leading-5 text-ink/60">默认关闭。勾选并点击生成后才会调用所选模型；所有保留内容均附原始 PDF 页码。</p>
      </div>
      <label className="inline-flex cursor-pointer items-center gap-2 text-sm font-semibold text-ink"><input type="checkbox" checked={enabled} onChange={(event) => onEnabledChange(event.target.checked)} className="h-4 w-4 accent-verify" />启用双语总结</label>
    </div>
    {enabled && <div className="border-t border-ink/15 p-4 sm:p-5">
      {!data && summary?.status !== "failed" && <div className="flex flex-wrap items-center justify-between gap-3"><p className="max-w-2xl text-sm leading-6 text-ink/70">当前没有自动任务。选择 OpenAI、DeepSeek 或 Claude 可生成中英对照；本地验证模式不会伪造翻译，只显示可核查的原文证据。</p><button type="button" disabled={inProgress} onClick={onGenerate} className="bg-verify px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-ink disabled:cursor-not-allowed disabled:bg-verify/45">{inProgress ? "正在整理证据…" : "生成双语总结"}</button></div>}
      {summary?.status === "failed" && <div className="flex flex-wrap items-center justify-between gap-3 border-l-4 border-marker bg-marker/10 px-4 py-3"><p className="text-sm text-ink">生成未完成：{summary.error_message || "未知错误"}</p><button type="button" onClick={onGenerate} className="text-sm font-semibold text-verify underline underline-offset-4">重试</button></div>}
      {data && <BilingualSummaryView data={data} language={language} onLanguageChange={onLanguageChange} isRefreshing={inProgress} onRegenerate={onGenerate} />}
    </div>}
  </section>;
}

function BilingualSummaryView({ data, language, onLanguageChange, isRefreshing, onRegenerate }: {
  data: BilingualSummaryData;
  language: "both" | "zh" | "en";
  onLanguageChange: (language: "both" | "zh" | "en") => void;
  isRefreshing: boolean;
  onRegenerate: () => void;
}) {
  const sections: Array<[string, SummaryClaim[]]> = [["核心发现", data.key_findings], ["方法", data.methods], ["局限", data.limitations]];
  return <div>
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div className="flex gap-2">{(["both", "zh", "en"] as const).map((option) => <button key={option} type="button" onClick={() => onLanguageChange(option)} className={`border px-2.5 py-1 text-xs font-semibold ${language === option ? "border-verify bg-verify text-white" : "border-ink/20 bg-white text-ink/65"}`}>{option === "both" ? "中英对照" : option === "zh" ? "仅中文" : "English"}</button>)}</div><button type="button" disabled={isRefreshing} onClick={onRegenerate} className="text-xs font-semibold text-verify underline decoration-verify/35 underline-offset-4 disabled:text-ink/35">{isRefreshing ? "正在更新…" : "使用当前模型重新生成"}</button></div>
    {data.translation_mode === "local_verification" && <p className="mb-4 border-l-4 border-signal bg-signal/10 px-4 py-3 text-sm leading-6 text-ink">本地验证模式未调用翻译模型。为避免将原文误写为译文，中文区域只显示说明；选择已配置的云端模型后可主动重新生成。</p>}
    <div className={`grid gap-4 ${language === "both" ? "lg:grid-cols-2" : "grid-cols-1"}`}>
      {language !== "zh" && <SummaryColumn language="English" content={data.english_summary} citations={data.overview_citations} />}
      {language !== "en" && <SummaryColumn language="中文" content={data.chinese_summary} citations={data.overview_citations} />}
    </div>
    {sections.filter(([, claims]) => claims.length > 0).map(([heading, claims]) => <section key={heading} className="mt-5 border-t border-ink/15 pt-4"><h3 className="mb-3 text-sm font-semibold text-ink">{heading}</h3><ol className="space-y-3">{claims.map((claim, index) => <li key={`${heading}-${index}`} className="border-l-2 border-verify/45 pl-3 text-sm leading-6 text-ink/80">{language !== "zh" && <p>{claim.english}</p>}{language !== "en" && <p className={language === "both" ? "mt-1 text-ink/65" : ""}>{claim.chinese}</p>}<SummaryCitationBadges citations={claim.citations} /></li>)}</ol></section>)}
  </div>;
}

function SummaryColumn({ language, content, citations }: { language: string; content: string; citations: SummaryCitation[] }) {
  return <article className="border border-ink/15 bg-fog/35 p-4"><p className="mb-2 text-xs font-semibold tracking-[0.12em] text-verify">{language.toUpperCase()} OVERVIEW</p><p className="whitespace-pre-wrap text-sm leading-6 text-ink/80">{content}</p><SummaryCitationBadges citations={citations} /></article>;
}

function SummaryCitationBadges({ citations }: { citations: SummaryCitation[] }) {
  return <div className="mt-3 flex flex-wrap gap-2">{citations.map((citation) => <span key={`${citation.chunk_id}-${citation.page_number}`} title={`来源片段 ${citation.chunk_id}`} className="border border-signal/35 bg-signal/10 px-2 py-0.5 text-[11px] font-semibold text-signal">证据 p. {citation.page_number}</span>)}</div>;
}

function PaperRecord({ paper, onImport, isImporting }: { paper: Paper; onImport: (paper: Paper) => void; isImporting: boolean }) {
  const access = paper.access_type === "open_access"
    ? { label: "开放版本", className: "border-verify/40 bg-verify/10 text-verify" }
    : paper.access_type === "institution_login"
      ? { label: "需学校登录", className: "border-signal/40 bg-signal/10 text-signal" }
      : { label: "仅元数据", className: "border-ink/20 bg-fog text-ink/65" };
  const authorLine = paper.authors.slice(0, 4).join("、") || "作者信息未提供";
  const primaryUrl = paper.open_access_url || paper.landing_page_url;
  return <li className="grid gap-4 px-5 py-5 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-start">
    <div className="min-w-0">
      <div className="mb-2 flex flex-wrap items-center gap-2"><span className={`border px-2 py-1 text-xs font-semibold ${access.className}`}>{access.label}</span>{paper.sources.map((source) => <span key={source} className="text-xs text-ink/50">{source}</span>)}</div>
      <h2 className="evidence-serif max-w-4xl text-lg font-semibold leading-6 text-ink">{paper.title}</h2>
      <p className="mt-2 text-sm text-ink/65">{authorLine}{paper.publication_year ? ` · ${paper.publication_year}` : ""}{paper.citation_count !== null ? ` · ${paper.citation_count.toLocaleString()} citations` : ""}</p>
      {paper.doi && <p className="mt-2 break-all text-xs text-ink/50">DOI {paper.doi}</p>}
      {paper.abstract && <p className="mt-3 line-clamp-3 max-w-4xl text-sm leading-6 text-ink/70">{paper.abstract}</p>}
    </div>
    <div className="flex shrink-0 flex-wrap gap-2 lg:max-w-48 lg:justify-end">
      {primaryUrl && <a href={primaryUrl} target="_blank" rel="noreferrer" className="border border-ink/20 bg-white px-3 py-2 text-xs font-semibold text-ink transition-colors hover:border-verify hover:text-verify">{paper.open_access_url ? "打开开放版本" : "查看出处"}</a>}
      {paper.open_access_pdf_url && <button type="button" onClick={() => void onImport(paper)} disabled={isImporting} className="border border-verify bg-verify px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-ink disabled:cursor-not-allowed disabled:bg-verify/45">{isImporting ? "正在导入…" : "导入开放 PDF"}</button>}
      {paper.institution_access_url && <a href={paper.institution_access_url} target="_blank" rel="noreferrer" className="border border-signal/35 bg-signal/5 px-3 py-2 text-xs font-semibold text-signal transition-colors hover:bg-signal hover:text-white">在 {paper.institution_name} 检索</a>}
    </div>
  </li>;
}

function EmptyAnswer() {
  return <div className="py-20 text-center"><p className="evidence-serif text-2xl text-ink/65">问题会从证据开始。</p><p className="mx-auto mt-3 max-w-md text-sm leading-6 text-ink/50">上传 PDF 后提出具体问题；系统将返回可展开到页码和原文片段的回答。</p></div>;
}

function EmptyEvidence() {
  return <div className="border border-dashed border-ink/20 px-4 py-8 text-center text-sm leading-6 text-ink/55">尚未生成证据地图。<br />答案中的每一条引用都会在这里留下来源轨迹。</div>;
}

function EvidenceMap({ groups, selected, onSelect }: { groups: Record<string, Citation[]>; selected: Citation | null; onSelect: (citation: Citation) => void }) {
  return <div className="space-y-5">{Object.entries(groups).map(([document, items]) => <section key={document}><div className="mb-2 flex items-baseline justify-between gap-3"><h2 className="max-w-[18ch] truncate text-sm font-semibold text-ink">{document}</h2><span className="text-xs text-ink/50">{items.length} 条依据</span></div><div className="flex flex-wrap gap-2">{items.map((citation) => <button key={citation.chunk_id} type="button" onClick={() => onSelect(citation)} className={`border px-2.5 py-1.5 text-xs transition-colors ${selected?.chunk_id === citation.chunk_id ? "border-signal bg-signal text-white" : "border-ink/20 bg-paper text-ink hover:border-verify"}`}>p. {citation.page_number}</button>)}</div></section>)}</div>;
}

function SelectedExcerpt({ citation }: { citation: Citation }) {
  return <section className="mt-7 border-t border-ink/15 pt-5"><p className="mb-2 text-xs font-semibold tracking-[0.12em] text-verify">SOURCE EXCERPT · PAGE {citation.page_number}</p><blockquote className="evidence-serif border-l-2 border-marker pl-3 text-sm leading-6 text-ink/80">{citation.excerpt}</blockquote><p className="mt-3 text-xs text-ink/50">RRF {citation.rrf_score.toFixed(4)} · rerank {citation.rerank_score.toFixed(3)}</p></section>;
}

function CitationText({ text, citations, onSelect }: { text: string; citations: Citation[]; onSelect: (citation: Citation) => void }) {
  const nodes: ReactNode[] = [];
  let cursor = 0;
  for (const match of text.matchAll(citationPattern)) {
    const [label, documentId, page] = match;
    const start = match.index ?? 0;
    if (start > cursor) nodes.push(<span key={`text-${cursor}`}>{text.slice(cursor, start)}</span>);
    const citation = citations.find((item) => item.document_id === documentId && item.page_number === Number(page));
    nodes.push(citation ? <button key={`citation-${start}`} type="button" onClick={() => onSelect(citation)} className="citation-link">{label}</button> : <span key={`citation-${start}`}>{label}</span>);
    cursor = start + label.length;
  }
  if (cursor < text.length) nodes.push(<span key={`text-${cursor}`}>{text.slice(cursor)}</span>);
  return <>{nodes}</>;
}

async function consumeSSE(response: Response, onFrame: (frame: SSEFrame) => void) {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Response body is unavailable");
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const rawFrame of frames) {
      const fields: Record<string, string> = Object.fromEntries(rawFrame.split("\n").filter(Boolean).map((line) => {
        const [key, ...rest] = line.split(": ");
        return [key, rest.join(": ")];
      }));
      if (fields.event && fields.data && fields.id) onFrame({ id: Number(fields.id), event: fields.event as SSEFrame["event"], data: JSON.parse(fields.data) });
    }
    if (done) return;
  }
}
