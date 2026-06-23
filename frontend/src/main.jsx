import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  BookOpen,
  Check,
  ChevronDown,
  ClipboardList,
  ExternalLink,
  FileText,
  ImagePlus,
  Loader2,
  MessageSquareText,
  RotateCcw,
  Search,
  Send,
  Settings2,
  Sparkles,
  X
} from "lucide-react";
import campusImage from "./assets/uob-aston-webb.jpg";
import uobLogo from "./assets/uob-logo.svg";
import "./styles.css";

const DEFAULT_SETTINGS = {
  llm_provider: "gemini",
  embedding_provider: "sentence_transformers",
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
  top_k: 5,
  use_reranker: true,
  use_llm: true
};

const DEMO_QUESTIONS = [
  "What temperature monitoring methods are used during thermal ablation?",
  "How is ultrasound thermometry used in HIFU treatment monitoring?",
  "What are the limitations of using thermal imaging or thermometry for ablation monitoring?",
  "Which evidence mentions spectral CT for in vivo thermometry during thermal ablation?"
];

const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const SUPPORTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];

function App() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      answer:
        "Hi. I can answer questions using the imported ultrasound and thermal ablation literature, with cited evidence shown on the right.",
      evidence: [],
      provider: "local",
      model: null,
      used_llm: false
    }
  ]);
  const [question, setQuestion] = useState(DEMO_QUESTIONS[0]);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [modelConfig, setModelConfig] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [health, setHealth] = useState(null);
  const [selectedMessageId, setSelectedMessageId] = useState("welcome");
  const [selectedEvidenceIndex, setSelectedEvidenceIndex] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isImageLoading, setIsImageLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedImagePreview, setSelectedImagePreview] = useState("");
  const [imageModality, setImageModality] = useState("thermal");
  const messagesEndRef = useRef(null);
  const imageInputRef = useRef(null);

  useEffect(() => {
    refreshRuntimeState();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isLoading]);

  const activeMessage = useMemo(() => {
    return messages.find((message) => message.id === selectedMessageId) || messages[messages.length - 1];
  }, [messages, selectedMessageId]);

  const corpusStats = useMemo(() => {
    const ragDocs = documents.filter(
      (document) => document.metadata?.source_tag === "rag_v1_ultrasound_heat_papers"
    );
    const docs = ragDocs.length || documents.length;
    const chunks = (ragDocs.length ? ragDocs : documents).reduce(
      (sum, document) => sum + Number(document.chunks_count || 0),
      0
    );
    return { docs, chunks };
  }, [documents]);

  async function refreshRuntimeState() {
    const [healthResult, configResult, docsResult] = await Promise.allSettled([
      getJson("/api/health"),
      getJson("/api/model-config"),
      getJson("/api/documents")
    ]);
    if (healthResult.status === "fulfilled") {
      setHealth(healthResult.value);
    }
    if (configResult.status === "fulfilled") {
      setModelConfig(configResult.value);
      setSettings((current) => ({
        ...current,
        llm_provider: configResult.value.llm_provider || current.llm_provider
      }));
    }
    if (docsResult.status === "fulfilled") {
      setDocuments(docsResult.value);
    }
  }

  async function askQuestion(nextQuestion = question) {
    const trimmed = nextQuestion.trim();
    if (!trimmed || isLoading) return;

    const attachedImage = selectedImage;
    const attachedImagePreview = selectedImagePreview;
    const attachedImageModality = imageModality;
    setError("");
    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      answer: trimmed,
      evidence: [],
      image: attachedImage
        ? {
            name: attachedImage.name,
            preview: attachedImagePreview,
            modality: attachedImageModality,
            size: attachedImage.size
          }
        : null
    };
    setMessages((current) => [...current, userMessage]);
    setSelectedMessageId(userMessage.id);
    setSelectedEvidenceIndex(null);
    setSelectedImage(null);
    setSelectedImagePreview("");
    if (imageInputRef.current) imageInputRef.current.value = "";
    setIsLoading(true);
    setIsImageLoading(Boolean(attachedImage));

    try {
      let body;
      if (attachedImage) {
        const formData = new FormData();
        formData.append("image", attachedImage);
        formData.append("question", trimmed);
        formData.append("image_modality", attachedImageModality);
        formData.append("top_k", String(Number(settings.top_k)));
        formData.append("llm_provider", settings.llm_provider);
        formData.append("embedding_provider", settings.embedding_provider);
        formData.append("embedding_model", settings.embedding_model);
        formData.append("use_reranker", String(settings.use_reranker));
        formData.append("use_llm", String(settings.use_llm));
        body = await postForm("/api/chat-with-image", formData);
      } else {
        body = await postJson("/api/chat", {
          question: trimmed,
          top_k: Number(settings.top_k),
          llm_provider: settings.llm_provider,
          embedding_provider: settings.embedding_provider,
          embedding_model: settings.embedding_model,
          use_reranker: settings.use_reranker,
          use_llm: settings.use_llm
        });
      }
      const assistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        answer: body.answer || "No answer returned.",
        evidence: body.evidence || [],
        provider: body.provider,
        model: body.model,
        used_llm: body.used_llm,
        safety_notice: body.safety_notice,
        visual_summary: body.visual_summary
      };
      setMessages((current) => [...current, assistantMessage]);
      setSelectedMessageId(assistantMessage.id);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      const failureMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        answer: `Request failed: ${message}`,
        evidence: [],
        provider: "local",
        used_llm: false,
        is_error: true
      };
      setMessages((current) => [...current, failureMessage]);
      setSelectedMessageId(failureMessage.id);
    } finally {
      setIsLoading(false);
      setIsImageLoading(false);
    }
  }

  function resetConversation() {
    messages.forEach((message) => {
      if (message.image?.preview) URL.revokeObjectURL(message.image.preview);
    });
    if (selectedImagePreview) URL.revokeObjectURL(selectedImagePreview);
    const welcome = {
      id: crypto.randomUUID(),
      role: "assistant",
      answer:
        "A new conversation has started. You can ask about HIFU, thermal ablation, ultrasound thermometry, or evidence comparison.",
      evidence: [],
      provider: "local",
      used_llm: false
    };
    setMessages([welcome]);
    setSelectedMessageId(welcome.id);
    setSelectedEvidenceIndex(null);
    setSelectedImage(null);
    setSelectedImagePreview("");
    setImageModality("thermal");
    if (imageInputRef.current) imageInputRef.current.value = "";
    setError("");
  }

  function chooseImage(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!SUPPORTED_IMAGE_TYPES.includes(file.type)) {
      setError("Supported image formats are PNG, JPEG, and WebP.");
      event.target.value = "";
      return;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      setError("Image must be no larger than 10 MB.");
      event.target.value = "";
      return;
    }
    if (selectedImagePreview) URL.revokeObjectURL(selectedImagePreview);
    setSelectedImage(file);
    setSelectedImagePreview(URL.createObjectURL(file));
    setError("");
  }

  function removeSelectedImage() {
    if (selectedImagePreview) URL.revokeObjectURL(selectedImagePreview);
    setSelectedImage(null);
    setSelectedImagePreview("");
    if (imageInputRef.current) imageInputRef.current.value = "";
  }

  return (
    <div className="app-shell" style={{ "--campus-image": `url(${campusImage})` }}>
      <aside className="sidebar">
        <div className="brand-block">
          <img className="uob-logo" src={uobLogo} alt="University of Birmingham" />
          <div className="project-title">
            <span>MSc Final Project Demonstrator</span>
            <h1>Ultrasound Thermal RAG</h1>
            <p>Evidence-grounded research assistant for thermal ablation literature.</p>
          </div>
        </div>

        <div className="campus-panel">
          <img src={campusImage} alt="Aston Webb Building on the University of Birmingham campus" />
          <div>
            <span>Original redbrick university</span>
            <strong>Aston Webb inspired RAG demo</strong>
          </div>
        </div>

        <section className="side-section corpus-panel">
          <div className="section-title">
            <BookOpen size={17} />
            <span>Corpus</span>
          </div>
          <div className="metric-grid">
            <Metric label="papers" value={formatNumber(corpusStats.docs)} />
            <Metric label="chunks" value={formatNumber(corpusStats.chunks)} />
          </div>
          <StatusLine health={health} modelConfig={modelConfig} />
        </section>

        <section className="side-section">
          <div className="section-title">
            <Settings2 size={17} />
            <span>Retrieval</span>
          </div>
          <label className="field-label" htmlFor="embedding-provider">
            Embedding
          </label>
          <div className="select-wrap">
            <select
              id="embedding-provider"
              value={settings.embedding_provider}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  embedding_provider: event.target.value,
                  embedding_model:
                    modelConfig?.embedding_model_options?.[event.target.value]?.[0] || current.embedding_model
                }))
              }
            >
              <option value="sentence_transformers">SentenceTransformers</option>
              <option value="gemini">Gemini embeddings</option>
              <option value="openai">OpenAI embeddings</option>
              <option value="qwen">Qwen embeddings</option>
              <option value="local">Local hash</option>
            </select>
            <ChevronDown size={16} />
          </div>

          <label className="field-label" htmlFor="embedding-model">
            Embedding model
          </label>
          <input
            id="embedding-model"
            value={settings.embedding_model}
            onChange={(event) =>
              setSettings((current) => ({ ...current, embedding_model: event.target.value }))
            }
          />

          <div className="control-row">
            <label className="field-label" htmlFor="top-k">
              Top K
            </label>
            <input
              id="top-k"
              className="small-input"
              type="number"
              min="1"
              max="20"
              value={settings.top_k}
              onChange={(event) =>
                setSettings((current) => ({ ...current, top_k: event.target.value }))
              }
            />
          </div>

          <Toggle
            label="CrossEncoder reranker"
            checked={settings.use_reranker}
            onChange={(checked) => setSettings((current) => ({ ...current, use_reranker: checked }))}
          />
          <Toggle
            label="LLM answer generation"
            checked={settings.use_llm}
            onChange={(checked) => setSettings((current) => ({ ...current, use_llm: checked }))}
          />
        </section>

        <section className="side-section">
          <div className="section-title">
            <ClipboardList size={17} />
            <span>Demo Questions</span>
          </div>
          <div className="preset-list">
            {DEMO_QUESTIONS.map((item) => (
              <button
                className="preset-button"
                type="button"
                key={item}
                onClick={() => setQuestion(item)}
              >
                {item}
              </button>
            ))}
          </div>
        </section>
      </aside>

      <main className="chat-column">
        <header className="chat-header">
          <div>
            <p className="eyebrow">University of Birmingham final project</p>
            <h2>Scientific RAG Assistant</h2>
          </div>
          <div className="header-actions">
            <a className="icon-link" href="/debug" title="Open debug UI">
              <Search size={17} />
              <span>Debug</span>
            </a>
            <button className="ghost-button" type="button" onClick={resetConversation}>
              <RotateCcw size={17} />
              <span>Reset</span>
            </button>
          </div>
        </header>

        {error ? (
          <div className="error-banner">
            <AlertTriangle size={18} />
            <span>{error}</span>
            <button type="button" onClick={() => setError("")} aria-label="Dismiss error">
              <X size={16} />
            </button>
          </div>
        ) : null}

        <section className="messages-panel">
          {messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              isSelected={message.id === selectedMessageId}
              onSelect={() => {
                setSelectedMessageId(message.id);
                setSelectedEvidenceIndex(null);
              }}
              onCitationClick={(index) => {
                setSelectedMessageId(message.id);
                setSelectedEvidenceIndex(index);
              }}
            />
          ))}
          {isLoading ? <ThinkingBubble isImage={isImageLoading} /> : null}
          <div ref={messagesEndRef} />
        </section>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            askQuestion();
          }}
        >
          {selectedImage ? (
            <div className="image-attachment">
              <img src={selectedImagePreview} alt="Selected upload preview" />
              <div className="image-attachment-details">
                <strong>{selectedImage.name}</strong>
                <span>{formatFileSize(selectedImage.size)}</span>
              </div>
              <label className="image-modality-label" htmlFor="image-modality">
                Image type
                <select
                  id="image-modality"
                  value={imageModality}
                  onChange={(event) => setImageModality(event.target.value)}
                >
                  <option value="thermal">Thermal image</option>
                  <option value="ultrasound">Ultrasound image</option>
                  <option value="scientific_figure">Scientific figure</option>
                  <option value="unknown">Other image</option>
                </select>
              </label>
              <button
                className="remove-image-button"
                type="button"
                onClick={removeSelectedImage}
                aria-label="Remove selected image"
                title="Remove selected image"
              >
                <X size={17} />
              </button>
            </div>
          ) : null}
          <div className="composer-main">
            <input
              ref={imageInputRef}
              className="visually-hidden"
              type="file"
              accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
              onChange={chooseImage}
            />
            <button
              className="attach-image-button"
              type="button"
              onClick={() => imageInputRef.current?.click()}
              aria-label="Attach an image"
              title="Attach an image"
              disabled={isLoading}
            >
              <ImagePlus size={20} />
            </button>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={
                selectedImage
                  ? "Ask a question about the uploaded image and the scientific literature..."
                  : "Ask a question about HIFU, thermometry, thermal ablation, or evidence limitations..."
              }
              rows={3}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  askQuestion();
                }
              }}
            />
            <button className="send-button" type="submit" disabled={isLoading || !question.trim()}>
              {isLoading ? <Loader2 className="spin" size={19} /> : <Send size={19} />}
              <span>Ask</span>
            </button>
          </div>
        </form>
      </main>

      <EvidencePanel
        message={activeMessage}
        selectedIndex={selectedEvidenceIndex}
        onSelectEvidence={setSelectedEvidenceIndex}
      />
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function StatusLine({ health, modelConfig }) {
  const configured = health?.configured_providers || {};
  const llmReady = configured[modelConfig?.llm_provider || "gemini"];
  const stReady = configured.sentence_transformers;
  return (
    <div className="status-stack">
      <span className={llmReady ? "status-chip ready" : "status-chip muted"}>
        {llmReady ? <Check size={14} /> : <AlertTriangle size={14} />}
        LLM {llmReady ? "ready" : "missing"}
      </span>
      <span className={stReady ? "status-chip ready" : "status-chip muted"}>
        {stReady ? <Check size={14} /> : <AlertTriangle size={14} />}
        ST {stReady ? "ready" : "missing"}
      </span>
    </div>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="toggle-row">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="toggle-track" aria-hidden="true">
        <span className="toggle-thumb" />
      </span>
    </label>
  );
}

function MessageBubble({ message, isSelected, onSelect, onCitationClick }) {
  const isAssistant = message.role === "assistant";
  return (
    <article
      className={`message-bubble ${message.role} ${isSelected ? "selected" : ""} ${
        message.is_error ? "error" : ""
      }`}
      onClick={onSelect}
    >
      <div className="message-meta">
        <span className="avatar">{isAssistant ? <Sparkles size={16} /> : <MessageSquareText size={16} />}</span>
        <span>{isAssistant ? "Assistant" : "You"}</span>
        {isAssistant && message.provider ? (
          <span className="model-pill">
            {message.provider}
            {message.model ? ` / ${message.model}` : ""}
          </span>
        ) : null}
      </div>
      <div className="message-content">
        {message.image ? (
          <div className="message-image">
            <img src={message.image.preview} alt={message.image.name} />
            <div>
              <strong>{message.image.name}</strong>
              <span>{imageModalityLabel(message.image.modality)}</span>
            </div>
          </div>
        ) : null}
        <AnswerText text={message.answer} onCitationClick={onCitationClick} />
      </div>
      {isAssistant && message.visual_summary ? (
        <details className="visual-summary">
          <summary>Image analysis used for retrieval</summary>
          <p>{message.visual_summary}</p>
        </details>
      ) : null}
      {message.evidence?.length ? (
        <div className="citation-row">
          {message.evidence.map((item, index) => (
            <button
              className="citation-button"
              type="button"
              key={`${item.chunk_id}-${index}`}
              onClick={(event) => {
                event.stopPropagation();
                onCitationClick(index);
              }}
            >
              [{index + 1}]
            </button>
          ))}
        </div>
      ) : null}
    </article>
  );
}

function AnswerText({ text, onCitationClick }) {
  const lines = String(text || "").split("\n");
  return lines.map((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) return <div className="answer-space" key={index} />;
    if (/^#{1,3}\s+/.test(trimmed)) {
      return (
        <h3 key={index} className="answer-heading">
          {renderInline(trimmed.replace(/^#{1,3}\s+/, ""), onCitationClick)}
        </h3>
      );
    }
    if (/^[-*]\s+/.test(trimmed)) {
      return (
        <p key={index} className="answer-list">
          {renderInline(trimmed.replace(/^[-*]\s+/, ""), onCitationClick)}
        </p>
      );
    }
    return <p key={index}>{renderInline(trimmed, onCitationClick)}</p>;
  });
}

function renderInline(text, onCitationClick) {
  const parts = String(text).split(/(\[\d+\])/g);
  return parts.map((part, index) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) return <React.Fragment key={`${part}-${index}`}>{part}</React.Fragment>;
    const evidenceIndex = Number(match[1]) - 1;
    return (
      <button
        className="inline-citation"
        key={`${part}-${index}`}
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          onCitationClick(evidenceIndex);
        }}
      >
        {part}
      </button>
    );
  });
}

function ThinkingBubble({ isImage }) {
  return (
    <article className="message-bubble assistant thinking">
      <div className="message-meta">
        <span className="avatar">
          <Sparkles size={16} />
        </span>
        <span>Assistant</span>
      </div>
      <div className="thinking-line">
        <Loader2 className="spin" size={18} />
        <span>
          {isImage
            ? "Analysing image, retrieving evidence, and generating answer..."
            : "Retrieving evidence and generating answer..."}
        </span>
      </div>
    </article>
  );
}

function EvidencePanel({ message, selectedIndex, onSelectEvidence }) {
  const evidence = message?.evidence || [];
  return (
    <aside className="evidence-column">
      <div className="evidence-header">
        <div>
          <p className="eyebrow">Retrieved evidence</p>
          <h2>{evidence.length ? `${evidence.length} sources` : "No evidence"}</h2>
        </div>
        <FileText size={22} />
      </div>

      <div className="evidence-list">
        {evidence.length ? (
          evidence.map((item, index) => (
            <EvidenceCard
              key={`${item.chunk_id}-${index}`}
              item={item}
              index={index}
              selected={selectedIndex === index}
              onClick={() => onSelectEvidence(index)}
            />
          ))
        ) : (
          <div className="empty-evidence">
            <BookOpen size={24} />
            <span>Select an answer with citations.</span>
          </div>
        )}
      </div>
    </aside>
  );
}

function EvidenceCard({ item, index, selected, onClick }) {
  const meta = item.metadata || {};
  const doi = asText(meta.doi);
  const pages = citationPages(meta);
  const method = asText(meta.retrieval_method || "retrieval");
  const chunkId = asText(meta.source_chunk_id || item.chunk_id);
  const section = asText(meta.section || "unknown section");
  const year = asText(meta.year || "n.d.");
  const sourceUrl = doi && doi !== "no DOI" ? `https://doi.org/${doi}` : item.source_path;

  return (
    <article className={`evidence-card ${selected ? "active" : ""}`} onClick={onClick}>
      <div className="evidence-topline">
        <span className="source-index">[{index + 1}]</span>
        <span className="score">{Number(item.score).toFixed(3)}</span>
      </div>
      <h3>{item.title}</h3>
      <div className="evidence-facts">
        <span>{year}</span>
        <span>{pages}</span>
        <span>{section}</span>
      </div>
      <p>{compact(item.content, 360)}</p>
      <div className="evidence-footer">
        <span className="chunk-id">{chunkId}</span>
        {sourceUrl ? (
          <a href={sourceUrl} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}>
            DOI
            <ExternalLink size={13} />
          </a>
        ) : null}
      </div>
      <div className="method-pill">{method}</div>
    </article>
  );
}

async function getJson(url) {
  const response = await fetch(url);
  const body = await readBody(response);
  if (!response.ok) throw new Error(formatApiError(body, response.status));
  return body;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const body = await readBody(response);
  if (!response.ok) throw new Error(formatApiError(body, response.status));
  return body;
}

async function postForm(url, formData) {
  const response = await fetch(url, {
    method: "POST",
    body: formData
  });
  const body = await readBody(response);
  if (!response.ok) throw new Error(formatApiError(body, response.status));
  return body;
}

async function readBody(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch (_error) {
    return { detail: text };
  }
}

function formatApiError(body, status) {
  if (typeof body?.detail === "string") return body.detail;
  return `HTTP ${status}`;
}

function citationPages(metadata) {
  const start = metadata.page_start;
  const end = metadata.page_end;
  const hasStart = start !== undefined && start !== null && start !== "";
  const hasEnd = end !== undefined && end !== null && end !== "";
  if (!hasStart && !hasEnd) return "page unknown";
  if (hasStart && hasEnd && String(start) !== String(end)) return `pp. ${start}-${end}`;
  return `p. ${hasStart ? start : end}`;
}

function compact(value, maxLength) {
  const text = asText(value).replace(/\s+/g, " ").trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength).trim()}...`;
}

function asText(value) {
  if (value === undefined || value === null || value === "") return "";
  return String(value);
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(Number(value || 0));
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function imageModalityLabel(value) {
  if (value === "thermal") return "Thermal image";
  if (value === "ultrasound") return "Ultrasound image";
  if (value === "scientific_figure") return "Scientific figure";
  return "Other image";
}

createRoot(document.getElementById("root")).render(<App />);
