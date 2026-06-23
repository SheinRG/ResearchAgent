"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "motion/react";
import useToast from "@/stores/toastStore";
import { useAuth } from "@/hooks/useAuth";
import {
  ArrowUpIcon,
  PlusIcon,
  MicIcon,
  FileTextIcon,
  CloseIcon,
} from "@/components/Icons";

const PLACEHOLDERS = [
  "Ask anything — I'll research it and cite the sources",
  "What are the latest breakthroughs in fusion energy?",
  "How does CRISPR base editing actually work?",
  "Why is the AI chip supply chain so concentrated?",
];

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * The product's composer. `mode="large"` is the two-row home box (with example
 * chips rendered by the page); `mode="compact"` is the single-row sticky
 * follow-up bar. When `withTools` is set it shows the attach + dictate
 * affordances from the design. File attachments are uploaded at submit time:
 * extracted text is sent to the backend as a separate `documents` field (not
 * appended to the query string). Dictation uses the Web Speech API when available.
 */
export default function SearchBar({
  onSearch,
  mode = "large",
  disabled = false,
  placeholder = null,
  clearOnSubmit = false,
  withTools = true,
}) {
  const [query, setQuery] = useState("");
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [attachments, setAttachments] = useState([]);
  const [listening, setListening] = useState(false);
  const [uploading, setUploading] = useState(false);

  const fileRef = useRef(null);
  const recRef = useRef(null);
  const stopTimerRef = useRef(null);
  const showToast = useToast((s) => s.show);
  const { token } = useAuth();

  const isLarge = mode === "large";
  const isDisabled = disabled || uploading;

  // Rotate placeholder text — only when no fixed placeholder is supplied.
  useEffect(() => {
    if (placeholder) return;
    const interval = setInterval(() => {
      setPlaceholderIdx((prev) => (prev + 1) % PLACEHOLDERS.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [placeholder]);

  // Tidy up any in-flight dictation on unmount.
  useEffect(() => {
    return () => {
      if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
      try {
        recRef.current?.stop();
      } catch {
        /* ignore */
      }
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || isDisabled) return;

    if (attachments.length === 0) {
      onSearch(trimmed, []);
      if (clearOnSubmit) {
        setQuery("");
        setAttachments([]);
      }
      return;
    }

    // Upload all attachments and collect extracted text as structured documents.
    // The full returned text is forwarded — the backend decides how much to use.
    setUploading(true);
    const documents = [];

    try {
      for (const attachment of attachments) {
        try {
          const formData = new FormData();
          formData.append("file", attachment.file);

          const res = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            headers: {
              Authorization: `Bearer ${token}`,
            },
            body: formData,
          });

          if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            showToast(`Failed to upload ${attachment.name}: ${err.detail || res.statusText}`);
            continue;
          }

          const data = await res.json();
          documents.push({
            name: attachment.name,
            text: data.text,
            file: attachment.file,
            file_id: data.file_id || "",
            mime: data.mime || attachment.file?.type || "",
            size: data.size || attachment.file?.size || 0,
          });
        } catch {
          showToast(`Could not upload ${attachment.name} — skipping`);
        }
      }
    } finally {
      setUploading(false);
    }

    // Pass query as-is; documents travel as a separate argument so the backend
    // can handle them independently rather than as inline query context.
    onSearch(trimmed, documents);
    if (clearOnSubmit) {
      setQuery("");
      setAttachments([]);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  const handleFiles = (e) => {
    // Store actual File objects alongside display metadata
    const files = [...(e.target.files || [])].map((f) => ({
      id: `f${Date.now()}${Math.random().toString(36).slice(2, 6)}`,
      name: f.name,
      file: f,
    }));
    e.target.value = "";
    if (!files.length) return;
    setAttachments((prev) => [...prev, ...files]);
    showToast(
      files.length === 1 ? `Attached ${files[0].name}` : `Attached ${files.length} files`
    );
  };

  const removeAttachment = (id) =>
    setAttachments((prev) => prev.filter((a) => a.id !== id));

  const stopDictation = () => {
    if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
    try {
      recRef.current?.stop();
    } catch {
      /* ignore */
    }
    recRef.current = null;
    setListening(false);
  };

  const toggleDictation = () => {
    if (listening) {
      stopDictation();
      return;
    }
    setListening(true);
    showToast("Listening…");
    // Visual stays on regardless of mic permission; auto-stops after a few seconds.
    stopTimerRef.current = setTimeout(stopDictation, 6000);
    const SR =
      typeof window !== "undefined" &&
      (window.SpeechRecognition || window.webkitSpeechRecognition);
    if (SR) {
      try {
        const rec = new SR();
        rec.lang = "en-US";
        rec.interimResults = true;
        rec.continuous = true;
        rec.onresult = (ev) => {
          const text = [...ev.results].map((r) => r[0].transcript).join("");
          setQuery(text);
        };
        rec.onend = () => setListening(false);
        rec.start();
        recRef.current = rec;
      } catch {
        /* ignore — visual-only fallback */
      }
    }
  };

  const tools = withTools ? (
    <>
      <input
        ref={fileRef}
        type="file"
        multiple
        accept=".txt,.md,.pdf,.docx"
        onChange={handleFiles}
        style={{ display: "none" }}
      />
      <button
        type="button"
        className="ask-icon-btn"
        onClick={() => fileRef.current?.click()}
        title="Add files"
        aria-label="Add files"
        disabled={isDisabled}
      >
        <PlusIcon width={isLarge ? 19 : 18} height={isLarge ? 19 : 18} />
      </button>
    </>
  ) : null;

  const dictateBtn = withTools ? (
    <button
      type="button"
      className={`ask-icon-btn ${listening ? "is-listening" : ""}`}
      onClick={toggleDictation}
      title={listening ? "Stop dictation" : "Dictate"}
      aria-label={listening ? "Stop dictation" : "Dictate"}
      disabled={isDisabled}
    >
      <MicIcon width={17} height={17} />
    </button>
  ) : null;

  const submitBtn = (
    <button
      type="submit"
      className="ask-submit"
      disabled={!query.trim() || isDisabled}
      aria-label={uploading ? "Uploading files…" : "Start research"}
    >
      {uploading ? (
        <span style={{ fontSize: "11px", fontWeight: 600, letterSpacing: "0.02em" }}>
          …
        </span>
      ) : (
        <ArrowUpIcon width={isLarge ? 18 : 17} height={isLarge ? 18 : 17} />
      )}
    </button>
  );

  return (
    <motion.div
      className="search-container"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
    >
      <form
        onSubmit={handleSubmit}
        className={`ask-box ${isLarge ? "" : "ask-box-compact"}`}
      >
        {isLarge && attachments.length > 0 && (
          <div className="ask-attachments">
            {attachments.map((a) => (
              <span key={a.id} className="ask-chip">
                <FileTextIcon width={12} height={12} />
                <span className="ask-chip-name">{a.name}</span>
                <button
                  type="button"
                  className="ask-chip-remove"
                  onClick={() => removeAttachment(a.id)}
                  title="Remove"
                  aria-label={`Remove ${a.name}`}
                >
                  <CloseIcon width={12} height={12} />
                </button>
              </span>
            ))}
          </div>
        )}

        {isLarge ? (
          <>
            <input
              id="search-input"
              type="text"
              className="ask-input"
              placeholder={
                uploading
                  ? "Uploading files…"
                  : placeholder || PLACEHOLDERS[placeholderIdx]
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isDisabled}
              autoComplete="off"
              aria-label="Research question"
            />
            <div className="ask-toolbar">
              {tools}
              <div className="ask-tools-right">
                {dictateBtn}
                {submitBtn}
              </div>
            </div>
          </>
        ) : (
          <>
            {tools}
            <input
              type="text"
              className="ask-input"
              placeholder={
                uploading
                  ? "Uploading files…"
                  : placeholder || PLACEHOLDERS[placeholderIdx]
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isDisabled}
              autoComplete="off"
              aria-label="Ask a follow-up"
            />
            {dictateBtn}
            {submitBtn}
          </>
        )}
      </form>
    </motion.div>
  );
}
