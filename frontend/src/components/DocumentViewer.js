"use client";

import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { FileTextIcon, DownloadIcon, CloseIcon } from "@/components/Icons";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Slide-in document viewer panel. Renders a PDF or text file in an iframe
 * (with native browser controls), falls back to extracted plain text, or
 * offers a download link for unsupported types. Works for both live-session
 * files (browser File object) and restored sessions (served via file_id).
 *
 * Props:
 *   document – { name, text, file, file_id, mime, size } | null
 *   onClose  – () => void
 */
export default function DocumentViewer({ document, onClose }) {
  const [objectUrl, setObjectUrl] = useState(null);

  const isPdf =
    document &&
    ((document.mime || "").includes("pdf") ||
      /\.pdf$/i.test(document.name || ""));

  const isTextLike =
    document &&
    (/\.(txt|md)$/i.test(document.name || "") ||
      (document.mime || "").startsWith("text/"));

  // Create / revoke an object URL for any raw File (live session).
  useEffect(() => {
    if (!document?.file) return;
    const url = URL.createObjectURL(document.file);
    setObjectUrl(url);
    return () => {
      URL.revokeObjectURL(url);
      setObjectUrl(null);
    };
  }, [document]);

  // Close on Escape.
  useEffect(() => {
    if (!document) return;
    const onKeyDown = (e) => {
      if (e.key === "Escape") onClose?.();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [document, onClose]);

  if (!document) return null;

  // Remote URL available when the file was persisted server-side.
  const remoteUrl = document.file_id
    ? `${API_BASE}/api/files/${document.file_id}`
    : null;

  // Prefer the local object URL (live session); fall back to the server URL.
  const fileUrl = objectUrl || remoteUrl;

  // ---- Size formatting -------------------------------------------------
  const sizeLabel =
    document.size > 0
      ? `${(document.size / 1024).toFixed(1)} KB`
      : null;

  // ---- Text paragraphs -------------------------------------------------
  const paragraphs = document.text
    ? document.text.split(/\n{2,}/).filter(Boolean)
    : [];

  // ---- Body rendering --------------------------------------------------
  let body;
  if ((isPdf || isTextLike) && fileUrl) {
    body = (
      <iframe
        className="doc-viewer-frame"
        src={fileUrl}
        title={document.name}
      />
    );
  } else if (paragraphs.length > 0) {
    body = (
      <article className="doc-viewer-text">
        {paragraphs.map((para, i) => (
          <p key={i}>{para}</p>
        ))}
      </article>
    );
  } else if (fileUrl) {
    body = (
      <article className="doc-viewer-text">
        <p style={{ color: "var(--text-tertiary)", fontStyle: "italic" }}>
          Preview isn&apos;t available for this file type.
        </p>
        <p style={{ marginTop: "0.75rem" }}>
          <a
            href={fileUrl}
            download={document.name}
            style={{ color: "var(--accent)", textDecoration: "underline" }}
          >
            Download {document.name}
          </a>
        </p>
      </article>
    );
  } else {
    body = (
      <article className="doc-viewer-text">
        <p style={{ color: "var(--text-tertiary)", fontStyle: "italic" }}>
          No preview available for this file.
        </p>
      </article>
    );
  }

  // ---- Download button -------------------------------------------------
  const downloadBtn = document.file ? (
    // Live session: use the blob object URL (already created above).
    <a
      role="button"
      className="doc-viewer-btn"
      onClick={() => {
        try {
          const url = objectUrl || URL.createObjectURL(document.file);
          const a = window.document.createElement("a");
          a.href = url;
          a.download = document.name;
          a.click();
          if (!objectUrl) setTimeout(() => URL.revokeObjectURL(url), 1000);
        } catch { /* ignore */ }
      }}
      title="Download"
      aria-label="Download file"
      style={{ cursor: "pointer" }}
    >
      <DownloadIcon width={15} height={15} />
    </a>
  ) : remoteUrl ? (
    // Restored session: real anchor so the browser handles the download.
    <a
      href={remoteUrl}
      download={document.name}
      className="doc-viewer-btn"
      title="Download"
      aria-label="Download file"
    >
      <DownloadIcon width={15} height={15} />
    </a>
  ) : null;

  return (
    <>
      {/* Backdrop — only visible (and interactive) on narrow screens via CSS */}
      <div className="doc-viewer-backdrop" onClick={onClose} />

      <motion.aside
        className="doc-viewer"
        initial={{ x: 32, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 32, opacity: 0 }}
        transition={{ duration: 0.25 }}
      >
        {/* Header */}
        <div className="doc-viewer-header">
          <FileTextIcon width={16} height={16} style={{ flexShrink: 0, color: "var(--text-tertiary)" }} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="doc-viewer-title" title={document.name}>
              {document.name}
            </div>
            {sizeLabel && (
              <div className="doc-viewer-meta">{sizeLabel}</div>
            )}
          </div>
          <div className="doc-viewer-actions">
            {downloadBtn}
            <button
              type="button"
              className="doc-viewer-btn"
              onClick={onClose}
              title="Close"
              aria-label="Close viewer"
            >
              <CloseIcon width={15} height={15} />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="doc-viewer-body">
          {body}
        </div>
      </motion.aside>
    </>
  );
}
