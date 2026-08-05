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
  const [loadError, setLoadError] = useState(false);

  const isPdf =
    document &&
    ((document.mime || "").includes("pdf") ||
      /\.pdf$/i.test(document.name || ""));

  const isTextLike =
    document &&
    (/\.(txt|md)$/i.test(document.name || "") ||
      (document.mime || "").startsWith("text/"));

  // Build a blob URL for the file: straight from the in-memory File in a live
  // session, or by fetching the persisted bytes for a restored one.
  //
  // The fetch is why this isn't just <iframe src={apiUrl}>. /api/files/{id}
  // requires an Authorization header, and the browser won't attach one to an
  // iframe or anchor navigation — so the bytes have to come through fetch()
  // and become a blob URL we own.
  useEffect(() => {
    if (!document) return;
    let cancelled = false;
    let created = null;

    const publish = (url) => {
      if (cancelled) {
        URL.revokeObjectURL(url);
        return;
      }
      created = url;
      setObjectUrl(url);
    };

    setLoadError(false);

    if (document.file) {
      publish(URL.createObjectURL(document.file));
    } else if (document.file_id) {
      (async () => {
        try {
          const token = localStorage.getItem("auth_token");
          const res = await fetch(`${API_BASE}/api/files/${document.file_id}`, {
            headers: token ? { Authorization: `Bearer ${token}` } : {},
          });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const blob = await res.blob();
          publish(URL.createObjectURL(blob));
        } catch {
          if (!cancelled) setLoadError(true);
        }
      })();
    }

    return () => {
      cancelled = true;
      if (created) URL.revokeObjectURL(created);
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

  // Always a blob URL we created — never a direct link to the API route.
  const fileUrl = objectUrl;

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
  } else if (loadError) {
    body = (
      <article className="doc-viewer-text">
        <p style={{ color: "var(--text-tertiary)", fontStyle: "italic" }}>
          This file couldn&apos;t be loaded. It may belong to another account,
          or your session may have expired.
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
  // One path for both live and restored sessions now that the bytes always
  // arrive as a blob we hold.
  const downloadBtn = fileUrl ? (
    <a
      href={fileUrl}
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
