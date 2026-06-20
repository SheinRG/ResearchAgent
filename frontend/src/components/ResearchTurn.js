"use client";

import { useState, useCallback } from "react";
import { motion } from "motion/react";
import ResearchTabs from "@/components/ResearchTabs";
import PhaseIndicator from "@/components/PhaseIndicator";
import StreamingAnswer from "@/components/StreamingAnswer";
import FollowUpChips from "@/components/FollowUpChips";
import SkeletonLoader from "@/components/SkeletonLoader";
import useToast from "@/stores/toastStore";
import {
  AlertIcon,
  CheckIcon,
  CopyIcon,
  RefreshIcon,
  ThumbsUpIcon,
  ThumbsDownIcon,
  PenIcon,
} from "@/components/Icons";

/** Strip markdown and citation markers to produce clean plain text. */
function toPlainText(md = "") {
  return md
    .replace(/\[(\d+)\]/g, "")                   // [1] [2] citation markers
    .replace(/#{1,6}\s+/g, "")                    // headings
    .replace(/\*\*(.+?)\*\*/gs, "$1")             // bold
    .replace(/\*(.+?)\*/gs, "$1")                 // italic
    .replace(/`{3}[\s\S]*?`{3}/g, "")            // fenced code blocks
    .replace(/`(.+?)`/g, "$1")                    // inline code
    .replace(/^\s*[-*+]\s+/gm, "")               // unordered list bullets
    .replace(/^\s*\d+\.\s+/gm, "")               // ordered list numbers
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")      // [text](url) → text
    .replace(/\n{3,}/g, "\n\n")                   // collapse excess blank lines
    .trim();
}

/** Copy-to-clipboard button with a transient "Copied" flash. */
function CopyButton({ text, title = "Copy", size = 14, clean = false }) {
  const [copied, setCopied] = useState(false);
  const onCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(clean ? toPlainText(text) : text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  }, [text]);

  return (
    <button
      className={`msg-action-btn ${copied ? "is-active" : ""}`}
      onClick={onCopy}
      title={copied ? "Copied" : title}
      aria-label={title}
    >
      {copied ? (
        <CheckIcon width={size} height={size} />
      ) : (
        <CopyIcon width={size} height={size} />
      )}
    </button>
  );
}

/**
 * A single question/answer turn in a research thread. The question is a
 * right-aligned bubble with hover actions (copy / resend / edit); below it a
 * tabbed answer card (Answer / Sources / Images) followed, once complete, by a
 * done bar, feedback actions (copy / regenerate / like / dislike) and
 * follow-ups.
 */
export default function ResearchTurn({
  query,
  sources = [],
  images = [],
  answer = "",
  isStreaming = false,
  isLive = false,
  phase = null,
  phaseMessage = "",
  showSkeleton = false,
  error = null,
  doneData = null,
  followUps = [],
  onFollowUp = null,
  onRetry = null,
}) {
  const showToast = useToast((s) => s.show);
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(query);
  const [feedback, setFeedback] = useState(null); // "up" | "down" | null

  const canAct = Boolean(onFollowUp);
  const isDone = Boolean(doneData) && !isStreaming;

  // Parse typed errors — prefix:message format from useResearch
  const errorType = error?.startsWith("rate_limit:") ? "rate_limit"
    : error?.startsWith("network:") ? "network"
    : error ? "generic" : null;
  const errorMessage = error?.includes(":") ? error.split(":").slice(1).join(":") : error;

  const errorTitles = {
    rate_limit: "Query limit reached",
    network: "Connection error",
    generic: "Something went wrong",
  };

  const resend = () => onFollowUp?.(query);

  const saveEdit = () => {
    const next = editText.trim();
    setEditing(false);
    if (next) onFollowUp?.(next);
  };

  const setVote = (vote) => {
    setFeedback((prev) => {
      const next = prev === vote ? null : vote;
      if (next === "up") showToast("Thanks for the feedback");
      else if (next === "down") showToast("Noted — we'll do better");
      return next;
    });
  };

  const answerPanel = (
    <>
      {isLive && phase && !error && !doneData && (
        <PhaseIndicator phase={phase} message={phaseMessage} />
      )}

      {isLive && errorType && (
        <motion.div
          className={`error-container error-type-${errorType}`}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <div className="error-icon">
            <AlertIcon width={32} height={32} />
          </div>
          <div className="error-title">{errorTitles[errorType]}</div>
          <div className="error-message">{errorMessage}</div>
          {onRetry && errorType !== "rate_limit" && (
            <button className="error-retry" onClick={onRetry}>
              Try again
            </button>
          )}
        </motion.div>
      )}

      {/* No-results state: done but empty answer */}
      {doneData && !answer && !isStreaming && !errorType && (
        <motion.div
          className="error-container error-type-empty"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <div className="error-title">No results found</div>
          <div className="error-message">The search didn't return enough information to answer this question. Try rephrasing or asking something more specific.</div>
          {onRetry && (
            <button className="error-retry" onClick={onRetry}>Try again</button>
          )}
        </motion.div>
      )}

      {isLive && showSkeleton && !error && <SkeletonLoader />}

      {answer && (
        <StreamingAnswer answer={answer} isStreaming={isStreaming} sources={sources} />
      )}

      {doneData && (
        <div className="done-bar">
          <span className="done-confidence">
            <CheckIcon width={13} height={13} />
            {Math.round((doneData.confidence || 0) * 100)}% confidence
          </span>
          <span className="done-separator">/</span>
          <span>{doneData.total_sources || sources.length || 0} sources</span>
          <span className="done-separator">/</span>
          <span>
            {doneData.iterations || 1}{" "}
            {(doneData.iterations || 1) === 1 ? "iteration" : "iterations"}
          </span>
        </div>
      )}

      {/* Feedback actions — once the answer is complete */}
      {isDone && answer && (
        <div className="msg-actions msg-actions-ai">
          <CopyButton text={answer} title="Copy answer" size={15} clean />
          {canAct && (
            <button
              className="msg-action-btn"
              onClick={resend}
              title="Regenerate"
              aria-label="Regenerate answer"
            >
              <RefreshIcon width={15} height={15} />
            </button>
          )}
          <button
            className={`msg-action-btn ${feedback === "up" ? "is-active" : ""}`}
            onClick={() => setVote("up")}
            title="Good answer"
            aria-label="Good answer"
          >
            <ThumbsUpIcon width={15} height={15} />
          </button>
          <button
            className={`msg-action-btn ${feedback === "down" ? "is-active" : ""}`}
            onClick={() => setVote("down")}
            title="Needs work"
            aria-label="Needs work"
          >
            <ThumbsDownIcon width={15} height={15} />
          </button>
        </div>
      )}

      {followUps.length > 0 && (
        <FollowUpChips
          suggestions={followUps}
          onSelect={onFollowUp}
          disabled={!onFollowUp}
        />
      )}
    </>
  );

  return (
    <section className="chat-turn">
      {/* User question — right-aligned bubble with hover actions. */}
      <div className="chat-question-row">
        {editing ? (
          <div className="chat-question-edit">
            <textarea
              rows={2}
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              autoFocus
            />
            <div className="chat-edit-actions">
              <button
                className="btn-ghost"
                onClick={() => {
                  setEditing(false);
                  setEditText(query);
                }}
              >
                Cancel
              </button>
              <button className="btn-accent" onClick={saveEdit}>
                Update
              </button>
            </div>
          </div>
        ) : (
          <>
            <motion.div
              className="chat-question"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              {query}
            </motion.div>
            <div className="msg-actions">
              <CopyButton text={query} title="Copy" />
              {canAct && (
                <>
                  <button
                    className="msg-action-btn"
                    onClick={resend}
                    title="Resend"
                    aria-label="Resend question"
                  >
                    <RefreshIcon width={14} height={14} />
                  </button>
                  <button
                    className="msg-action-btn"
                    onClick={() => {
                      setEditText(query);
                      setEditing(true);
                    }}
                    title="Edit"
                    aria-label="Edit question"
                  >
                    <PenIcon width={14} height={14} />
                  </button>
                </>
              )}
            </div>
          </>
        )}
      </div>

      {/* Answer area — tabs then content. */}
      <div className="chat-answer">
        <ResearchTabs sources={sources} images={images}>
          {answerPanel}
        </ResearchTabs>
      </div>
    </section>
  );
}
