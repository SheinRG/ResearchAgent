"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "motion/react";
import { ArrowUpIcon, SparklesIcon } from "@/components/Icons";

const PLACEHOLDERS = [
  "Ask anything...",
  "What are the latest breakthroughs in quantum computing?",
  "How does CRISPR gene editing work?",
  "What is the current state of nuclear fusion research?",
  "Explain the implications of recent AI regulation proposals",
];

export default function SearchBar({
  onSearch,
  mode = "large",
  disabled = false,
  placeholder = null,
  clearOnSubmit = false,
}) {
  const [query, setQuery] = useState("");
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const inputRef = useRef(null);

  // Rotate placeholder text — only when no fixed placeholder is supplied.
  useEffect(() => {
    if (placeholder) return;
    const interval = setInterval(() => {
      setPlaceholderIdx((prev) => (prev + 1) % PLACEHOLDERS.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [placeholder]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (trimmed && !disabled) {
      onSearch(trimmed);
      if (clearOnSubmit) setQuery("");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  const isLarge = mode === "large";

  return (
    <motion.div
      className="search-container"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.15 }}
    >
      <form
        onSubmit={handleSubmit}
        className={`ask-box ${isLarge ? "" : "ask-box-compact"}`}
      >
        <input
          ref={inputRef}
          id="search-input"
          type="text"
          className="ask-input"
          placeholder={placeholder || PLACEHOLDERS[placeholderIdx]}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          autoComplete="off"
          aria-label="Research question"
        />
        {isLarge ? (
          <div className="ask-toolbar">
            <span className="ask-badge" title="Multi-step research with cited sources">
              <SparklesIcon width={13} height={13} />
              deep research
            </span>
            <button
              type="submit"
              className="ask-submit"
              disabled={!query.trim() || disabled}
              aria-label="Start research"
            >
              <ArrowUpIcon width={17} height={17} />
            </button>
          </div>
        ) : (
          <button
            type="submit"
            className="ask-submit"
            disabled={!query.trim() || disabled}
            aria-label="Start research"
          >
            <ArrowUpIcon width={16} height={16} />
          </button>
        )}
      </form>
    </motion.div>
  );
}
