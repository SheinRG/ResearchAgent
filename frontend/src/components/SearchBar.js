"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "motion/react";

const PLACEHOLDERS = [
  "What are the latest breakthroughs in quantum computing?",
  "How does CRISPR gene editing work and what are its applications?",
  "Compare the economic policies of keynesian vs austrian economics",
  "What is the current state of nuclear fusion research?",
  "Explain the implications of recent AI regulation proposals",
];

export default function SearchBar({ onSearch, mode = "large", disabled = false }) {
  const [query, setQuery] = useState("");
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const inputRef = useRef(null);

  // Rotate placeholder text
  useEffect(() => {
    const interval = setInterval(() => {
      setPlaceholderIdx((prev) => (prev + 1) % PLACEHOLDERS.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (trimmed && !disabled) {
      onSearch(trimmed);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  return (
    <motion.div
      className="search-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <form
        onSubmit={handleSubmit}
        className={`search-bar ${mode === "large" ? "search-bar-large" : "search-bar-compact"}`}
      >
        <input
          ref={inputRef}
          id="search-input"
          type="text"
          className="search-input"
          placeholder={PLACEHOLDERS[placeholderIdx]}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          autoComplete="off"
          aria-label="Research question"
        />
        <motion.button
          type="submit"
          className="search-submit"
          disabled={!query.trim() || disabled}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          aria-label="Start research"
        >
          →
        </motion.button>
      </form>
    </motion.div>
  );
}
