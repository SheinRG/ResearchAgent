"use client";

import { motion } from "motion/react";

export default function CitationTooltip({ source }) {
  if (!source) return null;

  return (
    <motion.div
      className="citation-tooltip"
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 4 }}
      transition={{ duration: 0.15 }}
    >
      <div className="citation-tooltip-header">
        <img
          src={source.favicon || `https://www.google.com/s2/favicons?domain=${source.domain}&sz=32`}
          alt=""
          className="citation-tooltip-favicon"
          onError={(e) => {
            e.target.style.display = "none";
          }}
        />
        <span className="citation-tooltip-domain">{source.domain}</span>
      </div>
      <div className="citation-tooltip-title">{source.title}</div>
      {source.snippet && (
        <div className="citation-tooltip-snippet">{source.snippet}</div>
      )}
    </motion.div>
  );
}
