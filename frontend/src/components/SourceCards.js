"use client";

import { motion, AnimatePresence } from "motion/react";

export default function SourceCards({ sources = [] }) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-section">
      <div className="sources-label">Sources</div>
      <div className="sources-strip">
        <AnimatePresence>
          {sources.map((source, index) => (
            <motion.a
              key={source.url || index}
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="source-card"
              initial={{ opacity: 0, x: 20, scale: 0.95 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              transition={{
                duration: 0.3,
                delay: index * 0.08,
                ease: [0.25, 0.46, 0.45, 0.94],
              }}
            >
              <div className="source-card-header">
                <span className="source-index">{index + 1}</span>
                <img
                  src={source.favicon || `https://www.google.com/s2/favicons?domain=${source.domain}&sz=32`}
                  alt=""
                  className="source-favicon"
                  onError={(e) => {
                    e.target.style.display = "none";
                  }}
                />
                <span className="source-domain">{source.domain}</span>
              </div>
              <div className="source-title">{source.title}</div>
            </motion.a>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
