"use client";

import { motion, AnimatePresence } from "motion/react";

const PHASE_CONFIG = {
  planning: { icon: "🔍", label: "Planning" },
  searching: { icon: "🌐", label: "Searching" },
  reading: { icon: "📖", label: "Reading" },
  writing: { icon: "✍️", label: "Writing" },
  reflecting: { icon: "🔎", label: "Reflecting" },
  done: { icon: "✅", label: "Done" },
};

export default function PhaseIndicator({ phase, message }) {
  if (!phase) return null;

  const config = PHASE_CONFIG[phase] || { icon: "⏳", label: phase };
  const isDone = phase === "done";

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={phase}
        className="phase-indicator"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 10 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
      >
        <motion.span
          className="phase-icon"
          initial={{ scale: 0.5, rotate: -20 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 15 }}
        >
          {config.icon}
        </motion.span>
        <span className="phase-text">
          {message || config.label}
        </span>
        {!isDone && <div className="phase-spinner" />}
      </motion.div>
    </AnimatePresence>
  );
}
