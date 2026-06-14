"use client";

import { motion, AnimatePresence } from "motion/react";
import { useRouter } from "next/navigation";
import { ArrowRightIcon } from "@/components/Icons";

export default function FollowUpChips({ suggestions = [], onSelect = null, disabled = false }) {
  const router = useRouter();

  if (!suggestions || suggestions.length === 0) return null;

  const handleClick = (question) => {
    if (disabled) return;
    // In a thread, append the follow-up in place; otherwise fall back to a
    // fresh research navigation (preserves standalone use of this component).
    if (onSelect) {
      onSelect(question);
      return;
    }
    const encoded = encodeURIComponent(question);
    router.push(`/research?q=${encoded}`);
  };

  return (
    <div className="followup-section">
      <div className="followup-label">Related</div>
      <div className="followup-chips">
        <AnimatePresence>
          {suggestions.map((suggestion, index) => (
            <motion.button
              key={suggestion}
              className="followup-chip"
              onClick={() => handleClick(suggestion)}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{
                duration: 0.3,
                delay: index * 0.1,
                ease: [0.25, 0.46, 0.45, 0.94],
              }}
              whileHover={{ x: 4 }}
            >
              <span className="followup-chip-icon">
                <ArrowRightIcon width={16} height={16} />
              </span>
              {suggestion}
            </motion.button>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
