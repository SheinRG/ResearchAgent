"use client";

import { motion, AnimatePresence } from "motion/react";
import { useRouter } from "next/navigation";

export default function FollowUpChips({ suggestions = [] }) {
  const router = useRouter();

  if (!suggestions || suggestions.length === 0) return null;

  const handleClick = (question) => {
    const encoded = encodeURIComponent(question);
    router.push(`/research?q=${encoded}`);
  };

  return (
    <div className="followup-section">
      <div className="followup-label">Related Questions</div>
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
              <span className="followup-chip-icon">→</span>
              {suggestion}
            </motion.button>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
