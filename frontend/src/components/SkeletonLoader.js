"use client";

import { motion } from "motion/react";

export default function SkeletonLoader() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      {/* Source card skeletons */}
      <div className="sources-section">
        <div className="sources-label">Sources</div>
        <div className="skeleton-sources-row">
          {[0, 1, 2, 3].map((i) => (
            <motion.div
              key={i}
              className="skeleton skeleton-source-card"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1, duration: 0.3 }}
            />
          ))}
        </div>
      </div>

      {/* Answer skeleton */}
      <div className="skeleton-answer-block">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <motion.div
            key={i}
            className="skeleton skeleton-text-line"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 + i * 0.08, duration: 0.3 }}
          />
        ))}
      </div>
    </motion.div>
  );
}
