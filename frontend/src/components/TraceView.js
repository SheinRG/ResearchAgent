"use client";

import { motion } from "motion/react";
import { CheckIcon } from "@/components/Icons";
import { safeUrl } from "@/lib/safeUrl";

/**
 * The Trace tab — what the pipeline considered, and what it did with it.
 *
 * The answer only ever shows the sources it quoted. This shows the rest of the
 * shortlist: everything the reranker scored, whether it reached the model, and
 * whether the model used it. All of it is derived from work the pipeline had
 * already done and was discarding.
 *
 * Three states, in decreasing order of usefulness to the reader:
 *   cited       the answer referenced it with a [n] marker
 *   sent        the model saw it and chose not to use it
 *   considered  ranked, but it didn't make the prompt's budget
 */

const STATUS_LABEL = {
  cited: "cited",
  sent: "sent to model",
  considered: "not used",
};

function Stat({ value, label }) {
  return (
    <div className="trace-stat">
      <span className="trace-stat-value">{value}</span>
      <span className="trace-stat-label">{label}</span>
    </div>
  );
}

export default function TraceView({ trace }) {
  if (!trace) return null;

  const counts = trace.counts || {};
  const sources = trace.sources || [];
  const subQueries = trace.sub_queries || [];

  return (
    <div className="trace-view">
      {/* The funnel: how much was gathered, how much survived each narrowing. */}
      <div className="trace-funnel">
        <Stat value={counts.chunks_ranked ?? 0} label="passages ranked" />
        <span className="trace-arrow">→</span>
        <Stat value={counts.chunks_kept ?? 0} label="kept" />
        <span className="trace-arrow">→</span>
        <Stat value={counts.sources_sent ?? 0} label="sent to model" />
        <span className="trace-arrow">→</span>
        <Stat value={counts.sources_cited ?? 0} label="cited" />
      </div>

      {subQueries.length > 0 && (
        <section className="trace-section">
          <h4 className="trace-heading">Searched for</h4>
          <ul className="trace-subqueries">
            {subQueries.map((q, i) => (
              <li key={`${q}-${i}`}>{q}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="trace-section">
        <h4 className="trace-heading">
          Ranked sources
          {trace.reranker_model && (
            <span className="trace-model">{trace.reranker_model}</span>
          )}
        </h4>

        {sources.length === 0 ? (
          <p className="trace-empty">
            Nothing was retrieved for this question, so there was nothing to rank.
          </p>
        ) : (
          <ol className="trace-list">
            {sources.map((s, i) => {
              // Uploaded documents use file:// URLs; safeUrl rejects them, so
              // they render as plain text rather than a broken link.
              const href = safeUrl(s.url);
              const pct = Math.max(0, Math.min(1, s.score || 0)) * 100;
              return (
                <motion.li
                  key={`${s.url}-${i}`}
                  className={`trace-item trace-item-${s.status}`}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2, delay: Math.min(i * 0.02, 0.3) }}
                >
                  <div className="trace-score" title={`Relevance score ${s.score}`}>
                    <div className="trace-score-bar">
                      <div
                        className="trace-score-fill"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="trace-score-value">
                      {(s.score ?? 0).toFixed(2)}
                    </span>
                  </div>

                  <div className="trace-body">
                    <div className="trace-item-head">
                      {href ? (
                        <a
                          href={href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="trace-title"
                        >
                          {s.title || s.domain || s.url}
                        </a>
                      ) : (
                        <span className="trace-title">
                          {s.title || s.domain || s.url}
                        </span>
                      )}

                      <span className={`trace-badge trace-badge-${s.status}`}>
                        {s.status === "cited" && (
                          <CheckIcon width={10} height={10} />
                        )}
                        {s.status === "cited"
                          ? `cited [${s.citation_index}]`
                          : STATUS_LABEL[s.status] || s.status}
                      </span>
                    </div>

                    <div className="trace-meta">
                      <span>{s.domain}</span>
                      {s.chunks > 1 && (
                        <>
                          <span className="trace-dot">·</span>
                          <span>{s.chunks} passages</span>
                        </>
                      )}
                    </div>

                    {s.preview && <p className="trace-preview">{s.preview}</p>}
                  </div>
                </motion.li>
              );
            })}
          </ol>
        )}
      </section>

      <p className="trace-footnote">
        Scores come from the re-ranker, which reads every retrieved passage and
        rates it against your question. Only the top passages fit in the model&apos;s
        prompt — the rest are shown here so you can see what was left out.
      </p>
    </div>
  );
}
