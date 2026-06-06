"use client";

import { useEffect, useRef, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { motion } from "motion/react";
import SearchBar from "@/components/SearchBar";
import PhaseIndicator from "@/components/PhaseIndicator";
import SourceCards from "@/components/SourceCards";
import StreamingAnswer from "@/components/StreamingAnswer";
import FollowUpChips from "@/components/FollowUpChips";
import SkeletonLoader from "@/components/SkeletonLoader";
import ThemeToggle from "@/components/ThemeToggle";
import useResearch from "@/hooks/useResearch";
import useResearchStore from "@/stores/researchStore";
import { useAuth } from "@/hooks/useAuth";

function ResearchContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const query = searchParams.get("q") || "";
  const hasStarted = useRef(false);
  const lastQuery = useRef("");

  const { token, user, isAuthenticated, isLoading, logout } = useAuth();

  const {
    phase,
    phaseMessage,
    subQueries,
    sources,
    answer,
    followUps,
    isStreaming,
    isDone,
    doneData,
    error,
    startResearch,
    stopResearch,
  } = useResearch();

  const { addRecentSearch } = useResearchStore();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/login");
    }
  }, [isAuthenticated, isLoading, router]);

  // Start research when query changes
  useEffect(() => {
    if (query && query !== lastQuery.current && token) {
      lastQuery.current = query;
      hasStarted.current = true;
      addRecentSearch(query);
      startResearch(query, 1, token);
    }
  }, [query, startResearch, addRecentSearch, token]);

  const handleNewSearch = (newQuery) => {
    const encoded = encodeURIComponent(newQuery);
    router.push(`/research?q=${encoded}`);
  };

  // Determine confidence class
  const getConfidenceClass = (confidence) => {
    if (confidence >= 0.8) return "confidence-high";
    if (confidence >= 0.6) return "confidence-medium";
    return "confidence-low";
  };

  const showSkeleton = isStreaming && !answer && sources.length === 0;

  if (isLoading || !isAuthenticated) return null;

  return (
    <>
      {/* Navbar with compact search */}
      <nav className="navbar">
        <a href="/" className="navbar-brand">
          <span className="navbar-brand-icon">🔬</span>
          Research Agent
        </a>
        <div className="navbar-actions">
          {user && <span className="navbar-user">{user.name || user.email}</span>}
          <button className="logout-button" onClick={logout}>Log out</button>
          <ThemeToggle />
        </div>
      </nav>

      <main className="main-content">
        <div className="research-page">
          {/* Query Display */}
          <motion.h1
            className="research-query-display"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            {query}
          </motion.h1>

          {/* Sources */}
          {sources.length > 0 && <SourceCards sources={sources} />}

          {/* Phase Indicator */}
          {phase && !isDone && (
            <PhaseIndicator phase={phase} message={phaseMessage} />
          )}

          {/* Error State */}
          {error && (
            <motion.div
              className="error-container"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <div className="error-icon">⚠️</div>
              <div className="error-title">Something went wrong</div>
              <div className="error-message">{error}</div>
              <button
                className="error-retry"
                onClick={() => startResearch(query)}
              >
                Try Again
              </button>
            </motion.div>
          )}

          {/* Skeleton Loading */}
          {showSkeleton && !error && <SkeletonLoader />}

          {/* Streaming Answer */}
          {answer && (
            <StreamingAnswer
              answer={answer}
              isStreaming={isStreaming}
              sources={sources}
            />
          )}

          {/* Done Bar */}
          {isDone && doneData && (
            <motion.div
              className="done-bar"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <span
                className={`confidence-badge ${getConfidenceClass(
                  doneData.confidence
                )}`}
              >
                {Math.round((doneData.confidence || 0) * 100)}% confidence
              </span>
              <span className="done-separator" />
              <span className="done-stat">
                <span className="done-stat-value">
                  {doneData.total_sources || 0}
                </span>{" "}
                sources
              </span>
              <span className="done-separator" />
              <span className="done-stat">
                <span className="done-stat-value">
                  {doneData.iterations || 1}
                </span>{" "}
                {doneData.iterations === 1 ? "iteration" : "iterations"}
              </span>
            </motion.div>
          )}

          {/* Follow-up Suggestions */}
          {followUps.length > 0 && <FollowUpChips suggestions={followUps} />}

          {/* New Search (after done) */}
          {isDone && (
            <motion.div
              style={{ marginTop: "48px" }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              <SearchBar onSearch={handleNewSearch} mode="compact" />
            </motion.div>
          )}
        </div>
      </main>
    </>
  );
}

export default function ResearchPage() {
  return (
    <Suspense
      fallback={
        <div className="main-content">
          <div className="research-page">
            <SkeletonLoader />
          </div>
        </div>
      }
    >
      <ResearchContent />
    </Suspense>
  );
}
