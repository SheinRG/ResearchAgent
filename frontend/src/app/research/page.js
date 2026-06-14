"use client";

import { useCallback, useEffect, useRef, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import SearchBar from "@/components/SearchBar";
import ResearchTurn from "@/components/ResearchTurn";
import SkeletonLoader from "@/components/SkeletonLoader";
import useResearch from "@/hooks/useResearch";
import useResearchStore from "@/stores/researchStore";
import { useAuth } from "@/hooks/useAuth";

function ResearchContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const urlQuery = searchParams.get("q") || "";

  const { token, isAuthenticated, isLoading } = useAuth();

  const {
    phase,
    phaseMessage,
    sources,
    answer,
    isStreaming,
    error,
    startResearch,
  } = useResearch();

  const { addRecentSearch } = useResearchStore();

  // The conversation thread: completed turns, plus the one currently streaming.
  const [turns, setTurns] = useState([]);
  const [activeQuery, setActiveQuery] = useState(null);

  const seedRef = useRef(null); // URL query that seeded the current thread
  const sessionIdRef = useRef(null); // session id shared by every turn in the thread
  const liveRef = useRef(null); // wrapper around the streaming turn (for auto-scroll)

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/login");
    }
  }, [isAuthenticated, isLoading, router]);

  // Freeze a finished run into the thread. Runs from useResearch's SSE handler
  // (an event callback, not an effect) once the answer/sources are final.
  const handleComplete = useCallback((turn) => {
    const sid = turn.doneData?.session_id;
    if (sid && !sessionIdRef.current) sessionIdRef.current = sid;
    setTurns((prev) => [
      ...prev,
      {
        id: sid ? `${sid}-${prev.length}` : `${turn.query}-${Date.now()}`,
        query: turn.query,
        answer: turn.answer,
        sources: turn.sources,
        followUps: turn.followUps,
        doneData: turn.doneData,
      },
    ]);
    setActiveQuery(null);
  }, []);

  // A new URL query (from the home page or sidebar) starts a fresh thread.
  // Follow-ups stay in-page and never touch the URL, so they don't reset it.
  useEffect(() => {
    if (!token || !urlQuery || urlQuery === seedRef.current) return;
    seedRef.current = urlQuery;
    sessionIdRef.current = null;
    setTurns([]);
    setActiveQuery(urlQuery);
    addRecentSearch(urlQuery);
    startResearch(urlQuery, 1, token, [], null, handleComplete);
  }, [urlQuery, token, startResearch, addRecentSearch, handleComplete]);

  // Ask a follow-up within the current thread, carrying the prior turns as
  // context so the agent answers in continuation instead of from scratch.
  const submitQuery = useCallback(
    (q) => {
      const trimmed = (q || "").trim();
      if (!trimmed || isStreaming || activeQuery || !token) return;
      const history = turns.map((t) => ({ query: t.query, answer: t.answer }));
      addRecentSearch(trimmed);
      setActiveQuery(trimmed);
      startResearch(trimmed, 1, token, history, sessionIdRef.current, handleComplete);
    },
    [turns, isStreaming, activeQuery, token, startResearch, addRecentSearch, handleComplete]
  );

  // Bring each new question into view as it starts streaming.
  useEffect(() => {
    if (activeQuery && liveRef.current) {
      liveRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [activeQuery]);

  const retry = useCallback(() => {
    if (!activeQuery) return;
    const history = turns.map((t) => ({ query: t.query, answer: t.answer }));
    startResearch(activeQuery, 1, token, history, sessionIdRef.current, handleComplete);
  }, [activeQuery, turns, token, startResearch, handleComplete]);

  const showSkeleton = isStreaming && !answer && sources.length === 0;
  const hasThread = turns.length > 0 || Boolean(activeQuery);

  if (isLoading || !isAuthenticated) return null;

  return (
    <main className="main-content">
      <div className="research-page research-thread">
        {turns.map((turn) => (
          <ResearchTurn
            key={turn.id}
            query={turn.query}
            sources={turn.sources}
            answer={turn.answer}
            isStreaming={false}
            doneData={turn.doneData}
            followUps={turn.followUps}
            onFollowUp={submitQuery}
          />
        ))}

        {activeQuery && (
          <div ref={liveRef}>
            <ResearchTurn
              query={activeQuery}
              sources={sources}
              answer={answer}
              isStreaming={isStreaming}
              isLive
              phase={phase}
              phaseMessage={phaseMessage}
              showSkeleton={showSkeleton}
              error={error}
              onRetry={retry}
            />
          </div>
        )}

        {hasThread && (
          <div className="followup-composer">
            <SearchBar
              onSearch={submitQuery}
              mode="compact"
              placeholder="Ask a follow-up…"
              disabled={isStreaming || Boolean(activeQuery)}
              clearOnSubmit
            />
          </div>
        )}
      </div>
    </main>
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
