"use client";

import { useCallback, useEffect, useRef, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import SearchBar from "@/components/SearchBar";
import ResearchTurn from "@/components/ResearchTurn";
import SessionHeader from "@/components/SessionHeader";
import SkeletonLoader from "@/components/SkeletonLoader";
import useResearch from "@/hooks/useResearch";
import useResearchStore from "@/stores/researchStore";
import { useAuth } from "@/hooks/useAuth";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function ResearchContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const urlQuery  = searchParams.get("q")       || "";
  const sessionId = searchParams.get("session") || "";

  const { token, isAuthenticated, isLoading } = useAuth();

  const {
    phase,
    phaseMessage,
    sources,
    images,
    answer,
    isStreaming,
    error,
    startResearch,
  } = useResearch();

  const { addRecentSearch, bumpSessions, consumePendingDocuments } = useResearchStore();

  // The conversation thread: completed turns, plus the one currently streaming.
  const [turns, setTurns] = useState([]);
  const [activeQuery, setActiveQuery] = useState(null);
  const [sessionTitle, setSessionTitle] = useState(null); // user-renamed title

  // Loading / error state while fetching a stored thread from the DB.
  const [sessionLoading, setSessionLoading] = useState(false);
  const [sessionError, setSessionError]     = useState(null);

  /**
   * seedRef     — the ?q= value that seeded the current live research run.
   *               Prevents the ?q= effect from re-firing on re-renders.
   * sessionIdRef — session UUID shared by every turn; set from the first
   *               done event or from the ?session= param when restoring.
   * loadedSessionRef — the ?session= ID we already loaded (prevents double
   *               fetching when searchParams identity changes without the
   *               value changing, e.g. on HMR).
   * liveRef     — DOM wrapper for the currently-streaming turn (auto-scroll).
   * activeDocsRef — documents for the currently-active run; re-used by retry
   *               so the same files are re-sent without re-uploading.
   */
  const seedRef          = useRef(null);
  const sessionIdRef     = useRef(null);
  const loadedSessionRef = useRef(null);
  const liveRef          = useRef(null);
  const activeDocsRef    = useRef([]);

  // Only redirect to login when trying to run a new query without auth.
  // Shared session links (?session=) are public and work without login.
  useEffect(() => {
    if (!isLoading && !isAuthenticated && !sessionId) {
      router.push("/login");
    }
  }, [isAuthenticated, isLoading, router, sessionId]);

  // ---------------------------------------------------------------------------
  // handleComplete — freeze a finished SSE run into the thread
  // ---------------------------------------------------------------------------
  const handleComplete = useCallback((turn) => {
    const sid = turn.doneData?.session_id;
    if (sid && !sessionIdRef.current) sessionIdRef.current = sid;
    setTurns((prev) => [
      ...prev,
      {
        id: sid ? `${sid}-${prev.length}` : `${turn.query}-${Date.now()}`,
        query:    turn.query,
        answer:   turn.answer,
        sources:  turn.sources,
        images:   turn.images,
        followUps: turn.followUps,
        doneData:  turn.doneData,
      },
    ]);
    setActiveQuery(null);
    // Tell the sidebar to re-fetch the sessions list so this thread appears.
    bumpSessions();
  }, [bumpSessions]);

  // ---------------------------------------------------------------------------
  // Effect A — ?session=<id>  →  load stored thread from the DB (no re-run)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    // Skip if: no sessionId param, already loaded this exact id, or auth still loading.
    if (!sessionId || loadedSessionRef.current === sessionId || isLoading) return;

    loadedSessionRef.current = sessionId;

    // Also mark the seed so Effect B (the ?q= handler) never fires for this nav.
    seedRef.current = null;

    setSessionLoading(true);
    setSessionError(null);
    setTurns([]);
    setSessionTitle(null);
    setActiveQuery(null);

    // Public endpoint — send auth header if available, not required.
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    fetch(`${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}`, { headers })
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || `Failed to load session (${res.status})`);
        }
        return res.json();
      })
      .then((data) => {
        // data shape: { id, title, created_at, turns: [SessionTurn, ...] }
        const restoredTurns = (data.turns || []).map((t, i) => ({
          id:        `${data.id}-${i}`,
          query:     t.query,
          answer:    t.answer,
          sources:   t.sources   || [],
          images:    [],           // stored turns have no images; Images tab will be empty
          followUps: t.follow_up_suggestions || [],
          doneData: {
            session_id:   data.id,
            total_sources: (t.sources || []).length,
            confidence:   t.confidence,
            iterations:   t.iterations || 1,
          },
        }));

        sessionIdRef.current = data.id;
        setSessionTitle(data.title || null);
        setTurns(restoredTurns);
      })
      .catch((err) => {
        setSessionError(err.message || "Could not load this session.");
      })
      .finally(() => {
        setSessionLoading(false);
      });
  }, [sessionId, token, isLoading]);

  // ---------------------------------------------------------------------------
  // Effect B — ?q=<query>  →  start a fresh live research run
  // ---------------------------------------------------------------------------
  // A ?session= navigation must NOT trigger this effect.  We guard with two
  // conditions: (1) there must be no ?session= param, and (2) the query must
  // differ from the last seeded value.
  useEffect(() => {
    if (!token || !urlQuery || sessionId || urlQuery === seedRef.current) return;
    seedRef.current       = urlQuery;
    sessionIdRef.current  = null;
    loadedSessionRef.current = null;
    setTurns([]);
    setSessionTitle(null);
    setSessionError(null);
    setActiveQuery(urlQuery);
    addRecentSearch(urlQuery);
    // Consume any documents staged by the home page before navigating here.
    // consumePendingDocuments clears the store so a refresh doesn't replay them.
    const pendingDocs = consumePendingDocuments();
    activeDocsRef.current = pendingDocs;
    startResearch(urlQuery, 1, token, [], null, handleComplete, pendingDocs);
  }, [urlQuery, sessionId, token, startResearch, addRecentSearch, handleComplete, consumePendingDocuments]);

  // ---------------------------------------------------------------------------
  // submitQuery — follow-up within the current thread
  // ---------------------------------------------------------------------------
  const submitQuery = useCallback(
    (q, documents) => {
      const trimmed = (q || "").trim();
      if (!trimmed || isStreaming || activeQuery || !token) return;
      const history = turns.map((t) => ({ query: t.query, answer: t.answer }));
      const docs = documents || [];
      addRecentSearch(trimmed);
      setActiveQuery(trimmed);
      activeDocsRef.current = docs;
      // sessionIdRef.current is already set (either from a live run or from the
      // restored thread), so this follow-up correctly continues the same thread.
      startResearch(trimmed, 1, token, history, sessionIdRef.current, handleComplete, docs);
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
    // Re-send the same documents that were part of the failed run so the retry
    // is semantically identical to the original request.
    startResearch(activeQuery, 1, token, history, sessionIdRef.current, handleComplete, activeDocsRef.current);
  }, [activeQuery, turns, token, startResearch, handleComplete]);

  const showSkeleton = isStreaming && !answer && sources.length === 0;
  const hasThread    = turns.length > 0 || Boolean(activeQuery);
  const headerTitle  = sessionTitle || turns[0]?.query || activeQuery || "Untitled session";

  // For shared sessions, unauthenticated visitors are allowed to view.
  if (isLoading && !sessionId) return null;
  if (!isAuthenticated && !sessionId) return null;

  // Show a full-page skeleton while the stored thread is being fetched.
  if (sessionLoading) {
    return (
      <main className="research-page">
        <div className="research-thread">
          <SkeletonLoader />
        </div>
      </main>
    );
  }

  // Show a simple error state when the stored thread couldn't be loaded.
  if (sessionError) {
    return (
      <main className="research-page">
        <div className="research-thread">
          <div className="error-container">
            <div className="error-title">Could not load session</div>
            <div className="error-message">{sessionError}</div>
            <button className="error-retry" onClick={() => router.push("/")}>
              Go home
            </button>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="research-page">
      <div className="research-thread">
        {hasThread && (
          <SessionHeader title={headerTitle} onRename={setSessionTitle} turns={turns} />
        )}

        {turns.map((turn) => (
          <ResearchTurn
            key={turn.id}
            query={turn.query}
            sources={turn.sources}
            images={turn.images}
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
              images={images}
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

        {hasThread && isAuthenticated && (
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

        {hasThread && !isAuthenticated && (
          <div className="shared-session-cta">
            <p>Want to ask follow-up questions or save your own research?</p>
            <a href="/login" className="btn-accent" style={{ display: "inline-block", padding: "8px 20px", borderRadius: 8, textDecoration: "none", fontSize: 14 }}>
              Sign in to continue
            </a>
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
        <div className="research-page">
          <div className="research-thread">
            <SkeletonLoader />
          </div>
        </div>
      }
    >
      <ResearchContent />
    </Suspense>
  );
}
