"use client";

import { useState, useCallback, useRef } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Custom hook for SSE-based research.
 * Posts to /api/research, parses streaming SSE events for
 * phase updates, sources, tokens, follow-ups, and done signals.
 */
export default function useResearch() {
  const [phase, setPhase] = useState(null);
  const [phaseMessage, setPhaseMessage] = useState("");
  const [subQueries, setSubQueries] = useState([]);
  const [sources, setSources] = useState([]);
  const [answer, setAnswer] = useState("");
  const [followUps, setFollowUps] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [doneData, setDoneData] = useState(null);
  const [error, setError] = useState(null);

  const abortRef = useRef(null);

  const startResearch = useCallback(async (query, maxIterations = 2) => {
    // Reset state
    setPhase(null);
    setPhaseMessage("");
    setSubQueries([]);
    setSources([]);
    setAnswer("");
    setFollowUps([]);
    setIsStreaming(true);
    setIsDone(false);
    setDoneData(null);
    setError(null);

    // Abort any previous request
    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_BASE}/api/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, max_iterations: maxIterations }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from buffer
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        let eventType = null;

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith("data: ") && eventType) {
            try {
              const data = JSON.parse(line.slice(6));
              handleEvent(eventType, data);
            } catch {
              // Ignore malformed JSON
            }
            eventType = null;
          }
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        console.error("Research error:", err);
        setError(err.message || "An unexpected error occurred");
      }
    } finally {
      setIsStreaming(false);
    }
  }, []);

  const handleEvent = useCallback((type, data) => {
    switch (type) {
      case "phase":
        setPhase(data.phase);
        setPhaseMessage(data.message || "");
        break;

      case "sub_queries":
        setSubQueries(data.queries || []);
        break;

      case "sources":
        setSources((prev) => {
          const existing = new Set(prev.map((s) => s.url));
          const newSources = (data.sources || []).filter(
            (s) => !existing.has(s.url)
          );
          return [...prev, ...newSources];
        });
        break;

      case "token":
        setAnswer((prev) => prev + (data.token || ""));
        break;

      case "follow_up":
        setFollowUps(data.suggestions || []);
        break;

      case "done":
        setIsDone(true);
        setDoneData(data);
        setPhase("done");
        setPhaseMessage("Research complete");
        break;

      case "error":
        setError(data.message || "An error occurred during research");
        break;

      default:
        break;
    }
  }, []);

  const stopResearch = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  return {
    // State
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
    // Actions
    startResearch,
    stopResearch,
  };
}
