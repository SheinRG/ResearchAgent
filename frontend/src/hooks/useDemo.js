"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/hooks/useAuth";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * What a visitor is allowed to do right now.
 *
 * Answers two questions the UI needs before it renders anything: how many free
 * queries a logged-out visitor has left, and whether live research is currently
 * paused by the spend ceiling. The second applies to signed-in users too, so
 * this hook runs for everyone.
 *
 * Fails *open* on a network error, which is the opposite of how the server
 * behaves — and deliberately so. The server owns the limit and enforces it on
 * every request; this is presentation. Hiding the search box because a status
 * call timed out would break the app to protect a budget that is already
 * protected.
 */
export default function useDemo() {
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const [status, setStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/demo/status`, {
        // Sends and receives the visitor cookie — without it the server mints a
        // new id on every call and the allowance never counts down.
        credentials: "include",
      });
      if (!res.ok) throw new Error(`status ${res.status}`);
      setStatus(await res.json());
    } catch {
      setStatus(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Re-read once auth settles: the same endpoint answers differently for a
  // signed-in user (no quota) than for a visitor.
  useEffect(() => {
    if (authLoading) return;
    refresh();
  }, [authLoading, isAuthenticated, refresh]);

  const quota = status?.quota || null;

  return {
    isLoading: authLoading || isLoading,
    // True only when we know the visitor is out. Unknown status reads as "not
    // exhausted" so a failed status call never walls someone off by accident.
    isExhausted: !!quota?.exhausted,
    remaining: quota?.remaining ?? null,
    limit: quota?.limit ?? null,
    isAnonymous: status?.anonymous ?? !isAuthenticated,
    // The spend ceiling, if it is currently biting.
    livePaused: status?.live ? !status.live.available : false,
    liveMessage: status?.live?.message || "",
    liveReason: status?.live?.reason || "",
    refresh,
  };
}
