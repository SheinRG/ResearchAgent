"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

/**
 * Zustand store for research session state.
 *
 * Persisted (localStorage): recentSearches.
 * Transient (in-memory only): sessionsNonce — incremented whenever a research
 * turn completes so AppLayout knows to re-fetch the sessions list from the DB.
 *
 * Notes are no longer stored here — they live in the DB and are fetched via
 * GET /api/notes in AppLayout.
 */
const useResearchStore = create(
  persist(
    (set, get) => ({
      // -----------------------------------------------------------------------
      // Persisted state
      // -----------------------------------------------------------------------

      /** Recent searches (local fallback; sidebar now prefers the DB list). */
      recentSearches: [],

      /** Add a search to local recent-history (deduped, max 10). */
      addRecentSearch: (query) => {
        const { recentSearches } = get();
        const filtered = recentSearches.filter(
          (s) => s.query.toLowerCase() !== query.toLowerCase()
        );
        const updated = [
          { query, timestamp: Date.now() },
          ...filtered,
        ].slice(0, 10);
        set({ recentSearches: updated });
      },

      /** Wipe the local recent-search list. */
      clearRecentSearches: () => {
        set({ recentSearches: [] });
      },

      // -----------------------------------------------------------------------
      // Transient state (NOT persisted — see partialize below)
      // -----------------------------------------------------------------------

      /**
       * Monotonically-incrementing counter.  AppLayout watches this value; when
       * it changes the sidebar re-fetches GET /api/sessions so a freshly-saved
       * thread appears without a full page reload.
       */
      sessionsNonce: 0,

      /** Call this after every completed research turn to refresh the sidebar. */
      bumpSessions: () => set({ sessionsNonce: get().sessionsNonce + 1 }),

      // -----------------------------------------------------------------------
      // Transient: document hand-off across the home → research URL navigation.
      // NOT persisted (excluded from partialize below).
      // -----------------------------------------------------------------------

      /**
       * Documents staged on the home page when the user attaches files.
       * Consumed once by Effect B in research/page.js, then cleared.
       * Shape: [{ name: string, text: string }]
       */
      pendingDocuments: [],

      /** Stage documents before navigating to /research. */
      setPendingDocuments: (docs) => set({ pendingDocuments: docs || [] }),

      /**
       * Read and clear pendingDocuments atomically so a page-reload or a second
       * Effect B firing never replays the same files.
       */
      consumePendingDocuments: () => {
        const docs = get().pendingDocuments;
        set({ pendingDocuments: [] });
        return docs;
      },
    }),
    {
      name: "research-store",
      // sessionsNonce is intentionally excluded so it resets to 0 on every
      // page load and always triggers an initial fetch in AppLayout.
      partialize: (state) => ({
        recentSearches: state.recentSearches,
      }),
    }
  )
);

export default useResearchStore;
