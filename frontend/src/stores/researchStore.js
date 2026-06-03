"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

/**
 * Zustand store for research session state.
 * Persists recent searches to localStorage.
 */
const useResearchStore = create(
  persist(
    (set, get) => ({
      // Recent searches (persisted)
      recentSearches: [],

      // Add a search to recent history
      addRecentSearch: (query) => {
        const { recentSearches } = get();
        // Remove duplicate if exists
        const filtered = recentSearches.filter(
          (s) => s.query.toLowerCase() !== query.toLowerCase()
        );
        // Add to front, limit to 10
        const updated = [
          { query, timestamp: Date.now() },
          ...filtered,
        ].slice(0, 10);
        set({ recentSearches: updated });
      },

      // Clear recent searches
      clearRecentSearches: () => {
        set({ recentSearches: [] });
      },
    }),
    {
      name: "research-store",
      partialize: (state) => ({
        recentSearches: state.recentSearches,
      }),
    }
  )
);

export default useResearchStore;
