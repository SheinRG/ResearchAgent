"use client";

import { useRouter } from "next/navigation";
import { motion } from "motion/react";
import SearchBar from "@/components/SearchBar";
import ThemeToggle from "@/components/ThemeToggle";
import useResearchStore from "@/stores/researchStore";

const EXAMPLE_QUERIES = [
  "Latest breakthroughs in quantum computing",
  "How does mRNA vaccine technology work?",
  "Compare React, Vue, and Svelte in 2025",
  "What is the current state of nuclear fusion?",
  "Explain transformer architecture in AI",
  "How close are we to AGI?",
];

export default function HomePage() {
  const router = useRouter();
  const { recentSearches } = useResearchStore();

  const handleSearch = (query) => {
    const encoded = encodeURIComponent(query);
    router.push(`/research?q=${encoded}`);
  };

  const formatTime = (timestamp) => {
    const diff = Date.now() - timestamp;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return "just now";
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  return (
    <>
      {/* Navbar */}
      <nav className="navbar">
        <a href="/" className="navbar-brand">
          <span className="navbar-brand-icon">🔬</span>
          Research Agent
        </a>
        <div className="navbar-actions">
          <ThemeToggle />
        </div>
      </nav>

      {/* Hero Section */}
      <main className="main-content">
        <div className="hero">
          {/* Badge */}
          <motion.div
            className="hero-badge"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <span className="hero-badge-dot" />
            100% Local • Zero API Keys
          </motion.div>

          {/* Title */}
          <motion.h1
            className="hero-title"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            Research anything.
            <br />
            Get cited answers.
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            className="hero-subtitle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            An AI agent that searches the web, reads sources, and synthesizes
            comprehensive answers with citations — all running locally on your machine.
          </motion.p>

          {/* Search Bar */}
          <SearchBar onSearch={handleSearch} mode="large" />

          {/* Example Chips */}
          <motion.div
            className="example-chips"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            {EXAMPLE_QUERIES.map((q, i) => (
              <motion.button
                key={q}
                className="example-chip"
                onClick={() => handleSearch(q)}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 + i * 0.08 }}
                whileHover={{ y: -1 }}
              >
                {q}
              </motion.button>
            ))}
          </motion.div>

          {/* Recent Searches */}
          {recentSearches.length > 0 && (
            <motion.div
              className="recent-section"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
            >
              <div className="recent-label">Recent Searches</div>
              <div className="recent-list">
                {recentSearches.slice(0, 5).map((search, i) => (
                  <motion.a
                    key={search.timestamp}
                    className="recent-item"
                    onClick={() => handleSearch(search.query)}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.9 + i * 0.05 }}
                  >
                    <span className="recent-item-icon">🕐</span>
                    <span className="recent-item-text">{search.query}</span>
                    <span className="recent-item-time">
                      {formatTime(search.timestamp)}
                    </span>
                  </motion.a>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </main>
    </>
  );
}
