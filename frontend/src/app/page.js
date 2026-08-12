"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Link from "next/link";
import SearchBar from "@/components/SearchBar";
import { useAuth } from "@/hooks/useAuth";
import useDemo from "@/hooks/useDemo";
import useResearchStore from "@/stores/researchStore";

// Time-aware openers, framed for someone here to *work* — late hours nudge
// ("Working late") rather than sign off. Phrased so a ", Name" appends cleanly.
function greetingForHour(hour) {
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  if (hour < 22) return "Good evening";
  return "Working late";
}

// A wide pool of openers; the hero shows a rotating window of three so the
// page feels alive and nudges different directions on every visit.
const SUGGESTION_POOL = [
  "What are the latest breakthroughs in fusion energy?",
  "How does CRISPR base editing actually work?",
  "Why is the AI chip supply chain so concentrated?",
  "What's driving the surge in GLP-1 weight-loss drugs?",
  "How do large language models actually represent meaning?",
  "Is small modular nuclear finally becoming viable?",
  "How close are we to practical quantum error correction?",
  "What makes the new mRNA cancer vaccines different?",
  "How do solid-state batteries compare to lithium-ion?",
  "What's the scientific consensus on gut microbiome health?",
  "How are autonomous agents changing software engineering?",
  "What's the realistic timeline for commercial fusion power?",
  "Why are higher interest rates reshaping global startups?",
  "How does end-to-end encryption actually keep messages private?",
  "What's the state of room-temperature superconductor research?",
];

// Fisher-Yates shuffle — fresh order each mount so visits differ.
function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const VISIBLE_COUNT = 3;
const ROTATE_MS = 6000;

export default function HomePage() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading } = useAuth();
  const demo = useDemo();
  const setPendingDocuments = useResearchStore((s) => s.setPendingDocuments);

  // Greeting depends on the client clock, so resolve after mount to avoid a
  // server/client hydration mismatch.
  const [greeting, setGreeting] = useState("");

  // Rotating suggestions: a shuffled pool resolved after mount (avoids a
  // hydration mismatch), advanced by a window of three on a timer with a
  // brief cross-fade.
  const [pool, setPool] = useState([]);
  const [start, setStart] = useState(0);
  const [chipsOpacity, setChipsOpacity] = useState(1);

  // Playful one-time welcome: right after a fresh sign-in, greet the user as
  // "bade bhai" for a few seconds, then cross-fade to their real name. While
  // `welcomeName` is set it overrides the rendered name; `nameOpacity` drives
  // the fade so the swap itself stays invisible.
  const [welcomeName, setWelcomeName] = useState(null);
  const [nameOpacity, setNameOpacity] = useState(1);

  useEffect(() => {
    setGreeting(greetingForHour(new Date().getHours()));
    setPool(shuffle(SUGGESTION_POOL));
  }, []);

  // Consume the one-shot "just logged in" flag (set by useAuth on a real
  // sign-in) and run the bade-bhai → real-name transition exactly once.
  useEffect(() => {
    let fresh = false;
    try {
      fresh = sessionStorage.getItem("just_logged_in") === "1";
      if (fresh) sessionStorage.removeItem("just_logged_in");
    } catch {
      // sessionStorage unavailable — skip the welcome, no harm done.
    }
    if (!fresh) return;

    setWelcomeName("bade bhai");
    // Hold "bade bhai" ~3.5s, fade it out (0.4s), then swap in the real name and
    // fade it back. The 50ms gap after fade-out guarantees the text swap lands
    // while fully transparent, so the change reads as a clean cross-fade.
    const fadeOut = setTimeout(() => setNameOpacity(0), 3500);
    const swap = setTimeout(() => {
      setWelcomeName(null);
      setNameOpacity(1);
    }, 3950);
    return () => {
      clearTimeout(fadeOut);
      clearTimeout(swap);
    };
  }, []);

  useEffect(() => {
    if (pool.length <= VISIBLE_COUNT) return;
    const id = setInterval(() => {
      setChipsOpacity(0);
      setTimeout(() => {
        setStart((s) => (s + VISIBLE_COUNT) % pool.length);
        setChipsOpacity(1);
      }, 250);
    }, ROTATE_MS);
    return () => clearInterval(id);
  }, [pool.length]);

  const suggestions = pool.length
    ? Array.from({ length: VISIBLE_COUNT }, (_, i) => pool[(start + i) % pool.length])
    : [];

  // No redirect to /login. A product whose whole pitch is "watch it show its
  // work" cannot make you register before you can watch it work — logged-out
  // visitors get a small allowance of real queries instead. The server owns
  // that limit; see backend/app/services/anonymous.py.

  const handleSearch = (query, documents) => {
    // If the search comes with file documents, stage them in the store so
    // Effect B on the research page can pick them up after navigation.
    if (documents?.length) {
      setPendingDocuments(documents);
    }
    router.push(`/research?q=${encodeURIComponent(query)}`);
  };

  // Only the auth bootstrap gates the render. The demo status arrives a moment
  // later and only adds a pill and a banner, so waiting on it would delay the
  // hero for every visitor to show a line of small print.
  if (isLoading) return null;

  // Prefer the user's personalized name; fall back to their first name.
  const realName = user?.preferred_name?.trim() || user?.name?.split(" ")[0];
  // During the welcome window, show "bade bhai" in place of the real name.
  const shownName = welcomeName ?? realName;

  const showQuota =
    !isAuthenticated && !demo.isLoading && typeof demo.remaining === "number";

  return (
    <main className="home-hero">
      <div className="home-hero-inner">
        {/* The spend ceiling is up. Cached answers still work, so this says
            what is actually true rather than hiding the search box. */}
        {demo.livePaused ? (
          <div className="demo-banner" role="status">
            <span className="demo-banner-dot" aria-hidden="true" />
            <span>{demo.liveMessage}</span>
            {demo.liveReason === "anon_daily" ? (
              <Link href="/login" className="demo-banner-link">
                Sign in
              </Link>
            ) : null}
          </div>
        ) : null}

        <div className="home-greeting">
          <h1 className="home-greeting-title">
            {greeting}
            {shownName ? (
              <span
                className="home-greeting-name"
                style={{ opacity: nameOpacity, transition: "opacity 0.4s ease" }}
              >
                , {shownName}
              </span>
            ) : null}
            <span className="home-greeting-dot">.</span>
          </h1>
          <p className="home-greeting-sub">
            What do you want to understand today?
          </p>
        </div>

        <div className="home-search">
          <SearchBar onSearch={handleSearch} mode="large" />

          {showQuota && demo.isExhausted ? (
            <div className="demo-wall">
              <p className="demo-wall-title">
                That&apos;s your {demo.limit} free queries.
              </p>
              <p className="demo-wall-sub">
                Sign in to keep going — it&apos;s free, and your research gets
                saved so you can come back to it.
              </p>
              <Link href="/login" className="demo-wall-cta">
                Sign in
              </Link>
            </div>
          ) : (
            <div
              className="suggestion-row"
              style={{ opacity: chipsOpacity, transition: "opacity 0.25s ease" }}
            >
              {suggestions.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => handleSearch(ex)}
                >
                  {ex}
                </button>
              ))}
            </div>
          )}

          {showQuota && !demo.isExhausted ? (
            <p className="demo-quota">
              {demo.remaining} of {demo.limit} free {demo.limit === 1 ? "query" : "queries"} left ·{" "}
              <Link href="/login" className="demo-quota-link">
                Sign in
              </Link>{" "}
              for more
            </p>
          ) : null}
        </div>
      </div>
    </main>
  );
}
