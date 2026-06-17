"use client";

import { useRouter } from "next/navigation";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import SearchBar from "@/components/SearchBar";
import { useAuth } from "@/hooks/useAuth";

// Time-aware openers, framed for someone who's here to *work* — so the late
// hours nudge ("burning the midnight oil"), they don't sign off ("good night").
// Phrased as statements so a ", Name" can be appended cleanly.
const GREETINGS = {
  morning: ["Good morning", "Morning", "Rise and shine", "Fresh start"],
  afternoon: ["Good afternoon", "Afternoon", "Hope the day's going well"],
  evening: ["Good evening", "Evening", "Winding down or digging in"],
  night: [
    "Burning the midnight oil",
    "Working late",
    "Late-night deep dive",
    "Still going strong",
  ],
};

// Rotating curiosity sub-line under the greeting.
const PHRASES = [
  "What's going on?",
  "What are you curious about?",
  "What do you want to know?",
  "What's on your mind?",
  "Let's dig into something.",
  "Ask me anything.",
];

function bucketForHour(hour) {
  if (hour >= 5 && hour < 12) return "morning";
  if (hour >= 12 && hour < 17) return "afternoon";
  if (hour >= 17 && hour < 22) return "evening";
  return "night";
}

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

export default function HomePage() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading } = useAuth();

  // Greeting + phrase depend on the client clock and Math.random, so resolve
  // them after mount to avoid a server/client hydration mismatch.
  const [greeting, setGreeting] = useState("");
  const [phrase, setPhrase] = useState("");

  useEffect(() => {
    setGreeting(pick(GREETINGS[bucketForHour(new Date().getHours())]));
    setPhrase(pick(PHRASES));
  }, []);

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/login");
    }
  }, [isAuthenticated, isLoading, router]);

  const handleSearch = (query) => {
    const encoded = encodeURIComponent(query);
    router.push(`/research?q=${encoded}`);
  };

  if (isLoading || !isAuthenticated) return null;

  const name = user?.name?.split(" ")[0];

  return (
    <main className="home-hero">
      <div className="home-hero-inner">
        <motion.div
          className="home-greeting"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.05 }}
        >
          <h1 className="home-greeting-title">
            {greeting}
            {name ? <span className="home-greeting-name">, {name}</span> : null}
          </h1>
          <p className="home-greeting-sub">{phrase}</p>
        </motion.div>

        <div className="home-search">
          <SearchBar onSearch={handleSearch} mode="large" />
        </div>
      </div>
    </main>
  );
}
