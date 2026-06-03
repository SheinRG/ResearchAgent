import "./globals.css";
import { ThemeProvider } from "@/components/ThemeProvider";

export const metadata = {
  title: "AI Research Agent — Deep Research, Cited Answers",
  description:
    "An autonomous AI research agent that searches the web, analyzes sources, and delivers comprehensive cited answers. Powered by local LLMs — 100% free, zero API keys.",
  keywords: ["AI research", "search agent", "cited answers", "Perplexity clone"],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <ThemeProvider>
          <div className="app-container">{children}</div>
        </ThemeProvider>
      </body>
    </html>
  );
}
