import "./globals.css";
import { Raleway } from "next/font/google";
import { ThemeProvider } from "@/components/ThemeProvider";
import { AuthProvider } from "@/components/AuthProvider";
import AppLayout from "@/components/AppLayout";

// Display typeface for the landing hero. Self-hosted at build time by
// next/font; exposed as a CSS variable so globals.css can scope it.
const raleway = Raleway({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-raleway",
});

export const metadata = {
  title: "goon.ai — Research, with receipts",
  description:
    "An autonomous AI research agent that searches the web, analyzes sources, and delivers comprehensive answers with verifiable citations.",
  keywords: ["AI research", "search agent", "cited answers", "deep research"],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={raleway.variable} suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <ThemeProvider>
          <AuthProvider>
            <AppLayout>
              <div className="app-container">{children}</div>
            </AppLayout>
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
