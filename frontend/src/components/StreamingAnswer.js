"use client";

import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import CitationTooltip from "./CitationTooltip";
import { SparklesIcon } from "@/components/Icons";

function CitationBadge({ idx, source }) {
  const [hovered, setHovered] = useState(false);
  return (
    <span
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <a
        href={source?.url || "#"}
        target="_blank"
        rel="noopener noreferrer"
        className="citation-badge"
        onClick={(e) => {
          if (!source?.url) e.preventDefault();
        }}
      >
        {idx}
      </a>
      {hovered && source && <CitationTooltip source={source} />}
    </span>
  );
}

export default function StreamingAnswer({ answer = "", isStreaming = false, sources = [] }) {
  // Custom components for react-markdown to render citation badges
  const markdownComponents = useMemo(() => ({
    // Override text rendering to replace [1], [2] with citation badges
    p: ({ children, ...props }) => {
      return (
        <p {...props}>
          {processChildren(children, sources)}
        </p>
      );
    },
    li: ({ children, ...props }) => {
      return (
        <li {...props}>
          {processChildren(children, sources)}
        </li>
      );
    },
    td: ({ children, ...props }) => {
      return (
        <td {...props}>
          {processChildren(children, sources)}
        </td>
      );
    },
    th: ({ children, ...props }) => {
      return (
        <th {...props}>
          {processChildren(children, sources)}
        </th>
      );
    },
  }), [sources]);

  if (!answer) return null;

  return (
    <div className="answer-section">
      <div className="answer-container">
        <div className="answer-content">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
          >
            {answer}
          </ReactMarkdown>
          {isStreaming && <span className="typing-cursor" />}
        </div>
      </div>
    </div>
  );
}

/**
 * Process React children to replace [N] patterns with citation badges.
 */
function processChildren(children, sources) {
  if (!children) return children;

  return Array.isArray(children)
    ? children.map((child, i) => processNode(child, i, sources))
    : processNode(children, 0, sources);
}

function processNode(node, key, sources) {
  if (typeof node !== "string") return node;

  // Match [1], [2], etc. in text
  const parts = node.split(/(\[\d+\])/g);
  if (parts.length === 1) return node;

  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      const idx = parseInt(match[1], 10);
      const source = sources[idx - 1]; // 1-indexed to 0-indexed
      return <CitationBadge key={`${key}-${i}`} idx={idx} source={source} />;
    }
    return part;
  });
}
