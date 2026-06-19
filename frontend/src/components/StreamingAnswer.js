"use client";

import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import CitationTooltip from "./CitationTooltip";
import { SparklesIcon } from "@/components/Icons";

export default function StreamingAnswer({ answer = "", isStreaming = false, sources = [] }) {
  const [hoveredCitation, setHoveredCitation] = useState(null);

  // Custom components for react-markdown to render citation badges
  const markdownComponents = useMemo(() => ({
    // Override text rendering to replace [1], [2] with citation badges
    p: ({ children, ...props }) => {
      return (
        <p {...props}>
          {processChildren(children, sources, hoveredCitation, setHoveredCitation)}
        </p>
      );
    },
    li: ({ children, ...props }) => {
      return (
        <li {...props}>
          {processChildren(children, sources, hoveredCitation, setHoveredCitation)}
        </li>
      );
    },
    td: ({ children, ...props }) => {
      return (
        <td {...props}>
          {processChildren(children, sources, hoveredCitation, setHoveredCitation)}
        </td>
      );
    },
    th: ({ children, ...props }) => {
      return (
        <th {...props}>
          {processChildren(children, sources, hoveredCitation, setHoveredCitation)}
        </th>
      );
    },
  }), [sources, hoveredCitation]);

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
function processChildren(children, sources, hoveredCitation, setHoveredCitation) {
  if (!children) return children;

  return Array.isArray(children)
    ? children.map((child, i) =>
        processNode(child, i, sources, hoveredCitation, setHoveredCitation)
      )
    : processNode(children, 0, sources, hoveredCitation, setHoveredCitation);
}

function processNode(node, key, sources, hoveredCitation, setHoveredCitation) {
  if (typeof node !== "string") return node;

  // Match [1], [2], etc. in text
  const parts = node.split(/(\[\d+\])/g);
  if (parts.length === 1) return node;

  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      const idx = parseInt(match[1], 10);
      const source = sources[idx - 1]; // 1-indexed to 0-indexed

      return (
        <span
          key={`${key}-${i}`}
          style={{ position: "relative", display: "inline-block" }}
          onMouseEnter={() => setHoveredCitation(idx)}
          onMouseLeave={() => setHoveredCitation(null)}
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
          {hoveredCitation === idx && source && (
            <CitationTooltip source={source} />
          )}
        </span>
      );
    }
    return part;
  });
}
