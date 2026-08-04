/**
 * exportSession.js — real file-download helpers for PDF, Markdown, and DOCX.
 *
 * All three functions receive a `session` object:
 *   { title: string, createdBy: string, turns: Array<Turn> }
 *
 * A Turn looks like:
 *   { id, query, answer, sources: [{title, url, domain, snippet?, favicon?}],
 *     images, followUps, doneData }
 *
 * Dynamic imports keep jspdf and docx out of the initial bundle and safe from
 * SSR (these functions must only be called in browser event handlers).
 */

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Build a URL-safe filename from a session title.
 * @param {string} title
 * @param {string} ext  e.g. "pdf"
 * @returns {string}
 */
function makeFilename(title, ext) {
  const slug = (title || "session")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60);
  return `${slug || "session"}.${ext}`;
}

/**
 * Trigger a browser download from a Blob.
 * @param {Blob} blob
 * @param {string} filename
 */
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Parse a Markdown string into an array of block descriptors.
 * Recognised block types: "heading1", "heading2", "heading3",
 * "bullet", "table-row", "code", "blank", "paragraph".
 *
 * Only the text within is cleaned (strips inline markers like **bold**,
 * `code`, [n] citation refs) — the type carries structural intent so
 * PDF/DOCX renderers can use it without re-parsing.
 *
 * @param {string} md  Raw Markdown string
 * @returns {Array<{type: string, text: string, level?: number}>}
 */
function parseMarkdown(md) {
  if (!md) return [];
  const lines = md.split("\n");
  const blocks = [];
  let inCodeFence = false;

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();

    // Fenced code blocks (``` or ~~~). The fence lines themselves are markup,
    // not content, so they are dropped and everything between them is emitted
    // verbatim — inline cleaning would eat the *, _, and ` that code needs.
    if (/^\s*(```|~~~)/.test(line)) {
      inCodeFence = !inCodeFence;
      continue;
    }
    if (inCodeFence) {
      blocks.push({ type: "code", text: rawLine });
      continue;
    }

    // Blank line
    if (line.trim() === "") {
      blocks.push({ type: "blank", text: "" });
      continue;
    }

    // ATX headings ### / ## / #
    const headingMatch = line.match(/^(#{1,3})\s+(.*)/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      blocks.push({
        type: `heading${level}`,
        text: cleanInline(headingMatch[2]),
        level,
      });
      continue;
    }

    // Unordered bullets (-, *, +)
    const bulletMatch = line.match(/^[\s]*[-*+]\s+(.*)/);
    if (bulletMatch) {
      blocks.push({ type: "bullet", text: cleanInline(bulletMatch[1]) });
      continue;
    }

    // Numbered list
    const numberedMatch = line.match(/^[\s]*\d+\.\s+(.*)/);
    if (numberedMatch) {
      blocks.push({ type: "bullet", text: cleanInline(numberedMatch[1]) });
      continue;
    }

    // Pipe table rows — keep them as a single text line, cells separated by |
    if (line.includes("|") && line.trim().startsWith("|")) {
      // Skip divider rows (--|-- etc.)
      if (/^\|[\s:|-]+\|$/.test(line.trim())) continue;
      const cells = line
        .split("|")
        .map((c) => c.trim())
        .filter((c) => c.length > 0);
      blocks.push({ type: "table-row", text: cells.join("  |  ") });
      continue;
    }

    // Paragraph / everything else
    blocks.push({ type: "paragraph", text: cleanInline(line) });
  }

  return blocks;
}

/**
 * Strip inline Markdown markers and citation refs from a string.
 * Converts **bold** / *italic* / `code` to plain text and removes [n] refs.
 * @param {string} text
 * @returns {string}
 */
function cleanInline(text) {
  return (text || "")
    .replace(/\*\*\*(.+?)\*\*\*/g, "$1")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/__(.+?)__/g, "$1")
    .replace(/_(.+?)_/g, "$1")
    .replace(/`(.+?)`/g, "$1")
    .replace(/\[(\d+)\]/g, "") // citation markers
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1") // Markdown links → label
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "") // images → remove
    .trim();
}

/**
 * Format a date for document headers.
 * @returns {string}
 */
function formatDate() {
  return new Date().toLocaleDateString(undefined, {
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}

// ---------------------------------------------------------------------------
// 1. Markdown export
// ---------------------------------------------------------------------------

/**
 * Build a clean Markdown document from session data and trigger a .md download.
 * No external libraries required.
 *
 * @param {{ title: string, createdBy: string, turns: Array }} session
 */
export function exportMarkdown(session) {
  const { title = "Session", createdBy = "Researcher", turns = [] } = session;
  const lines = [];

  lines.push(`# ${title}`);
  lines.push(`*Created by ${createdBy} · ${formatDate()}*`);
  lines.push("");

  turns.forEach((turn, i) => {
    lines.push(`---`);
    lines.push(`## Q${i + 1}: ${turn.query || ""}`);
    lines.push("");

    if (turn.answer) {
      lines.push(turn.answer);
      lines.push("");
    }

    if (turn.sources && turn.sources.length > 0) {
      lines.push("### Sources");
      turn.sources.forEach((s, idx) => {
        const label = s.title || s.domain || s.url || `Source ${idx + 1}`;
        const url = s.url || "";
        lines.push(`${idx + 1}. [${label}](${url})`);
        if (s.snippet) lines.push(`   > ${s.snippet}`);
      });
      lines.push("");
    }
  });

  const content = lines.join("\n");
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  downloadBlob(blob, makeFilename(title, "md"));
}

// ---------------------------------------------------------------------------
// 2. PDF export
// ---------------------------------------------------------------------------

/**
 * Build a PDF document from session data and trigger a .pdf download.
 * Uses jsPDF (dynamically imported). Handles line-wrapping and page breaks.
 *
 * @param {{ title: string, createdBy: string, turns: Array }} session
 */
export async function exportPdf(session) {
  const { jsPDF } = await import("jspdf");
  const { title = "Session", createdBy = "Researcher", turns = [] } = session;

  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const pageW = doc.internal.pageSize.getWidth();
  const pageH = doc.internal.pageSize.getHeight();
  const marginL = 18;
  const marginR = 18;
  const marginT = 20;
  const marginB = 20;
  const usableW = pageW - marginL - marginR;
  let y = marginT;

  /** Add a page break if `needed` mm won't fit on the current page. */
  const ensureSpace = (needed = 8) => {
    if (y + needed > pageH - marginB) {
      doc.addPage();
      y = marginT;
    }
  };

  /** Render a wrapped text block and advance y. */
  const addWrappedText = (text, size, bold = false) => {
    doc.setFontSize(size);
    doc.setFont("helvetica", bold ? "bold" : "normal");
    const lines = doc.splitTextToSize(text || "", usableW);
    const lineH = size * 0.4;
    ensureSpace(lines.length * lineH + 2);
    doc.text(lines, marginL, y);
    y += lines.length * lineH + 2;
  };

  // ── Document header ──────────────────────────────────────────────────────
  doc.setFillColor(245, 245, 247);
  doc.rect(0, 0, pageW, 28, "F");

  doc.setFontSize(16);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(30, 30, 30);
  doc.text(title, marginL, 13);

  doc.setFontSize(9);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(100, 100, 100);
  doc.text(`${createdBy}  ·  ${formatDate()}`, marginL, 21);

  doc.setTextColor(30, 30, 30);
  y = 36;

  // ── Turns ────────────────────────────────────────────────────────────────
  turns.forEach((turn, i) => {
    ensureSpace(16);

    // Turn divider
    if (i > 0) {
      doc.setDrawColor(220, 220, 220);
      doc.line(marginL, y, pageW - marginR, y);
      y += 5;
    }

    // Question
    doc.setFillColor(230, 238, 255);
    const qLines = doc.splitTextToSize(`Q: ${turn.query || ""}`, usableW - 4);
    const qLineH = 10 * 0.4;
    const qH = qLines.length * qLineH + 4;
    ensureSpace(qH + 4);
    doc.roundedRect(marginL, y, usableW, qH, 2, 2, "F");
    doc.setFontSize(10);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(40, 60, 120);
    doc.text(qLines, marginL + 2, y + qLineH + 0.5);
    y += qH + 4;
    doc.setTextColor(30, 30, 30);

    // Answer blocks
    if (turn.answer) {
      const blocks = parseMarkdown(turn.answer);
      for (const block of blocks) {
        if (block.type === "blank") {
          y += 2;
          continue;
        }
        if (block.type === "heading1") {
          ensureSpace(8);
          addWrappedText(block.text, 13, true);
        } else if (block.type === "heading2") {
          ensureSpace(7);
          addWrappedText(block.text, 11, true);
        } else if (block.type === "heading3") {
          ensureSpace(6);
          addWrappedText(block.text, 10, true);
        } else if (block.type === "bullet") {
          ensureSpace(5);
          doc.setFontSize(9.5);
          doc.setFont("helvetica", "normal");
          const bulletLines = doc.splitTextToSize(block.text, usableW - 6);
          const bLineH = 9.5 * 0.4;
          ensureSpace(bulletLines.length * bLineH + 1);
          doc.text("•", marginL + 2, y);
          doc.text(bulletLines, marginL + 6, y);
          y += bulletLines.length * bLineH + 1.5;
        } else if (block.type === "table-row") {
          ensureSpace(5);
          doc.setFontSize(8.5);
          doc.setFont("courier", "normal");
          const tLines = doc.splitTextToSize(block.text, usableW);
          const tLineH = 8.5 * 0.4;
          ensureSpace(tLines.length * tLineH + 1);
          doc.text(tLines, marginL, y);
          y += tLines.length * tLineH + 1;
        } else if (block.type === "code") {
          ensureSpace(5);
          doc.setFontSize(8.5);
          doc.setFont("courier", "normal");
          doc.setTextColor(70, 70, 90);
          const cLines = doc.splitTextToSize(block.text || " ", usableW - 4);
          const cLineH = 8.5 * 0.4;
          ensureSpace(cLines.length * cLineH + 1);
          doc.text(cLines, marginL + 3, y);
          y += cLines.length * cLineH + 1;
          doc.setTextColor(30, 30, 30);
        } else {
          addWrappedText(block.text, 9.5);
        }
      }
      y += 3;
    }

    // Sources list
    if (turn.sources && turn.sources.length > 0) {
      ensureSpace(8);
      doc.setFontSize(9);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(80, 80, 80);
      doc.text("Sources", marginL, y);
      y += 5;
      doc.setFont("helvetica", "normal");

      turn.sources.slice(0, 20).forEach((s, idx) => {
        ensureSpace(7);
        const label = s.title || s.domain || s.url || `Source ${idx + 1}`;
        const urlText = s.url || "";
        doc.setFontSize(8.5);
        doc.setTextColor(60, 60, 60);
        const srcLines = doc.splitTextToSize(`[${idx + 1}] ${label}`, usableW - 4);
        const sLineH = 8.5 * 0.4;
        doc.text(srcLines, marginL + 3, y);
        y += srcLines.length * sLineH + 0.5;
        if (urlText) {
          doc.setFontSize(7.5);
          doc.setTextColor(80, 100, 200);
          const urlLines = doc.splitTextToSize(urlText, usableW - 10);
          doc.text(urlLines, marginL + 6, y);
          y += urlLines.length * 7.5 * 0.4 + 1;
        }
        doc.setTextColor(30, 30, 30);
      });
      y += 3;
    }
  });

  // ── Footer page numbers ───────────────────────────────────────────────────
  const totalPages = doc.internal.getNumberOfPages();
  for (let p = 1; p <= totalPages; p++) {
    doc.setPage(p);
    doc.setFontSize(8);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(160, 160, 160);
    doc.text(`Page ${p} of ${totalPages}`, pageW / 2, pageH - 8, { align: "center" });
  }

  doc.save(makeFilename(title, "pdf"));
}

// ---------------------------------------------------------------------------
// 3. DOCX export
// ---------------------------------------------------------------------------

/**
 * Build a Word document from session data and trigger a .docx download.
 * Uses the `docx` library (dynamically imported).
 *
 * @param {{ title: string, createdBy: string, turns: Array }} session
 */
export async function exportDocx(session) {
  const docx = await import("docx");
  const {
    Document,
    Packer,
    Paragraph,
    TextRun,
    HeadingLevel,
    AlignmentType,
    BorderStyle,
  } = docx;

  const { title = "Session", createdBy = "Researcher", turns = [] } = session;

  /** Helper: create a plain paragraph. */
  const para = (text, opts = {}) =>
    new Paragraph({ children: [new TextRun({ text: text || "", ...opts })] });

  /** Helper: create a blank spacer paragraph. */
  const blank = () => new Paragraph({ text: "" });

  const children = [];

  // ── Document title ────────────────────────────────────────────────────────
  children.push(
    new Paragraph({
      text: title,
      heading: HeadingLevel.TITLE,
    })
  );
  children.push(
    para(`Created by ${createdBy}  ·  ${formatDate()}`, {
      color: "888888",
      size: 18,
    })
  );
  children.push(blank());

  // ── Turns ────────────────────────────────────────────────────────────────
  turns.forEach((turn, i) => {
    // Divider paragraph (top border)
    children.push(
      new Paragraph({
        text: "",
        border: {
          top: { style: BorderStyle.SINGLE, size: 6, color: "CCCCCC" },
        },
      })
    );

    // Question heading
    children.push(
      new Paragraph({
        children: [
          new TextRun({
            text: `Q${i + 1}: ${turn.query || ""}`,
            bold: true,
            color: "2444AA",
            size: 24,
          }),
        ],
        heading: HeadingLevel.HEADING_2,
      })
    );
    children.push(blank());

    // Answer blocks
    if (turn.answer) {
      const blocks = parseMarkdown(turn.answer);
      for (const block of blocks) {
        if (block.type === "blank") {
          children.push(blank());
        } else if (block.type === "heading1") {
          children.push(
            new Paragraph({ text: block.text, heading: HeadingLevel.HEADING_1 })
          );
        } else if (block.type === "heading2") {
          children.push(
            new Paragraph({ text: block.text, heading: HeadingLevel.HEADING_2 })
          );
        } else if (block.type === "heading3") {
          children.push(
            new Paragraph({ text: block.text, heading: HeadingLevel.HEADING_3 })
          );
        } else if (block.type === "bullet") {
          children.push(
            new Paragraph({
              children: [new TextRun({ text: block.text })],
              bullet: { level: 0 },
            })
          );
        } else if (block.type === "table-row" || block.type === "code") {
          children.push(
            new Paragraph({
              children: [
                new TextRun({ text: block.text, font: "Courier New", size: 18 }),
              ],
            })
          );
        } else {
          children.push(para(block.text));
        }
      }
    }
    children.push(blank());

    // Sources
    if (turn.sources && turn.sources.length > 0) {
      children.push(
        new Paragraph({
          text: "Sources",
          heading: HeadingLevel.HEADING_3,
        })
      );
      turn.sources.forEach((s, idx) => {
        const label = s.title || s.domain || s.url || `Source ${idx + 1}`;
        children.push(
          new Paragraph({
            children: [
              new TextRun({
                text: `[${idx + 1}] ${label}`,
                bold: true,
                size: 18,
              }),
              ...(s.url
                ? [
                    new TextRun({ text: "  " }),
                    new TextRun({ text: s.url, color: "4060CC", size: 18 }),
                  ]
                : []),
            ],
          })
        );
        if (s.snippet) {
          children.push(
            para(s.snippet, { color: "666666", size: 16, italics: true })
          );
        }
      });
      children.push(blank());
    }
  });

  const doc = new Document({
    sections: [{ children }],
  });

  const blob = await Packer.toBlob(doc);
  downloadBlob(blob, makeFilename(title, "docx"));
}
