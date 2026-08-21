"""
Run the evaluation set through the real pipeline and score the results.

Usage:
    python -m evals.run_eval                 # full live run, compare to baseline
    python -m evals.run_eval --validate      # parse the query set only, no API calls
    python -m evals.run_eval --only prose-rag,table-vector-dbs
    python -m evals.run_eval --update-baseline
    python -m evals.run_eval --capture evals/data/pairs.jsonl

Exit code is 0 when every hard invariant held and no tracked metric regressed
past its tolerance; 1 otherwise, so CI fails on a real regression.

Why this calls the graph directly instead of POST /api/research:

- The answer cache sits in the endpoint, in front of the graph. Going through
  HTTP means a second run inside the cache TTL would replay stored answers and
  score the cache instead of the pipeline.
- The graph's final state carries everything worth measuring — rerank scores,
  sub-queries, the citation-ordered source list — while the SSE stream only
  carries what the UI needs.
- No auth, no rate limiting, no stream parsing in the way.

Note that the *search* cache is inside the researcher node and does still
apply, with a one-hour TTL. A re-run within the hour therefore measures
synthesis against identical retrieval. That is cheaper and less noisy, but it
is not a test of retrieval — flush Redis first if that is what you want.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import yaml

from evals.scorers import QueryScore, VALID_FORMATS, score_run

if TYPE_CHECKING:  # the capture module pulls in app.*, which --validate must not need
    from evals.capture import PairSink

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evals")

EVALS_DIR = Path(__file__).parent
QUERIES_PATH = EVALS_DIR / "queries.yaml"
BASELINE_PATH = EVALS_DIR / "baseline.json"
RESULTS_DIR = EVALS_DIR / "results"

# How many queries run at once. Groq enforces per-minute request limits and the
# pipeline makes 3+ calls per query, so this is deliberately modest — a run that
# spends its time collecting 429s measures nothing.
CONCURRENCY = 4

# How far a tracked metric may fall below the baseline before the run fails.
# Generous on purpose: the synthesizer runs at temperature 0.4 and the web
# itself changes between runs, so small movements are noise. This catches
# "citations stopped working", not "citations got 2% worse".
TOLERANCE = {
    "mean_coverage": 0.10,
    "mean_utilization": 0.10,
    "format_compliance_rate": 0.15,
    "triage_format_match_rate": 0.15,
}


# ---------------------------------------------------------------------------
# Query set
# ---------------------------------------------------------------------------

def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    """Load and validate the query set. Raises ValueError on a malformed file."""
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path.name} must contain a non-empty list of queries")

    seen: set[str] = set()
    for i, spec in enumerate(raw):
        if not isinstance(spec, dict):
            raise ValueError(f"query #{i} is not a mapping")

        qid = spec.get("id")
        if not qid:
            raise ValueError(f"query #{i} has no id")
        if qid in seen:
            raise ValueError(f"duplicate query id: {qid!r}")
        seen.add(qid)

        if not spec.get("query"):
            raise ValueError(f"{qid}: empty query text")

        fmt = spec.get("expected_format", "")
        if fmt and fmt not in VALID_FORMATS and fmt != "chat":
            raise ValueError(
                f"{qid}: expected_format {fmt!r} is not one of "
                f"{VALID_FORMATS + ('chat',)}"
            )

        for turn in spec.get("history") or []:
            if not isinstance(turn, dict) or "query" not in turn:
                raise ValueError(f"{qid}: history turns need at least a 'query'")

    return raw


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _initial_state(spec: dict) -> dict:
    """
    Build the graph's input state.

    Mirrors what routers.research sends, minus sse_callback: every node guards
    its callback use, so None simply means nothing is streamed.
    """
    history = [
        {"query": t.get("query", ""), "answer": t.get("answer", "")}
        for t in (spec.get("history") or [])
    ]
    return {
        "query": spec["query"],
        "max_iterations": 1,
        "history": history,
        "user_name": "",
        "iteration": 0,
        "sub_queries": [],
        "documents": [],
        "document_chunks": [],
        "search_results": [],
        "scraped_content": [],
        "ranked_chunks": [],
        "draft_answer": "",
        "citations": [],
        "all_sources": [],
        "reflection": {},
        "confidence": 0.0,
        "follow_up_suggestions": [],
        "phase": "starting",
        "error": "",
        "needs_web": True,
        "sse_callback": None,
    }


async def run_one(
    spec: dict,
    semaphore: asyncio.Semaphore,
    sink: Optional[PairSink] = None,
) -> QueryScore:
    """Run a single query through the graph and score whatever comes back."""
    # Imported here rather than at module scope so --validate works without the
    # full backend environment installed.
    from app.agents.graph import get_research_graph
    from app.services.usage import usage_scope

    async with semaphore:
        started = time.monotonic()
        accumulated: dict = {}
        error = ""

        try:
            with usage_scope() as usage:
                graph = get_research_graph()
                async for chunk in graph.astream(_initial_state(spec)):
                    if isinstance(chunk, dict):
                        for node_output in chunk.values():
                            if isinstance(node_output, dict):
                                accumulated.update(node_output)
            usage_payload = usage.as_dict()
        except Exception as e:
            # One broken query must not abort the run — the other nineteen
            # still carry information, and the failure is itself a result.
            logger.error("%s failed: %s", spec.get("id"), e)
            error = f"{type(e).__name__}: {e}"
            usage_payload = {}

        latency_ms = int((time.monotonic() - started) * 1000)
        score = score_run(
            spec=spec,
            final_state=accumulated,
            latency_ms=latency_ms,
            usage=usage_payload,
            error=error,
        )
        if sink is not None and sink.enabled and not error:
            sink.add(_capture_from(spec, accumulated))

        print(_line_for(score, spec), flush=True)
        return score


def _capture_from(spec: dict, state: dict) -> list[dict]:
    """
    Extract judge candidates from one run's final state.

    Guarded rather than trusted: capture is a side errand of the eval run, and a
    malformed answer must not take down the scoring it rides along with.
    """
    from app.config import get_settings
    from evals.capture import capture_pairs

    try:
        settings = get_settings()
        return capture_pairs(
            query_id=spec["id"],
            query=spec["query"],
            answer=state.get("draft_answer", ""),
            ranked_chunks=state.get("ranked_chunks", []),
            cited_sources=state.get("all_sources", []),
            max_sources=settings.max_cited_sources,
            max_chunks=settings.rerank_top_k,
        )
    except Exception as e:
        logger.error("capture failed for %s: %s", spec.get("id"), e)
        return []


def _line_for(score: QueryScore, spec: dict) -> str:
    problems = score.failures(allow_fallback=bool(spec.get("allow_fallback")))
    mark = "FAIL" if problems else "ok  "
    detail = f"  <- {'; '.join(problems)}" if problems else ""
    return (
        f"  {mark} {score.id:<32} {score.latency_ms/1000:5.1f}s  "
        f"cov {score.coverage:4.0%}  util {score.utilization:4.0%}  "
        f"${score.cost_usd:.5f}{detail}"
    )


async def run_all(
    specs: list[dict],
    sink: Optional[PairSink] = None,
) -> list[QueryScore]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    return list(await asyncio.gather(*(run_one(s, semaphore, sink) for s in specs)))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize(scores: list[QueryScore], specs_by_id: dict[str, dict]) -> dict:
    """Roll individual scores into the numbers that get compared over time."""
    ok = [s for s in scores if not s.error]
    research = [s for s in ok if s.mode == "research" and not s.fallback]

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 4) if values else 0.0

    hard_failures = {
        s.id: s.failures(allow_fallback=bool(specs_by_id[s.id].get("allow_fallback")))
        for s in scores
    }
    hard_failures = {k: v for k, v in hard_failures.items() if v}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "queries": len(scores),
        "errors": len(scores) - len(ok),
        "hard_failures": hard_failures,
        # Faithfulness / integrity
        "total_invalid_citations": sum(len(s.invalid_citation_markers) for s in scores),
        "mean_coverage": mean([s.coverage for s in research]),
        "mean_utilization": mean([s.utilization for s in research]),
        # Contract between stages
        "format_compliance_rate": mean([1.0 if s.format_ok else 0.0 for s in research]),
        "triage_format_match_rate": mean(
            [1.0 if s.triage_format_match else 0.0 for s in ok]
        ),
        # Retrieval health
        "mean_sources": mean([float(s.sources_provided) for s in research]),
        "mean_ranked_chunks": mean([float(s.ranked_chunks) for s in research]),
        "mean_top_rerank_score": mean([s.top_rerank_score for s in research]),
        # Cost / latency
        "mean_latency_ms": int(mean([float(s.latency_ms) for s in ok])),
        "total_cost_usd": round(sum(s.cost_usd for s in scores), 6),
        "total_tokens": sum(s.total_tokens for s in scores),
    }


def compare_to_baseline(summary: dict, baseline: Optional[dict]) -> list[str]:
    """Tracked metrics that fell further than their tolerance allows."""
    if not baseline:
        return []

    regressions: list[str] = []
    for metric, tolerance in TOLERANCE.items():
        before = baseline.get(metric)
        after = summary.get(metric)
        if before is None or after is None:
            continue
        if after < before - tolerance:
            regressions.append(
                f"{metric}: {after:.2%} vs baseline {before:.2%} "
                f"(tolerance {tolerance:.0%})"
            )
    return regressions


def baseline_update_blocked(summary: dict) -> str:
    """
    Why this run must not become the baseline, or "" if it may.

    A baseline is a standing promise that these numbers are acceptable. A run
    containing a fabricated citation or a crashed query is not acceptable at any
    number, so it cannot be allowed to set the bar — otherwise the one command
    that defines "good" is also the one command that can never fail.

    Regressions deliberately do NOT block: moving the bar is the entire purpose
    of --update-baseline, and a deliberate drop (a cheaper model, a smaller
    source cap) is a legitimate reason to run it.
    """
    failures = summary.get("hard_failures") or {}
    if failures:
        return (
            f"{len(failures)} query/queries violated a hard invariant: "
            f"{', '.join(sorted(failures))}"
        )
    return ""


def render_markdown(
    summary: dict, scores: list[QueryScore], regressions: list[str]
) -> str:
    """A report readable in the GitHub Actions job summary."""
    passed = not summary["hard_failures"] and not regressions
    lines = [
        f"# Eval run — {'PASS' if passed else 'FAIL'}",
        "",
        f"_{summary['generated_at']} · {summary['queries']} queries · "
        f"${summary['total_cost_usd']:.4f} · "
        f"{summary['mean_latency_ms'] / 1000:.1f}s mean_",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Fabricated citations | **{summary['total_invalid_citations']}** |",
        f"| Citation coverage | {summary['mean_coverage']:.0%} |",
        f"| Source utilization | {summary['mean_utilization']:.0%} |",
        f"| Format compliance | {summary['format_compliance_rate']:.0%} |",
        f"| Triage format match | {summary['triage_format_match_rate']:.0%} |",
        f"| Mean sources | {summary['mean_sources']:.1f} |",
        f"| Mean top rerank score | {summary['mean_top_rerank_score']:.3f} |",
        f"| Errored runs | {summary['errors']} |",
        "",
    ]

    if regressions:
        lines += ["## Regressions", ""]
        lines += [f"- {r}" for r in regressions] + [""]

    if summary["hard_failures"]:
        lines += ["## Hard failures", ""]
        for qid, problems in summary["hard_failures"].items():
            lines.append(f"- **{qid}** — {'; '.join(problems)}")
        lines.append("")

    lines += [
        "## Per query",
        "",
        "| Query | Mode | Format | Cov | Util | Cites | Latency | Cost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in scores:
        fmt = f"{s.declared_format or '—'}{'' if s.format_ok else ' ⚠'}"
        lines.append(
            f"| {s.id} | {s.mode or '—'} | {fmt} | {s.coverage:.0%} | "
            f"{s.utilization:.0%} | {len(s.invalid_citation_markers) or ''} | "
            f"{s.latency_ms / 1000:.1f}s | ${s.cost_usd:.5f} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> int:
    specs = load_queries()

    if args.only:
        wanted = {q.strip() for q in args.only.split(",") if q.strip()}
        unknown = wanted - {s["id"] for s in specs}
        if unknown:
            print(f"Unknown query ids: {sorted(unknown)}", file=sys.stderr)
            return 1
        specs = [s for s in specs if s["id"] in wanted]

    if args.validate:
        print(f"Query set OK — {len(specs)} queries, no duplicate ids, formats valid.")
        return 0

    # Imported here, like the graph itself, so --validate stays dependency-free.
    from evals.capture import PairSink

    sink = PairSink(Path(args.capture) if args.capture else None)
    if sink.enabled:
        print(f"Capturing judge candidates to {sink.path}")

    print(f"Running {len(specs)} queries (concurrency {CONCURRENCY})...\n")
    scores = await run_all(specs, sink)

    if sink.enabled:
        written = sink.flush()
        skipped = len(sink.records) - written
        note = f" ({skipped} already present)" if skipped else ""
        print(f"Captured {written} new judge candidates{note} -> {sink.path}")

    specs_by_id = {s["id"]: s for s in specs}
    summary = summarize(scores, specs_by_id)

    baseline = None
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    regressions = compare_to_baseline(summary, baseline)

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {"summary": summary, "results": [s.as_dict() for s in scores]}
    (RESULTS_DIR / "latest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    report = render_markdown(summary, scores, regressions)
    print("\n" + report)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")

    if args.update_baseline:
        blocked = baseline_update_blocked(summary)
        if blocked:
            print(
                f"\nRefusing to update the baseline — {blocked}.\n"
                "Fix those first, then re-run with --update-baseline.",
                file=sys.stderr,
            )
            return 1
        BASELINE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nBaseline updated: {BASELINE_PATH}")
        return 0

    if baseline is None:
        print(
            "\nNo baseline yet — run with --update-baseline once these numbers "
            "look right, and future runs will be compared against them."
        )

    return 1 if (summary["hard_failures"] or regressions) else 0


def main() -> int:
    # The report contains em dashes and a warning glyph. A Windows console
    # defaults to cp1252 and raises UnicodeEncodeError on those, which would
    # fail the run for a cosmetic reason.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover — not a TextIOWrapper
        pass

    parser = argparse.ArgumentParser(description="Run the research eval set.")
    parser.add_argument(
        "--validate", action="store_true",
        help="Parse and check the query set without calling any API.",
    )
    parser.add_argument(
        "--only", default="",
        help="Comma-separated query ids to run instead of the whole set.",
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Write this run's summary as the new baseline.",
    )
    parser.add_argument(
        "--capture", default="",
        help=(
            "Also append citation-judge candidates to this JSONL file. "
            "The full chunk text behind each [n] marker exists only while the "
            "graph is running, so this is the one chance to record it."
        ),
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
