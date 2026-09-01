"""
Download the public slice and write it out as citation-support training pairs.

This is the thin half. Every decision that affects what the model learns —
the label mapping, the gold gate, de-duplication, class balance — lives in
``backend/evals/trainset.py``, which is stdlib-only and covered by CI. What is
here is field names, network calls, and a report, because those are the parts
that need `datasets` and cannot run in the backend test suite.

    python build_dataset.py --per-source 4000

Sources, and why these:

**VitaminC** (`tals/vitaminc`, CC BY-SA 3.0, 371k train rows) is the closest
public analogue of the deployed task, and not by accident. It is built from
Wikipedia revision pairs, so it contains *near-identical evidence chunks where
one supports a claim and the other does not*. That is precisely the case
`ROADMAP.md` argues lexical overlap cannot handle — same topic, shared
vocabulary, different evidence — which makes it the slice most likely to teach
what the fine-tune is being measured on.

**MultiNLI** (`nyu-mll/multi_nli`, CC BY / CC BY-SA / MIT) adds genre breadth
so the judge does not overfit to encyclopedic register. Its premises are
longer and more varied than SNLI's image captions, which matters given the
evidence-length gap noted in ``trainset.py``.

**FEVER** is deliberately *not* wired up despite having a label mapping: the
HF loader's evidence columns are wiki page/sentence pointers, and resolving
them to text needs a second join against the `wiki_pages` config. VitaminC is
FEVER-derived with the evidence already inlined, so the join buys nothing.

Excluded on licence: **ANLI** (CC BY-NC 4.0) and **SciFact** (CC BY-NC 2.0).
See LICENCES.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# The contract half lives in the backend. Importing it rather than vendoring a
# copy is what keeps the CI-tested rules and the Colab build from drifting;
# it costs nothing, because trainset/contamination are stdlib-only.
sys.path.insert(0, str(REPO / "backend"))

from evals import trainset as ts  # noqa: E402
from evals.contamination import load_manifest  # noqa: E402

DEFAULT_MANIFEST = HERE / "gold_manifest.json"
DEFAULT_OUT = HERE / "data" / "public.jsonl"

SOURCES = {
    "vitaminc": {
        "path": "tals/vitaminc",
        "split": "train",
        "sentence": "claim",
        "evidence": "evidence",
        "label": "label",
    },
    "mnli": {
        "path": "nyu-mll/multi_nli",
        "split": "train",
        "sentence": "hypothesis",
        "evidence": "premise",
        "label": "label",
    },
}


def stream_rows(spec: dict, limit: int):
    """
    Yield ``{sentence, evidence, label}`` dicts from one corpus.

    Streamed rather than downloaded whole: MultiNLI and VitaminC are ~400k
    rows each and the build only wants a few thousand. On a Colab box the
    difference is a minute against a disk-quota failure.
    """
    from datasets import load_dataset  # imported here so --help works bare

    stream = load_dataset(spec["path"], split=spec["split"], streaming=True)
    for i, row in enumerate(stream):
        if i >= limit:
            break
        yield {
            "sentence": row[spec["sentence"]],
            "evidence": row[spec["evidence"]],
            "label": row[spec["label"]],
        }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - not a TextIOWrapper
            pass

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--per-source", type=int, default=4000,
        help="Rows to read per corpus before mapping and filtering.",
    )
    parser.add_argument(
        "--ratio", type=float, default=1.0,
        help="Max majority-class examples per minority example.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sources", nargs="*", default=sorted(SOURCES),
        choices=sorted(SOURCES),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(
            f"No gold manifest at {manifest_path}.\n"
            "The build refuses to run without one: a build with no gold gate "
            "is how the gold set ends up in the training data.\n"
            "  cd backend && python -m evals.label --manifest --limit 200",
            file=sys.stderr,
        )
        return 1
    manifest = load_manifest(manifest_path)

    examples: list[dict] = []
    seen: set[str] = set()
    reports: dict[str, dict] = {}

    for name in args.sources:
        print(f"  {name}: reading {args.per_source} rows from {SOURCES[name]['path']} ...")
        rows = stream_rows(SOURCES[name], args.per_source)
        kept, report = ts.build(rows, manifest, source=name, seen=seen)
        examples.extend(kept)
        reports[name] = report
        print(f"    kept {report['kept']} of {report['rows']}")

    before = ts.compose(examples)
    examples = ts.balance(examples, seed=args.seed, ratio=args.ratio)
    after = ts.compose(examples)

    written = ts.write_jsonl(examples, args.out)
    summary = {
        "gold_manifest": {"n": manifest.get("n"), "created": manifest.get("created")},
        "per_source": reports,
        "before_balance": before,
        "after_balance": after,
        "written": written,
    }
    Path(args.out).with_suffix(".report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n  wrote {written} examples -> {args.out}")
    print(f"  by source: {after['by_source']}")
    print(f"  supported share: {before['supported_share']} -> {after['supported_share']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
