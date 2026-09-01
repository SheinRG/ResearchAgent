"""
Tests for the public-data training set (evals/trainset.py).

Same standard as the judge and contamination tests: guard the properties that
would let a *wrong* training set look like a right one. A mislabelled corpus
does not crash — it produces a model that scores plausibly and answers a
different question than the one deployed.

The properties worth defending:

* **`neutral` becomes `unsupported`.** The decision the module exists to make.
  Dropping neutral instead would train a contradiction detector, which is the
  failure that looks *best* on public benchmarks and worst in production.
* **An unmapped label raises.** Defaulting is how a corpus gets silently
  mislabelled.
* **The gold gate is applied to public data too.** Cheap, and the day someone
  adds a corpus built from the same web pages is the day it matters.
* **De-duplication spans sources.** The same pair from two corpora is one pair;
  counted twice it inflates the apparent dataset size and leaks across splits.
* **Balancing is deterministic and actually balances.** The mapping is 1:2 by
  construction, and an unbalanced set teaches the majority-class guess that
  macro-F1 exists to punish.
* **The report makes losses visible.** A slice that loses 90% of its rows to
  the gate must not look like a successful build.
"""

import pytest

from evals import trainset as ts
from evals.contamination import build_manifest
from evals.label import SUPPORTED, UNSUPPORTED


GOLD = [
    {
        "pair_id": "gold1",
        "query_id": "steps-cloudflare-tunnel",
        "sentence": "Cloudflare Tunnel requires the cloudflared daemon on the origin host.",
        "evidence": "To expose a local service you install cloudflared and authenticate it.",
    },
]


@pytest.fixture
def manifest():
    return build_manifest(GOLD, seed=0, limit=1)


def row(sentence, evidence, label):
    return {"sentence": sentence, "evidence": evidence, "label": label}


# --- the mapping decision --------------------------------------------------

def test_neutral_maps_to_unsupported_not_dropped():
    """
    The module's central decision.

    A citation pointing at a chunk that neither confirms nor denies the claim
    is a bad citation, and it is the most common real failure. Dropping these
    rows would train a contradiction detector instead.
    """
    assert ts.map_label("mnli", "neutral") == UNSUPPORTED
    assert ts.map_label("mnli", 1) == UNSUPPORTED
    assert ts.map_label("fever", "NOT ENOUGH INFO") == UNSUPPORTED


def test_entailment_maps_to_supported():
    assert ts.map_label("mnli", "entailment") == SUPPORTED
    assert ts.map_label("mnli", 0) == SUPPORTED
    assert ts.map_label("fever", "SUPPORTS") == SUPPORTED


def test_contradiction_and_neutral_are_indistinguishable_after_mapping():
    """Both are 'the citation does not establish this' — that is the point."""
    assert ts.map_label("mnli", "contradiction") == ts.map_label("mnli", "neutral")


def test_origin_label_survives_so_the_mapping_stays_auditable():
    """Without it, nobody can re-derive the set under a different mapping."""
    example = ts.to_example(
        sentence="s", evidence="e", label=UNSUPPORTED,
        source="mnli", origin_label="neutral",
    )
    assert example["origin_label"] == "neutral"
    assert example["label"] == UNSUPPORTED


def test_no_gold_label_is_dropped_not_guessed():
    """SNLI/MNLI use -1 when annotators failed to agree."""
    assert ts.map_label("mnli", -1) is None


def test_unmapped_label_raises(manifest):
    with pytest.raises(ts.UnknownLabel, match="unmapped label"):
        ts.map_label("fever", "PROBABLY")


def test_unknown_source_raises_and_names_the_known_ones():
    with pytest.raises(ts.UnknownLabel, match="no label mapping"):
        ts.map_label("some-new-corpus", "SUPPORTS")


@pytest.mark.parametrize("source", ["anli", "scifact"])
def test_non_commercial_corpora_have_no_mapping(source):
    """
    ANLI (CC BY-NC 4.0) and SciFact (CC BY-NC 2.0) are the two obvious picks
    that the licence rule excludes. A mapping is how a corpus ends up in a
    build, so the absence of one is the enforcement -- assert it, or someone
    adds it back as a convenience in six months.
    """
    with pytest.raises(ts.UnknownLabel, match="no label mapping"):
        ts.map_label(source, "SUPPORTS")


# --- the gate applies here too ---------------------------------------------

def test_public_rows_are_still_checked_against_gold(manifest):
    """Cheap insurance, and the corpora are built from the same open web."""
    rows = [
        row("Cloudflare Tunnel requires the cloudflared daemon on the origin host.",
            "Unrelated evidence text with nothing in common.", "SUPPORTS"),
        row("A completely independent claim about something else.",
            "Independent supporting evidence for that other claim.", "SUPPORTS"),
    ]
    kept, report = ts.build(rows, manifest, source="fever")
    assert report["dropped_contaminated"] == 1
    assert report["contaminated_sentence"] == 1
    assert [e["sentence"] for e in kept] == [
        "A completely independent claim about something else."
    ]


def test_report_makes_losses_visible(manifest):
    """A build that drops most of its input must not look like a success."""
    rows = [
        row("", "no sentence", "SUPPORTS"),
        row("no evidence", "", "SUPPORTS"),
        row("A fine claim about an unrelated topic.", "Fine evidence for it.", "SUPPORTS"),
    ]
    kept, report = ts.build(rows, manifest, source="fever")
    assert report["rows"] == 3
    assert report["dropped_empty"] == 2
    assert report["kept"] == 1 == len(kept)


# --- de-duplication --------------------------------------------------------

def test_duplicates_are_dropped_within_a_source(manifest):
    rows = [
        row("The same claim.", "The same evidence.", "SUPPORTS"),
        row("The  same   claim.", "The same evidence.", "SUPPORTS"),
    ]
    kept, report = ts.build(rows, manifest, source="fever")
    assert len(kept) == 1
    assert report["dropped_duplicate"] == 1


def test_deduplication_spans_sources(manifest):
    """Counted twice, the same pair inflates the set and leaks across splits."""
    seen: set[str] = set()
    pair = [row("A shared claim.", "Shared evidence.", "SUPPORTS")]
    first, _ = ts.build(pair, manifest, source="fever", seen=seen)
    second, report = ts.build(
        [row("A shared claim.", "Shared evidence.", "entailment")],
        manifest, source="mnli", seen=seen,
    )
    assert len(first) == 1 and len(second) == 0
    assert report["dropped_duplicate"] == 1


# --- balancing -------------------------------------------------------------

def make(n, label, tag):
    return [
        ts.to_example(sentence=f"{tag} sentence {i}", evidence=f"{tag} evidence {i}",
                      label=label, source="fever", origin_label=label)
        for i in range(n)
    ]


def test_balance_caps_the_majority_class():
    examples = make(10, SUPPORTED, "s") + make(40, UNSUPPORTED, "u")
    out = ts.balance(examples, seed=0)
    counts = {}
    for e in out:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    assert counts == {SUPPORTED: 10, UNSUPPORTED: 10}


def test_balance_allows_a_deliberate_ratio():
    examples = make(10, SUPPORTED, "s") + make(40, UNSUPPORTED, "u")
    out = ts.balance(examples, seed=0, ratio=2.0)
    counts = {}
    for e in out:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    assert counts == {SUPPORTED: 10, UNSUPPORTED: 20}


def test_balance_is_deterministic():
    examples = make(10, SUPPORTED, "s") + make(40, UNSUPPORTED, "u")
    a = ts.balance(examples, seed=7)
    b = ts.balance(examples, seed=7)
    assert [ts.example_key(e) for e in a] == [ts.example_key(e) for e in b]


def test_balance_rejects_a_ratio_below_one():
    """Below 1 it truncates both classes, which no caller means by 'balance'."""
    with pytest.raises(ValueError, match=">= 1"):
        ts.balance(make(4, SUPPORTED, "s") + make(4, UNSUPPORTED, "u"), ratio=0.5)


def test_balance_passes_through_a_single_class():
    examples = make(5, SUPPORTED, "s")
    assert len(ts.balance(examples)) == 5


# --- composition report ----------------------------------------------------

def test_compose_reports_the_share_that_matters():
    examples = make(3, SUPPORTED, "s") + make(9, UNSUPPORTED, "u")
    summary = ts.compose(examples)
    assert summary["total"] == 12
    assert summary["by_label"] == {SUPPORTED: 3, UNSUPPORTED: 9}
    assert summary["supported_share"] == 0.25


def test_roundtrip_through_jsonl(tmp_path):
    examples = make(3, SUPPORTED, "s")
    out = tmp_path / "train.jsonl"
    assert ts.write_jsonl(examples, out) == 3
    assert list(ts.read_jsonl(out)) == examples
