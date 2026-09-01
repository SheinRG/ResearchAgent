"""
Tests for the gold/train contamination guard (evals/contamination.py).

Like the judge tests, these guard the properties that would let a *wrong*
number look plausible rather than the ones that would make the code crash. A
contaminated training set does not raise; it just quietly produces a better
score than the fine-tune earned, which is the single most expensive way this
project could fail.

So the cases here are the ways overlap sneaks past a naive check:

* **A different pair_id is not a different pair.** Same run, same evidence
  chunk, different sentence — the pair ids differ and the evidence does not.
* **Reformatting must not defeat a hash.** Whitespace and case are the two
  things that change for free between a capture run and a dataset build.
* **Paraphrase must not defeat a hash.** The exact-hash tier cannot see it,
  which is the whole reason the near-duplicate tier exists.
* **The manifest must not contain the answer key.** It is committed; a
  committed copy of the gold sentences would defeat the blind protocol.
* **The guard raises rather than filters.** A silent drop turns contamination
  into a smaller dataset nobody notices.
"""

import json

import pytest

from evals import contamination as cm


# --- fixtures --------------------------------------------------------------

GOLD = [
    {
        "pair_id": "gold1",
        "query_id": "steps-cloudflare-tunnel",
        "sentence": "Cloudflare Tunnel requires the cloudflared daemon on the origin host.",
        "evidence": "To expose a local service you install cloudflared and authenticate it.",
    },
    {
        "pair_id": "gold2",
        "query_id": "table-postgres-mysql",
        "sentence": "Postgres supports partial indexes; MySQL does not.",
        "evidence": "Partial indexes are a Postgres feature with no MySQL equivalent.",
    },
]


@pytest.fixture
def manifest():
    return cm.build_manifest(GOLD, seed=0, limit=2)


# --- the manifest is not the answer key ------------------------------------

def test_manifest_carries_no_evidence_and_no_labels(manifest):
    """
    It gets committed, so what it contains is public.

    Two things must never be in it. **Evidence text**, because the evidence is
    most of what a labeller reads and reproducing it in the repo is
    reproducing the gold set. **Labels**, because those are the answer key and
    they live in gold.jsonl and nowhere else.

    Sentence *content words* are deliberately present — the near-duplicate
    tier has to compare against something, and a bag of stemless words carries
    no label. That is a real trade, so it is asserted rather than assumed.
    """
    blob = json.dumps(manifest)
    assert "To expose a local service" not in blob
    assert "Partial indexes are a Postgres feature" not in blob

    assert set(manifest) == {
        "created", "seed", "limit", "n", "note",
        "pair_ids", "query_ids", "evidence_sha", "sentence_sha", "sentence_words",
    }
    assert not any("label" in key for key in manifest)
    assert "unsupported" not in blob


def test_manifest_is_stable_across_rebuilds():
    """Sorted output, so re-exporting an unchanged gold set is a no-op diff."""
    a = cm.build_manifest(GOLD, seed=0, limit=2)
    b = cm.build_manifest(list(reversed(GOLD)), seed=0, limit=2)
    for key in ("pair_ids", "query_ids", "evidence_sha", "sentence_sha"):
        assert a[key] == b[key]


# --- the tiers -------------------------------------------------------------

def test_clean_pair_passes(manifest):
    fresh = {
        "pair_id": "new1",
        "query_id": "some-other-run",
        "sentence": "Redis evicts keys under maxmemory using the configured policy.",
        "evidence": "The maxmemory-policy setting controls which keys Redis evicts.",
    }
    assert cm.reasons(fresh, manifest) == []


def test_same_run_is_rejected_even_with_a_new_pair_id(manifest):
    """
    The case that motivated the whole module.

    A different sentence from the same query is a different pair_id and the
    same retrieved corpus. On the real capture this is 97 of 111 candidates.
    """
    sibling = {
        "pair_id": "brand-new",
        "query_id": "steps-cloudflare-tunnel",
        "sentence": "A completely unrelated claim about something else entirely.",
        "evidence": "Different text, different chunk, nothing shared at all here.",
    }
    assert cm.reasons(sibling, manifest) == ["query_id"]


def test_reused_evidence_chunk_is_rejected(manifest):
    """Same chunk reached through a different run still leaks the test corpus."""
    pair = {
        "pair_id": "new2",
        "query_id": "unseen-run",
        "sentence": "An entirely fresh claim nobody has written down before now.",
        "evidence": "To expose a local service you install cloudflared and authenticate it.",
    }
    assert cm.reasons(pair, manifest) == ["evidence"]


def test_reformatting_does_not_defeat_the_hash(manifest):
    """Case and whitespace change for free between a capture and a build."""
    pair = {
        "pair_id": "new3",
        "query_id": "unseen-run",
        "sentence": "Fresh unrelated wording with no bearing on anything.",
        "evidence": "  TO EXPOSE a local service   you install\n cloudflared and AUTHENTICATE it.  ",
    }
    assert "evidence" in cm.reasons(pair, manifest)


def test_paraphrased_sentence_is_caught_by_the_near_duplicate_tier(manifest):
    """The exact-hash tier structurally cannot see this one."""
    pair = {
        "pair_id": "new4",
        "query_id": "unseen-run",
        # Same content words, reordered and re-punctuated.
        "sentence": "The cloudflared daemon is required on the origin host by Cloudflare Tunnel.",
        "evidence": "Some completely different supporting text about another topic.",
    }
    why = cm.reasons(pair, manifest)
    assert "sentence~" in why, why


def test_near_duplicate_does_not_fire_on_merely_same_topic(manifest):
    """
    A guard that rejects everything on-topic would starve the in-domain slice.

    Shared vocabulary is the normal case for same-domain data; only near-total
    overlap should count as the same claim.
    """
    pair = {
        "pair_id": "new5",
        "query_id": "unseen-run",
        "sentence": "Cloudflare offers a free plan with unlimited bandwidth for static sites.",
        "evidence": "Some completely different supporting text about another topic.",
    }
    assert cm.reasons(pair, manifest) == []


def test_reasons_reports_every_tier_not_just_the_first(manifest):
    """Which tier fires tells you whether it is a sourcing or paraphrase bug."""
    why = cm.reasons(GOLD[0], manifest)
    assert set(why) == {"pair_id", "query_id", "evidence", "sentence"}


def test_empty_sentence_does_not_match_everything(manifest):
    """A pair with no content words must not silently near-match all of gold."""
    pair = {
        "pair_id": "new6",
        "query_id": "unseen-run",
        "sentence": "   ",
        "evidence": "Some completely different supporting text about another topic.",
    }
    assert cm.reasons(pair, manifest) == []


# --- the guard behaves like a gate, not a filter ---------------------------

def test_partition_splits_and_explains(manifest):
    clean, dirty = cm.partition([GOLD[0], {
        "pair_id": "ok",
        "query_id": "unseen-run",
        "sentence": "Something entirely unrelated to the gold pairs above.",
        "evidence": "And evidence that shares nothing with them either.",
    }], manifest)
    assert [p["pair_id"] for p in clean] == ["ok"]
    assert len(dirty) == 1 and dirty[0][0]["pair_id"] == "gold1"


def test_assert_clean_raises_and_names_the_tiers(manifest):
    """Silent filtering would turn contamination into a dataset nobody audits."""
    with pytest.raises(ValueError) as exc:
        cm.assert_clean(GOLD, manifest)
    message = str(exc.value)
    assert "2 training pair(s)" in message
    assert "query_id=2" in message


def test_assert_clean_passes_on_clean_data(manifest):
    cm.assert_clean([{
        "pair_id": "ok",
        "query_id": "unseen-run",
        "sentence": "Something entirely unrelated to the gold pairs above.",
        "evidence": "And evidence that shares nothing with them either.",
    }], manifest)
