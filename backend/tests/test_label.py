"""
Tests for the gold-set labelling tool (evals/label.py).

The gold set is the only artifact licensed to produce a quotable number, so the
tests here are less about the code working and more about the discipline
holding. Three properties matter, and all three are silent when broken:

* **The labeller stays blind.** If `pair_type`, the query, or the source domain
  ever leaks into the display, the labels start agreeing with what capture
  already assumed and the gold set stops being independent evidence.
* **Order is shuffled but reproducible.** Adjacent cited/swapped pairs for the
  same sentence would be judged relative to each other; a random order that
  changes between sessions would make a resumed session inconsistent.
* **Labels survive.** Every answer is written as it is given, so a closed
  window costs one pair rather than an hour.
"""

import json

import pytest

from evals import label as lab


# --- fixtures --------------------------------------------------------------

_SENTENCE = "Commercial output was reached in 2031 at four separate coastal sites."
_EVIDENCE = (
    "Alpha reactors reached commercial output in 2031 across four separate "
    "coastal sites, according to the operator's annual regulatory filing."
)


def _candidate(pair_id: str, pair_type: str = "cited", **extra) -> dict:
    record = {
        "pair_id": pair_id,
        "query_id": "q-1",
        "query": "what happened with alpha reactors",
        "sentence": _SENTENCE,
        "cited_indices": [1],
        "citation_index": 1,
        "pair_type": pair_type,
        "source_url": "https://a.test/1",
        "source_title": "Alpha Annual Filing",
        "source_domain": "a.test",
        "evidence": _EVIDENCE,
        "evidence_chunks": 1,
        "label": None,
    }
    record.update(extra)
    return record


def _flat(text: str) -> str:
    """Collapse wrapping so a wrapped phrase can still be searched for."""
    return " ".join(text.split())


def _candidates(n: int = 6) -> list[dict]:
    return [
        _candidate(f"pair-{i:02d}", "cited" if i % 2 == 0 else "swapped")
        for i in range(n)
    ]


class _Scripted:
    """Feeds canned answers to run_session and records what was displayed."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.output: list[str] = []

    def prompt(self, _msg: str) -> str:
        return self.answers.pop(0) if self.answers else "q"

    def write(self, text: str) -> None:
        self.output.append(text)

    @property
    def shown(self) -> str:
        return "".join(self.output)


# --- blinding --------------------------------------------------------------

def test_render_shows_only_sentence_and_evidence():
    """
    The judge sees sentence + evidence. A labeller who sees more is holding the
    model to a standard it structurally cannot meet.
    """
    # Normalised because render wraps to the terminal width, so long evidence is
    # not a contiguous substring of the screen.
    screen = _flat(lab.render(_candidate("pair-01"), 1, 10))
    assert _SENTENCE in screen
    assert _EVIDENCE in screen


def test_render_hides_the_capture_pair_type():
    """`swapped` is a hint about the answer — showing it manufactures agreement."""
    for pair_type in ("cited", "swapped"):
        screen = _flat(lab.render(_candidate("pair-01", pair_type), 1, 10))
        assert pair_type not in screen


def test_render_hides_query_source_and_url():
    candidate = _candidate("pair-01")
    screen = _flat(lab.render(candidate, 1, 10))
    assert candidate["query"] not in screen
    assert candidate["source_title"] not in screen
    assert candidate["source_domain"] not in screen
    assert candidate["source_url"] not in screen


def test_session_never_displays_the_pair_type(tmp_path):
    scripted = _Scripted("s", "u", "k", "q")
    lab.run_session(
        _candidates(6), tmp_path / "gold.jsonl",
        prompt=scripted.prompt, write=scripted.write,
    )
    assert "swapped" not in scripted.shown
    assert "cited" not in scripted.shown


# --- ordering --------------------------------------------------------------

def test_pending_excludes_already_labelled():
    candidates = _candidates(6)
    labelled = {"pair-00": {"label": lab.SUPPORTED}, "pair-03": {"label": lab.UNSUPPORTED}}
    queue = lab.pending(candidates, labelled)
    ids = {c["pair_id"] for c in queue}
    assert ids == {"pair-01", "pair-02", "pair-04", "pair-05"}


def test_pending_is_shuffled_but_reproducible():
    candidates = _candidates(12)
    first = [c["pair_id"] for c in lab.pending(candidates, {}, seed=7)]
    second = [c["pair_id"] for c in lab.pending(candidates, {}, seed=7)]
    original = [c["pair_id"] for c in candidates]

    assert first == second, "a resumed session must continue the same order"
    assert first != original, "adjacent pairs for one sentence must not stay adjacent"
    assert sorted(first) == sorted(original), "shuffling must not drop or add pairs"


def test_recheck_sample_only_returns_labelled_pairs():
    candidates = _candidates(6)
    labelled = {"pair-01": {}, "pair-04": {}}
    sample = lab.recheck_sample(candidates, labelled, count=5)
    assert {c["pair_id"] for c in sample} == {"pair-01", "pair-04"}


# --- persistence -----------------------------------------------------------

def test_labels_are_written_as_they_are_given(tmp_path):
    """An hour of labelling must not depend on exiting cleanly."""
    path = tmp_path / "gold.jsonl"
    scripted = _Scripted("s", "u", "q")  # quits before the queue empties

    recorded = lab.run_session(
        _candidates(6), path, prompt=scripted.prompt, write=scripted.write
    )

    assert recorded == 2
    first, _ = lab.load_labels(path)
    assert len(first) == 2
    assert set(r["label"] for r in first.values()) == {lab.SUPPORTED, lab.UNSUPPORTED}


def test_label_file_carries_no_scraped_text(tmp_path):
    """
    Only ids and labels, so the gold set is committable while the candidates it
    refers to are not.
    """
    path = tmp_path / "gold.jsonl"
    scripted = _Scripted("s", "q")
    lab.run_session(_candidates(2), path, prompt=scripted.prompt, write=scripted.write)

    body = path.read_text(encoding="utf-8")
    assert _SENTENCE not in body
    assert _EVIDENCE not in body
    record = json.loads(body.splitlines()[0])
    assert set(record) == {"pair_id", "label", "note", "pass", "at"}


def test_invalid_answers_are_rejected_without_recording(tmp_path):
    path = tmp_path / "gold.jsonl"
    scripted = _Scripted("yes", "maybe", "s", "q")
    recorded = lab.run_session(
        _candidates(3), path, prompt=scripted.prompt, write=scripted.write
    )
    assert recorded == 1
    assert "Enter s, u, k, or q." in scripted.shown


def test_append_label_rejects_an_unknown_label(tmp_path):
    with pytest.raises(ValueError):
        lab.append_label(tmp_path / "gold.jsonl", "pair-01", "probably")


def test_a_later_label_supersedes_an_earlier_one(tmp_path):
    """Correcting a mistake is another append, not an edit."""
    path = tmp_path / "gold.jsonl"
    lab.append_label(path, "pair-01", lab.SUPPORTED)
    lab.append_label(path, "pair-01", lab.UNSUPPORTED)
    first, _ = lab.load_labels(path)
    assert first["pair-01"]["label"] == lab.UNSUPPORTED


def test_passes_are_kept_apart(tmp_path):
    path = tmp_path / "gold.jsonl"
    lab.append_label(path, "pair-01", lab.SUPPORTED, pass_no=1)
    lab.append_label(path, "pair-01", lab.UNSUPPORTED, pass_no=2)
    first, recheck = lab.load_labels(path)
    assert first["pair-01"]["label"] == lab.SUPPORTED
    assert recheck["pair-01"]["label"] == lab.UNSUPPORTED


def test_truncated_tail_does_not_break_loading(tmp_path):
    path = tmp_path / "gold.jsonl"
    lab.append_label(path, "pair-01", lab.SUPPORTED)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"pair_id": "pair-02", "lab')
    first, _ = lab.load_labels(path)
    assert set(first) == {"pair-01"}


def test_missing_candidates_file_explains_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        lab.load_candidates(tmp_path / "nope.jsonl")
    assert "--capture" in str(exc.value)


# --- reporting -------------------------------------------------------------

def test_stats_report_balance_and_progress():
    candidates = _candidates(6)
    labelled = {
        "pair-00": {"label": lab.SUPPORTED},
        "pair-01": {"label": lab.UNSUPPORTED},
        "pair-02": {"label": lab.UNCLEAR},
    }
    stats = lab.label_stats(candidates, labelled)

    assert stats["candidates"] == 6
    assert stats["labelled"] == 3
    assert stats["remaining"] == 3
    assert stats["scoreable"] == 2
    assert stats["supported_rate"] == 0.5
    assert stats["unclear_rate"] == pytest.approx(1 / 3)


def test_stats_break_down_by_capture_type_after_the_fact():
    """
    Hidden while labelling, reported afterwards: if swapped pairs come back
    mostly 'supported', the negatives are not negatives and the data plan needs
    revisiting.
    """
    candidates = _candidates(4)  # pair-00/02 cited, pair-01/03 swapped
    labelled = {
        "pair-00": {"label": lab.SUPPORTED},
        "pair-01": {"label": lab.UNSUPPORTED},
        "pair-02": {"label": lab.SUPPORTED},
        "pair-03": {"label": lab.SUPPORTED},
    }
    by_type = lab.label_stats(candidates, labelled)["by_pair_type"]
    assert by_type["cited"][lab.SUPPORTED] == 2
    assert by_type["swapped"][lab.UNSUPPORTED] == 1
    assert by_type["swapped"][lab.SUPPORTED] == 1


def test_agreement_counts_only_pairs_labelled_twice():
    first = {"a": {"label": lab.SUPPORTED}, "b": {"label": lab.UNSUPPORTED}, "c": {"label": lab.SUPPORTED}}
    second = {"a": {"label": lab.SUPPORTED}, "b": {"label": lab.SUPPORTED}}
    result = lab.agreement(first, second)

    assert result["compared"] == 2
    assert result["agreed"] == 1
    assert result["rate"] == 0.5
    assert result["disagreements"] == [
        {"pair_id": "b", "first": lab.UNSUPPORTED, "second": lab.SUPPORTED}
    ]


def test_agreement_counts_a_flip_to_unclear_as_disagreement():
    """Dropping these would flatter the ceiling the model gets compared against."""
    result = lab.agreement(
        {"a": {"label": lab.SUPPORTED}},
        {"a": {"label": lab.UNCLEAR}},
    )
    assert result["rate"] == 0.0


def test_agreement_with_no_overlap_is_reported_as_zero_compared():
    result = lab.agreement({"a": {"label": lab.SUPPORTED}}, {})
    assert result["compared"] == 0
    assert result["disagreements"] == []
