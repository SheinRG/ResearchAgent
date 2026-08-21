"""
Tests for citation-judge candidate capture (evals/capture.py).

Capture runs once per eval run and the data it produces is what a model gets
trained on, so a quiet bug here does not fail anything — it produces a dataset
that is subtly wrong and a result that cannot be trusted. The properties worth
pinning are the ones a later reader would otherwise have to take on faith:

* The evidence recorded for ``[n]`` is exactly what the synthesizer put behind
  ``[n]`` — same function, not a re-implementation of the same rules.
* Nothing is auto-labelled. Cited pairs are not assumed supported and swapped
  pairs are not assumed unsupported; both are candidates.
* The same inputs produce the same dataset, including which source got swapped
  in, so a re-capture is reproducible rather than merely similar.
* Appending is idempotent by pair_id, because the in-domain set is accumulated
  over many runs rather than produced by one.
"""

import json

from app.utils.citations import build_cited_context, select_evidence
from evals.capture import capture_pairs, write_pairs, PairSink


# --- fixtures --------------------------------------------------------------

def _chunk(url: str, text: str, title: str = "", domain: str = "") -> dict:
    return {
        "text": text,
        "score": 1.0,
        "source_url": url,
        "source_title": title or f"Title for {url}",
        "source_domain": domain or "example.test",
    }


# Sized like real chunks (config chunk_size is 500), so a single chunk clears
# _MIN_EVIDENCE_CHARS on its own — a one-source run must still produce pairs.
_A1 = ("Alpha reactors reached commercial output in 2031 across four separate coastal "
       "sites, according to the operator's annual regulatory filing.")
_A2 = ("Alpha's second phase added grid storage rated at nine gigawatt hours, "
       "commissioned eighteen months after the first units came online.")
_B1 = ("Beta province rejected the proposal after a lengthy public consultation "
       "that drew more than eleven thousand written submissions.")
_C1 = ("Gamma's tariff structure was revised twice during the same fiscal year, "
       "with the second revision backdated to the preceding quarter.")


def _ranked():
    return [
        _chunk("https://a.test/1", _A1),
        _chunk("https://b.test/1", _B1),
        _chunk("https://a.test/1", _A2),
        _chunk("https://c.test/1", _C1),
    ]


def _sources():
    return [
        {"url": "https://a.test/1", "title": "A", "domain": "a.test"},
        {"url": "https://b.test/1", "title": "B", "domain": "b.test"},
        {"url": "https://c.test/1", "title": "C", "domain": "c.test"},
    ]


_ANSWER = (
    "Commercial output was reached in 2031 at four separate sites [1].\n\n"
    "The provincial government rejected the proposal after consultation [2]."
)


def _capture(**overrides):
    kwargs = dict(
        query_id="q-1",
        query="what happened with alpha reactors",
        answer=_ANSWER,
        ranked_chunks=_ranked(),
        cited_sources=_sources(),
        swaps_per_sentence=1,
    )
    kwargs.update(overrides)
    return capture_pairs(**kwargs)


# --- the evidence must be what the model actually read ---------------------

def test_evidence_matches_what_the_synthesizer_put_behind_the_marker():
    """
    The one property the whole dataset rests on. If capture recovered different
    text than the prompt carried, every label would describe a pair the model
    was never shown — and nothing would look wrong.
    """
    _sources_out, context = build_cited_context(_ranked(), [], max_sources=8, max_chunks=12)
    records = _capture()

    cited = [r for r in records if r["pair_type"] == "cited" and r["citation_index"] == 1]
    assert cited, "expected a captured pair for marker [1]"
    evidence = cited[0]["evidence"]

    # Both chunks from source [1], in relevance order, exactly as prompted.
    assert evidence == f"{_A1}\n{_A2}"
    assert evidence in context


def test_source_order_matches_citation_numbering():
    order, _ = select_evidence(_ranked(), 8, 12)
    records = _capture()
    for record in records:
        expected_url = order[record["citation_index"] - 1]
        assert record["source_url"] == expected_url


# --- nothing is auto-labelled ----------------------------------------------

def test_no_record_is_labelled():
    """
    Cited != supported and swapped != unsupported. Auto-labelling either would
    assume the answer to the question the judge exists to ask.
    """
    for record in _capture():
        assert record["label"] is None


def test_pair_types_are_marked_but_not_scored():
    types = {r["pair_type"] for r in _capture()}
    assert types == {"cited", "swapped"}


def test_cited_indices_are_recorded_for_context():
    """A labeller needs to see what the sentence *meant* to rest on."""
    for record in _capture():
        assert record["cited_indices"]
        if record["pair_type"] == "cited":
            assert record["citation_index"] in record["cited_indices"]
        else:
            assert record["citation_index"] not in record["cited_indices"]


# --- swap selection --------------------------------------------------------

def test_swapped_source_is_never_one_the_sentence_cited():
    for record in _capture():
        if record["pair_type"] == "swapped":
            assert record["citation_index"] not in record["cited_indices"]


def test_capture_is_deterministic():
    """Same answers in, same dataset out — including which source got swapped."""
    first = _capture()
    second = _capture()
    assert [r["pair_id"] for r in first] == [r["pair_id"] for r in second]
    assert [r["citation_index"] for r in first] == [r["citation_index"] for r in second]


def test_swaps_can_be_disabled():
    records = _capture(swaps_per_sentence=0)
    assert records
    assert all(r["pair_type"] == "cited" for r in records)


def test_no_swap_when_there_is_no_alternative_source():
    single_chunk = [_chunk("https://a.test/1", _A1)]
    single_source = [{"url": "https://a.test/1", "title": "A", "domain": "a.test"}]
    records = _capture(
        answer="Commercial output was reached in 2031 at four separate sites [1].",
        ranked_chunks=single_chunk,
        cited_sources=single_source,
    )
    assert records
    assert all(r["pair_type"] == "cited" for r in records)


# --- what gets skipped -----------------------------------------------------

def test_uncited_sentences_are_skipped():
    """An uncited sentence is a coverage problem; there is no pair to judge."""
    records = _capture(answer="This sentence asserts something but cites nothing at all.")
    assert records == []


def test_markers_pointing_past_the_source_list_are_skipped():
    """A fabricated [9] has no evidence behind it — it is a different metric."""
    records = _capture(
        answer="Commercial output was reached in 2031 at four separate sites [9]."
    )
    assert records == []


def test_scrap_evidence_is_skipped():
    """Sub-threshold chunks are extraction noise, not evidence."""
    records = _capture(
        ranked_chunks=[_chunk("https://a.test/1", "Cookies.")],
        cited_sources=[{"url": "https://a.test/1", "title": "A", "domain": "a.test"}],
        answer="Commercial output was reached in 2031 at four separate sites [1].",
    )
    assert records == []


def test_empty_inputs_are_handled():
    assert _capture(answer="") == []
    assert _capture(cited_sources=[]) == []
    assert _capture(ranked_chunks=[]) == []


def test_more_sources_than_chunks_does_not_misalign():
    """
    A run that fell back to raw search results has sources with no chunks
    behind them. Pairing must cap at what both lists agree on rather than
    indexing off the end.
    """
    records = _capture(
        ranked_chunks=[_chunk("https://a.test/1", _A1)],
        cited_sources=_sources(),  # three sources, one chunked
        answer=(
            "Commercial output was reached in 2031 at four separate sites [1].\n\n"
            "The provincial government rejected the proposal after consultation [2]."
        ),
    )
    assert all(r["citation_index"] == 1 for r in records)


# --- writing ---------------------------------------------------------------

def test_write_pairs_appends_and_skips_duplicates(tmp_path):
    path = tmp_path / "data" / "pairs.jsonl"
    records = _capture()

    assert write_pairs(records, path) == len(records)
    # Re-running the same queries must not duplicate the dataset.
    assert write_pairs(records, path) == 0

    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == len(records)
    assert len({l["pair_id"] for l in lines}) == len(records)


def test_write_pairs_survives_a_truncated_tail(tmp_path):
    """A run killed mid-write must not make the file unappendable."""
    path = tmp_path / "pairs.jsonl"
    records = _capture()
    write_pairs(records, path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"pair_id": "half')

    assert write_pairs(records, path) == 0  # existing ids still recognised


def test_sink_is_inert_without_a_path():
    sink = PairSink(None)
    assert not sink.enabled
    sink.add(_capture())
    assert sink.flush() == 0


def test_sink_collects_then_writes_once(tmp_path):
    path = tmp_path / "pairs.jsonl"
    sink = PairSink(path)
    sink.add(_capture(query_id="q-1"))
    sink.add(_capture(query_id="q-2"))
    assert not path.exists()  # nothing written until flush

    written = sink.flush()
    assert written == len(sink.records)
    assert path.exists()
