"""
Tests for the citation-judge baselines (evals/judge.py).

The measurement is the deliverable here, so these tests guard the properties
that would let a *wrong* number look plausible rather than the ones that would
make the code crash:

* **The judge stays as blind as the labeller.** If the query, source title, or
  pair type ever reach the prompt, the model is answering an easier question
  than the human did and the comparison to gold is void.
* **A hedging reply parses as its conclusion.** Reasoning models argue with
  themselves before answering; taking the first verdict word inverts them.
* **macro-F1 punishes the majority-class guesser.** On a set that is ~80%
  supported, plain accuracy would hand "always supported" a 0.8 and the
  fine-tune would be measured against a bar that means nothing.
* **The oracle threshold really is the baseline's best case.** It is meant to
  flatter the incumbent — a sweep that misses the optimum would understate the
  bar the fine-tune has to clear.
"""

import json

import pytest

from evals import judge as jd
from evals.label import SUPPORTED, UNSUPPORTED


# --- fixtures --------------------------------------------------------------

_SENTENCE = "Commercial output was reached in 2031 at four separate coastal sites."
_EVIDENCE = (
    "Alpha reactors reached commercial output in 2031 across four separate "
    "coastal sites, according to the operator's annual regulatory filing."
)


def _candidate(pair_id: str, **extra) -> dict:
    record = {
        "pair_id": pair_id,
        "query_id": "q-1",
        "query": "what happened with alpha reactors",
        "sentence": _SENTENCE,
        "pair_type": "cited",
        "source_url": "https://a.test/1",
        "source_title": "Alpha Annual Filing",
        "source_domain": "a.test",
        "evidence": _EVIDENCE,
        "label": None,
    }
    record.update(extra)
    return record


# --- blindness -------------------------------------------------------------

def test_prompt_shows_only_sentence_and_evidence():
    """The judge's view must match the hand-labeller's exactly."""
    cand = _candidate("p1")
    prompt = jd.build_prompt(cand["sentence"], cand["evidence"])

    assert cand["sentence"] in prompt
    assert cand["evidence"] in prompt

    # Everything the labeller is denied, the judge is denied too.
    for leak in (cand["query"], cand["source_title"], cand["source_domain"],
                 cand["source_url"], cand["pair_type"]):
        assert leak not in prompt


def test_prompt_names_the_failure_modes_overlap_misses():
    """Numbers, dates and polarity are exactly where lexical overlap agrees
    and the truth differs; the prompt has to ask about them."""
    prompt = jd.build_prompt(_SENTENCE, _EVIDENCE).lower()
    assert "number" in prompt and "polarity" in prompt
    assert "same topic is not enough" in prompt


# --- verdict parsing -------------------------------------------------------

@pytest.mark.parametrize("reply,expected", [
    ("supported", SUPPORTED),
    ("unsupported", UNSUPPORTED),
    ("Supported", SUPPORTED),
    ("  UNSUPPORTED\n", UNSUPPORTED),
    ("The verdict is: supported.", SUPPORTED),
])
def test_parse_verdict_plain(reply, expected):
    assert jd.parse_verdict(reply) == expected


def test_parse_verdict_takes_the_conclusion_not_the_first_thought():
    """A model that reasons out loud states the wrong answer first."""
    reply = (
        "At first glance this looks supported, but the evidence says 2031 "
        "while the sentence says 2013, so: unsupported"
    )
    assert jd.parse_verdict(reply) == UNSUPPORTED


def test_parse_verdict_does_not_match_supported_inside_unsupported():
    """Word boundaries, or every negative reply reads as positive."""
    assert jd.parse_verdict("unsupported") == UNSUPPORTED


@pytest.mark.parametrize("reply", ["", None, "I cannot determine this.", "maybe"])
def test_parse_verdict_unparseable_is_none(reply):
    """None, not a guess — an unparseable reply is a judge failure and has to
    be countable as one."""
    assert jd.parse_verdict(reply) is None


# --- lexical variants ------------------------------------------------------

def test_coverage_ignores_evidence_length_but_jaccard_does_not():
    """The point of the second variant: padding the evidence must not make a
    supported pair look unsupported."""
    padding = " Unrelated boilerplate about shipping policy and cookies." * 20
    long_evidence = _EVIDENCE + padding

    assert jd.coverage(_SENTENCE, long_evidence) == pytest.approx(
        jd.coverage(_SENTENCE, _EVIDENCE)
    )
    assert jd.token_overlap(_SENTENCE, long_evidence) < jd.token_overlap(_SENTENCE, _EVIDENCE)


def test_coverage_of_empty_sentence_is_zero():
    assert jd.coverage("", _EVIDENCE) == 0.0


def test_coverage_is_high_for_shared_vocabulary_with_different_claim():
    """Why the incumbent is untrustworthy, stated as a test: same words,
    opposite number, and the lexical score barely notices."""
    contradicting = _SENTENCE.replace("2031", "1997").replace("four", "nine")
    assert jd.coverage(contradicting, _EVIDENCE) > 0.6


# --- macro-F1 --------------------------------------------------------------

def test_macro_f1_punishes_the_majority_class_guesser():
    truths = [SUPPORTED] * 8 + [UNSUPPORTED] * 2
    always_supported = [SUPPORTED] * 10

    # Accuracy would be 0.80. Macro-F1 refuses to reward it.
    assert jd.macro_f1(truths, always_supported) < 0.5


def test_macro_f1_perfect_and_inverted():
    truths = [SUPPORTED, UNSUPPORTED, SUPPORTED, UNSUPPORTED]
    assert jd.macro_f1(truths, truths) == pytest.approx(1.0)
    flipped = [UNSUPPORTED, SUPPORTED, UNSUPPORTED, SUPPORTED]
    assert jd.macro_f1(truths, flipped) == pytest.approx(0.0)


def test_macro_f1_counts_unparseable_as_wrong():
    """An unparseable reply must cost the judge, not be quietly dropped."""
    truths = [SUPPORTED, UNSUPPORTED]
    assert jd.macro_f1(truths, [None, None]) == 0.0


# --- oracle threshold ------------------------------------------------------

def test_oracle_threshold_finds_a_perfect_split():
    scored = [(0.1, UNSUPPORTED), (0.2, UNSUPPORTED), (0.8, SUPPORTED), (0.9, SUPPORTED)]
    cut, f1 = jd.oracle_threshold(scored)
    assert f1 == pytest.approx(1.0)
    assert 0.2 < cut <= 0.8


def test_oracle_threshold_is_the_best_available_not_merely_a_good_one():
    """The baseline is meant to be flattered; a sweep that settles for second
    best would understate the bar."""
    scored = [
        (0.10, UNSUPPORTED), (0.30, SUPPORTED), (0.35, UNSUPPORTED),
        (0.50, SUPPORTED), (0.55, SUPPORTED), (0.90, SUPPORTED),
    ]
    _, best = jd.oracle_threshold(scored)

    brute = max(
        jd.macro_f1(
            [t for _, t in scored],
            [SUPPORTED if s >= cut else UNSUPPORTED for s, _ in scored],
        )
        for cut in [i / 1000 for i in range(1001)]
    )
    assert best >= brute - 1e-9


def test_oracle_threshold_on_empty_input():
    assert jd.oracle_threshold([]) == (0.0, 0.0)


# --- scoring ---------------------------------------------------------------

def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def test_score_judges_excludes_unclear_pairs(tmp_path):
    """`unclear` is excluded by design — it is a data problem reported
    separately, not a judgement the model got wrong."""
    candidates = [_candidate(f"p{i}") for i in range(3)]
    gold = {
        "p0": {"pair_id": "p0", "label": SUPPORTED},
        "p1": {"pair_id": "p1", "label": UNSUPPORTED},
        "p2": {"pair_id": "p2", "label": "unclear"},
    }
    pred_dir = tmp_path / "judge"
    _write(pred_dir / "m.jsonl", [
        {"pair_id": "p0", "verdict": SUPPORTED, "score": 1.0},
        {"pair_id": "p1", "verdict": UNSUPPORTED, "score": 0.0},
        {"pair_id": "p2", "verdict": SUPPORTED, "score": 1.0},
    ])

    rows = jd.score_judges(candidates, gold, pred_dir=pred_dir)
    assert rows[0]["n"] == 2
    assert rows[0]["macro_f1"] == pytest.approx(1.0)


def test_score_judges_ignores_pairs_absent_from_the_candidate_file(tmp_path):
    """Gold labels for pairs not in this candidate set must not be scored —
    otherwise a stale label file silently changes the denominator."""
    candidates = [_candidate("p0")]
    gold = {
        "p0": {"pair_id": "p0", "label": SUPPORTED},
        "ghost": {"pair_id": "ghost", "label": UNSUPPORTED},
    }
    pred_dir = tmp_path / "judge"
    _write(pred_dir / "m.jsonl", [
        {"pair_id": "p0", "verdict": SUPPORTED, "score": 1.0},
        {"pair_id": "ghost", "verdict": SUPPORTED, "score": 1.0},
    ])

    rows = jd.score_judges(candidates, gold, pred_dir=pred_dir)
    assert rows[0]["n"] == 1


def test_score_judges_uses_oracle_threshold_when_there_are_no_verdicts(tmp_path):
    """Lexical files carry scores only; scoring has to threshold them."""
    candidates = [_candidate(f"p{i}") for i in range(4)]
    gold = {
        "p0": {"pair_id": "p0", "label": UNSUPPORTED},
        "p1": {"pair_id": "p1", "label": UNSUPPORTED},
        "p2": {"pair_id": "p2", "label": SUPPORTED},
        "p3": {"pair_id": "p3", "label": SUPPORTED},
    }
    pred_dir = tmp_path / "judge"
    _write(pred_dir / "lexical-coverage.jsonl", [
        {"pair_id": "p0", "verdict": None, "score": 0.1},
        {"pair_id": "p1", "verdict": None, "score": 0.2},
        {"pair_id": "p2", "verdict": None, "score": 0.7},
        {"pair_id": "p3", "verdict": None, "score": 0.9},
    ])

    rows = jd.score_judges(candidates, gold, pred_dir=pred_dir)
    assert "oracle_threshold" in rows[0]
    assert rows[0]["macro_f1"] == pytest.approx(1.0)


def test_score_judges_counts_unparseable_replies(tmp_path):
    candidates = [_candidate(f"p{i}") for i in range(2)]
    gold = {
        "p0": {"pair_id": "p0", "label": SUPPORTED},
        "p1": {"pair_id": "p1", "label": UNSUPPORTED},
    }
    pred_dir = tmp_path / "judge"
    _write(pred_dir / "m.jsonl", [
        {"pair_id": "p0", "verdict": SUPPORTED, "score": 1.0},
        {"pair_id": "p1", "verdict": None, "score": 0.0},
    ])

    rows = jd.score_judges(candidates, gold, pred_dir=pred_dir)
    assert rows[0]["unparseable"] == 1


# --- prediction storage ----------------------------------------------------

def test_predictions_are_resumable_and_corrections_win(tmp_path):
    """Append-only: a rerun corrects in place rather than duplicating, so an
    interrupted Groq run resumes without re-paying for what it finished."""
    path = tmp_path / "m.jsonl"
    jd.append_prediction(path, jd._record("p0", "m", SUPPORTED, 1.0))
    jd.append_prediction(path, jd._record("p0", "m", UNSUPPORTED, 0.0))

    loaded = jd.load_predictions(path)
    assert len(loaded) == 1
    assert loaded["p0"]["verdict"] == UNSUPPORTED


def test_load_predictions_survives_a_truncated_tail(tmp_path):
    """A killed process mid-write must not make the whole file unreadable."""
    path = tmp_path / "m.jsonl"
    jd.append_prediction(path, jd._record("p0", "m", SUPPORTED, 1.0))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"pair_id": "p1", "verdi')

    loaded = jd.load_predictions(path)
    assert set(loaded) == {"p0"}


# --- the pre-registered bar ------------------------------------------------

def _row(judge, f1, n=100):
    return {"judge": judge, "n": n, "macro_f1": f1}


def test_bar_is_measured_against_the_stronger_lexical_variant():
    """Taking the weaker variant would lower the bar for free — the whole
    point of computing both is to be held to the harder one."""
    rows = [
        _row("lexical-jaccard", 0.55),
        _row("lexical-coverage", 0.62),
        _row("openai-gpt-oss-20b", 0.80),
    ]
    bar = jd.verdict_on_bar(rows, "openai/gpt-oss-20b")

    assert bar["strongest_lexical"]["judge"] == "lexical-coverage"
    assert bar["targets"]["beat_lexical"] == pytest.approx(0.77)


def test_gate_fires_when_the_prompted_judge_leaves_no_headroom():
    rows = [_row("lexical-coverage", 0.60), _row("openai-gpt-oss-20b", 0.95)]
    bar = jd.verdict_on_bar(rows, "openai/gpt-oss-20b")
    assert bar["gate"]["no_headroom"] is True


def test_gate_stays_open_when_there_is_room_to_improve():
    rows = [_row("lexical-coverage", 0.60), _row("openai-gpt-oss-20b", 0.82)]
    bar = jd.verdict_on_bar(rows, "openai/gpt-oss-20b")
    assert bar["gate"]["no_headroom"] is False
    assert bar["targets"]["stretch_reference"] == pytest.approx(0.79)


def test_zero_shot_target_is_absent_until_qwen_has_been_run():
    """The second must-hit clause cannot be invented from the judges that
    happen to exist — it stays missing until its baseline is measured."""
    rows = [_row("lexical-coverage", 0.60), _row("openai-gpt-oss-20b", 0.82)]
    bar = jd.verdict_on_bar(rows, "openai/gpt-oss-20b")
    assert "beat_zero_shot" not in bar["targets"]

    rows.append(_row("qwen3-1.7b-zero-shot", 0.65))
    bar = jd.verdict_on_bar(rows, "openai/gpt-oss-20b")
    assert bar["targets"]["beat_zero_shot"] == pytest.approx(0.75)


def test_bar_ignores_judges_with_no_scored_pairs():
    """A judge with n=0 has no number; letting it win 'strongest lexical'
    would set the bar off an empty file."""
    rows = [
        {"judge": "lexical-jaccard", "n": 0, "macro_f1": 0.0},
        _row("lexical-coverage", 0.60),
    ]
    bar = jd.verdict_on_bar(rows, "")
    assert bar["strongest_lexical"]["judge"] == "lexical-coverage"


# --- truncated reasoning ---------------------------------------------------

class _StubClient:
    """Records the token budget of every call and replays scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.budgets = []

    async def generate(self, prompt, *, system="", temperature=0.0, model="",
                       max_tokens=0, stage=""):
        self.budgets.append(max_tokens)
        return self.replies.pop(0) if self.replies else ""


def _run_predict(monkeypatch, tmp_path, replies, candidates):
    import app.services.llm as llm_module

    client = _StubClient(replies)
    monkeypatch.setattr(llm_module, "get_llm_client", lambda: client)
    monkeypatch.setattr(jd, "PRED_DIR", tmp_path)
    import asyncio
    asyncio.run(jd.predict_groq(candidates, "m", tpm_budget=10 ** 9))
    return client


def test_empty_reply_is_retried_with_more_room(monkeypatch, tmp_path):
    """An empty reply means reasoning ran out of budget, not that the model had
    no opinion. Recording it as wrong would score the harness and understate a
    baseline the fine-tune has to beat."""
    client = _run_predict(monkeypatch, tmp_path, ["", "supported"], [_candidate("p0")])

    recorded = jd.load_predictions(tmp_path / "m.jsonl")
    assert recorded["p0"]["verdict"] == SUPPORTED
    assert len(client.budgets) == 2
    assert client.budgets[1] > client.budgets[0]


def test_a_genuinely_unparseable_reply_is_recorded_as_a_failure(monkeypatch, tmp_path):
    """The retry must not turn into a loop that manufactures an answer — one
    extra attempt, then it counts against the judge."""
    client = _run_predict(
        monkeypatch, tmp_path, ["no idea", "still no idea"], [_candidate("p0")]
    )

    recorded = jd.load_predictions(tmp_path / "m.jsonl")
    assert recorded["p0"]["verdict"] is None
    assert len(client.budgets) == 2


def test_raw_is_kept_only_when_parsing_failed(monkeypatch, tmp_path):
    """Raw replies can quote the evidence, and these files are committed —
    so they are stored only where they say something the verdict doesn't."""
    _run_predict(
        monkeypatch, tmp_path,
        ["supported", "hmm", "hmm"],
        [_candidate("p0"), _candidate("p1")],
    )

    recorded = jd.load_predictions(tmp_path / "m.jsonl")
    assert "raw" not in recorded["p0"]
    assert recorded["p1"]["raw"] == "hmm"


def test_predict_lexical_is_idempotent(tmp_path, monkeypatch):
    """Re-running must not double-count — the file is a cache, not a log of
    attempts."""
    monkeypatch.setattr(jd, "PRED_DIR", tmp_path)
    candidates = [_candidate(f"p{i}") for i in range(5)]

    first = jd.predict_lexical(candidates)
    second = jd.predict_lexical(candidates)

    assert first["lexical-jaccard"] == 5
    assert second["lexical-jaccard"] == 0
