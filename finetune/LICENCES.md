# Data licences

The pre-registration promises a notebook a stranger can re-run. That rules out
gated corpora (no click-through) and, because this repo is public and the
project is a portfolio piece rather than a private experiment, it rules out
non-commercial ones too. Every licence below was checked against the dataset
card on 2026-09-02 rather than recalled.

## Included

| Corpus | HF path | Licence | Gated | Why |
| --- | --- | --- | --- | --- |
| VitaminC | `tals/vitaminc` | CC BY-SA 3.0 | No | Contrastive Wikipedia revision pairs — near-identical evidence where one version supports a claim and the other does not. The closest public analogue of the deployed task. |
| MultiNLI | `nyu-mll/multi_nli` | CC BY 3.0 / CC BY-SA 3.0 / MIT (mixed, by genre) | No | Genre breadth, so the judge does not overfit to encyclopedic register. |

## Excluded on licence

| Corpus | Licence | Note |
| --- | --- | --- |
| ANLI | CC BY-NC 4.0 | Non-commercial. The obvious pick for adversarial NLI, and the obvious trap. |
| SciFact | CC BY-NC 2.0 | Non-commercial. Scientific-claim verification would otherwise fit the task well. |

Neither has an entry in `LABEL_MAP` in `backend/evals/trainset.py`, and a test
asserts they never gain one. A mapping is how a corpus quietly ends up in a
build six months later.

## Excluded for other reasons

| Corpus | Note |
| --- | --- |
| FEVER (`fever/fever`) | CC BY-SA 3.0 and ungated, so the licence is fine, and it keeps a label mapping. Not wired up because its evidence columns are wiki page/sentence *pointers* — resolving them to text needs a second join against the `wiki_pages` config. VitaminC is FEVER-derived with evidence already inlined, so the join buys nothing. |
| SNLI | Premises are image captions: one short sentence, concrete, present-tense. The evidence-length and register gap against 174-word retrieved chunks is worse than MultiNLI's for no added breadth. Mapping retained (it shares MultiNLI's integer scheme). |

## What this means for the derived dataset

Both included corpora are **share-alike** (CC BY-SA 3.0 for VitaminC, and
partly so for MultiNLI). The training set built from them inherits that: the
derived data is CC BY-SA 3.0, and redistributing it means attribution plus the
same licence.

This is separate from the repo's MIT licence, which covers the **code**. Code
MIT, data CC BY-SA — the distinction is worth stating plainly in the writeup
rather than letting a reader assume the repo licence covers everything in it.

The trained adapter weights are a derivative of the *base model* (Qwen3-1.7B,
Apache-2.0) trained on this data. Whether SA reaches through model weights is
genuinely unsettled and this project does not pretend to resolve it; the
honest move is to state what went in and let a user apply their own reading.
