# Nimble sufficiency and contextual evidence — 9 October 2026

The enriched evidence profile resolves the observed false abstentions in this focused diagnostic, but it still accepts an unsupported award request. It is implemented as an explicit experimental version, **v2.3**, with its own qualification and baseline identities. The qualified v2.2 profile remains the default. The existing production baseline and qualification record are preserved.

## Changes

- Freeze contextual facts for 19 affected catalog items in `evaluation/fixtures/context_evidence_v2.3.json`. TMDB supplies production companies and positive Israel streaming offers with capture timestamps and source URLs. Best Picture wins for two films are sourced from the Academy's [1995](https://www.oscars.org/oscars/ceremonies/1995/memorable-moments) and [2002](https://www.oscars.org/oscars/ceremonies/2002/memorable-moments) records.
- Preserve unknown availability, missing award records, and incomplete coverage. An empty or partial offer list does not establish provider absence. An Israel offer does not establish availability in Canada. Renting or buying an item does not establish subscription access.
- Deliver the same frozen facts to qualification and pooled evaluation. Live request watch options cannot fill gaps in the frozen contextual snapshot.
- Retain the native sufficiency probability in `JudgeOutput`, the disk cache, and each model's adjudication record. The acceptance threshold remains 0.5. Old cached probabilities remain unknown; no number is reconstructed from a boolean.
- Give enriched evidence its own prompt/qualification fingerprint and judgment-cache namespace. A model/input evidence-version mismatch is rejected. The v2.2 character limit remains 4,096; v2.3 permits 8,192 characters for the additional facts, without truncating evidence.
- Add 14 factual sufficiency controls to enriched-profile qualification. These check clear matches, known violations, supported contextual facts and unknown mandatory facts. At least 95% correct usable decisions are required in addition to the existing 400 distinct pilot pairs, 100 repeats and 100 factual controls. The original qualification cannot authorize the enriched profile.

## Measured diagnostic

The inputs and expected decisions were frozen before inference. The judge received queries and evidence only; expected sufficiency and grade bounds were not included in model requests. Both profiles used the same pinned Nimble artifact and grading instructions. Two real uncached requests were made for each of 14 cases per profile: **56 native requests**, all successful, with no malformed or over-limit outputs. Repeats are not independent cases.

| Result | Original v2.2 | Enriched v2.3 |
| --- | ---: | ---: |
| Correct usable decisions | 12/28 | 26/28 |
| Distinct cases correct on both repeats | 6/14 | 13/14 |
| False abstentions | 10 | 0 |
| False sufficiency decisions | 4 | 2 |
| Execution failures | 0 | 0 |

The Matrix, Star Wars, Return of the Jedi, Toy Story 4, and Predator examples no longer abstained under v2.3. The enriched studio, Netflix/Israel and documented award examples also passed. Both requests for unknown Canadian availability correctly remained unjudged.

The remaining error is `unknown_award`: Predator is asked to satisfy an Academy Award Best Picture condition without a supplied award record. Nimble nevertheless reports sufficient evidence with probability **0.7858** and assigns grade 1. The required fact is unknown, so a usable grade is unsupported. The original profile also gave the wrong nonzero grade for a known runtime violation; the enriched profile rejected it correctly.

The effect combines additional facts and their presentation. This experiment does not isolate their individual contributions, measure recommendation improvements, estimate broad model accuracy, or rejudge the full 101-query baseline. Five cases were selected because of observed baseline abstentions; this is a diagnostic corpus, not a blind holdout.

**Decision:** v2.3 reaches 13/14 = 92.86%, below the 95% sufficiency-control threshold. It is not production-qualified or promoted. No new qualification pass flag or baseline is fabricated. The remaining award error needs a separately validated judgment-policy or model repair before adoption.

## Reproduce and compare

```text
python -m evaluation.evidence_assessment --output evaluation/artifacts/new_sufficiency_assessment.json
python -m evaluation.evaluate --v2 --split dev --evidence-version v2.3 --baseline ann_only --candidate default --backend elasticsearch
```

The second command is exploratory. Full/regression evaluations and baseline capture require the corresponding qualification record. A v2.2 reference is incompatible with v2.3, and v2.2 remains the default for normal evaluation. Subsequent adoption requires fresh full qualification, successful sufficiency controls, and a newly named production reference.

The qualification command selects its evidence explicitly and writes a separate record:

```text
python -m evaluation.evaluate --qualify-judges --evidence-version v2.3
```

The committed measurements are [native outputs](nimble_sufficiency_assessment_20261009.json), [frozen model inputs](nimble_sufficiency_assessment_20261009.inputs.json), and [the control manifest](../evaluation/fixtures/sufficiency_cases_v1.json). Native outputs retain sufficiency probabilities and provenance. The original snapshot's 94 unresolved judgments remain historical results; they have not been silently reduced using this small diagnostic.
