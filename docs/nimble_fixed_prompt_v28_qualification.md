# Nimble fixed-prompt qualification — 8 October 2026

**Nimble qualifies as a fixed-prompt single judge under v2.8.**

Repository cleanup later on 8 October makes Ollama Nimble the default and removes the other judge integrations. The test counts and coverage table in this qualification report describe the source at qualification time. Current cleanup verification is recorded in [the delivery report](evaluation_suite_v2_report.md). Frozen qualification inputs, native requests and blinded reference labels remain unchanged.

The committed [validation summary](evaluation_suite_v2_validation.json) and [active qualification record](../evaluation/.judge_qualification_ollama.json) are available to PR reviewers. Links into `evaluation/artifacts/` below refer to local-only audit files, intentionally excluded from Git.

## Revised deployment contract

The user approved qualifying the frozen production prompt rather than requiring invariance to a different prompt. Qualification protocol is **v2.8**; the actual adapter and grading prompt remain **v2.7**, evidence remains v2.2, and query interpretation remains v1. The canonical grading order is 0, 1, 2, 3.

Qualification requires at least 400 distinct pairs across 20 query families, 100 fixed-prompt repeats with ≥99% published-decision agreement, 100 represented factual controls with ≥95% usable correct decisions, and zero unresolved production-request failures or malformed outputs. A published decision is the accepted grade when evidence is sufficient, otherwise unjudged. Stable abstention is repeatable; it is not control accuracy.

Reversed options remain an optional diagnostic, enabled with `--option-order-diagnostic`. They do not veto fixed-prompt qualification. A missing or failed diagnostic is never reported as passing; its execution failures are counted separately from production requests. Historical 81% and 83% option-order results remain unchanged and auditable under their original protocols.

Changed qualification policy produces a new fingerprint and cache identity. Old v2.7 pass flags cannot authorize v2.8. The frozen model, runtime, prompt, rubric and interpretation identities must match; changes require revalidation. No old failed report is rewritten as a fresh pass.

## Additional repairs

- Positive binary controls require a successful, evidence-sufficient grade ≥2. A weak grade 1 no longer counts as a correct positive.
- Control evidence uses a neutral synopsis, without “satisfying” or “exceeding” answer hints. This run uses historical years, not future-dated “classic” films.
- The public qualification command now saves the complete measured report. It previously dropped protocol, scope and sample-count fields that its own qualification validator requires.
- New regression tests cover disagreeing, skipped and failed optional diagnostics; production failures and failed repeats still block qualification. Stale v2.7 records are rejected.

## Frozen real run and independent reference

The installed model is `nimble:latest`, Q8_0, on Ollama 0.35.1. Weight digest: `9b953de7a5336756ece1cb1e8632e374b3dbdabe3d02d405cf8291da2d43a131`. The v2.8 qualification fingerprint is `1f005ca2ac68f282d51b2dd9ccc9a8ded5b01a34c0dd8e25e25888e2ecd860cd`.

The sample contains 400 previously unjudged query/item combinations across 20 families. Sampling excluded 2,032 historical combinations. Query wording can have appeared before. There are 600 planned real requests: 400 initial judgments, 100 repeats and 100 controls. No reversed requests or development examples are executed in this run. The prompt and inputs were frozen before inference; no model weights were downloaded or external services started.

Luna independently reviewed 80 cases selected before Nimble produced any outputs: four per family, including two reference-related candidates, one genre-overlap candidate and one catalog comparison. Luna was given the rubric and evidence without Nimble's judgments. Its input and labels were hash-locked before comparison. This is an independent LLM reference, not human truth or a controlled comparison of identical inference runtimes. Its labels contain six clear/complete positives, 22 weak matches and 52 irrelevant cases, so the sample gives limited evidence about positive-grade accuracy.

Qualification establishes operational reliability for the frozen prompt and represented controls. Subjective four-grade accuracy and the ability to identify the better recommendation system need separate evidence. Evidence coverage and LLM agreement are not accuracy estimates. No recommendation baseline or ranking weights are changed by judge qualification.

## Evidence

- [Frozen manifest](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/manifest.json)
- [Production prompt](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/frozen_prompt.json)
- [Fresh pilot inputs](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/pilot_inputs.json)
- [Neutral factual controls](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/factual_controls.json)
- [Real native requests](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/native_calls.jsonl)
- [Blinded Luna labels](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/blind_luna_labels.json)
- [Locked Luna provenance](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/blind_luna_manifest.json)

The previous v2.7 run and recommendation checks remain historical evidence in [the verified experiment archive](../evaluation/artifacts/archive/cleanup_20261008.zip), under their original paths. Their failed legacy recommendation gates are separate from this judge qualification.

## Completed result and decision

**Nimble qualifies as a fixed-prompt single judge under v2.8.** The model and native grading prompt remain unchanged. The qualification policy now treats option-order changes as optional diagnostics, as approved by the user.

The fresh run completed **600 real native requests**, **600 successful**, with **0 unresolved production failures** and **0 malformed outputs**. It evaluated 400 distinct pairs across 20 families, 100 fixed-prompt repeats and 100 neutral historical factual controls. Repeatability is **100%** (required ≥99%); usable factual-control correctness is **100%** (required ≥95%). Initial evidence coverage is **373/400 = 93.25%**, which is not accuracy. Elapsed time was **2448.719319 seconds**. No reversed calls or development checks were used to qualify this run.

Independent blinded Luna comparison is complete for **80/80 cases**. Published decisions agree on **51/80**; sufficiency agrees on **77/80**. Among **77 mutually sufficient cases**, exact grade agreement is **51/77**, within-one agreement is **77/77**, and binary relevance agreement is **72/77**. There are **0 differences of two or more grades**. These are LLM agreement measurements, not human accuracy.

The reference labels contain only six clear/complete positives. Of the **6 mutually sufficient Luna positives**, Nimble accepts **6**; of the **11 mutually sufficient Nimble positives**, Luna accepts **6**. The small positive sample limits generalization. Luna's self-check flagged nine rationales (three factual rationale issues, six arguable interpretations); a supplementary check confirmed another contradiction with an explicitly supplied Science Fiction genre. Locked labels and numerical agreement measurements were preserved. These reviewer checks are self-review, not new independent truth.

The saved record selects **bespoke-nimble-9b**, with `panel_mode=single_judge`. This is fixed-prompt qualification for automated single-judge evaluation, with deterministic constraints and explicit unjudged outputs. It does not establish human calibration of subjective grades, qualify a multi-model consensus panel, independently prove that judge-based A/B decisions select the better recommender, or promote recommendation weights/baselines. Existing missing provider/studio/franchise-phase facts remain a limitation.

**784 tests pass**, six dependency warnings, **58.87 seconds**. Exact overall coverage is **10609/12202 = 86.94476315358138%** across API, ETL and evaluation; all **40** modified production modules pass the unrounded ≥85% gate. Repository-wide and explicit changed-file hooks pass, including Black, Ruff and Mypy. The full coverage run was repeated after formatting so coverage line numbers match the saved source. Unit contracts use isolated stubs; they are separate from the real Nimble requests and Luna's authored reference judgments.


- [Measured qualification](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/qualification_report.json)
- [Complete Luna comparison and all disagreements](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/luna_comparison.json)
- [Scoped promotion decision](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/promotion_decision.json)
- [Reference rationale self-check](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/reference_rationale_audit.json)
- [Supplementary reference check](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/reference_rationale_addendum.json)

## Precise changed-module coverage

The gate compares counts before rounding.

| Production module | Covered / statements | Coverage |
|---|---:|---:|
| `api/core/cross_encoder.py` | 214/240 | 89.16666667% |
| `api/core/elasticsearch_client.py` | 28/28 | 100.00000000% |
| `api/core/filter_matcher.py` | 409/455 | 89.89010989% |
| `api/core/inference_metrics.py` | 47/47 | 100.00000000% |
| `api/core/persistent_cache.py` | 109/109 | 100.00000000% |
| `api/core/reranker.py` | 825/890 | 92.69662921% |
| `api/db/session.py` | 61/71 | 85.91549296% |
| `api/pipeline/context.py` | 218/225 | 96.88888889% |
| `api/pipeline/models.py` | 101/102 | 99.01960784% |
| `api/pipeline/reranker.py` | 81/81 | 100.00000000% |
| `api/pipeline/retriever/fusion.py` | 126/136 | 92.64705882% |
| `api/pipeline/runner.py` | 64/65 | 98.46153846% |
| `api/pipeline/scorer.py` | 351/380 | 92.36842105% |
| `evaluation/build_dataset_splits.py` | 101/103 | 98.05825243% |
| `evaluation/catalog_references.py` | 17/17 | 100.00000000% |
| `evaluation/coverage_gate.py` | 42/42 | 100.00000000% |
| `evaluation/datasets.py` | 167/177 | 94.35028249% |
| `evaluation/deterministic.py` | 171/192 | 89.06250000% |
| `evaluation/evaluate.py` | 952/1077 | 88.39368617% |
| `evaluation/evidence.py` | 64/64 | 100.00000000% |
| `evaluation/judge/__init__.py` | 9/9 | 100.00000000% |
| `evaluation/judge/base.py` | 51/54 | 94.44444444% |
| `evaluation/judge/clef.py` | 18/18 | 100.00000000% |
| `evaluation/judge/clm.py` | 18/18 | 100.00000000% |
| `evaluation/judge/consensus.py` | 178/182 | 97.80219780% |
| `evaluation/judge/kev.py` | 12/12 | 100.00000000% |
| `evaluation/judge/nimble.py` | 70/81 | 86.41975309% |
| `evaluation/judge/ollama.py` | 39/39 | 100.00000000% |
| `evaluation/judge/qualification.py` | 285/297 | 95.95959596% |
| `evaluation/judge/rubric.py` | 31/31 | 100.00000000% |
| `evaluation/judge/stub.py` | 56/58 | 96.55172414% |
| `evaluation/judge/systemone.py` | 60/60 | 100.00000000% |
| `evaluation/latency.py` | 120/123 | 97.56097561% |
| `evaluation/metrics.py` | 178/193 | 92.22797927% |
| `evaluation/models.py` | 285/299 | 95.31772575% |
| `evaluation/personalization.py` | 120/132 | 90.90909091% |
| `evaluation/private_benchmark.py` | 34/36 | 94.44444444% |
| `evaluation/query_interpretation.py` | 15/15 | 100.00000000% |
| `evaluation/runner.py` | 160/173 | 92.48554913% |
| `evaluation/trace.py` | 47/53 | 88.67924528% |

## Recommendation checks and environmental timing

PostgreSQL, Elasticsearch and the stored baseline were available. With model downloads disabled and Nimble idle, both legacy checks completed and returned 1. A/B nDCG@10 rose from 0.4430057237 to 0.4495396785; MAP rose from 0.3834023757 to 0.3875877871. Its quality non-regression checks pass, but the stored-baseline mean/P95 latency gate fails: mean 42.2736870960→768.4932193595 ms and P95 5.8750300072→546.3254799601 ms. This is a legacy stored-snapshot comparison, not a new primary ABBA/BAAB cache-cold experiment or a causal attribution of performance changes.

The first regression run overlapped host Modern Standby: Windows Kernel-Power events 506/507 show entry at 03:28:37 and exit at 09:53:36 local time on 8 October. One request therefore recorded 23,125,095.07 ms. That run and its measured values remain archived; its latency is excluded as performance evidence. Nimble qualification completed before standby and is unaffected.

The awake regression rerun completed in 37.111301 seconds, with no standby transition in its measured interval. It still fails nDCG@10 (approximately 0.4495 versus required 0.72), while ILD passes (approximately 0.6091 versus 0.45). Its descriptive request latency is mean 532.5 ms, P50 197.8 ms and P95 383.5 ms. No recommendation baseline or weights were promoted.

Complete real MovieLens/Tag Genome, synthetic personalization and private-holdout promotion tracks were not executed in this follow-up. No new real Clef, Kev or CLM model execution was performed. Mocked unit contracts do not substitute for these integrations. Subjective four-grade accuracy and the ability of judge-based A/B decisions to select the better recommender remain unproven by this bounded audit.

- [Available-service check outcomes](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/v28-legacy-checks.json)
- [Actual legacy A/B metrics](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/legacy_ab_comparison_report.json)
- [Preserved standby-interrupted run](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/legacy_regression_standby.log)
- [Verified standby interval](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/standby_timing_evidence.json)
- [Awake rerun timing validation](../evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/awake_power_validation.json)

## Use the qualified single judge

The CLI now defaults to the installed Ollama Nimble model and single-judge mode. The explicit equivalent is:

```text
python -m evaluation.evaluate --v2 --judge-runtime ollama --judge-config nimble --judgment-mode single_judge --track product --split dev --baseline ann_only --candidate default --backend elasticsearch --in-process
```

This starts a separate recommendation evaluation; it is not part of the 600-request qualification above. Unjudged outputs remain unjudged. Optional robustness measurement is available through `--qualify-judges --judge-runtime ollama --option-order-diagnostic`.
