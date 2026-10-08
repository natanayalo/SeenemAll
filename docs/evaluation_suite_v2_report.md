# Evaluation Suite v2 delivery report

Updated 8 October 2026 after the Nimble-only cleanup.

Reviewers can inspect the committed [validation summary](evaluation_suite_v2_validation.json), including exact module coverage, real qualification totals, reference agreement and legacy recommendation deltas. Links into `evaluation/artifacts/` below refer to local audit evidence, intentionally excluded from Git; raw logs and the recovery archive are not part of the PR.

## Current judge

Nimble through Ollama is the only configured production judge. The evaluation CLI defaults to `--judge-config nimble --judge-runtime ollama --judgment-mode single_judge`. Clef, Kev and CLM adapters, their discovery entries and the unused Transformers Nimble loader are removed. Production consensus requests are rejected; deterministic stubs and the shared consensus/qrels engine remain for isolated contract tests and existing qrels compatibility. Qualification never falls back to a stub or an unrelated model.

The frozen grading prompt is unchanged (adapter v2.7), and qualification remains v2.8. The active qualification record retains Nimble only; earlier records remain archived. Qualification requires at least 400 distinct pairs across 20 families, 100 repeats, 100 factual controls, at least 99% repeatability and 95% usable control correctness, and zero unresolved failures or malformed responses. Option-order sensitivity remains an optional diagnostic. Distribution parsing, abstention handling, deterministic constraints, provenance fingerprints and judgment-cache invalidation are retained.

The real qualification completed 600 successful native requests: 400 distinct pairs, 100 repeats and 100 neutral factual controls. Repeatability and factual controls each passed 100/100. The independently authored Luna reference covers 80 cases; exact grade agreement is 51/77 mutually sufficient cases, and all 77 agree within one grade. These establish bounded reliability and LLM agreement, not human-calibrated subjective accuracy. See [the frozen qualification report](nimble_fixed_prompt_v28_qualification.md).

```text
python -m evaluation.evaluate --v2 --track product --split dev --baseline ann_only --candidate default --backend elasticsearch --in-process
```

This command runs a separate recommendation evaluation. It does not rerun qualification or imply that recommendation promotion has passed. Model weights are never automatically downloaded and external services are never automatically started.

## Preserved implementation

The request-scoped inference collector, executor propagation, synchronized counters, separate provider calls/items/cache/failure telemetry, and exception/early-return isolation remain. `actual_inferences_performed` counts successful scoring calls. Cache-cold latency checks continue to reject request errors, empty responses, fallback and result-cache hits; ANN-only accepts zero reranker calls, while enabled stages must execute successfully. Measurement starts after cache preparation, with the real monotonic clock and ABBA/BAAB ordering.

Catalog evidence takes precedence over benchmark annotations; typed movie/TV identities and repaired title references are retained. Frozen datasets, baseline snapshots, historical qrels, recommendation weights, ranking behavior and the runtime intent cache remain. MovieLens rankable positives, metric gain modes, family/user sampling and existing missing-catalog/history/title/index-page regression checks are retained.

Git attributes preserve exact bytes for the frozen v2.2 catalog and historical v2.0 qrels across platforms. The staged catalog blob was checked against its recorded SHA-256; line-ending normalization cannot silently invalidate that fingerprint. Historical v2.0 qrels are retained for audit, not newly qualified Nimble labels; new judging uses the v2.7 namespace.

## Cleanup and audit archive

The current `evaluation/artifacts/nimble_fixed_prompt_v2_8_20261008/` folder remains intact. Its frozen manifests, raw native requests, controls, blinded Luna labels, qualification, reference limitations and prior recommendation checks are preserved byte-for-byte.

Seven superseded experiment folders, six historical review documents, retired judge sources, accessible temporary test files, one-off experiment scripts, duplicate logs and generated result/cache files were archived before their active copies were removed. The archive verifies 788 readable files (25,530,396 source bytes) with per-file SHA-256 checks. Historical reports retain their original dates and outcomes under their original archive paths. The previous long delivery report and qualification file are included.

- [Verified recovery archive](../evaluation/artifacts/archive/cleanup_20261008.zip)
- [Archive manifest and preserved current-file hashes](../evaluation/artifacts/archive/cleanup_20261008_manifest.json)
- [Cleanup result and Windows permission errors](../evaluation/artifacts/archive/cleanup_20261008_result.json)

Windows denies reading, deleting or moving 49 older temporary test folders under `.cache`. They remain, and their unreadable contents are not claimed to be archived. No ownership or access-control changes were made. The installed Ollama model weights were not uninstalled; this cleanup removes unused repository integrations.

## Verification

The cleaned source passes **777 tests**, with six dependency warnings, in **44.65 seconds**. Exact overall statement coverage across API, ETL and evaluation is **10423/12004 = 86.82939020326558%**. All **36 remaining modified production modules** pass the unrounded 85% threshold. The evaluation entrypoint covers **913/1037 = 88.04243009%**; qualification covers **267/279 = 95.69892473%**; the Ollama and System One modules each cover 100%.

All changed-file repository hooks pass, including Black, Ruff, Mypy and secret checks. `git diff --check` passes. Formatting completed before the full coverage run.

Evaluation contracts passed 266 isolated tests before cleanup of the archived source copies. Unit tests use pinned synthetic model metadata or a local HTTP test server; they do not count as real model execution. The lower full-suite count reflects removal of obsolete direct-loading tests and repeated adapter tests, while keeping the shared distribution, permutation, provenance, service/authentication and failure checks on the retained Nimble adapter. New tests cover the Nimble defaults, rejection of retired presets, rejection of unsupported production consensus and absence of qualification fallback.

Live read-only checks confirm the installed Nimble artifact is available and its qualification fingerprint remains `1f005ca2ac68f282d51b2dd9ccc9a8ded5b01a34c0dd8e25e25888e2ecd860cd`. All 34 files in the current qualification artifact folder are unchanged. No fresh model scoring run was needed: the actual scoring request, response parsing, rubric and model fingerprint remain unchanged.

- [Full-suite results](../evaluation/artifacts/cleanup_20261008/full_suite.log)
- [Exact coverage for all modified modules](../evaluation/artifacts/cleanup_20261008/module_coverage.log)
- [Preservation and installed-judge verification](../evaluation/artifacts/cleanup_20261008/verification.json)
- [Formatting, lint, typing and safety hooks](../evaluation/artifacts/cleanup_20261008/hooks.log)

## Recommendation promotion and remaining evidence

The latest real legacy recommendation checks remain failures: quality non-regression passed in A/B, but its stored-baseline latency gate failed. The awake regression rerun failed nDCG@10 (approximately 0.4495 versus required 0.72), while diversity passed. The first regression run crossed host standby; its timing remains archived and excluded as performance evidence. No new A/B or regression run is needed to attribute this judge/configuration cleanup, which does not change recommendation ranking. No weights or baseline were promoted.

The historical controlled 400-request cache-cold latency validation remains archived. Complete real MovieLens/Tag Genome, synthetic personalization and private-holdout promotion tracks have not been executed in this cleanup. Subjective accuracy and whether judge-based A/B decisions consistently select the better recommender remain separate evidence requirements. Missing provider, studio and franchise-phase facts remain limitations; unknown evidence remains unjudged.
