# Evaluation runtime

Routine comparisons reuse the existing local judgment cache. A pair is reused only when its query, evidence, model, rubric and qualification identities match. There is no new shared judgment bank. Unknown evidence remains unknown and prevents conclusive promotion.

## Workflow

| Mode | Queries | Purpose |
|---|---:|---|
| `make eval-v2-quick` | 14 development families | Fixed diverse diagnostic; never promotion evidence |
| `make eval-v2` | 51 development families | Normal iteration |
| `make eval-v2-promotion` | 50 regression families | Statistical promotion checks with all existing family and critical-slice requirements |
| `make eval-v2 SPLIT=full` | 101 families | Broader release validation |

Each command accepts `CONFIG` and `BASELINE`. The quick diagnostic selects three independent development families from each of franchise, vibe, constraint and entity, plus one typo and one multi-constraint family. Selection is fixed by case ID and never consults candidate scores. `--quick-dev` rejects full/regression, private evaluation, qualification, latency, personalization, hardware inspection, rescore and baseline capture modes. Reports mark the diagnostic as ineligible for promotion.

All selected queries retrieve at depth 100. Before any grading, the evaluator checks execution, fallbacks, empty outputs, duplicates, hard constraints, dislikes, completeness, canonical order and promotion sample sizes. If these checks fail, the report identifies affected queries and explicitly marks relevance, nDCG confidence intervals and recall comparison as skipped. A successful preflight only starts grading; it is not a promotion decision. Use `--continue-on-preflight-failure` to collect full graded diagnostics while retaining the final gates.

Only uncached pairs are sent to the judge. `--judge-workers` defaults to four and accepts one through four. Independent calls can run concurrently; cache writes and adjudication stay on the calling thread. Consensus test panels still grade both independent judges and only consult the tie-breaker on disagreement. `--baseline-judge-workers` also defaults to four; the stored baseline retains the worker count used when it was captured.

New judgments are durably appended to a small local journal. The full cache is compacted every 25 additions and at pool boundaries using an atomic replacement. Restart replays complete journal entries and repairs an incomplete final append. A failed replacement preserves the preceding snapshot and the recovery journal. Neither cache nor journal is committed. Run one evaluation writer at a time against a given cache.

Latency remains a separate, unchanged measurement: ten representative queries, 400 timed requests and ten warmups, with warm models and result caches bypassed. Do not run grading concurrently with the latency benchmark.

## Measured on 9 October 2026

The qualified v2.2 Nimble worker diagnostic made **85 fresh native requests**: one warmup plus 14 distinct development pairs repeated twice at each of one, two and four workers. It bypassed the judgment cache. [Raw outputs and verified input manifest](evaluation_judge_workers_20261009.json) retain the pinned judge fingerprint.

| Workers | Fresh judgments | Elapsed | Grade/sufficiency agreement with sequential controls | Execution failures |
|---|---:|---:|---:|---:|
| 1 | 28 | 144.07 seconds | 100% | 0 |
| 2 | 28 | 118.37 seconds | 100% | 0 |
| 4 | 28 | 110.89 seconds | 100% | 0 |

Four workers reduced elapsed grading time by **23.03%** on this diagnostic; this is not a fourfold speedup. Repeated decisions were stable in every arm. The small fixed sample and sequential-first arm order limit generalization. These measurements do not change judge qualification or promote the experimental v2.3 evidence profile. Rerun `make eval-v2-judge-benchmark` after changing hardware or the service setup; the benchmark recommends parallelism only with repeatable, matching decisions, zero failures, an unchanged pinned artifact and at least a 10% measured elapsed improvement.

Real production comparisons against the frozen reference measured:

| Run | Queries | Elapsed | New judge calls | Result |
|---|---:|---:|---:|---|
| Quick development | 14 | 49.40 seconds | 0 | Preflight FAIL: existing constraint/order and completeness defects |
| Promotion preflight | 50 | 63.61 seconds | 0 | Preflight INVALID: existing empty outputs and other defects |
| Full graded diagnostic, forcing continuation | 101 | 104.49 seconds | 0 | Final INVALID: five existing empty outputs |

The full diagnostic reused all judgments, retained **94 unresolved pairs**, and had exactly zero per-query baseline/candidate nDCG and Recall@100 deltas. These results demonstrate unchanged grading semantics and safe rejection of existing defects, not production qualification. Timings exclude the separate latency benchmark and are specific to this host, existing local cache and workload. A new model or evidence contract still needs fresh qualification and grading.

Verification: **903 tests pass**, exact overall coverage **10,958/12,523 = 87.50299449%**, and all **40 modified production modules** meet the unrounded 85% gate. Repository hooks pass. [Machine-readable runtime and verification evidence](evaluation_runtime_20261009.json).
