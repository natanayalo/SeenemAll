# Production reference with Evaluation Suite v2

The v2 reference records the current production configuration under the new measurement contract. It does not claim an improvement, change recommendation weights, or waive quality gates. The legacy `evaluation/baseline.json` remains historical evidence; its scores and cached timings cannot be compared directly with v2.

## Capture

```text
python -m evaluation.evaluate --v2 --track product --split full --config default --backend elasticsearch --save-v2-baseline evaluation/baseline_v2.json --v2-report evaluation/artifacts/production_v2_report.json
```

The snapshot is immutable: choose a new file name to establish a subsequent reference. The full product dataset must be present, and the pinned Nimble artifact must match its qualification record. Stub judging cannot establish a production reference.

The corresponding make targets are `make eval-v2-baseline` and `make eval-v2 CONFIG=YOUR_CONFIG SPLIT=full`. Pass `BASELINE=path/to/new_reference.json` to select a different snapshot path.

Capture requests current production once per quality query. It stores typed ranked IDs to depth 100, independently judged top-ten and declared reference items, judgment execution statuses and evidence fingerprints, per-query metrics, deterministic constraint and canonical-order defects, dataset/catalog hashes, the live search-index content fingerprint, model provenance, request parameters, and code/runtime identity. An index change during capture invalidates the reference. Scores are averaged within query families and then across families. Positive hits require grade >=2; nDCG uses exponential graded gains. AP@K uses all known positive judgments as its denominator.

Successful evidence abstentions are stored as unresolved, without invented grades. A snapshot with these gaps has `quality_status: inconclusive` and `judgments_complete: false`; its arithmetic scores are provisional because unknown grades count as zero. Recording production's defects does not certify it for promotion. Execution failures prevent a usable reference and leave a diagnostic report.

Successful empty responses are recorded as quality defects with zero retrieval scores. Completeness is capped at 100%. The judge's conservative input guard accepts up to 4,096 characters without truncating evidence; a cached resource rejection is retried only when its unchanged input now fits that guard.

## Compare

```text
python -m evaluation.evaluate --v2 --track product --split dev --baseline-file evaluation/baseline_v2.json --candidate YOUR_CONFIG --backend elasticsearch
python -m evaluation.evaluate --v2 --track product --split full --baseline-file evaluation/baseline_v2.json --candidate YOUR_CONFIG --backend elasticsearch
```

Quality comparisons use the snapshot's frozen production rankings. They retain its known positives in the shared judgment denominator, run the candidate, and judge new pooled results. The checksum, selected cases, frozen catalog, live index artifact, judge fingerprint, backend, K and gain mode must match. A full reference can serve its unchanged development or regression subset. A legacy baseline, modified case, changed catalog/index or incompatible judge is rejected. Failed judging remains INVALID; unresolved top-K or declared references remains INCONCLUSIVE.

## Workload and latency

Product development has **51 queries/51 families**, regression has **50/50**, and full has **101/101**. There are **379 declared reference entries** across the full split, with overlap deduplicated within each query. At K=10, capture judges at most **1,389 query-item pairs** (1,010 top-ten results plus 379 references); empty/short results and overlap reduce that number. It does not judge every result at retrieval depth 100. Fingerprinted judgments can be reused, and independent model calls use two workers by default; disk cache writes remain sequential.

The full suite is appropriate for establishing a baseline or validating a release. Use the development split for iteration and a small representative query sample for quick diagnostics. Such a sample is not a promotion check: promotion requires at least 50 independent families and at least ten families in each critical slice.

Latency is a separate measurement. Capture selects ten queries across franchise, vibe, constraints and entity slices, warms both measurement arms with five queries each, then executes five repetitions of ABBA/BAAB per query. This produces **400 timed requests plus ten warmup requests**, using the same production configuration in both arms. Models remain warm and result caches are bypassed; inference counters verify actual scoring. Timings are specific to the recorded host and runtime.

For future speed comparisons, rerun the baseline and candidate together with the latency harness, using their corresponding code/configurations and the same host/cache policy. Historical snapshot timings are descriptive, not a latency regression gate. The legacy 0.72 nDCG threshold belongs to the old label/query contract and is not an established absolute target for v2.
