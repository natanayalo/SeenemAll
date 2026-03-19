# Recommendation KPI Contract

Date: 2026-03-14
Owner: Recommendation pipeline (`api/routes/recommend.py`, `api/core/`)

## Scope
- Applies to ranking-path changes in ANN retrieval, mixer, business rules, MMR, and reranker.
- Used for offline evaluation and release-gate decisions in low/no-traffic environments.

## Primary KPIs
- `nDCG@10` (offline relevance quality): target `>= 0.45`, release gate `>= baseline - 0.02`.
- `MAP@10` (offline ranking precision): target `>= 0.35`, release gate `>= baseline - 0.02`.
- `HitRate@10` (offline success frequency): target `>= 0.70`, release gate `>= baseline - 0.03`.
- `MRR@10` (first good result position): track as secondary quality indicator.
- `NegativeRate@10` (hard-negative leakage): target toward `0.00` on labeled cases.

## Offline Guardrails
- Critical slices must not regress more than:
  - `nDCG@10`: `-0.05`
  - `MAP@10`: `-0.05`
  - `HitRate@10`: `-0.07`
- Overall metrics must not regress more than:
  - `nDCG@10`: `-0.02`
  - `MAP@10`: `-0.02`
  - `HitRate@10`: `-0.03`
- Use bootstrap CIs from evaluator output when inspecting borderline deltas.

## Required Slices
- Query mode: `query`, `no_query`.
- User state: `warm_start`, `cold_start`.
- Media type: `movie`, `tv`.
- Constraint slices: `provider_filtered`, `maturity_capped`, `runtime_capped`.

## Release Gate
1. Freeze a baseline summary JSON from a known-good version.
2. Run candidate evaluation with same dataset and settings.
3. Compare candidate vs baseline using `evaluation/check_release_gate.py`.
4. Block promotion when gate script returns non-zero.
5. Prefer `--significance-mode strict` with paired CSVs for high-confidence changes.

## Optional Online Metrics (When Traffic Exists)
- `CTR@10`, `completion@10`, and negative feedback rate can be enabled later.
- Impression logging fields (`request_id`, `rank`, `profile`, `query_present`, `algo_version`) are already in place for future use.
