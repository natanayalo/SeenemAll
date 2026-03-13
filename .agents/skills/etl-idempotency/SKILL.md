---
name: etl-idempotency
description: Keep ETL synchronization jobs safe and repeatable. Use when changing TMDB, embeddings, or JustWatch ingestion logic, upsert behavior, deduplication, retries, or failure handling.
---
# ETL Idempotency Skill

Use this skill for ETL pipeline changes.

## Fast Path
1. Identify the authoritative key used for dedupe/upsert.
2. Verify reruns do not create duplicate rows.
3. Confirm partial failure paths are resumable.
4. Keep normalization deterministic.
5. Validate downstream API assumptions are preserved.

## Guardrails
- Prefer upsert semantics over insert-only flows.
- Do not couple ETL success to non-critical external enrichment.
- Keep progress logging sufficient for replay.
- Avoid schema changes without migration plan.

Use [references/checklist.md](references/checklist.md) for command sequence.
