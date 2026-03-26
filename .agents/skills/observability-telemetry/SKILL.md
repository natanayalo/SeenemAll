---
name: observability-telemetry
description: Maintain and extend logging, metrics, and debug diagnostics safely. Use when modifying structured logs, request tracing, counters/histograms, metric endpoints, or recommendation debug payloads.
---
# Observability and Telemetry Skill

Use this skill for instrumentation changes.

## Fast Path
1. Identify telemetry surfaces touched (logs, metrics, debug API).
2. Preserve parseable log shape and stable metric names.
3. Keep latency instrumentation around expensive stages.
4. Ensure diagnostic payloads remain useful but bounded.
5. Add tests for edge cases and empty-state snapshots.

## Guardrails
- Avoid noisy high-cardinality labels.
- Avoid plain-text fallbacks when structured output is expected.
- Keep metric key names stable to protect dashboards.
- Include fallback behavior when telemetry serialization fails.

Use [references/checklist.md](references/checklist.md) for command sequence.
