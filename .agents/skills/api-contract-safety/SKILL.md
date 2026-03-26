---
name: api-contract-safety
description: Protect API contract compatibility when changing routes, request models, response payloads, validation rules, or pagination behavior. Use for FastAPI route edits that can break clients or integration tests.
---
# API Contract Safety Skill

Use this skill when touching API interfaces.

## Fast Path
1. Identify all changed endpoints and payload fields.
2. Determine if changes are additive, behavioral, or breaking.
3. Keep backward compatibility unless explicitly requested.
4. Update unit and integration tests for touched contracts.
5. Validate response shape, status codes, and pagination behavior.

## Guardrails
- Prefer additive fields over renames/removals.
- Do not silently change field semantics.
- Keep error response formats stable.
- If a breaking change is required, document it clearly in PR notes.

Use [references/checklist.md](references/checklist.md) for command sequence.
