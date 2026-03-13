---
name: review-resolution
description: Resolve pull request review feedback end-to-end. Use when asked to resolve PR comments, address review feedback, or fix unresolved review threads. Covers fetching latest review state, triaging actionable vs optional comments, applying minimal fixes, validating changes, pushing updates, and resolving addressed threads.
---
# Review Resolution Skill

Use this skill for repeatable, high-signal review resolution on GitHub pull requests.

## Fast Path
1. Identify the PR number and active branch.
2. Fetch latest review summaries, inline comments, and unresolved threads.
3. Triage each thread:
   - implement now
   - intentionally defer with rationale
   - blocked (needs user decision)
4. Apply the smallest safe fix set.
5. Run required validation checks.
6. Push commits to the PR branch.
7. Resolve only threads that are fully addressed in pushed code.
8. Verify thread status after resolving.
9. Report results with thread links and commit hash.

## Guardrails
- Prioritize correctness and behavior safety over large refactors.
- Prefer targeted fixes to satisfy review intent with minimal blast radius.
- Do not resolve threads that were not implemented.
- Preserve existing tests unless updating expected behavior.
- If feedback is optional and not worth complexity, state that clearly and leave thread open.

## Required Data Collection
- PR overview and reviews
- Inline review comments
- Review thread status (resolved/unresolved/outdated)

Use the command checklist in [references/checklist.md](references/checklist.md).

## Validation
- Run repository-required checks before finalizing.
- At minimum, run tests relevant to changed files.
- If pre-commit modifies files, re-stage and re-run.

## Output Contract
- List implemented comments with links.
- List intentionally deferred comments with rationale.
- Include validation commands run and outcomes.
- Confirm final unresolved thread count.
