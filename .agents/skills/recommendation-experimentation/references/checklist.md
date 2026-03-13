# Recommendation Experimentation Checklist

## Experiment Setup
- Hypothesis: <one sentence>
- Metrics: <precision/latency/diversity/etc>
- Rollback trigger: <condition>

## Change Safety
- Feature flags/overrides wired correctly.
- Cache keys include all new behavior-driving params.
- Fallback ordering remains available.

## Validation Commands
```bash
pre-commit run --all-files
pytest tests/unit/test_recommend_route.py tests/unit/test_metrics.py
pytest tests/integration/test_routes.py
```

## Documentation
- Update impact notes in `recommendation_flags_impact.md` when tuning logic changes.

## Report Template
- Hypothesis outcome: <met/not met>
- Behavioral change summary: <details>
- Latency impact: <details>
- Rollback status: <needed/not needed>
