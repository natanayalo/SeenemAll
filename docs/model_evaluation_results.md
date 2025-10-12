# Movie Domain Model Evaluation Results

## Test Results (October 2025)

### 1. all-MiniLM-L6-v2 (Baseline)
- MSE: 0.1982
- MAE: 0.4316
- Encode time: 0.15s per batch
- Pros:
  - Well-established
  - Small model size
  - Good documentation
- Cons:
  - Higher error on movie similarity
  - Less accurate on genre relationships

### 2. E5-small-v2
- MSE: 0.1421 (28% improvement)
- MAE: 0.2680 (38% improvement)
- Encode time: 0.12s per batch
- Pros:
  - Better semantic understanding
  - Faster inference
  - Same dimension (384) = easy migration
  - Better cross-lingual performance
- Cons:
  - Larger model size (133MB vs 91MB)
  - Newer, less battle-tested

### 3. MovieCLIP (Pending)
- Requires:
  - GPU infrastructure
  - Visual data pipeline
  - Additional storage
- Benefits to evaluate:
  - Visual understanding
  - Multi-modal features
  - Domain-specific knowledge

### 4. MovieBERT (Pending)
- Text-only alternative
- Potential benefits:
  - Movie-specific vocabulary
  - Plot understanding
  - Character relationship modeling

## Recommendations

1. Short-term:
   - Migrate to E5-small-v2
   - Benefits outweigh slightly larger model size
   - Drop-in replacement (same dimensions)

2. Medium-term:
   - Evaluate MovieBERT
   - Lower infrastructure requirements
   - Could complement E5 embeddings

3. Long-term:
   - Consider MovieCLIP if visual features prove valuable
   - Requires infrastructure planning
   - Higher implementation complexity

## Migration Plan

1. Phase 1: E5-small-v2 Migration
   - Add version field to embeddings table
   - Create migration script
   - A/B test recommendations
   - Monitor quality metrics

2. Phase 2: Hybrid Approach
   - Evaluate combining E5 with MovieBERT
   - Test ensemble methods
   - Measure improvement vs complexity

3. Phase 3: Visual Integration
   - Cost-benefit analysis of MovieCLIP
   - Infrastructure planning
   - Pilot with subset of content
