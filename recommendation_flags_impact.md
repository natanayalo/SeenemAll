# Recommendation System Flag Impact Analysis

This document provides a detailed analysis of the flags and their impact on the recommendation system in `api/routes/recommend.py`.

## Hybrid and Mixer Weights

These weights are used in the `_apply_mixer_scores` function to calculate a `retrieval_score` for each candidate item. This score is then used to rank the items before the final diversification and reranking steps.

### `_HYBRID_ANN_WEIGHT` / `mixer_ann_weight`

- **Purpose:** Controls the weight of the Approximate Nearest Neighbor (ANN) score in the final retrieval score. The ANN score is based on the cosine similarity between the user's vector and the item's vector.
- **Default Value:** `0.5`
- **Impact:**
    - **Higher Value:** Increases the importance of the ANN score, leading to recommendations that are more similar to the user's taste based on their viewing history or the rewritten query.
    - **Lower Value:** Decreases the importance of the ANN score, allowing other signals like popularity and trending to have a greater influence.
- **Component:** Scoring and Reranking (`_apply_mixer_scores`)

### `_MIXER_COLLAB_WEIGHT` / `mixer_collab_weight`

- **Purpose:** Controls the weight of the collaborative filtering score. This score is based on the items liked by users with similar tastes.
- **Default Value:** `0.3`
- **Impact:**
    - **Higher Value:** Increases the influence of collaborative filtering, leading to recommendations that are popular among similar users.
    - **Lower Value:** Reduces the influence of collaborative filtering, making the recommendations more personalized to the individual user's vector.
- **Component:** Scoring and Reranking (`_apply_mixer_scores`)

### `_MIXER_TRENDING_WEIGHT` / `mixer_trending_weight`

- **Purpose:** Controls the weight of the trending score. This score is based on the item's trending rank.
- **Default Value:** `0.2`
- **Impact:**
    - **Higher Value:** Boosts items that are currently trending, making the recommendations more reflective of current popular content.
    - **Lower Value:** Reduces the influence of trending items, focusing more on personalization and other signals.
- **Component:** Scoring and Reranking (`_apply_mixer_scores`)

### `_HYBRID_POPULARITY_WEIGHT` / `mixer_popularity_weight`

- **Purpose:** Controls the weight of the popularity score, which is based on the item's overall popularity.
- **Default Value:** `0.25`
- **Impact:**
    - **Higher Value:** Favors items that are generally popular, which can be a safe choice but might lead to less diverse recommendations.
    - **Lower Value:** Reduces the influence of overall popularity, giving more weight to other, more personalized signals.
- **Component:** Scoring and Reranking (`_apply_mixer_scores`)

### `_HYBRID_VOTE_WEIGHT` / `mixer_vote_weight`

- **Purpose:** Controls the weight of the vote count in the retrieval score.
- **Default Value:** `0.2`
- **Impact:**
    - **Higher Value:** Gives a boost to items with a higher number of votes, which can be an indicator of quality or popularity.
    - **Lower Value:** Reduces the influence of the vote count, which might allow for the recommendation of more niche or newer items.
- **Component:** Scoring and Reranking (`_apply_mixer_scores`)

### `_MIXER_NOVELTY_WEIGHT` / `mixer_novelty_weight`

- **Purpose:** Controls the weight of the novelty score. The novelty score is inversely related to the popularity of an item.
- **Default Value:** `0.1`
- **Impact:**
    - **Higher Value:** Promotes less popular items, leading to more diverse and serendipitous recommendations.
    - **Lower Value:** Reduces the promotion of less popular items, favoring more mainstream content.
- **Component:** Scoring and Reranking (`_apply_mixer_scores`)

### `_HYBRID_MIN_ANN_WEIGHT`

- **Purpose:** Sets a minimum weight for the ANN score to ensure that it always has some influence on the final score.
- **Default Value:** `0.05`
- **Impact:** Prevents the ANN score from being completely ignored, even if the `mixer_ann_weight` is set to 0. This ensures a baseline level of personalization.
- **Component:** Scoring and Reranking (`_apply_mixer_scores`)

## Rewrite Vector Weights

These weights are used in the `_build_rewrite_vector` function to create a query vector that is a blend of the rewritten query and the ANN description.

### `_ANN_DESCRIPTION_WEIGHT` / `ann_weight_override`

- **Purpose:** Controls the weight of the ANN description in the blended rewrite vector.
- **Default Value:** `1.2`
- **Impact:**
    - **Higher Value:** Gives more importance to the ANN description, which is a concise, factual summary of the item.
    - **Lower Value:** Reduces the influence of the ANN description, giving more weight to the rewritten user query.
- **Component:** Rewrite Vector Generation (`_build_rewrite_vector`)

### `_REWRITE_TEXT_WEIGHT` / `rewrite_weight_override`

- **Purpose:** Controls the weight of the rewritten user query in the blended rewrite vector.
- **Default Value:** `1.0`
- **Impact:**
    - **Higher Value:** Gives more importance to the user's query, making the recommendations more specific to their stated intent.
    - **Lower Value:** Reduces the influence of the user's query, relying more on the factual ANN description.
- **Component:** Rewrite Vector Generation (`_build_rewrite_vector`)

## Other Flags

### `_REWRITE_BLEND_ALPHA`

- **Purpose:** Controls the blending of the user's short-term vector and the rewrite vector.
- **Default Value:** `0.5`
- **Impact:**
    - **Value of 1.0:** Uses only the user's short-term vector.
    - **Value of 0.0:** Uses only the rewrite vector.
    - **Value of 0.5:** Creates a 50/50 blend of the two vectors.
- **Component:** Candidate Generation (`recommend` function)

### `_SERENDIPITY_RATIO`

- **Purpose:** Determines the proportion of "long-tail" (less popular) items to be included in the final recommendations.
- **Default Value:** `0.15`
- **Impact:**
    - **Higher Value:** Increases the number of serendipitous recommendations, potentially introducing the user to new and interesting content.
    - **Lower Value:** Reduces the number of serendipitous recommendations, leading to a more predictable set of popular items.
- **Component:** Scoring and Reranking (`_apply_serendipity_slot`)
