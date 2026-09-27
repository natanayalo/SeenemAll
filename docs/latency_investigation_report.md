# Recommendation latency investigation

**Updated:** 2026-09-27
**Status:** Corrected stage benchmark and frozen-set request evaluation complete; production telemetry pending.

## Findings

The corrected retrieval and ranking replay measured **72.28 ms mean**, **68.15 ms p50**, and **106.84 ms p95**. Compared with the earlier 326 ms estimate, the corrected mean is lower by a factor of about **4.51**. This came from correcting the benchmark measurement, not from changing production ranking behavior.

The measured boundary starts at query embedding and ends after Cross-Encoder scoring. It includes Elasticsearch retrieval, SQL hydration, and reranking. It does **not** include intent parsing or the complete `/recommend` HTTP request.

| Retrieval/ranking stage | Mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| Query embedding | 6.09 ms | 3.61 ms | 18.33 ms |
| Elasticsearch (ANN + BM25) | 26.81 ms | 27.09 ms | 34.01 ms |
| SQL hydration | 3.57 ms | 3.31 ms | 5.80 ms |
| Cross-Encoder | 35.80 ms | 33.56 ms | 59.44 ms |
| **Retrieval/ranking total** | **72.28 ms** | **68.15 ms** | **106.84 ms** |

The run used 50 frozen holdout queries with three randomized, interleaved repetitions (150 executions), 25 candidates, and the OpenVINO Cross-Encoder on GPU. Stage percentiles describe each stage separately and must not be added to infer the total p95.

## Latency target

Against a 200 ms target, the retrieval/ranking p95 leaves **93.16 ms of margin**. The target-to-latency ratio is **1.87×** (`200 / 106.84`), not 7.3×.

The separate uncached `parse_intent` benchmark measured **27.3 ms p95 across 220 queries**, with no parser sample over 200 ms. Its parser-only target-to-latency ratio is about **7.3×**. The two p95 values come from different boundaries and samples; they cannot be added to assert a whole-API p95. Measure the complete `/recommend` request boundary before making a whole-API SLA claim.

## Query formulation decision

Raw Query remains the production default. In the corrected latency replay, Raw Query measured **72.28 ms mean / 106.84 ms p95**, and Strategy F measured **74.83 ms mean / 109.72 ms p95**. That difference is small compared with the total latency.

On the 50-query quality holdout, Raw Query had the higher nDCG@10 point estimate (**0.2906** vs. **0.2806** for F). The bootstrap comparison did not establish a reliable difference (F vs. Raw: −1.01%, 95% interval −3.10% to +0.61%, p=0.266). Keep Strategy F available for experiments on noisy-query classes, but there is no evidence here to make it the default.

The earlier holdout run supplied the quality comparison; the corrected latency replay supplied the final latency numbers. Raw experiment inputs, scripts, and result dumps have since been removed after consolidating the decision here.

## Selected implementation

The current working copy wires the experiment winner into the synchronous request path:

1. Deterministic intent rules run first; unresolved query text triggers GLiNER v2.1 through OpenVINO on CPU. The result stays structured and is merged with canonical filters.
2. The raw query is the default text for both Elasticsearch BM25 and the MiniLM query embedding. Structured constraints remain separate filters. Strategy F can be selected with `RETRIEVAL_QUERY_FORMULATION=strategy_f` for experiments.
3. Elasticsearch combines BM25 and kNN retrieval, then the app hydrates and scores candidates.
4. The local `ms-marco-MiniLM-L-6-v2` Cross-Encoder reranks up to 25 candidates through OpenVINO GPU; explanations are deterministic and feature-grounded. Remote LLM reranking is ignored on this request path.

The app now records parser path/latency, query formulation, retrieval stages, Cross-Encoder latency and candidate counts, zero-result counts, pipeline latency, and whole `/recommend` HTTP latency percentiles. Those whole-request percentiles need live traffic before they can support an API SLA claim. The parser uses deterministic-only fallback if its OpenVINO model artifact cannot load; embedding and reranking loaders retain a PyTorch fallback if OpenVINO initialization fails.

The Docker image exports the selected GLiNER v2.1 checkpoint to OpenVINO IR during the build, pinned to revision `4e091416cf7c3481db542c2a3d26156916f3a47f`. The local generated IR is ignored by Git and the Docker build context; the image creates its own artifact under `/opt/models/gliner_small_ov` from the [upstream Apache-2.0 model](https://huggingface.co/urchade/gliner_small-v2.1). For a native local run, `python scripts/export_gliner_openvino.py` writes the export to `models/gliner_small_ov`.

The 62-query Elasticsearch A/B run compared the previous `ann_cross_encoder` snapshot with the production `default` path using hybrid fast intent and a disabled persistent cache. nDCG@10 improved from **0.2049** to **0.2276** (+11.1%), MAP from **0.1568** to **0.1661** (+5.9%), and P95 request latency from **453.3 ms** to **336.2 ms** (−25.8%). Mean latency was **382.9 ms** vs. **382.0 ms** (+0.9 ms, +0.23%), so the strict A/B gate reported a failure only on its mean-latency check. A fresh run saved the `default` production configuration as the new baseline at nDCG@10 **0.2268**, MAP **0.1648**, and P95 **361.0 ms**. The quality regression gate then passed against the old baseline floors (nDCG@10 **0.2049**, ILD **0.45**). A further local request replay measured P95 **800.9 ms**, showing enough run-to-run latency variation that none of these request-level timings should be treated as a stable SLA; validate latency with production telemetry.

These changes are in the local working tree and have not been deployed.

## Cross-Encoder interpretation and next step

Cross-Encoder scoring averaged **35.80 ms**, close to half of the retrieval/ranking mean. A separate microbenchmark reported 8.53 ms p95 using 25 short, curated query-document pairs over 10 timed trials; that result is not directly comparable to catalog documents with potentially long overviews.

There is also a payload difference to keep in mind: the production reranker can serialize title, year, media type, genres, directors, cast, and overview, while the latency decomposition script passes title, genres, overview, year, runtime, and media type. It does not pass directors or cast. Treat the 35.80 ms figure as a controlled replay measurement, not an exact measurement of every production Cross-Encoder input.

The current retrieval/ranking profile is already below the 200 ms target. Do not pursue speculative Cross-Encoder optimization based on this profile alone. Use production telemetry to identify whether latency, recommendation quality, or a recurring real-query failure deserves the next iteration.

## Evidence

The benchmark data and one-off runners were temporary investigation artifacts. The corrected measurements, holdout comparison, sample sizes, and measurement boundaries are summarized above.
