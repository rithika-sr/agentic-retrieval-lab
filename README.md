# Agentic Retrieval Lab

*Small-scale, open-data experiments on first-stage retrieval for agentic RAG — inspired by Perplexity's Q2D-Web paper (Schall et al., 2026).*

> 🚧 Work in progress. Results below will be filled in only from files in `results/`.

## The questions
1. **Does BM25 still earn its seat in hybrid search?** BM25 + one embedding model vs two embedding models, fused the same way.
2. **What happens to retrieval when an AI agent writes the query?** Human vs agent primary vs agent support queries, with deeper LLM-judged labels.
3. **Why does retrieval fail, and which fix helps?** A failure-taxonomy analyzer with a fix simulator.

## Headline results
First-stage retrieval on the 200k-passage MS MARCO subcorpus, 300 human queries (from `results/m2/`).

| Retriever | Recall@100 | MRR@10 | Results |
|---|---|---|---|
| BM25 | 0.6422 | 0.1772 | [`2026-11-23_bm25_subcorpus`](results/m2/2026-11-23_bm25_subcorpus/) |
| bge-small-en-v1.5 | 0.9633 | 0.3872 | [`2026-11-23_dense_bge-small-en-v1.5`](results/m2/2026-11-23_dense_bge-small-en-v1.5/) |
| e5-small-v2 | 0.9622 | 0.4047 | [`2026-11-23_dense_e5-small-v2`](results/m2/2026-11-23_dense_e5-small-v2/) |
| Hybrid: BM25 + bge (RRF, k=60) | 0.9439 | 0.2810 | [`2026-11-23_hybrid_bm25_bge`](results/m2/2026-11-23_hybrid_bm25_bge/) |

> **Read this first: these results are provisional.**
> (a) The 200k subcorpus was built from BM25's top-100 pools, so it holds every passage that is hard *for BM25* but few that are hard for dense models. That favours the dense retrievers (and penalises a hybrid that mixes BM25 in).
> (b) A small length bias is confirmed: sorting BM25's candidates by length alone beats random order, MRR@10 0.0483 vs 0.0277 ([`length_check.json`](results/m2/2026-11-23_bm25_subcorpus/length_check.json)).
> **Next steps:** a length-matched control corpus and dense-pooled distractors, before reading "does BM25 earn its seat" from these numbers.

## Foundation data (Milestone 1)
All numbers from `results/foundation/`.
- **Queries:** 300 human queries from MS MARCO dev/small (seed 42), each rewritten by Claude Haiku 4.5 into 1 primary + 3 support queries → 1,200 agent queries, 0 rewrite failures (`2026-10-08_rewrites_300/`).
- **BM25 over the full 8.8M-passage collection:** the official positive is in the top 100 for 63.7% of human queries, 60.3% of primaries, 41.7% of supports, and 77.7% of questions when any of the 5 variants counts (`2026-10-08_bm25/pool_stats.json`).
- **Subcorpus:** 200,000 passages = 317 official positives + 104,877 BM25-pooled distractors + 94,806 random backfill (`2026-10-09_corpus/corpus_stats.json`).
- **Answer leakage (rule v3, dev-set evidence only):** among queries whose answer is a factoid, 68 of 366 support queries and 5 of 122 primaries contain an answer word not in the question (`2026-10-08_leak_v3/leak_v3_counts.json`). Known false positives on generic words; held-out validation is planned for Milestone 4.

## BM25 on the 200k subcorpus (Milestone 2, part A)
All numbers from `results/m2/2026-11-23_bm25_subcorpus/`. Agent queries are scored against their parent question's official MS MARCO positive.

| Query type | n | Recall@100 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| Human | 300 | 0.6422 | 0.1772 | 0.2129 |
| Agent primary | 300 | 0.5922 | 0.1642 | 0.1949 |
| Agent support | 900 | 0.3807 | 0.0819 | 0.1011 |

- Human − primary Recall@100: +0.050 (paired bootstrap 95% CI 0.0067 to 0.0933).
- **Length check:** on the 196 human queries with a positive in BM25's top 100, sorting those candidates by length alone (shortest first) gives MRR@10 0.0483 vs 0.0277 for random order (difference +0.0206, CI 0.0036 to 0.0408). Positives average 56.3 words vs 64.47 for the other candidates. That is a small but real length signal in this corpus (`length_check.json`).

## Setup
```bash
uv sync
cp .env.example .env   # add your keys
uv run pytest
```

## Limitations
Small scale (≈300 human queries, ≈200k MS MARCO passages), single-turn rewrites, English only, BM25-pooled distractors, LLM-judge bias, short passages. This replicates *ideas* from Q2D-Web; it is not a reproduction of their (private) benchmark.

## Credits
See `docs/SOURCES.md`.
