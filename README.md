# Agentic Retrieval Lab

*Small-scale, open-data experiments on first-stage retrieval for agentic RAG — inspired by Perplexity's Q2D-Web paper (Schall et al., 2026).*

> 🚧 Work in progress. Results below will be filled in only from files in `results/`.

## The questions
1. **Does BM25 still earn its seat in hybrid search?** BM25 + one embedding model vs two embedding models, fused the same way.
2. **What happens to retrieval when an AI agent writes the query?** Human vs agent primary vs agent support queries, with deeper LLM-judged labels.
3. **Why does retrieval fail, and which fix helps?** A failure-taxonomy analyzer with a fix simulator.

## Headline results
_TBD — filled after each experiment._

## Foundation data (Milestone 1)
All numbers from `results/foundation/`.
- **Queries:** 300 human queries from MS MARCO dev/small (seed 42), each rewritten by Claude Haiku 4.5 into 1 primary + 3 support queries → 1,200 agent queries, 0 rewrite failures (`2026-10-08_rewrites_300/`).
- **BM25 over the full 8.8M-passage collection:** the official positive is in the top 100 for 63.7% of human queries, 60.3% of primaries, 41.7% of supports, and 77.7% of questions when any of the 5 variants counts (`2026-10-08_bm25/pool_stats.json`).
- **Subcorpus:** 200,000 passages = 317 official positives + 104,877 BM25-pooled distractors + 94,806 random backfill (`2026-10-09_corpus/corpus_stats.json`).
- **Answer leakage (rule v3, dev-set evidence only):** among queries whose answer is a factoid, 68 of 366 support queries and 5 of 122 primaries contain an answer word not in the question (`2026-10-08_leak_v3/leak_v3_counts.json`). Known false positives on generic words; held-out validation is planned for Milestone 4.

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
