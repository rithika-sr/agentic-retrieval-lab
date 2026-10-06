# Agentic Retrieval Lab

*Small-scale, open-data experiments on first-stage retrieval for agentic RAG — inspired by Perplexity's Q2D-Web paper (Schall et al., 2026).*

> 🚧 Work in progress. Results below will be filled in only from files in `results/`.

## The questions
1. **Does BM25 still earn its seat in hybrid search?** BM25 + one embedding model vs two embedding models, fused the same way.
2. **What happens to retrieval when an AI agent writes the query?** Human vs agent primary vs agent support queries, with deeper LLM-judged labels.
3. **Why does retrieval fail, and which fix helps?** A failure-taxonomy analyzer with a fix simulator.

## Headline results
_TBD — filled after each experiment._

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
