# SOURCES.md

## Paper that inspired this project
- M. Schall, S. Eslami, M. Krimmel, A. Chaffin, L. Milliken, B. Wang, D. Bykov. *Q2D-Web: A Large-Scale Benchmark for Retrieval in Agentic RAG Systems.* arXiv:2609.08887v2, Sept 2026. Leaderboard: https://huggingface.co/spaces/perplexity-ai/q2d-web-leaderboard
  - Borrowed ideas: primary/support query structure, pooling + LLM judge, multiple label sets, holdout principle, RRF subcorpus sampling, unique-positives analysis, hard-positive/negative taxonomies.

## Datasets
- MS MARCO Passage Ranking (Bajaj et al., 2016) via `ir_datasets`.

## Methods
- BM25 (Robertson & Zaragoza, 2009) — via `bm25s`
- Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) — via `ranx`

## Models (add license notes as used)
- BAAI/bge-small-en-v1.5
- intfloat/e5-small-v2
- Qwen/Qwen3-Embedding-0.6B
- google/embeddinggemma-300m (gated)
- LLM: Claude Haiku 4.5 (Anthropic) or a local Ollama model
