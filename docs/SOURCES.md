# SOURCES.md

## Paper that inspired this project
- M. Schall, S. Eslami, M. Krimmel, A. Chaffin, L. Milliken, B. Wang, D. Bykov. *Q2D-Web: A Large-Scale Benchmark for Retrieval in Agentic RAG Systems.* arXiv:2609.08887v2, Sept 2026. Leaderboard: https://huggingface.co/spaces/perplexity-ai/q2d-web-leaderboard
  - Borrowed ideas: primary/support query structure, pooling + LLM judge, multiple label sets, holdout principle, RRF subcorpus sampling, unique-positives analysis, hard-positive/negative taxonomies.

## Datasets
- MS MARCO Passage Ranking (Bajaj et al., 2016, arXiv:1611.09268) via `ir_datasets` 0.6.3 (MacAvaney et al., SIGIR 2021), dataset id `msmarco-passage/dev/small`: 6,980 dev queries, official qrels, 8.8M-passage collection. First used 2026-10-07.
  - Source archive: `collectionandqueries.tar.gz` (1,057,717,952 bytes, md5 `31644046b18952c1386cd4564ba2ae69`) from `https://msmarco.z22.web.core.windows.net/msmarcoranking/`, cached at `~/.ir_datasets/msmarco-passage/`.
  - Passage text uses ir_datasets' `FixEncoding` (repairs MS MARCO's double-encoded UTF-8).
  - Terms: MS MARCO data usage agreement, http://www.msmarco.org/dataset.aspx (non-commercial research use).
  - Full 8,841,823-passage `collection.tsv` streamed from the same cached archive (no extra download) to build the full-collection BM25 index and the 200k subcorpus (2026-10-08/09). Passage ids are 0..8,841,822 in file order (checked while streaming).

## Methods
- BM25 (Robertson & Zaragoza, 2009) — via `bm25s` 0.3.12 (X. H. Lù, *BM25S: Orders of magnitude faster lexical search via eager sparse scoring*, arXiv:2407.03618, 2024), `method="lucene"`. Variant naming follows Kamphuis, de Vries, Boytsov & Lin, *Which BM25 Do You Mean? A Large-Scale Reproducibility Study of Scoring Variants*, ECIR 2020. Index built with our chunked builder (`arl.retrievers.bm25.build_index`), verified bit-identical to stock bm25s indexing on a 500k slice. First used 2026-10-08.
  - k1 = 0.9, b = 0.4: the Anserini MS MARCO passage defaults (Yang, Fang & Lin, *Anserini: Enabling the Use of Lucene for Information Retrieval Research*, SIGIR 2017).
  - Stemming: Snowball English stemmer (M. Porter, *Snowball: A language for stemming algorithms*, 2001) via `PyStemmer` 3.1.0. Stopwords: bm25s's built-in English list.
- Leak rule v3 stopwords: scikit-learn's `ENGLISH_STOP_WORDS` (318 words; scikit-learn, Pedregosa et al., JMLR 2011). First used 2026-10-08.
- Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) — via `ranx`

## Models (add license notes as used)
- BAAI/bge-small-en-v1.5 — MIT license; first used in the M0 smoke test (revision `5c38ec7c`)
- intfloat/e5-small-v2
- Qwen/Qwen3-Embedding-0.6B
- google/embeddinggemma-300m (gated)
- LLM: Claude Haiku 4.5 (Anthropic) or a local Ollama model — `claude-haiku-4-5-20251001` first called in the M0 smoke test (one call); M1 query rewriting and leak judging (2026-10-07/08); answer extraction for leak rule v3 (2026-10-08)
