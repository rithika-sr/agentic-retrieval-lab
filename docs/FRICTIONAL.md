# FRICTIONAL.md — decisions, surprises, limitations

Dated log. Newest entries at the bottom. Record *why*, not just *what*.

## 2026-10-05 — Initial design decisions
- **One repo, three experiments, shared foundation.** All three ideas share the corpus, retrievers, and eval harness. Build order: Foundation → Exp 2 v1 → Exp 1 → Exp 3 → Exp 2 v2.
- **Dataset:** MS MARCO passage dev/small (public, web queries). Q2D-Web data is private.
- **Subcorpus (~200k):** all official positives + BM25 top-100 (full collection) for every query variant + random backfill. Reason: dense-encoding 8.8M passages on a laptop is impractical.
- **Known bias:** distractors are BM25-pooled, so they may favor dense models. To be measured (Exp 1 stretch) and stated publicly.
- **Primary metric = Recall@100**, not Recall@1000, because top-1000 of 200k is a far larger slice than top-1000 of 190M.
- **Holdout principle:** pool/label models (BM25, bge-small, e5-small) are older than holdout models (Qwen3-Embedding-0.6B, EmbeddingGemma-300M).
- **Expected limitation:** short passages → the truncation failure category will be rare.

## Setup report (M0)
_To be filled by Claude Code._
