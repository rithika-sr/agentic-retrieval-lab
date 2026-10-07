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
_Session: 2026-10-05 to 2026-10-07. Numbers below are from console output of the commands shown (M0 writes nothing to `results/`)._

**Machine**
- MacBook, Apple M5 (arm64), 4 performance + 6 efficiency cores, 16 GB RAM
- macOS 26.6.2 (build 25G83); 208 GB free of 460 GB
- Torch device: **mps** (checked with a real tensor op, not just `is_available()`)

**Tools**
- git 2.50.1 (Apple Git-155; the `git` on PATH is Apple's, not Homebrew's)
- uv 0.12.23 (Homebrew), gh 2.102.0 (authenticated as `rithika-sr`, `repo` scope), Homebrew 7.0.8
- Python 3.11.15 (Homebrew `python@3.11`, pinned via `.python-version`)
- Key packages (from `uv.lock`): torch 2.14.1, sentence-transformers 6.1.0, ranx 0.3.21, bm25s 0.3.12, faiss-cpu 1.15.1, anthropic 1.11.0, ir-datasets 0.6.3

**Smoke test (`uv run python scripts/smoke_test.py --model --api`), all PASS**
- bm25s: "eiffel tower paris" → correct doc #1 (score 2.379); a doc sharing no terms scored exactly 0.000.
- bge-small on mps, dim 384: cos(cat/sofa, kitten/couch) = 0.823; unrelated pairs = 0.308 / 0.330.
- ranx Recall@2 on toy qrels/run = 0.75 (q1 = 0.5, q2 = 1.0), matching the hand calculation.
- Anthropic `claude-haiku-4-5-20251001`: replied "OK", 15 input + 4 output tokens, ≈ $0.000035.

**Decisions**
- **Wrote `pyproject.toml` by hand instead of `uv init`.** `uv init --package` would have run `git init` early and tried to create files that already exist. Build backend: `uv_build`.
- **Smoke test makes paid/large calls opt-in** (`--model`, `--api` flags), so the default run is free and offline.
- **No Claude attribution in commits/PRs.** Rule added to CLAUDE.md rule 12; `.claude/settings.json` sets `attribution` to empty strings with `sessionUrl: false` (object form, because older Claude Code versions reject a bare boolean and then ignore the whole file). `.claude/settings.local.json` is gitignored.
- **GitHub repo is private** (`rithika-sr/agentic-retrieval-lab`) until the README has real results.
- Commit author is `rithika-sr <torits0311@gmail.com>` (personal email, chosen to outlive the university address).

**Surprises and how we handled them**
- **`uv_build` needs `src/arl/__init__.py` to exist before `uv sync`.** Created it first, then filled it in.
- **`uv.lock` has 151 packages but 128 were installed.** The extra entries (`nvidia-*`, `cuda-*`, `triton`) are Linux-only GPU packages that uv skips on macOS. Verified none were installed.
- **First run is slow, later runs are fast.** First import: sentence-transformers 37.0 s, ranx 10.4 s (numba compiling), bm25s 5.8 s; whole first run ≈ 66 s. Second full run: 13.6 s. One-time bytecode/JIT caching, nothing to fix.
- **ranx prints `NumbaTypeSafetyWarning: unsafe cast from uint64 to int64`** in its recall code. Harmless at our scale (only matters above ~9×10¹⁸). Left as is.
- **Cosine similarity doesn't bottom out at 0.** Unrelated sentences scored about 0.31 with bge-small. Lesson: raw cosine is only meaningful relative to other candidates for the same query; never threshold it. This is also why fusion uses RRF on ranks, not raw scores.
- **The bge-small cache folder looked empty (1 MB).** This `huggingface_hub` version stores weights in a shared `~/.cache/huggingface/hub/blobs/` store and symlinks to it. The 134 MB download is there. Cached revision `5c38ec7c`.
- **HF Hub "unauthenticated requests" warning.** Harmless for public models. `HF_TOKEN` will be needed in M2 for the gated `google/embeddinggemma-300m`.
- **`ruff format` reported "12 files" when there are only 4 Python files.** Verified with a modification-time check that only `scripts/smoke_test.py` changed (line wrapping); no Markdown was touched. Cause of the count not found.
