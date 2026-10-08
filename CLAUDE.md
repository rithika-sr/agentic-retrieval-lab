# CLAUDE.md — Agentic Retrieval Lab

Claude Code: read this file and `docs/PROJECT_BRIEF.md` at the start of every session.

## Who you're working with
Rithika — MS Data Analytics Engineering student. Comfortable with Python, pandas, SQL, ML basics. **New to information retrieval (IR) evaluation.** Treat her as a beginner in IR and teach as you build.

## Working rules (follow every time)
1. **Explain, then build.** Before each step, write 3–6 plain-English sentences: what you're about to do, why, and what could go wrong. Define new terms the first time they appear.
2. **Ask her to predict.** Before running an evaluation, ask what she expects (e.g. "Will BM25 beat bge-small on Recall@100?"). Then compare her prediction to the result.
3. **Explain outputs.** After running anything, interpret the output: what the numbers mean and whether they look sane.
4. **Small first.** Every script takes `--limit`. Run on 10–20 queries, show the output, and only scale up after she says go.
5. **Never spend money silently.** Before any batch of LLM calls, print the call count, estimated tokens, and estimated cost, then wait for her confirmation. All LLM calls go through `src/arl/llm.py`, with the SQLite cache at `data/cache/llm_cache.sqlite`. Respect `llm.max_calls_per_run`.
6. **Never download large files without saying the size first** and getting a yes.
7. **Secrets:** read keys from `.env` only. Never print them, and never ask her to paste a key into the chat.
8. **Never fabricate results.** Tables and numbers in docs must come from files in `results/`. If something wasn't run, say so.
9. **Don't overwrite results.** Write to `results/<exp>/<run_id>/` and save a copy of the config there.
10. **Honest framing.** This is a small-scale replication of *ideas* from the Q2D-Web paper on public data, not a reproduction of their benchmark.
11. **Update living docs in place:**
    - `docs/FRICTIONAL.md`: decisions, surprises, bugs, limitations (dated).
    - `README.md`: results sections (only real numbers).
    - `docs/SOURCES.md`: any paper, dataset, or model newly used.
12. **Git:** commit at the end of each milestone with a clear message. Ask before pushing. Never add Co-Authored-By trailers or 'Generated with Claude Code' lines to commits or PR descriptions.
13. **Code quality:** type hints, small functions, docstrings, `ruff format`, pytest for metrics, fusion, and schemas.
14. **Apple Silicon:** use `mps` if available, otherwise `cpu`. Report which was used.

## End-of-session summary (always finish with this)
- What we built (files changed)
- What Rithika should have learned (3 bullets)
- **Verification commands** she can run herself to confirm the state
- Key numbers (if any) and where they live
- Open issues and the suggested next milestone

## Repo map
- `configs/config.yaml` — all tunable settings
- `src/arl/` — shared foundation (data, llm, rewrite, corpus, retrievers, fusion, evaluate)
- `experiments/exp1_mini_q2d/` — agent queries, LLM judge, label sets
- `experiments/exp2_bm25_fusion/` — hybrid vs all-neural ensembles
- `experiments/exp3_failure_analyzer/` — failure taxonomy and fix simulator
- `app/` — Streamlit apps (labeling tool, analyzer dashboard)
- `data/` — gitignored; `results/` — committed metrics and charts (no big files)
- `prompts/` — Claude Code prompts per milestone

## Commands (fill in as they are created)
- `uv sync` — install the environment
- `uv run pytest` — run tests
- `uv run ruff format . && uv run ruff check .`
- `uv run ruff format . && uv run ruff check . && uv run pytest` — full pre-commit check
- `uv run python scripts/smoke_test.py` — free, offline environment checks (imports, device, BM25, ranx)
- `uv run python scripts/smoke_test.py --model --api` — also loads bge-small (~134 MB, once) and makes one paid Claude call (~$0.00005)
- `uv run python -m arl.data [--limit N]` — sample human queries, write qrels, and stream the positive passages from the cached archive (~3 min; free)
- `uv run python -m arl.rewrite [--limit N] --dry-run` — estimate rewrite cost; drop `--dry-run` to run (typed `yes`), add `--review` to print rewrites
- `uv run python -m arl.leak [--limit N] --dry-run` — estimate leak-judge cost; drop `--dry-run` to run (judge v2 currently fails its sanity check — see FRICTIONAL.md)
- `uv run python scripts/summarize_rewrites.py --run-id <id> [--logs ...]` — write rewrite/leak stats + config to `results/foundation/<id>/` (free, refuses to overwrite)
- `git push` — upload commits to `origin` (GitHub, private); ask first
