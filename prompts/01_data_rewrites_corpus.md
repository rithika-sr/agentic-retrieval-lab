Read CLAUDE.md and docs/PROJECT_BRIEF.md (especially Sections 5, 6, 7, and 12) first.
This session is Milestone 1: data + LLM client + agent rewrites + subcorpus.
No retrievers beyond the BM25 needed for corpus pooling, and no evaluation yet.

Rules for this session:
- Follow "explain, then build". After each numbered checkpoint below, STOP and wait
  for me to type "next".
- Never print .env. Ask before any download over 50 MB (say the size first) and
  before any paid LLM batch (show the dry-run estimate first).
- No Co-Authored-By or Claude attribution in commits.

0. Save this prompt verbatim to prompts/01_data_rewrites_corpus.md.

1. data.py: load msmarco-passage/dev/small via ir_datasets. Tell me the download
   size(s) and where ir_datasets stores files before downloading anything. Sample
   n_human_queries (config) with the config seed; support --limit. Save
   data/processed/queries_human.jsonl (query_id, text, query_type="human") and
   data/processed/qrels_official.tsv (only the sampled queries). Print: number of
   queries, positives per query (mean/min/max), and 5 example queries. Add tests.
   Run with --limit 20 first, then the full 300 after I say go.

2. llm.py, the provider-agnostic client (anthropic now, ollama interface stubbed):
   - SQLite cache at llm.cache_path keyed by a hash of provider+model+prompt+params;
     a cache hit makes no API call.
   - dry_run(prompts) -> call count (excluding cache hits), estimated input/output
     tokens, estimated cost (pricing constants in config, with a comment to verify
     on the console); require typed confirmation when require_confirmation=true.
   - Hard stop at max_calls_per_run; retries with backoff on rate limits.
   - temperature 0 by default for reproducibility.
   - Tests with a mocked client: cache hit/miss, the call cap, the dry-run math.
   No real API calls in this checkpoint.

3. rewrite.py: human query -> {primary: str, support: [str, str, str]} as JSON,
   validated with pydantic. Prompt (written in our own words, inspired by the paper's
   primary/support idea): primary restates the intent; support queries vary
   phrasing, pull in background, reach adjacent entities, and at least one is
   keyword-dense. Each query is 3-15 words and must be a search query, not an answer.
   On invalid JSON: one retry, then log the failure and skip (never crash the batch).
   Save data/processed/queries_agent.jsonl with query_id, parent_id, query_type
   (primary/support), text. Flag duplicates (support identical to primary or human).
   Run on --limit 20 (show the dry run first), then print all 20 human queries with
   their 4 rewrites so I can review them by hand. Wait for my review before the
   full 300.

4. BM25 over the FULL 8.8M collection (for distractor pooling only):
   - Tell me the collection download size and the expected extracted size first.
   - Memory test before the full build: index a 500k-passage slice, report peak RAM
     (e.g. /usr/bin/time -l) and time, then extrapolate to 8.8M. If the projected
     peak is above ~11 GB on my 16 GB Mac, propose a fallback (chunked tokenization,
     or a smaller pool collection) instead of running it.
   - Save the index to data/processed/bm25_full/ so it is built once.
   - Retrieve top bm25_pool_depth (100) for every query variant (human + primary +
     support) and save data/processed/bm25_full_top100.tsv (needed later to compare
     full-collection vs subcorpus BM25).

5. corpus.py: subcorpus = all official positives + the BM25 pools + random backfill
   (config seed) up to target_size. Save data/processed/corpus.jsonl and
   corpus_stats.json with: positives count, unique pooled passages, overlap between
   query types' pools, backfill count, total size, mean passage length. Copy the
   stats and the config to results/foundation/<run_id>/. Before running, ask me to
   predict how many unique pooled passages 1,200 queries x top-100 will give.
   Interpret the stats against my prediction.

6. Tests and quality: unit tests for the sampling (deterministic with seed), the
   rewrite schema, cache behavior, and the corpus (all positives present, no
   duplicate IDs, size == target). Run ruff format, ruff check, and pytest.

7. Docs and git: add dated FRICTIONAL.md entries (decisions, surprises, the BM25
   memory numbers, and any rewrite-quality issues I flagged), add ir_datasets/MS
   MARCO details to SOURCES.md, update the CLAUDE.md Commands section. Show git
   status, commit "feat: M1 data, llm client, agent rewrites, subcorpus", then ask
   before pushing.

8. End with the standard end-of-session summary from CLAUDE.md.
