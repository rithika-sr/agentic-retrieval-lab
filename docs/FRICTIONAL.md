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

## 2026-10-07 to 2026-10-08 — M1 part 1: data, LLM client, rewrites, leak judge
_Numbers from `results/foundation/2026-10-08_rewrites_300/` (summary JSONs + run logs) unless marked as console output. Checkpoints 4–7 (full-collection BM25, subcorpus, final tests/docs) are not done yet._

**Data (checkpoint 1)**
- `msmarco-passage/dev/small` queries and qrels live **inside** `collectionandqueries.tar.gz` (1.06 GB), so the small dev files cost the full download. ir_datasets saved it at `~/.ir_datasets/msmarco-passage/collectionandqueries.tar.gz` (not a `downloads/` folder, as first stated).
- 300 queries sampled with seed 42: 317 positives, mean 1.057 / min 1 / max 3 per query (console). Full dev/small: 6,590 / 331 / 51 / 8 queries with 1 / 2 / 3 / 4 positives, mean 1.065. Labels are sparse: expect false negatives.
- Sampling = sort IDs, shuffle once with the seed, take a prefix, so `--limit 20` is exactly the first 20 of the 300.
- Positive passage text is **streamed** from the cached archive with ir_datasets' own `TarExtract` + `FixEncoding` (identical text, nothing extracted): 2.78 GB scanned in 182 s at 222 MB peak RAM (console, `/usr/bin/time -l`). The decode/encoding fix is CPU-bound at ~15 MB/s — relevant for the BM25 build.

**LLM client (checkpoint 2)**
- **Model kept at Haiku 4.5** ($1 / $5 per MTok, checked 2026-10-07) over Haiku 5.5 ($0.10 / $0.50), because Haiku 5.5 rejects non-default `temperature`. Revisit for Exp 1's ~24k judge calls.
- **anthropic 1.x removed `temperature` from `messages.create()`** (TypeError before any request; nothing billed). The API still accepts it for Haiku 4.5, so it is sent via `extra_body`. A test now checks every argument we send is in the SDK signature.
- SDK retries disabled; our own backoff honours `retry-after` and logs the error type. Rate limits read from response headers: 10,000 requests/min, 10M input tokens/min. The 4 retries in the first 20-query run were therefore **not** a per-minute request limit (type unknown — logged since). Throttle set to 0.2 s.
- Dry-run estimates ran 5–30% high (chars/3.5 token estimate and 100-token output guess), as intended.

**Rewrites (checkpoint 3)**
- Prompt tightened after hand review of 20 (distinct angles; related entity must not be the answer; invented Titanic counter-example, checked absent from our 300). **Leakage did not drop** on the 20 (manual count 4 → 5); the model reuses answers it knows despite the instruction. Angle diversity improved; primaries identical to the human query rose 3 → 5, and some replies had 4 supports.
- Full 300: 300/300 rewritten, 0 failures, 1,200 agent queries; 293 calls, $0.2244 (`rewrite_run.log`). 15 queries needed the one retry: 11 returned 4 supports, 4 returned **clarifying questions instead of JSON** for truncated/odd MS MARCO queries (e.g. "what are the road c", "unspecific synonym"). Rewrites for truncated queries are the model's guesses.
- 54/300 primaries are identical to the human query (kept, flagged, as decided). 0 exact or near-duplicate (Jaccard ≥ 0.8) supports.
- Mean words: human 5.96, primary 6.91, support 6.36.
- **The retry must be a different prompt**: at temperature 0 with a cache, resending the same request returns the same bad reply. The retry appends the validation error.

**Leak flag — the judge is not usable yet**
- Judge v1 (question + query only) on the 20 reviewed queries: found 1 of 4 manual leaks (v1 rewrites), 2 of 5 (tightened rewrites), with a false positive (#3 "meaning of the name Nora"). It misses bare-entity leaks ("adonis creed", "xinjiang", "robot patent").
- Manual labels agreed with Rithika: count #20 "robot patent"; don't count #18; #15 "verapamil" only if the question asks for the generic name; #3 is a judge false positive.
- Judge v2 (`passage_v2`, per Rithika's spec: question + official passage + query, "contains the specific answer as stated in the passage, even as a bare keyword") on all 1,200: primary 172 leak / 128 not, support 253 / 647; $0.2379 (`leak_judge_run.log`).
- **v2 fails a negative control** (`leak_judge_sanity.json`): 26 of 54 primaries that are word-for-word identical to the question were flagged, which is impossible by definition. Samples show it flags topic words shared by query and passage (e.g. "Goodland Florida", "Kashi China"). On the first 20 it flags 38/80, agrees with 2 of 5 manual leaks and misses 3. **Do not use `leaks_answer` (passage_v2) in any analysis.** It was not tuned on the 20, per the plan.
- Next (not done): fix the judge before the held-out labeling, and keep the identical-to-question negative control as an automatic check for any future judge.

**Spend:** ≈ $0.50 total in M1 so far ($0.0129 + $0.0067 + $0.0164 + $0.0048 on the 20-query trials; $0.2244 + $0.2379 on the full run), all from run output.

## 2026-10-08 — Leak flag v3: LLM answer extraction + deterministic token rule
_Numbers from `results/foundation/2026-10-08_leak_v3/` (`leak_v3_counts.json`, `leak_v3_sanity.json`, `answer_run.log`) unless marked as console output._

**Design (why v3).** v2 asked the LLM "does this query leak?" and confused topic words with answers. v3 splits the job: Haiku only extracts the passage's answer and labels it `factoid` / `descriptive` (`arl.answers`, one call per human query); a plain Python rule (`arl.leak_v3`, `token_v3`) flags a query if it contains a content token of a *factoid* answer that is not in the human question (lowercase, accent-folded, scikit-learn's 318 English stopwords removed, single letters dropped, simple plural stemming). Descriptive answers and failed extractions → `leaks_answer_v3 = null` ("not checkable"), reported separately. v2 fields are kept in `queries_agent.jsonl` for the record.

**Change after the 20-query trial (logged per the plan): word cap on factoids only.** The 6-word answer cap caused 4/20 extraction failures (console), all descriptive gists of 7–12 words. Descriptive answers never feed the rule, so the cap now applies to factoid answers only; descriptive gists may be up to 25 words (a guard against essays). Free to apply — replies were cached and re-parsed. Side effect on the dev set: the Nora question's first reply ("English, Greek and Latin origin; means light or woman of honor") is now accepted as descriptive, so its earlier false positive (`origin`) became "not checkable". This changed answer *validation*, not the leak rule.

**Full 300** (`answer_run.log`): 298 calls, $0.1663 (dry run said $0.1577 + retries). 122 factoid / 168 descriptive / 10 failed. All 10 failures are factoid answers still over 6 words after the one retry (e.g. a cast list, "South of Hominy and West of Pawhuska, Oklahoma", a gold price table); kept as "extraction_failed", not tuned away.

**Results** (`leak_v3_counts.json`): checkable = factoid rows only (488 of 1,200 agent queries).
- primary: 5 leak / 117 no leak (factoid); 178 not checkable (168 descriptive + 10 failed).
- support: 68 leak / 298 no leak (factoid); 534 not checkable (504 + 30).
- 53 of 300 human queries have at least one flagged agent query.
- Compared with v2 on the same 488 factoid rows (console): both leak 46, v2 only 157, v3 only 27, neither 258.

**Sanity checks** (`leak_v3_sanity.json`):
- (a) Negative control **passed**: 0 of 54 primaries identical to the question flagged (v2: 26). This guards against bugs; it is not an accuracy score.
- (b) Dev set (first 20; reviewed before v3 existed, so optimistic): **caught 4 of 5** manual leaks (hodgenville/kentucky, rockingham, adonis, xinjiang); **missed** `1085008_s3` "robot patent" — extracted as factoid "Provided Education for Serfs", which doesn't contain the leaked term; **2 false positives**: `meter` (Mount Kenya "elevation meters feet"; answer "5199 meters") and `region` (Kashi "what region province"; answer "southwestern Xinjiang Uygur Autonomous Region"). v2 on the same 20: 2 of 5 caught, 38 of 80 flagged.

**Known limitation — generic-word false positives (test in M4, rule unchanged).** Unit and category words in the extracted answer (`meter`, `region`, earlier `origin`) count as leaked tokens even though any normal query would use them. The most frequent leak tokens across all 300 include `county`, `mile`, `country`, `city`, `state`, `year`, `time`, `daily` (console), so this is not rare. Per Rithika: do not change the rule to fit the dev set; measure its precision/recall on held-out hand labels in M4.

**Deferred to Milestone 4:** the held-out hand-labeling of leaks (queries 21+), done with the M4 labeling tool rather than a one-off script. Until then, v3 numbers are dev-set evidence only, and `leaks_answer` (v2) must still not be used.

## 2026-10-08 to 2026-10-09 — M1 part 2: full-collection BM25, pools, subcorpus
_Numbers from `results/foundation/2026-10-08_bm25/` (`bm25_exactness_500000.json`, `bm25_full_build.log`, `pool_stats.json`, `pool_breakdown.json`) and `results/foundation/2026-10-09_corpus/corpus_stats.json` unless marked as console output._

**BM25 setup.** bm25s 0.3.12, Lucene scoring, k1 = 0.9, b = 0.4 (config), English stopwords, Snowball English stemming via **PyStemmer 3.1.0 (new dependency)** to match standard stemmed MS MARCO BM25 setups. No new download: `collection.tsv` is streamed from the archive cached in M1 part 1.

**Memory test (stock bm25s), before any full build** (console, `/usr/bin/time -l`): 500k passages → 1.11 GB peak, 21.9 s; 1M → 1.80 GB, 42.0 s. Linear extrapolation to 8,841,823 passages: **10.8–12.6 GB**, above the ~11 GB limit for a 16 GB Mac, so the full stock build was **not run**. Cause (from the bm25s source): every passage's token ids are kept as Python lists (~0.56 GB per 1M passages), and they are still alive while indexing builds its (score, doc, word) arrays and sorts them into CSC (~0.7 GB per 1M).

**Fallback chosen (option A of A/B/C; Rithika picked it): a chunked builder with the same math.** `build_index` keeps only compact (word id int32, tf uint16) postings per 100k-passage chunk, computes df / avgdl / IDF with bm25s's own functions, then scores each chunk with bm25s's tf function and places it into CSC slots by counting sort (no global argsort). The stock path is kept as `build_index_reference`. Rejected: B (Pyserini prebuilt Lucene index: Java 21 + multi-GB download), C (smaller pool collection: pools would stop being "full collection").
- **Exactness check, 500k slice:** data/indices/indptr arrays identical in dtype and value (14,273,374 nonzeros), vocabularies identical, top-100 identical for all 1,500 query variants, max score difference 0.0. Also a unit test at chunk sizes 1/2/3/100.
- Chunked peak RAM (console): 500k 1.08 GB; 1M 1.29 GB; 2M 1.67 GB (≈0.38 GB per 1M vs ≈1.2–1.4 for stock).
- **Full build** (`bm25_full_build.log`): 8,841,823 passages, vocab 1,392,568, 255,977,138 nonzeros, **329 s, 3.54 GB peak** (projected ~4.3 GB). Index on disk: 2.0 GB at `data/processed/bm25_full/` (console `du`). The peak occurs while saving.
- Small bug fixed: the build CLI crashed on its final print when `--out` was outside the repo (the index had already been saved).

**Pools** (`pool_stats.json`): top-100 for 1,500 variants (300 human + 300 primary + 900 support) → `data/processed/bm25_full_top100.tsv` (TREC run format, 150,000 rows), 10.9 s search. Official positive in the top-100 of 8.8M:
- human 191/300 (63.7%), primary 181/300 (60.3%), support 375/900 (41.7%); any agent query per parent 220/300 (73.3%); any of the 5 variants 233/300 (77.7%).
- Rithika predicted human 60–65% (correct) and primaries same-or-better (they were slightly worse). On the 246 primaries that differ from the human query (`pool_breakdown.json`): human 155 hits vs primary 145; human-only 27, primary-only 17. Likely cause: rewording swaps the user's exact words for synonyms, which hurts exact-match BM25. To recheck with dense models in M2.
- The 20-query smoke test showed human 75% (console); the 300 gave 63.7%. Small-sample noise, a reminder not to quote dev-20 rates.
- Factoid supports flagged by leak v3 hit 45/68 vs 130/298 for unflagged (`pool_breakdown.json`). Consistent with leaked answer words helping retrieval, but correlational, and v3 has known false positives.

**Subcorpus** (`corpus_stats.json`): 317 positives + 104,877 pooled non-positive + 94,806 seeded backfill = **200,000**. 150,000 pool slots → **105,120 unique** pooled passages (Rithika predicted 90k, revised to ~100k after the 20-query run gave 6,887 from 10,000 slots). Pools barely overlap *across* questions; the overlap is between variants of the same question (8,219 passages in all three type pools). Support queries contribute most of the new material: 65,442 passages appear only in support pools (human-only 9,236, primary-only 8,038). 243 of 317 positives were already pooled.
- `corpus.jsonl` tags each passage with `source` (positive / pool / backfill) so later analyses can split by it. Backfill uses `numpy.random.default_rng(42)` over the collection minus positives and pools.
- **Corpus property — length bias:** pool passages are longer (mean 64.6 words, median 56) than positives (54.2 / 50) and backfill (56.1 / 50). BM25 with weak length normalization (b = 0.4) favors long passages. In this corpus, "long" weakly signals "BM25 distractor"; keep this in mind when comparing retrievers in M2.

**Tests (checkpoint 6):** 103 pass, ruff clean. New: `test_leak_v3.py`, `test_bm25.py` (incl. bit-exactness vs stock bm25s), `test_pool.py`, `test_corpus.py` (toy invariants + seeded determinism), `test_corpus_file.py` (real-file invariants: size == target, unique ids, all positives tagged `positive`, no empty text). The last one **skips** when `data/` is not built, e.g. on a fresh clone.

**Spend this session:** $0.0163 (answers, 20) + $0.1663 (answers, 300) = $0.18, from run output. M1 total ≈ $0.68. BM25, pools, and corpus are free.
