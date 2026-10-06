# Agentic Retrieval Lab — Project Brief

> **What this file is:** the single source of truth for this project. Upload it to the Claude.ai Project as knowledge, and keep a copy in the repo at `docs/PROJECT_BRIEF.md`. When a decision changes, update this file and log *why* in `docs/FRICTIONAL.md`.

---

## 0. How the assistant should use this brief

You (Claude, in the Claude.ai Project) are my **build partner and teacher**. You work alongside Claude Code:

- **Claude.ai Project (planning layer):** explains concepts, reviews results, catches mistakes, and writes **paste-ready prompts** for Claude Code, one milestone at a time. I paste them into Claude Code without reformatting.
- **Claude Code (execution layer):** writes and runs the code in my local repo, following `CLAUDE.md`.

The loop for every milestone:

1. I tell you where I am, and paste the end-of-session summary from Claude Code.
2. You **verify the real state** first. Ask me to run a check command (e.g. `ls results/`, `uv run pytest`, `head` a file) rather than trusting a summary.
3. You explain the next milestone in plain English: what we're building, why, and what the numbers will mean.
4. You give me **one** paste-ready Claude Code prompt for that milestone.
5. After it runs, we interpret the results together and update the docs.

**Teach me like a beginner in information retrieval (IR).** I'm comfortable with Python, pandas, SQL, and ML basics, but IR evaluation is new to me. Define every new term the first time it appears. Ask me to *predict* a result before we run it, because that's how I'll actually learn. Never invent numbers; every result must come from a file in `results/`.

---

## 1. The pitch

**Agentic Retrieval Lab** is a small-scale, open-data study of *first-stage retrieval for agentic RAG*, inspired by Perplexity's paper **"Q2D-Web: A Large-Scale Benchmark for Retrieval in Agentic RAG Systems"** (Schall et al., arXiv:2609.08887, Sept 2026).

Q2D-Web's data is private, so this project rebuilds its core ideas on public data at laptop scale and answers three questions:

1. **Exp 1 — Mini Q2D:** When an LLM agent rewrites human queries into *primary* and *support* queries, how does retrieval quality change? And how much do shallow labels hide?
2. **Exp 2 — Does BM25 earn its seat?** Is a hybrid of BM25 plus one embedding model better than two embedding models fused the same way? The paper explicitly leaves this untested.
3. **Exp 3 — Failure Analyzer:** A reusable tool that explains *why* a retriever missed relevant documents, using a failure taxonomy based on the paper's, and measures which fix (hybrid search, query rewriting, chunking) recovers the most misses.

All three run on **one shared foundation**: same corpus, same retrievers, same evaluation harness.

---

## 2. About me (context for the assistant)

- **Rithika Sankar Rajeswari**, MS in Data Analytics Engineering at Northeastern University (graduating Dec 2026). Currently a Graduate TA for IE 7615 Deep Learning for AI and a peer mentor.
- Job searching for **Data Science / Data Engineering / ML Engineering / AI Engineering** roles starting **January 2027**. This project is a portfolio piece and the basis of a LinkedIn post series.
- Previous experience: AI/ML Data Science Intern at Intuitive (Summer 2026). My project there was KB Question Intelligence: clustering and surfacing what users ask an internal RAG assistant. I also contributed **early ideation and literature review only** for a RAG evaluation framework that another team built. **Never overclaim this** in posts, the README, or interview framing. This lab is my own hands-on RAG evaluation work, and that's the honest link between the two.
- Workflow I like: Claude.ai for planning and review, Claude Code for execution, living docs updated in place (`README.md`, `FRICTIONAL.md`, `SOURCES.md`), and verifying actual output before writing the next prompt.
- Machine: MacBook (Apple Silicon assumed; confirm in Milestone 0). No dedicated GPU; Google Colab is available as a fallback.

---

## 3. What we borrow from the paper, and what we can't

**Borrowing (with credit):**

- The **primary vs support query** structure of agent searches.
- **Pooling plus an LLM judge** to build deeper relevance labels and reduce false negatives.
- **Multiple label sets**, and checking whether model rankings stay stable across them.
- A **holdout principle**: models that help create the labels are kept separate from (older than) the models being tested.
- **Subcorpus sampling**: keep all positives plus hard distractors, so a big corpus becomes a laptop-sized one.
- **RRF** fusion, **unique positives** analysis, and the **hard-positive / hard-negative taxonomies**.

**Can't do (and must say so publicly):**

- Their data is private, so we use **MS MARCO passages** (public web search queries plus Bing passages).
- Our scale is thousands of queries and about 200k passages, not 70k queries and 190M documents. **This is a small-scale replication of ideas, not a reproduction of their benchmark.**
- Their agent rewrote queries using conversation history. MS MARCO queries are single-turn, so our agent rewrites without context. This is a known limitation.
- MS MARCO passages are short (~60 words), so the **truncation** failure category will barely appear. That's a finding in itself, and Exp 3 has a stretch goal to add long documents.

---

## 4. Architecture: one lab, three experiments

```
                     ┌──────────────────────────────────────────┐
                     │            SHARED FOUNDATION             │
                     │  data loader → query sampler → rewriter  │
                     │  → subcorpus builder → retrievers        │
                     │  (BM25 + dense) → RRF fusion → evaluator │
                     │  + LLM client (cached, cost-guarded)     │
                     └───────────────┬──────────────────────────┘
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
  Exp 2: BM25 fusion        Exp 1: Mini Q2D            Exp 3: Failure Analyzer
  (official qrels first;    (agent rewrites, LLM       (taxonomy classifier +
   re-run on deep labels)    judge, label sets)         fix simulator + app)
          │                          │                          │
          └──────── LinkedIn Post 1 ─┴── Post 2 ────────────────┴── Post 3
```

**Build order:** Foundation → **Exp 2 v1** (fastest, needs no LLM judging) → **Exp 1** (creates deep labels) → **Exp 3** (uses Exp 1's labels and runs) → **Exp 2 v2** (re-run on deep labels).

---

## 5. Data and scale decisions

**Dataset:** MS MARCO Passage, loaded via `ir_datasets` as `msmarco-passage/dev/small`. That gives 6,980 dev queries with official relevance labels (about 1 positive per query), over a collection of 8.8M passages. The compressed download is about 1 GB.

**Query sample:** `n_human_queries = 300` (configurable), sampled with a fixed seed. Each query gets **1 primary + 3 support** agent rewrites, for 1,200 queries total across human, primary, and support. Start with 20 for smoke tests.

**Why not use the full 8.8M passages?** BM25 can index all 8.8M on a laptop. Embedding all 8.8M with even a small model would take many hours on a Mac. So we copy the paper's trick and build a **subcorpus**:

1. Keep **every officially relevant passage** for the sampled queries.
2. Add **hard distractors**: the BM25 top-100 over the *full* collection, for every query variant (human + primary + support).
3. **Random backfill** up to `target_size = 200,000` passages, so it isn't only BM25-flavored distractors.

**Known bias, logged in FRICTIONAL.md:** the distractors come from BM25 only, so they are hard *for BM25* but not specifically hard for dense models. That may make dense retrievers look better than they would on the full corpus. Mitigations: (a) state it openly, (b) Exp 1 stretch goal measures how much model *rankings* change between a BM25-pooled and a random subcorpus, using Kendall's τ (a mini version of the paper's Section 4.2), (c) optional Colab run to embed the full collection with one small model and add dense-pooled distractors.

**Metric cutoffs at our scale:** the paper uses Recall@1000 over 190M documents. Over 200k passages, top-1000 is a much bigger slice (0.5% vs 0.0005%), so Recall@1000 will look inflated. We report **Recall@100 as the primary metric**, plus Recall@1000, nDCG@10, and MRR@10 (MS MARCO's official metric). The assistant should explain why the cutoff shifts.

---

## 6. Tech stack (and why)

| Tool | Role | Why this one |
|---|---|---|
| Python 3.11 + **uv** | Environment and packages | Fast, modern, one command to reproduce the environment |
| **ir_datasets** | Load MS MARCO plus qrels | Standard IR data loader, handles downloads and caching |
| **bm25s** | BM25 lexical retrieval | Pure Python, fast, no Java (Pyserini needs Java) |
| **sentence-transformers** | Dense embedding models | Standard API, works on Apple MPS |
| **faiss-cpu** | Vector search | Exact (flat) search is easily fast enough at 200k |
| **ranx** | Metrics, RRF fusion, significance tests | Implements everything we need, well tested |
| **anthropic** SDK (or **Ollama** local) | Query rewriting, judging, failure classification | Provider-agnostic wrapper so I can switch to free local models |
| pydantic | Validate LLM JSON outputs | Catches malformed rewrites and labels |
| pandas, matplotlib / plotly | Analysis and charts | — |
| **streamlit** | Exp 3 dashboard | Fastest way to a shareable demo |
| pytest, ruff | Tests and formatting | Shows engineering quality in the repo |

**Models:**

- **Pool models** (older, used to build labels and pools): BM25, `BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`.
- **Holdout models** (newer, never used to create labels): `Qwen/Qwen3-Embedding-0.6B`, `google/embeddinggemma-300m` (gated, so accept the license on Hugging Face and run `hf auth login`).
- **Pitfall:** each embedding model expects its own query/passage prefix or instruction template. For example, e5 expects `"query: "` and `"passage: "` prefixes. The paper used each model's default template, and we do the same. Getting this wrong silently tanks scores.
- **LLM:** `claude-haiku-4-5-20251001` for rewriting, judging, and classifying (cheap and fast). Ollama with a local model is the free alternative, configured in `configs/config.yaml`.

---

## 7. Shared foundation spec (Milestones 1–2)

Package: `src/arl/`. Every CLI script takes `--config` and `--limit`.

- `config.py`: loads `configs/config.yaml`, seeds everything.
- `data.py`: loads MS MARCO via ir_datasets, samples human queries, saves `data/processed/queries_human.jsonl` and `qrels_official.tsv`.
- `llm.py`: provider-agnostic client (Anthropic / Ollama). **Every call is cached** in SQLite, keyed by a hash of model + prompt, so re-runs are free. Before any batch it does a **dry run**: prints call count, estimated tokens, and estimated cost, then asks for confirmation. Hard stop at `max_calls_per_run`.
- `rewrite.py`: turns each human query into `{primary, support[3]}` JSON, validated with pydantic. The prompt asks the agent to: restate intent (primary); vary phrasing, pull in background, reach adjacent entities, and include at least one keyword-dense support query. Saves `queries_agent.jsonl` with `parent_id` and `query_type`.
- `corpus.py`: builds the subcorpus (positives + BM25 top-100 for all variants + random backfill). Saves `corpus.jsonl` and a `corpus_stats.json`.
- `retrievers/bm25.py`, `retrievers/dense.py`: a common interface, `retrieve(queries) -> run`, using the ranx Run format, top-1000. Dense: encode the corpus once, cache embeddings as `.npy`, FAISS flat inner-product index, normalized vectors, MPS if available.
- `fusion.py`: RRF (k=60) over any list of runs, via ranx `fuse`, with a unit test against a hand-computed example.
- `evaluate.py`: computes metrics per run, per label set, per query type. Writes `results/<exp>/<run_id>/metrics.csv` plus a copy of the config.
- `tests/`: metric sanity tests (a perfect run gives recall 1.0), a fusion test, and a rewrite schema test.

---

## 8. Experiment 2 — Does BM25 earn its seat? (Milestone 3, then re-run in Milestone 6)

**Question:** At an equal ensemble size, with the same fusion rule and candidate budget, does including BM25 beat an all-neural ensemble?

**Why it matters:** The paper found BM25 has the *lowest* recall but contributes the *most unique* relevant documents (36,073 vs 27,979 for the next model). They explicitly did not test whether those unique finds translate into better fused recall. Every team building hybrid search wants to know this.

**Design:**

- Components: BM25, bge-small (D1), e5-small (D2), Qwen3-0.6B (D3).
- Size-2 ensembles: {BM25+D1}, {BM25+D2}, {BM25+D3} vs {D1+D2}, {D1+D3}, {D2+D3}.
- Size-3 ensembles: {BM25+D1+D3} vs {D1+D2+D3}.
- Controlled: RRF k=60, each component contributes its top-1000, the same metric cutoffs.
- Also compute **unique positives per retriever**, as in the paper's Section 5.1.
- **Statistics:** paired significance tests via ranx `compare`. Report effect size, not just p-values. With 300 queries, small differences may not be significant, and that's an honest result.

**v1 (Milestone 3):** human and primary queries scored with official qrels.
**v2 (Milestone 6):** re-run on Exp 1's Combined labels, broken down by human / primary / support.

**Hypothesis (to test, not assume):** BM25 helps Recall@100 most on support queries and on queries with rare entities or numbers, but may not help nDCG@10.

**Outputs:** `results/exp2/` metrics table, a fused-vs-component bar chart, a unique-positives chart, and `experiments/exp2_bm25_fusion/RESULTS.md`.

---

## 9. Experiment 1 — Mini Q2D (Milestone 4)

**Question:** How do retrievers behave on agent-written queries compared with human ones, and how much do shallow labels hide?

**Steps:**

1. **Label sets:**
   - **Official:** MS MARCO qrels. These apply to human and primary queries (same intent).
   - **LLM-Judged:** for each of the 1,200 queries, RRF over the **pool models only** (BM25, bge-small, e5-small), then take the top-20 unlabeled passages and send them to the LLM judge for a strict YES/NO. For support queries, also add the parent query's official positive to the candidates.
   - **Combined:** the union of Official and LLM-Judged.
2. **Judge prompt:** strict constraint matching, written in our own words and inspired by the paper's Figure 5. Output: YES or NO, plus a one-line reason (the reason is used for auditing only).
3. **Judge validation** (critical for credibility):
   - (a) Run the judge on official positives. It should say YES most of the time; report the agreement rate.
   - (b) **I hand-label 50 judged pairs** using a small CLI or Streamlit labeling tool. Report the judge's precision against my labels.
4. **Analyses:**
   - Retrieval quality on human vs primary vs support queries, as a dumbbell chart like the paper's Figure 8.
   - **Label-set sensitivity:** Kendall's τ of the model ranking across Official / LLM-Judged / Combined (does the paper's "ranking is stable" finding hold here?).
   - **False-negative estimate:** what fraction of top-10 results that are unlabeled officially get judged relevant (compare to the 70% MS MARCO finding the paper cites).
   - **Holdout check:** report pool models and holdout models separately.
   - *Stretch:* subcorpus sensitivity (BM25-pooled vs random subcorpus, Kendall's τ).

**Outputs:** `data/processed/qrels_{llm,combined}.tsv`, a judge-validation report, charts, and `experiments/exp1_mini_q2d/RESULTS.md`.

**Budget:** about 1,200 queries × 20 candidates ≈ 24k judge calls. **Check current Haiku pricing on the Anthropic console before running.** `llm.py` prints an estimate and requires confirmation. Cut `pool_depth` to 10 if needed.

---

## 10. Experiment 3 — Retrieval Failure Analyzer (Milestone 5)

**Question:** When a retriever misses relevant documents, *why*, and which fix recovers the most?

**Build as a reusable tool:**
`uv run arl-analyze --run <run> --qrels <qrels> --queries <q> --corpus <c>` → a report, plus a Streamlit app (`app/analyzer_app.py`).

**Steps:**

1. **Mine failures:**
   - **Hard positives:** relevant passages missing from a retriever's top-k (also "missed by all").
   - **Hard negatives:** non-relevant passages that every retriever ranked in its top-10.
2. **Classify with an LLM**, using categories adapted from the paper's taxonomy (credit it):
   - Hard positives: `lexical`, `reasoning`, `truncation`, `multilingual`, `entity_numeric`, `generic_dup`, `other`.
   - Hard negatives: `false_negative`, `wrong_aspect`, `related_entity`, `partial`, `generic`, `temporal`, `other`.
3. **Validate the classifier:** I hand-label 50 cases. Report Cohen's κ (agreement beyond chance).
4. **Fix simulator:** apply each fix to the failure set and measure the **recovery rate** (share of missed positives that now appear in the top-k), by category:
   - **Hybrid:** fuse the failing run with BM25 using RRF.
   - **Query rewriting:** LLM rewrites the failed query, then re-retrieve.
   - **Chunking:** only meaningful for long documents. *Stretch:* add a long-document dataset (e.g. a small MS MARCO Documents sample) so truncation failures exist.
5. **Dashboard:** category distribution, an example browser (query, missed document, LLM reason), and a "which fix helps which failure" heatmap.

**Outputs:** the CLI, the Streamlit app, `experiments/exp3_failure_analyzer/RESULTS.md`, and a demo GIF for LinkedIn.

---

## 11. Milestones, timeline, and LinkedIn series

Today is Mon Oct 5, 2026. Estimates assume focused 2–3 hour sessions.

| # | Milestone | What exists at the end | Target |
|---|---|---|---|
| M0 | **Setup** | uv env, deps installed, smoke tests pass, API key works, git initialized | Oct 5–6 |
| M1 | **Data + rewrites + corpus** | sampled queries, agent rewrites (validated), 200k subcorpus | Oct 7–8 |
| M2 | **Retrievers + eval harness** | BM25 + 4 dense runs, metrics tables, tests | Oct 8–10 |
| M3 | **Exp 2 v1** | fusion results, unique positives, charts, RESULTS.md | Oct 10–11 |
| — | **LinkedIn Post 1** | "Does BM25 still earn its place in hybrid search?" | week of Oct 12 |
| M4 | **Exp 1** | LLM-judged labels, judge validation, human-vs-agent analysis | Oct 12–17 |
| — | **LinkedIn Post 2** | "What happens to retrieval when an AI agent writes the query?" | week of Oct 19 |
| M5 | **Exp 3** | analyzer CLI + Streamlit app + validated taxonomy | Oct 19–24 |
| M6 | **Exp 2 v2 + polish** | re-run on deep labels, README, final charts | Oct 24–26 |
| — | **LinkedIn Post 3** | "Why retrieval fails, and which fix actually helps" | week of Oct 26 |

**Post rules:** numbers only from `results/`; credit the Q2D-Web authors and link the paper; state the scale and that it's a small-scale replication; one clear chart per post; link the repo; no overclaiming about the Intuitive work.

---

## 12. Budget and guardrails

- **Compute:** laptop for everything at 200k passages. If dense encoding takes more than about an hour per model, reduce `target_size` or move encoding to Colab.
- **Disk:** about 10 GB free (MS MARCO collection, BM25 index, embeddings).
- **API:** dry-run estimates plus confirmation before every batch; SQLite cache; `max_calls_per_run` hard cap. The API is billed separately from a Claude.ai subscription.
- **Secrets:** API keys live only in `.env` (gitignored). Never paste keys into chat.
- **Reproducibility:** fixed seeds; each results folder stores the config used.

---

## 13. Definition of done (portfolio quality bar)

- The README opens with the question, one headline chart, and a 3-sentence answer.
- Anyone can reproduce it with `uv sync` plus documented commands.
- Tests pass; code is formatted with ruff.
- Every number in the README and posts traces to a file in `results/`.
- A **Limitations** section is written up front, not hidden.
- `SOURCES.md` credits the paper and every dataset and model.
- An interview-ready story: the problem, my design choices, what surprised me, and what I'd do at production scale.

---

## 14. Limitations to state honestly

Small scale (300 human queries, 200k passages); single-turn rewrites with no conversation context; an English-only corpus (the paper covers 10 languages); BM25-pooled distractors; LLM judge bias (the same model family judges and classifies); short passages (so truncation is underrepresented); binary labels.

---

## 15. Glossary (quick reference)

- **Corpus:** the documents being searched.
- **Qrels:** the answer key (query–document relevance labels).
- **Run:** a retriever's ranked results for each query.
- **First-stage retriever:** the fast model that fetches candidates from the whole index.
- **BM25:** keyword scoring that weights rare words more.
- **Dense retriever:** embeds queries and documents as single vectors, then matches them by similarity.
- **Late interaction:** one vector per token (ColBERT-style).
- **Recall@k:** the share of relevant documents found in the top k.
- **nDCG@10:** quality of the ordering in the top 10.
- **MRR@10:** 1 / rank of the first relevant hit.
- **RRF:** fuses rankings by summing 1/(60+rank) across retrievers.
- **Pooling:** building the candidate set from multiple systems' top results.
- **False negative (in labels):** a relevant document labeled not relevant.
- **Hard negative:** a non-relevant document that looks relevant.
- **Hard positive:** a relevant document that retrievers miss.
- **Kendall's τ:** agreement between two orderings (1.0 = identical).
- **Cohen's κ:** agreement between two labelers beyond chance.
- **Holdout:** keeping evaluated models out of label creation, to avoid home-field advantage.
