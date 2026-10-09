Read CLAUDE.md, docs/PROJECT_BRIEF.md, the latest entries in docs/FRICTIONAL.md, and prompts/01_data_rewrites_corpus.md. We are continuing Milestone 1 with the same rules as that prompt: explain then build; STOP after each checkpoint until I type "next"; ask before downloads over 50 MB and before paid batches (dry run first); no attribution in commits.

Save this prompt verbatim to prompts/01b_continue_m1.md.

A. Leak flag v3 (keep this small and bounded)
1. Answer extraction: one Haiku call per human query (300) via llm.py. Given the question and its official passage(s), return JSON {"answer_type": "factoid" | "descriptive", "answer": "<at most 6 words>"}. Factoid = a specific entity, place, person, date, number, or name. Descriptive = definitions, explanations, how-to. Validate with pydantic. Dry run first.
2. Deterministic leak rule (no LLM): for factoid answers only, an agent query leaks if it contains any content token of the answer (lowercased, stopwords removed, simple plural stemming) that does NOT appear in the human question. Descriptive answers get leaks_answer_v3 = null, meaning "not checkable", reported separately. Store answer, answer_type, leaks_answer_v3, and leak_tokens per row. Keep the v2 fields for the record.
3. Sanity checks, saved to results/foundation/<run_id>/:
   (a) primaries identical to the human question must have 0 leaks;
   (b) on the 20 queries already reviewed (the dev set, not a test set), report which of the 5 manual leaks it catches and any false positives.
   Do not tweak the rule to fit those 20 beyond fixing clear bugs; log any change in FRICTIONAL.md.
4. Report leak counts by query type (primary/support) and answer type (factoid/descriptive).
Defer the held-out hand-labeling to Milestone 4, where we build the labeling tool anyway, and note this in FRICTIONAL.md.

B. Then do checkpoint 4 from prompts/01_data_rewrites_corpus.md (BM25: memory test on a 500k-passage slice before any full build), and continue through checkpoints 5-8.
