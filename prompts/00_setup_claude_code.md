Read CLAUDE.md and docs/PROJECT_BRIEF.md first. This session is Milestone 0: setup only.
No data downloads (other than the one small model named below) and no experiments.
I'm a beginner in IR and new to Claude Code, so follow the "explain, then build" rule
for every step. IMPORTANT: after each numbered step, STOP and wait for me to type
"next" before continuing.

Already done by me (verify, don't redo): Homebrew, git, uv (via Homebrew), gh installed
and authenticated as rithika-sr, .env created with ANTHROPIC_API_KEY set. Never print,
cat, or echo the contents of .env; only report whether a key is "present" or "missing".

0. Save this prompt verbatim to prompts/00_setup_claude_code.md, replacing the old
   version, so the repo keeps the prompt that was actually used.

1. Check my machine. Report macOS version, chip, RAM, free disk space, and versions of
   git, uv, gh, and brew. Explain in one line what each tool is for in this project.

2. Initialize the Python project in this existing folder. Do NOT delete, recreate, or
   overwrite any existing file (README.md, CLAUDE.md, configs/, docs/, prompts/,
   .gitignore, .env.example, .gitkeep files). If `uv init` would create or overwrite
   any of these, set up pyproject.toml manually instead.
   - Package named `arl` using the existing src/arl/ layout, Python 3.11, with a build
     backend so `import arl` works.
   - Dependencies: ir_datasets bm25s sentence-transformers faiss-cpu ranx anthropic
     pydantic pandas numpy pyyaml python-dotenv tqdm matplotlib plotly streamlit
     scikit-learn
   - Dev dependencies: pytest ruff ipykernel
   - Run `uv sync`. Explain each dependency in one line and which experiment uses it.
   - Explain what uv.lock is and why it gets committed.

3. Create src/arl/__init__.py and src/arl/config.py with load_config(path) that reads
   configs/config.yaml and seeds python, numpy, and torch. Add tests/test_config.py.

4. Create scripts/smoke_test.py, printing PASS or FAIL for each check:
   a. every library imports (warn me that ranx's first import may be slow because
      numba compiles code);
   b. torch device detection (mps vs cpu);
   c. bm25s indexes 5 toy documents and retrieves the right one for a toy query;
   d. sentence-transformers loads BAAI/bge-small-en-v1.5 (tell me the download size
      and ask before it runs), embeds 3 sentences, prints cosine similarities, and you
      explain why the related pair scores higher;
   e. ranx computes recall@2 on a toy qrels and run that you work out by hand first,
      so I can see the metric matches;
   f. ONLY after asking me: one Anthropic API call using the model in
      configs/config.yaml, max_tokens=5, asking it to reply "OK". This direct SDK call
      is an exception for M0 only; from M1 on, every LLM call goes through
      src/arl/llm.py.
   Run it and interpret the output.

5. Run `uv run ruff format .`, `uv run ruff check .`, and `uv run pytest`. Show and
   explain the results.

6. Git: `git init` (branch: main). Confirm with `git check-ignore` that .env and data
   files are ignored, and show me `git status` before committing so I can see exactly
   what will be uploaded. Make the first commit:
   "chore: project scaffold and environment setup (M0)".

7. GitHub: ASK ME FIRST. Then use gh to create a PRIVATE repo named
   agentic-retrieval-lab under rithika-sr, add it as origin, and push main. Show me
   the repo URL.

8. Update docs: fill in the "Setup report (M0)" section of docs/FRICTIONAL.md (machine
   specs, tool versions, device, anything that went wrong and how we fixed it), and
   update the Commands section of CLAUDE.md. Commit ("docs: M0 setup report") and push.

9. End with the standard end-of-session summary from CLAUDE.md, including verification
   commands I can run myself.
