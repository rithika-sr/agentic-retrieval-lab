"""M0 smoke test: check that every piece of the environment actually runs.

Usage:
    uv run python scripts/smoke_test.py                  # free, offline checks (a, b, c, e)
    uv run python scripts/smoke_test.py --model          # + downloads bge-small (~134 MB, once)
    uv run python scripts/smoke_test.py --model --api    # + one paid Claude call (~$0.00005)

No --limit flag: the smoke test only uses fixed toy data.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

from arl.config import PROJECT_ROOT, load_config

LIBRARIES = [
    "ir_datasets",
    "bm25s",
    "sentence_transformers",
    "faiss",
    "ranx",
    "anthropic",
    "pydantic",
    "pandas",
    "numpy",
    "yaml",
    "dotenv",
    "tqdm",
    "matplotlib",
    "plotly",
    "streamlit",
    "sklearn",
    "torch",
    "arl",
]

results: dict[str, str] = {}


def report(name: str, ok: bool | None, detail: str = "") -> None:
    """Record and print one check as PASS, FAIL, or SKIP (ok=None)."""
    status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    results[name] = status
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""))


def check_imports() -> None:
    """(a) Import every library and time it; ranx is slow the first time (numba compiles)."""
    print("\n(a) Imports. Note: ranx's first import can take 10-60 s while numba compiles.")
    failed = []
    for module in LIBRARIES:
        start = time.perf_counter()
        try:
            importlib.import_module(module)
        except Exception as err:  # noqa: BLE001 - we want to report any import failure
            failed.append(f"{module} ({type(err).__name__}: {err})")
            continue
        elapsed = time.perf_counter() - start
        if elapsed > 1.0:
            print(f"      {module} took {elapsed:.1f} s")
    report("imports", not failed, "all libraries" if not failed else "; ".join(failed))


def detect_device() -> str:
    """(b) Return 'mps' if Apple's GPU backend works (verified with a real op), else 'cpu'."""
    import torch

    print("\n(b) Torch device")
    if torch.backends.mps.is_available():
        try:
            x = torch.ones(3, device="mps")
            assert float((x * 2).sum().cpu()) == 6.0
            report("device", True, f"mps (torch {torch.__version__})")
            return "mps"
        except Exception as err:  # noqa: BLE001
            print(f"      mps reported available but failed: {err}")
    report("device", True, f"cpu (torch {torch.__version__}; mps not usable)")
    return "cpu"


def check_bm25(k1: float, b: float) -> None:
    """(c) Index 5 toy documents with bm25s and check the right one ranks first."""
    import bm25s

    print(f"\n(c) BM25 (k1={k1}, b={b} from config)")
    docs = [
        "The cat sat on the warm windowsill and purred.",
        "Stock markets fell sharply after the interest rate announcement.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The Eiffel Tower is a wrought-iron tower in Paris, France.",
        "Python is a popular programming language for data analysis.",
    ]
    query, expected = "eiffel tower paris", 3

    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(bm25s.tokenize(docs, stopwords="en", show_progress=False), show_progress=False)
    ids, scores = retriever.retrieve(
        bm25s.tokenize([query], stopwords="en", show_progress=False), k=2, show_progress=False
    )
    top, runner_up = int(ids[0, 0]), int(ids[0, 1])
    print(f"      query: {query!r}")
    print(f"      #1 doc {top} (score {scores[0, 0]:.3f}): {docs[top]}")
    print(f"      #2 doc {runner_up} (score {scores[0, 1]:.3f}): {docs[runner_up]}")
    report("bm25", top == expected, f"top hit is doc {top}, expected doc {expected}")


def check_embeddings(device: str, run: bool) -> None:
    """(d) Embed 3 sentences with bge-small and check the related pair is most similar."""
    print("\n(d) Dense embeddings (BAAI/bge-small-en-v1.5)")
    if not run:
        report("embeddings", None, "pass --model to download (~134 MB) and run")
        return
    from sentence_transformers import SentenceTransformer

    sentences = [
        "A cat is sleeping on the sofa.",
        "A kitten naps on the couch.",
        "The central bank raised interest rates.",
    ]
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    emb = model.encode(sentences, normalize_embeddings=True)  # unit vectors: dot = cosine
    sims = emb @ emb.T
    print(f"      device={model.device}, embedding dim={emb.shape[1]}")
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        print(f"      cos(s{i}, s{j}) = {sims[i, j]:.3f}   {sentences[i]!r} vs {sentences[j]!r}")
    ok = sims[0, 1] > sims[0, 2] and sims[0, 1] > sims[1, 2]
    report("embeddings", ok, "related pair (s0, s1) scores highest" if ok else "unexpected order")


def check_ranx() -> None:
    """(e) Recall@2 on a toy qrels/run; hand-computed answer is q1=0.5, q2=1.0, mean=0.75."""
    from ranx import Qrels, Run, evaluate

    print("\n(e) ranx Recall@2 (hand-computed: q1=1/2=0.5, q2=1/1=1.0, mean=0.75)")
    qrels = Qrels({"q1": {"d1": 1, "d3": 1}, "q2": {"d2": 1}})
    run = Run({"q1": {"d1": 0.9, "d2": 0.8, "d3": 0.7}, "q2": {"d2": 0.9, "d4": 0.5}})
    per_query = evaluate(qrels, run, "recall@2", return_mean=False)
    mean = float(evaluate(qrels, run, "recall@2"))
    print(f"      ranx per-query: q1={per_query[0]:.2f}, q2={per_query[1]:.2f}; mean={mean:.2f}")
    ok = abs(mean - 0.75) < 1e-9 and abs(per_query[0] - 0.5) < 1e-9
    report("ranx", ok, f"recall@2 = {mean:.2f} (expected 0.75)")


def check_api(model_name: str, run: bool) -> None:
    """(f) One direct Anthropic SDK call (M0-only exception; later calls go via arl.llm)."""
    from dotenv import load_dotenv

    print(f"\n(f) Anthropic API ({model_name})")
    load_dotenv(PROJECT_ROOT / ".env")
    key_present = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    print(f"      ANTHROPIC_API_KEY: {'present' if key_present else 'missing'}")
    if not run:
        report("api", None, "pass --api to make one paid call")
        return
    if not key_present:
        report("api", False, "key missing from .env")
        return
    print("      planned: 1 call, ~20 input + <=5 output tokens, est. cost ~$0.00005")

    import anthropic

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=5,
            messages=[{"role": "user", "content": 'Reply with exactly the word "OK".'}],
        )
    except anthropic.AuthenticationError:
        report("api", False, "authentication failed (check the key in .env)")
        return
    except anthropic.NotFoundError:
        report("api", False, f"model {model_name!r} not found")
        return
    except anthropic.APIStatusError as err:
        report("api", False, f"HTTP {err.status_code}: {err.message}")
        return
    except anthropic.APIConnectionError:
        report("api", False, "network error")
        return

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    usage = response.usage
    cost = usage.input_tokens * 1e-6 + usage.output_tokens * 5e-6  # Haiku 4.5: $1 / $5 per MTok
    print(f"      reply={text!r}, stop_reason={response.stop_reason}")
    print(
        f"      tokens: input={usage.input_tokens}, output={usage.output_tokens}; cost≈${cost:.6f}"
    )
    report("api", "OK" in text.upper(), f"model replied {text!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", action="store_true", help="download + run bge-small (d)")
    parser.add_argument("--api", action="store_true", help="make one paid Claude call (f)")
    args = parser.parse_args()

    config = load_config(args.config)
    check_imports()
    device = detect_device()
    check_bm25(config["retrievers"]["bm25"]["k1"], config["retrievers"]["bm25"]["b"])
    check_embeddings(device, args.model)
    check_ranx()
    check_api(config["llm"]["model"], args.api)

    print("\nSummary: " + ", ".join(f"{k}={v}" for k, v in results.items()))
    return 1 if "FAIL" in results.values() else 0


if __name__ == "__main__":
    sys.exit(main())
