"""Check that the chunked BM25 builder reproduces stock bm25s exactly on a collection slice.

Usage:
  uv run python scripts/verify_bm25_exact.py --reference data/processed/bm25_slice500k \
      --limit 500000 --run-id 2026-10-08_bm25

Builds the chunked index in memory over the first --limit passages, loads the reference
index (built earlier with `build_index_reference` over the same slice), and compares the
raw CSC arrays, the vocabularies, and the top-100 results for every query variant
(human + agent). Writes bm25_exactness_<limit>.json to results/foundation/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
from itertools import islice

import numpy as np

from arl.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from arl.data import QUERIES_HUMAN_PATH, iter_collection, read_queries_jsonl
from arl.leak import read_jsonl
from arl.retrievers.bm25 import META_NAME, build_index, load_index, search
from arl.rewrite import QUERIES_AGENT_PATH

TOP_K = 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--reference", required=True, help="stock-built index directory")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    out_path = PROJECT_ROOT / "results" / "foundation" / args.run_id
    out_path = out_path / f"bm25_exactness_{args.limit}.json"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; results are never overwritten")
    ref_dir = PROJECT_ROOT / args.reference
    ref_meta = json.loads((ref_dir / META_NAME).read_text())
    if ref_meta["n_docs"] != args.limit:
        raise ValueError(f"reference has {ref_meta['n_docs']} docs, not {args.limit}")

    bm25_cfg = load_config(args.config)["retrievers"]["bm25"]
    new, new_tok, stats = build_index(
        islice(iter_collection(), args.limit), k1=bm25_cfg["k1"], b=bm25_cfg["b"]
    )
    ref, ref_tok = load_index(ref_dir)

    arrays = {
        key: {
            "dtype_equal": ref.scores[key].dtype == new.scores[key].dtype,
            "array_equal": bool(np.array_equal(ref.scores[key], new.scores[key])),
            "length": len(new.scores[key]),
        }
        for key in ("data", "indices", "indptr")
    }
    queries = [q["text"] for q in read_queries_jsonl(QUERIES_HUMAN_PATH)]
    queries += [r["text"] for r in read_jsonl(QUERIES_AGENT_PATH)]
    ref_docs, ref_scores = search(ref, ref_tok, queries, k=TOP_K)
    new_docs, new_scores = search(new, new_tok, queries, k=TOP_K)
    same_rows = (ref_docs == new_docs).all(axis=1)

    report = {
        "limit": args.limit,
        "reference_index": args.reference,
        "reference_build": {k: ref_meta[k] for k in ref_meta if k != "limit"},
        "chunked_build": stats,
        "arrays": arrays,
        "vocab_equal": ref.vocab_dict == new.vocab_dict,
        "tokenizer_vocab_equal": ref_tok.get_vocab_dict() == new_tok.get_vocab_dict(),
        "n_queries": len(queries),
        "top_k": TOP_K,
        "queries_with_identical_topk_docs": int(same_rows.sum()),
        "topk_scores_max_abs_diff": float(np.abs(ref_scores - new_scores).max()),
    }
    report["exact"] = bool(
        all(a["dtype_equal"] and a["array_equal"] for a in arrays.values())
        and report["vocab_equal"]
        and report["tokenizer_vocab_equal"]
        and report["queries_with_identical_topk_docs"] == len(queries)
        and report["topk_scores_max_abs_diff"] == 0.0
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
