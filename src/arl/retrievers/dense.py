"""Dense retrieval with sentence-transformers + exact faiss inner-product search.

CLI (timing only; the evaluation lives in `arl.eval.dense_subcorpus`):
  uv run python -m arl.retrievers.dense time --model BAAI/bge-small-en-v1.5 --n 2000

Embeddings are L2-normalized, so inner product == cosine similarity, and IndexFlatIP is an
exact (brute-force) search. Passage embeddings are cached as .npy under
data/processed/embeddings/ (gitignored: data/processed/* and *.npy) with a JSON fingerprint
of the model, prefixes, max length, and doc ids; a mismatched cache is refused, not reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from arl.config import DEFAULT_CONFIG_PATH, load_config
from arl.data import PROCESSED_DIR

EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"

# (query prefix, passage prefix), from each model card.
MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
}


def prefixes(model_name: str) -> tuple[str, str]:
    """(query prefix, passage prefix) for a model; unknown models raise."""
    if model_name not in MODEL_PREFIXES:
        raise KeyError(f"no prefixes configured for {model_name}; add them to MODEL_PREFIXES")
    return MODEL_PREFIXES[model_name]


def add_prefix(texts: list[str], prefix: str) -> list[str]:
    """Prepend a prefix to every text."""
    return [prefix + t for t in texts]


def model_slug(model_name: str) -> str:
    """Filesystem-safe short name, e.g. 'BAAI/bge-small-en-v1.5' -> 'bge-small-en-v1.5'."""
    return model_name.split("/")[-1]


def pick_device(requested: str = "auto") -> str:
    """'mps' if requested/auto and a real MPS op works, else 'cpu'."""
    import torch

    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        try:
            torch.ones(2, device="mps").sum().item()
            return "mps"
        except RuntimeError as err:
            print(f"mps available but failed ({err}); using cpu")
    return "cpu"


def load_model(model_name: str, device: str, max_seq_length: int) -> Any:
    """Load a SentenceTransformer on the device with a capped sequence length."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    return model


def encode(model: Any, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    """Normalized float32 embeddings, shape (len(texts), dim)."""
    emb = model.encode(
        add_prefix(texts, prefix),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(emb, dtype=np.float32)


def fingerprint(model_name: str, max_seq_length: int, doc_ids: list[str]) -> dict[str, Any]:
    """What a cached embedding file must match to be reused."""
    return {
        "model": model_name,
        "passage_prefix": prefixes(model_name)[1],
        "max_seq_length": max_seq_length,
        "normalized": True,
        "n_docs": len(doc_ids),
        "doc_ids_sha256": hashlib.sha256("\n".join(doc_ids).encode()).hexdigest(),
    }


def corpus_embeddings(
    model: Any,
    model_name: str,
    doc_ids: list[str],
    texts: list[str],
    max_seq_length: int,
    batch_size: int,
    cache_dir: Path = EMBEDDINGS_DIR,
    log: Any = print,
) -> np.ndarray:
    """Load cached passage embeddings if the fingerprint matches, else encode and cache."""
    npy = cache_dir / f"{model_slug(model_name)}.npy"
    meta_path = npy.with_suffix(".json")
    want = fingerprint(model_name, max_seq_length, doc_ids)
    if npy.exists() and meta_path.exists():
        have = json.loads(meta_path.read_text())
        if {k: have.get(k) for k in want} != want:
            raise ValueError(f"{npy} was built with different settings; delete it to rebuild")
        log(f"loaded cached passage embeddings {npy} ({have.get('encode_s')} s to build)")
        return np.load(npy)
    t0 = time.perf_counter()
    emb = encode(model, texts, prefixes(model_name)[1], batch_size)
    elapsed = time.perf_counter() - t0
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, emb)
    meta_path.write_text(json.dumps(want | {"encode_s": round(elapsed, 1)}, indent=2) + "\n")
    log(f"encoded {len(texts):,} passages in {elapsed:.1f} s ({len(texts) / elapsed:.0f}/s)")
    return emb


def search(corpus_emb: np.ndarray, query_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by inner product: (positions, scores), shape (n_queries, k).

    faiss runs with one OpenMP thread: with torch loaded in the same process on macOS, two
    OpenMP runtimes collide and multi-threaded faiss search segfaults (seen 2026-11-23).
    Flat search is mostly one BLAS matrix multiply, so this costs little.
    """
    import faiss

    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(corpus_emb)
    scores, positions = index.search(query_emb, min(k, corpus_emb.shape[0]))
    return positions, scores


def time_encoding(model_name: str, n: int, config: dict[str, Any], n_corpus: int) -> dict:
    """Embed the first n subcorpus passages and extrapolate to the full corpus."""
    from arl.corpus import CORPUS_PATH

    dense_cfg = config["retrievers"]["dense"]
    texts = []
    with CORPUS_PATH.open() as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if len(texts) == n:
                break
    device = pick_device(dense_cfg["device"])
    model = load_model(model_name, device, dense_cfg["max_seq_length"])
    encode(model, texts[:64], prefixes(model_name)[1], dense_cfg["batch_size"])  # warm-up
    t0 = time.perf_counter()
    encode(model, texts, prefixes(model_name)[1], dense_cfg["batch_size"])
    elapsed = time.perf_counter() - t0
    rate = n / elapsed
    return {
        "model": model_name,
        "device": device,
        "n_passages": n,
        "seconds": round(elapsed, 2),
        "passages_per_s": round(rate, 1),
        "est_full_corpus_min": round(n_corpus / rate / 60, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("time", help="time passage encoding on the first --n subcorpus passages")
    t.add_argument("--model", required=True, choices=sorted(MODEL_PREFIXES))
    t.add_argument("--n", type=int, default=2000)
    t.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    config = load_config(args.config)
    n_corpus = int(config["corpus"]["target_size"])
    print(json.dumps(time_encoding(args.model, args.n, config, n_corpus), indent=2))


if __name__ == "__main__":
    main()
