"""
CLI: evaluate static word embeddings against psycholinguistic benchmarks.

Loads a (V, k) embedding from a Stage-3 .npz file and a stoi vocabulary from
a meta.pkl file, runs all available benchmarks, prints a sorted results table,
and optionally writes a CSV.

Example (Shakespeare char, downloadable benchmarks only):
    python scripts/run_embeddings_benchmarks.py \\
        --embedding embeddings/shakespeare_char_emb.npz \\
        --meta data/shakespeare_char/meta.pkl

Example (with local datasets):
    python scripts/run_embeddings_benchmarks.py \\
        --embedding embeddings/coca_emb.npz \\
        --meta data/coca/meta.pkl \\
        --ws353  ~/Documents/data/wordsim353crowd.csv \\
        --swow   ~/Documents/data/SWOW-EN18/strength.SWOW-EN.R1.20180827.csv \\
        --usf    "~/Documents/data/Nelson-BRM-2004/Appendices .csv files/AppendixA1.csv" \\
        --spaml  ~/Documents/data/semantic_priming/en_answered_prime_trials.csv \\
        --spp    ~/Documents/data/semantic_priming_project.xlsx \\
        --output results/benchmarks.csv
"""

import argparse
import os
import pickle
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from shape.embeddings_benchmarks import run_benchmarks


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate static embeddings on psycholinguistic benchmarks.")
    p.add_argument("--embedding", required=True,
                   help="Path to .npz embedding file (must contain 'embedding' key).")
    p.add_argument("--meta", required=True,
                   help="Path to meta.pkl containing {'stoi': {word: int}}.")
    p.add_argument("--output", default=None,
                   help="Optional CSV path to write results.")
    p.add_argument("--method", default="cosine", choices=["cosine", "dot"],
                   help="Similarity metric (default: cosine).")
    p.add_argument("--cache-dir", default=None,
                   help="Directory for cached benchmark downloads (default: ~/.cache/semshape/benchmarks/).")
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Restrict to specific benchmark names (default: all available).")

    # Local-only benchmarks
    p.add_argument("--ws353",  default=None, help="Path to WordSim353 CSV.")
    p.add_argument("--swow",   default=None, help="Path to SWOW strength CSV.")
    p.add_argument("--usf",    default=None, help="Path to USF norms CSV.")
    p.add_argument("--spaml",  default=None, help="Path to SPAML CSV.")
    p.add_argument("--spp",    default=None, help="Path to Semantic Priming Project XLSX.")
    return p.parse_args()


def main():
    args = parse_args()

    data = np.load(args.embedding)
    if "embedding" not in data:
        sys.exit(f"Error: {args.embedding!r} has no 'embedding' key. Keys: {list(data.keys())}")
    embedding = data["embedding"]
    print(f"Embedding shape: {embedding.shape}")

    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    stoi = meta.get("stoi")
    if stoi is None:
        sys.exit(f"Error: {args.meta!r} has no 'stoi' key. Keys: {list(meta.keys())}")
    print(f"Vocabulary size: {len(stoi):,}")

    paths = {}
    for name, val in [("ws353", args.ws353), ("swow", args.swow),
                      ("usf", args.usf), ("spaml", args.spaml), ("spp", args.spp)]:
        if val is not None:
            paths[name] = os.path.expanduser(val)

    results = run_benchmarks(
        embedding, stoi,
        paths=paths,
        benchmarks=args.benchmarks,
        method=args.method,
        cache_dir=args.cache_dir,
    )

    if results.empty:
        print("No benchmarks were evaluated.")
        return

    display = (
        results
        .sort_values("spearman_r", ascending=False)
        [["benchmark", "spearman_r", "benchmark_size", "n_pairs", "task_type", "title"]]
        .copy()
    )
    display["spearman_r"] = display["spearman_r"].map(lambda x: f"{x:.4f}" if x == x else "nan")

    col_widths = {c: max(len(c), display[c].astype(str).str.len().max()) for c in display.columns}
    header = "  ".join(c.ljust(col_widths[c]) for c in display.columns)
    print("\n" + header)
    print("-" * len(header))
    for _, row in display.iterrows():
        print("  ".join(str(row[c]).ljust(col_widths[c]) for c in display.columns))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
