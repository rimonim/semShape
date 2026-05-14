"""
Compute pairwise semantic similarity quantities for token pairs from a CSV.

Reads a CSV with word pairs, maps words to token ids via a vocabulary file,
computes the requested similarity quantities, and writes the results as
additional columns in the output CSV.  Both plain .csv and gzip-compressed
.csv.gz files are accepted as input and output (inferred from filename).

Three modes (mutually exclusive):
  --probs   Precomputed (N, V) probability memmap from sample_gsm.py.
            No model needed; fastest for repeated analyses.
  --h-eff   Precomputed h_eff hidden states (no windowing).
  --data    Stream the corpus with probability-space window averaging.

Examples:

  # From precomputed GSM samples (fastest; run sample_gsm.py first)
  python scripts/compute_similarity.py \\
      --input pairs.csv --output pairs_sim.csv \\
      --vocab data/coca/meta.pkl \\
      --probs features/coca_gsm/coca_val_no_window_probs.npy

  # From h_eff (non-windowed, no corpus re-run)
  python scripts/compute_similarity.py \\
      --input pairs.csv --output pairs_sim.csv \\
      --ckpt out-coca/ckpt.pt --vocab data/coca/meta.pkl \\
      --h-eff features/coca/coca_val_h_eff.npy

  # Streaming windowed corpus pass
  python scripts/compute_similarity.py \\
      --input pairs.csv --output pairs_sim.csv \\
      --ckpt out-coca/ckpt.pt --vocab data/coca/meta.pkl \\
      --data data/coca/val.bin \\
      --window-size 100 --decay-type power --alpha 0.5 \\
      --direction backward --tokens-per-minute 150
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model import GPT, GPTConfig
from shape.similarity import (
    compute_pairwise_similarities,
    compute_pairwise_similarities_from_probs,
    compute_pairwise_similarities_prob_window,
)
from shape.windowing import build_weight_lookup


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute pairwise semantic similarity quantities from a CSV of word pairs.")

    # Input / output
    p.add_argument("--input", required=True,
                   help="Input CSV path with word pairs.")
    p.add_argument("--output", required=True,
                   help="Output CSV path (input rows + new quantity columns).")
    p.add_argument("--word1-col", default="word1",
                   help="Column name for the query word (default: 'word1').")
    p.add_argument("--word2-col", default="word2",
                   help="Column name for the target word (default: 'word2').")

    # Vocab (always required)
    p.add_argument("--vocab", required=True,
                   help="Path to meta.pkl containing 'stoi' and 'itos' dicts.")

    # Model checkpoint (required for --h-eff and --data, not for --probs)
    p.add_argument("--ckpt", default=None,
                   help="Path to GPT checkpoint .pt (required for --h-eff and --data).")

    # Mode (mutually exclusive)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probs",
                      help="Path to *_probs.npy from sample_gsm.py. No model needed.")
    mode.add_argument("--h-eff",
                      help="Path to h_eff .npy file (non-windowed, requires --ckpt).")
    mode.add_argument("--data",
                      help="Path to binary corpus .bin for streaming windowed pass "
                           "(requires --ckpt).")

    # Quantities
    p.add_argument("--quantities", nargs="+",
                   default=["expected_probability", "expected_surprisal", "kl_divergence"],
                   choices=["expected_probability", "expected_surprisal", "kl_divergence"],
                   help="Similarity quantities to compute (default: all three).")

    # Windowing (only for --data)
    p.add_argument("--window-size", type=int, default=0,
                   help="Half-window radius in tokens (default: 0).")
    p.add_argument("--decay-type", default="none",
                   choices=["linear", "harmonic", "exponential", "power", "none"],
                   help="Window weight decay function (default: none).")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Decay alpha for exponential/power (default: 1.0).")
    p.add_argument("--direction", default="symmetric",
                   choices=["symmetric", "forward", "backward"],
                   help="Window direction (default: symmetric).")
    p.add_argument("--no-include-target", dest="include_target", action="store_false",
                   help="Exclude d=0 from the window average.")
    p.set_defaults(include_target=True)
    p.add_argument("--tokens-per-minute", type=float, default=None,
                   help="Convert token distances to minutes before power decay.")

    # Compute
    p.add_argument("--device", default=None,
                   help="Device string (default: cuda if available, else cpu).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size. Defaults: 4096 (--probs), 2048 (--h-eff), 32 (--data).")
    p.add_argument("--block-size", type=int, default=None,
                   help="Override model block_size (--data only).")
    p.add_argument("--min-context", type=int, default=32,
                   help="Minimum left-context tokens (--data only, default 32).")
    p.add_argument("--compute-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Autocast dtype for --data mode (default: bfloat16).")

    return p, p.parse_args()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        meta = pickle.load(f)
    stoi = meta.get("stoi")
    if stoi is None:
        raise ValueError(f"meta.pkl at {vocab_path!r} has no 'stoi' key.")
    return stoi


def words_to_ids(words, stoi):
    ids = np.full(len(words), -1, dtype=np.int64)
    oov_mask = np.ones(len(words), dtype=bool)
    for i, w in enumerate(words):
        if w in stoi:
            ids[i] = stoi[w]
            oov_mask[i] = False
    return ids, oov_mask


def main():
    p, args = parse_args()

    if args.probs is None and args.ckpt is None:
        p.error("--ckpt is required for --h-eff and --data modes.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df = pd.read_csv(args.input)
    for col in (args.word1_col, args.word2_col):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {args.input!r}. "
                             f"Available: {list(df.columns)}")
    print(f"Loaded {len(df):,} rows from {args.input!r}.")

    stoi = load_vocab(args.vocab)
    words1 = df[args.word1_col].astype(str).tolist()
    words2 = df[args.word2_col].astype(str).tolist()
    t1_ids, oov1 = words_to_ids(words1, stoi)
    t2_ids, oov2 = words_to_ids(words2, stoi)

    oov_mask = oov1 | oov2
    n_oov = oov_mask.sum()
    if n_oov > 0:
        oov_words = sorted({w for w, m in zip(words1 + words2, list(oov1) + list(oov2)) if m})
        print(f"Warning: {n_oov} rows have OOV words (will be NaN in output).")
        print(f"  OOV tokens: {oov_words}")

    valid_mask = ~oov_mask
    n_valid = valid_mask.sum()
    print(f"Valid pairs: {n_valid:,} / {len(df):,}")
    if n_valid == 0:
        print("No valid pairs — writing output with all NaN.")
        for qty in args.quantities:
            df[qty] = np.nan
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved → {args.output}")
        return

    t1_valid = t1_ids[valid_mask]
    t2_valid = t2_ids[valid_mask]

    pairs_valid = np.stack([t1_valid, t2_valid], axis=1)
    unique_pairs, inverse_indices = np.unique(pairs_valid, axis=0, return_inverse=True)
    t1_unique = unique_pairs[:, 0]
    t2_unique = unique_pairs[:, 1]
    n_unique = len(unique_pairs)
    if n_unique < n_valid:
        print(f"Deduped: {n_valid:,} valid pairs → {n_unique:,} unique pairs")

    if args.probs is not None:
        batch_size = args.batch_size or 4096
        print(f"Mode: probs  |  {args.probs}")
        result = compute_pairwise_similarities_from_probs(
            args.probs, t1_unique, t2_unique,
            quantities=args.quantities,
            batch_size=batch_size,
            device=device,
            verbose=True,
        )

    elif args.h_eff is not None:
        batch_size = args.batch_size or 2048
        model = load_model(args.ckpt, device)
        W = model.lm_head.weight.detach().to(device).float()
        print(f"Model: V={model.config.vocab_size}, d={model.config.n_embd}")
        result = compute_pairwise_similarities(
            args.h_eff, W, t1_unique, t2_unique,
            quantities=args.quantities,
            batch_size=batch_size,
            device=device,
            verbose=True,
        )

    else:
        batch_size = args.batch_size or 32
        if args.batch_size is None:
            print("Note: --data mode uses batch_size=32 (set --batch-size to override).")
        model = load_model(args.ckpt, device)
        print(f"Model: V={model.config.vocab_size}, d={model.config.n_embd}")

        weights_lookup = (
            build_weight_lookup(
                window_size=args.window_size,
                decay_type=args.decay_type,
                alpha=args.alpha,
                direction=args.direction,
                include_target=args.include_target,
                tokens_per_minute=args.tokens_per_minute,
            )
            if args.window_size > 0
            else {0: 1.0}
        )

        data = np.memmap(args.data, dtype=np.uint16, mode="r")
        print(f"Corpus: {len(data):,} tokens")
        print(f"Window: size={args.window_size}, decay={args.decay_type}, "
              f"alpha={args.alpha}, direction={args.direction}, "
              f"include_target={args.include_target}, "
              f"tokens_per_minute={args.tokens_per_minute}")

        result = compute_pairwise_similarities_prob_window(
            model, data, weights_lookup,
            t1_unique, t2_unique,
            quantities=args.quantities,
            block_size=args.block_size,
            min_context=args.min_context,
            batch_size=batch_size,
            device=device,
            compute_dtype=args.compute_dtype,
            verbose=True,
        )

    for qty in args.quantities:
        col = np.full(len(df), np.nan)
        col[valid_mask] = result[qty][inverse_indices]
        df[qty] = col

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")
    print(f"Columns added: {args.quantities}")
    for qty in args.quantities:
        vals = result[qty]
        print(f"  {qty}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")


if __name__ == "__main__":
    main()
