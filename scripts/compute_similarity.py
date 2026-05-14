"""
Compute pairwise semantic similarity quantities for token pairs from a CSV.

Reads a CSV with word pairs, maps words to token ids via a vocabulary file,
computes the requested similarity quantities, and writes the results as
additional columns in the output CSV.

Two modes (mutually exclusive):
  --h-eff   Use precomputed h_eff hidden states (no windowing, faster).
  --data    Stream the corpus with probability-space window averaging.

Example (h_eff, no window):
    python scripts/compute_similarity.py \\
        --input pairs.csv \\
        --output pairs_with_sim.csv \\
        --ckpt out-coca/ckpt.pt \\
        --vocab data/coca/meta.pkl \\
        --h-eff features/coca/coca_val_h_eff.npy \\
        --quantities expected_probability expected_surprisal kl_divergence

Example (prob-space backward window):
    python scripts/compute_similarity.py \\
        --input pairs.csv \\
        --output pairs_sim_bwd.csv \\
        --ckpt out-coca/ckpt.pt \\
        --vocab data/coca/meta.pkl \\
        --data data/coca/val.bin \\
        --window-size 100 --decay-type power --alpha 0.5 \\
        --direction backward --no-include-target --tokens-per-minute 150
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

    # Model / vocab
    p.add_argument("--ckpt", required=True,
                   help="Path to GPT checkpoint .pt.")
    p.add_argument("--vocab", required=True,
                   help="Path to meta.pkl containing 'stoi' and 'itos' dicts.")

    # Mode: h_eff vs. streaming corpus (mutually exclusive)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--h-eff",
                      help="Path to h_eff .npy file (non-windowed mode).")
    mode.add_argument("--data",
                      help="Path to binary token corpus .bin (windowed mode).")

    # Quantities
    p.add_argument("--quantities", nargs="+",
                   default=["expected_probability", "expected_surprisal", "kl_divergence"],
                   choices=["expected_probability", "expected_surprisal", "kl_divergence"],
                   help="Similarity quantities to compute (default: all three).")

    # Windowing (only relevant with --data)
    p.add_argument("--window-size", type=int, default=0,
                   help="Half-window radius in tokens (default: 0 = single position).")
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
                   help="Convert token distances to minutes before power decay "
                        "(d_minutes = d_tokens / tokens_per_minute).")

    # Compute
    p.add_argument("--device", default=None,
                   help="Device string (default: cuda if available, else cpu).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size: h_eff rows (non-windowed) or sequences (windowed). "
                        "Defaults: 2048 for h_eff mode, 32 for windowed mode.")
    p.add_argument("--block-size", type=int, default=None,
                   help="Override model block_size (windowed mode only).")
    p.add_argument("--min-context", type=int, default=32,
                   help="Minimum left-context tokens (windowed mode, default 32).")
    p.add_argument("--compute-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Autocast dtype for windowed mode (default: bfloat16).")

    return p.parse_args()


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
    """Map words to token ids. Returns (ids, oov_mask). OOV entries get id -1."""
    ids = np.full(len(words), -1, dtype=np.int64)
    oov_mask = np.ones(len(words), dtype=bool)
    for i, w in enumerate(words):
        if w in stoi:
            ids[i] = stoi[w]
            oov_mask[i] = False
    return ids, oov_mask


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load input CSV
    df = pd.read_csv(args.input)
    for col in (args.word1_col, args.word2_col):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {args.input!r}. "
                             f"Available: {list(df.columns)}")
    print(f"Loaded {len(df):,} rows from {args.input!r}.")

    # Load vocab and map words to ids
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

    # Load model
    model = load_model(args.ckpt, device)
    V = model.config.vocab_size
    print(f"Model: V={V}, d={model.config.n_embd}, block_size={model.config.block_size}")

    # Compute similarities
    if args.h_eff is not None:
        # Non-windowed mode: use precomputed h_eff
        batch_size = args.batch_size or 2048
        W = model.lm_head.weight.detach().to(device).float()
        result = compute_pairwise_similarities(
            args.h_eff, W, t1_valid, t2_valid,
            quantities=args.quantities,
            batch_size=batch_size,
            device=device,
            verbose=True,
        )
    else:
        # Windowed mode: streaming corpus pass
        batch_size = args.batch_size or 32
        if args.batch_size is None:
            print("Note: windowed mode uses batch_size=32 (set --batch-size to override).")

        if args.window_size > 0:
            weights_lookup = build_weight_lookup(
                window_size=args.window_size,
                decay_type=args.decay_type,
                alpha=args.alpha,
                direction=args.direction,
                include_target=args.include_target,
                tokens_per_minute=args.tokens_per_minute,
            )
        else:
            weights_lookup = {0: 1.0}

        data = np.memmap(args.data, dtype=np.uint16, mode="r")
        print(f"Corpus: {len(data):,} tokens")
        print(f"Window: size={args.window_size}, decay={args.decay_type}, "
              f"alpha={args.alpha}, direction={args.direction}, "
              f"include_target={args.include_target}, "
              f"tokens_per_minute={args.tokens_per_minute}")

        result = compute_pairwise_similarities_prob_window(
            model, data, weights_lookup,
            t1_valid, t2_valid,
            quantities=args.quantities,
            block_size=args.block_size,
            min_context=args.min_context,
            batch_size=batch_size,
            device=device,
            compute_dtype=args.compute_dtype,
            verbose=True,
        )

    # Assemble output
    for qty in args.quantities:
        col = np.full(len(df), np.nan)
        col[valid_mask] = result[qty]
        df[qty] = col

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")
    print(f"Columns added: {args.quantities}")
    for qty in args.quantities:
        vals = result[qty]
        print(f"  {qty}: min={vals.min():.4f}, max={vals.max():.4f}, "
              f"mean={vals.mean():.4f}")


if __name__ == "__main__":
    main()
