"""
Compute per-token marginal probability sums Z[t] = Σ_i p(t|Y_i) and save as .npy.

Z is proportional to the corpus marginal P(t) and is the denominator used by
the importance-weighted similarity estimator.  Use this to retroactively apply
the base-rate correction to previously computed kl_divergence outputs:

    corrected_kl(t1, t2) = raw_kl(t1, t2) + log(Z[t2] / Z[t1])

Two modes (mutually exclusive):

  --probs   Precomputed (N, V) probability memmap from sample_gsm.py.
  --h-eff   Precomputed h_eff hidden states (requires --ckpt).

Examples:

    python scripts/compute_marginals.py \\
        --probs features/coca_gsm/coca_val_short_forward_probs.npy \\
        --output features/coca_gsm/marginals_short_forward.npy

    python scripts/compute_marginals.py \\
        --probs features/coca_gsm/coca_val_long_backward_probs.npy \\
        --output features/coca_gsm/marginals_long_backward.npy

    python scripts/compute_marginals.py \\
        --h-eff features/coca_gsm/coca_val_no_window_h_eff.npy \\
        --ckpt out-coca/ckpt.pt \\
        --output features/coca_gsm/marginals_no_window.npy
"""

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from shape.similarity import (
    compute_token_marginals_from_h_eff,
    compute_token_marginals_from_probs,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute and save per-token marginal probability sums.")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probs",
                      help="Path to (N, V) *_probs.npy from sample_gsm.py.")
    mode.add_argument("--h-eff",
                      help="Path to h_eff .npy file (requires --ckpt).")

    p.add_argument("--output", required=True,
                   help="Output path for the (V,) float64 marginals .npy.")
    p.add_argument("--ckpt", default=None,
                   help="Path to GPT checkpoint (required for --h-eff).")
    p.add_argument("--device", default=None,
                   help="Device string (default: cuda if available, else cpu).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size. Defaults: 4096 (--probs), 2048 (--h-eff).")

    return p, p.parse_args()


def load_model(ckpt_path, device):
    from model import GPT, GPTConfig
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def main():
    p, args = parse_args()

    if args.h_eff is not None and args.ckpt is None:
        p.error("--ckpt is required for --h-eff mode.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.probs is not None:
        batch_size = args.batch_size or 4096
        print(f"Mode: probs  |  {args.probs}")
        Z = compute_token_marginals_from_probs(
            args.probs,
            batch_size=batch_size,
            device=device,
            verbose=True,
        )
    else:
        batch_size = args.batch_size or 2048
        print(f"Mode: h-eff  |  {args.h_eff}")
        model = load_model(args.ckpt, device)
        W = model.lm_head.weight.detach().to(device).float()
        print(f"Model: V={model.config.vocab_size}, d={model.config.n_embd}")
        Z = compute_token_marginals_from_h_eff(
            args.h_eff, W,
            batch_size=batch_size,
            device=device,
            verbose=True,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, Z)
    print(f"\nSaved → {args.output}  (shape={Z.shape}, dtype={Z.dtype})")
    print(f"  sum={Z.sum():.6g}, min={Z.min():.6g}, max={Z.max():.6g}")
    print(f"  non-zero tokens: {(Z > 0).sum():,} / {len(Z):,}")


if __name__ == "__main__":
    main()
