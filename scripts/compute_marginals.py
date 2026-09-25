"""
Compute per-token marginal probability sums Z[t] = Σ_i p(t|Y_i) and save as .npy.

Z is proportional to the corpus marginal P(t) and is the denominator used by
the importance-weighted similarity estimator.  Use this to retroactively apply
the base-rate correction to previously computed kl_divergence outputs:

    corrected_kl(t1, t2) = raw_kl(t1, t2) + log(Z[t2] / Z[t1])

Samples are stored samples from extract_features: {name}_h.npy hidden states
(requires --ckpt) or {name}_probs.npy probability vectors. For new samples the
same quantity, normalized by N, is already written as {name}_Z.npy.

Examples:

    python scripts/compute_marginals.py \\
        --samples features/coca_gsm/coca_val_short_forward_probs.npy \\
        --output features/coca_gsm/marginals_short_forward.npy

    python scripts/compute_marginals.py \\
        --samples features/coca_gsm/coca_val_long_backward_probs.npy \\
        --output features/coca_gsm/marginals_long_backward.npy

    python scripts/compute_marginals.py \\
        --samples features/coca_gsm/coca_val_no_window_h.npy \\
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

from compute_similarity import check_probs, load_W  # noqa: E402
from shape.similarity import compute_token_marginals


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute and save per-token marginal probability sums.")

    p.add_argument("--samples", required=True,
                   help="Path to a *_h.npy or *_probs.npy sample file.")

    p.add_argument("--output", required=True,
                   help="Output path for the (V,) float64 marginals .npy.")
    p.add_argument("--ckpt", default=None,
                   help="Path to GPT checkpoint (required for hidden-state samples).")
    p.add_argument("--device", default=None,
                   help="Device string (default: cuda if available, else cpu).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Sample rows per device batch (default: 2048).")

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"Samples: {args.samples}")

    if args.ckpt is not None:
        W = load_W(args.ckpt, device)
        print(f"Checkpoint: V={W.shape[0]}, d={W.shape[1]}")
    else:
        W = None
        check_probs(args.samples)
    Z = compute_token_marginals(
        args.samples,
        W=W,
        batch_size=args.batch_size or 2048,
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
