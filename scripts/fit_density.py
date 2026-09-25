"""
Stage 2 CLI: fit a normalizing flow on sampled hidden states → flow .pt file.

Wraps shape.density.fit_flow. Takes the {dataset}_h.npy memmap written by
Stage 1 (no window or --averaging aitchison) and
trains a Neural Spline Flow (Zuko NSF), saving the result to models/.

Example (COCA test, defaults):
    python scripts/fit_density.py \\
        --h features/coca/coca_test_h.npy \\
        --out models/coca/coca_test_flow.pt

Example (higher capacity, more epochs):
    python scripts/fit_density.py \\
        --h features/coca/coca_test_w5_harm_h.npy \\
        --out models/coca/coca_test_w5_harm_flow.pt \\
        --transforms 10 --hidden 512 512 512 --bins 10 --epochs 10
"""

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from shape.density import fit_flow


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: fit NSF density on sampled hidden states.")

    # Required
    p.add_argument("--h", required=True,
                   help="Path to a hidden-state .npy memmap (N, d) from Stage 1.")
    p.add_argument("--out", required=True,
                   help="Output path for the flow .pt file.")

    # Flow architecture
    p.add_argument("--transforms", type=int, default=6,
                   help="Number of NSF coupling transforms (default: 6).")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512],
                   help="Hidden layer widths for the coupling nets (default: 512 512).")
    p.add_argument("--bins", type=int, default=8,
                   help="Spline bins per dimension (default: 8).")

    # Training
    p.add_argument("--epochs", type=int, default=5,
                   help="Training epochs (default: 5).")
    p.add_argument("--batch-size", type=int, default=4096,
                   help="SGD mini-batch size (default: 4096).")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate (default: 1e-3).")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Adam weight decay (default: 0.0).")
    p.add_argument("--val-frac", type=float, default=0.02,
                   help="Fraction of data held out for validation NLL (default: 0.02).")

    # Compute
    p.add_argument("--device", default=None,
                   help="Device string (default: cuda if available, else cpu).")
    p.add_argument("--seed", type=int, default=1337)

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.startswith("cuda"):
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fit_flow(
        args.h,
        out_path=args.out,
        transforms=args.transforms,
        hidden_features=tuple(args.hidden),
        bins=args.bins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
