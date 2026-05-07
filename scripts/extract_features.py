"""
Stage 1 CLI: corpus pass → h_eff + Z.

Wraps shape.extract.extract_features. Designed for fast iteration over window
size and shape; all windowing knobs are first-class arguments.

Example (single-position, COCA test split):
    python scripts/extract_features.py \\
        --ckpt out-coca/ckpt.pt \\
        --data data/coca/val.bin \\
        --out-dir features/coca \\
        --dataset coca_val

Example (symmetric window-5, harmonic decay):
    python scripts/extract_features.py \\
        --ckpt out-coca/ckpt.pt \\
        --data data/coca/test.bin \\
        --out-dir features/coca \\
        --dataset coca_test_w5_harm \\
        --window 5 --weights harmonic --direction symmetric
"""

import argparse
import gc
import os
import pickle
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model import GPT, GPTConfig
from shape.extract import extract_features


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: extract h_eff and Z from a trained GPT.")

    # Required
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file.")
    p.add_argument("--data", required=True, help="Path to binary token corpus (.bin, uint16).")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--dataset", required=True,
                   help="Dataset name prefix used in output filenames.")

    # Windowing (the main iteration axis)
    p.add_argument("--window", type=int, default=0,
                   help="Half-window size for Variant-H averaging. 0 = single position.")
    p.add_argument("--weights", default="linear",
                   choices=["linear", "harmonic", "exponential", "power", "none"],
                   help="Decay function for window weights (default: linear).")
    p.add_argument("--weights-alpha", type=float, default=1.0,
                   help="Alpha parameter for exponential/power decay (default: 1.0).")
    p.add_argument("--direction", default="symmetric",
                   choices=["symmetric", "forward", "backward"],
                   help="Which side(s) of the window to include (default: symmetric).")
    p.add_argument("--no-include-target", dest="include_target", action="store_false",
                   help="Exclude the target position (d=0) from the window average.")
    p.set_defaults(include_target=True)

    # Geometry
    p.add_argument("--no-project-degenerate", dest="project_degenerate",
                   action="store_false",
                   help="Keep the degenerate direction (skip projection onto complement).")
    p.set_defaults(project_degenerate=True)
    p.add_argument("--min-context", type=int, default=32,
                   help="Minimum left-context tokens required per position (default: 32).")

    # Storage
    p.add_argument("--no-save-h-eff", dest="save_h_eff", action="store_false",
                   help="Skip writing the full (N, d) h_eff memmap (use with --save-subsample).")
    p.set_defaults(save_h_eff=True)
    p.add_argument("--save-subsample", type=int, default=0,
                   help="Reservoir-sample this many h_eff rows to disk (0 = skip).")

    # Compute
    p.add_argument("--batch-size", type=int, default=128,
                   help="Sequences per forward pass (default: 128).")
    p.add_argument("--block-size", type=int, default=None,
                   help="Override block_size from checkpoint (must be ≤ model's block_size).")
    p.add_argument("--device", default=None,
                   help="Device string (default: cuda if available, else cpu).")
    p.add_argument("--compute-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Autocast dtype for the forward pass (default: bfloat16).")
    p.add_argument("--seed", type=int, default=1337)

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.startswith("cuda"):
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"Model: V={cfg.vocab_size}, d={cfg.n_embd}, L={cfg.block_size}, "
          f"n_layer={cfg.n_layer}")

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    print(f"Corpus: {len(data):,} tokens  ({len(data) * 2 / 1e6:.1f} MB on disk)")

    os.makedirs(args.out_dir, exist_ok=True)
    result = extract_features(
        model, data,
        out_dir=args.out_dir,
        dataset_name=args.dataset,
        block_size=args.block_size,
        window=args.window,
        weights=args.weights,
        weights_alpha=args.weights_alpha,
        direction=args.direction,
        include_target=args.include_target,
        min_context=args.min_context,
        project_degenerate=args.project_degenerate,
        batch_size=args.batch_size,
        device=device,
        compute_dtype=args.compute_dtype,
        save_h_eff=args.save_h_eff,
        save_subsample=args.save_subsample,
        seed=args.seed,
        checkpoint_path=args.ckpt,
    )

    Z = result["Z"]
    print(f"\nN_valid = {result['n_valid']:,}, Σ_w Z_w = {Z.sum():.6f}")

    model.to("cpu")
    del model, ckpt
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"GPU free after cleanup: {free / 1e9:.2f} / {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
