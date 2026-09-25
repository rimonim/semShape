"""
Stage 3 CLI: samples → PMI → [ILR] → SVD → static word embeddings.

Wraps shape.embeddings.compute_embeddings. Loads W from a checkpoint and the
samples ({dataset}_h.npy or {dataset}_probs.npy) + Z from a Stage-1 features
directory; writes the low-dim embedding and the
full PMI matrix to disk.

Example (shakespeare_char, defaults):
    python scripts/compute_embeddings.py \\
        --ckpt out-shakespeare-char/ckpt.pt \\
        --features-dir features/shakespeare_char \\
        --dataset shakespeare_char \\
        --out embeddings/shakespeare_char_emb.npz \\
        --k 32 --neighbors-for "the and a"

Example (COCA test, row-subset over top-5000 frequent tokens):
    python scripts/compute_embeddings.py \\
        --ckpt out-coca/ckpt.pt \\
        --features-dir features/coca \\
        --dataset coca_val \\
        --out embeddings/coca_val_emb.npz \\
        --k 300 --top-tokens 5000 --no-ilr
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model import GPT, GPTConfig
from shape.embeddings import compute_embeddings, nearest_neighbors
from shape.samples import samples_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 3: compute static word embeddings via PMI+ILR+SVD.")

    # Inputs
    p.add_argument("--ckpt", required=True, help="Path to GPT checkpoint .pt.")
    p.add_argument("--features-dir", required=True,
                   help="Directory containing {dataset}_{h|probs}.npy, _Z.npy, _meta.json.")
    p.add_argument("--dataset", required=True,
                   help="Dataset name prefix (matches Stage 1 output).")
    p.add_argument("--out", required=True,
                   help="Output .npz path for PMI + embedding + SVD.")

    # Embedding config
    p.add_argument("--k", type=int, default=300,
                   help="Embedding dimensionality (default: 300).")
    p.add_argument("--eig-weight", type=float, default=0.5,
                   help="SVD singular-value exponent α in U·Σ^α (default: 0.5).")
    p.add_argument("--no-ilr", dest="use_ilr", action="store_false",
                   help="SVD on raw PMI instead of Ψ·PMI (ILR).")
    p.set_defaults(use_ilr=True)
    p.add_argument("--center", action="store_true",
                   help="Subtract per-column mean before SVD.")
    p.add_argument("--eps", type=float, default=1e-30,
                   help="Floor for log(M) and log(Z) (default: 1e-30).")

    # Row subsetting for large V
    p.add_argument("--top-tokens", type=int, default=0,
                   help="Restrict PMI rows to top-N most frequent tokens (0 = all V). "
                        "Uses the Stage 1 Z as a proxy for frequency. "
                        "Forces --no-ilr because Ψ needs full-V rows.")
    p.add_argument("--target-tokens-file", default=None,
                   help="Path to .npy with explicit token-id vector for row subsetting.")

    # Compute
    p.add_argument("--batch-size", type=int, default=2048,
                   help="Sample rows per device batch (default: 2048).")
    p.add_argument("--device", default=None,
                   help="Device string (default: cuda if available, else cpu).")

    # Z source
    p.add_argument("--z-from", choices=["stage1", "empirical"], default="stage1",
                   help="'stage1' uses the saved {dataset}_Z.npy. 'empirical' recomputes "
                        "Z from the samples (identical for current extract_features "
                        "outputs; differs for legacy windowed runs).")

    # Sanity
    p.add_argument("--neighbors-for", default="",
                   help="Space-separated tokens to print nearest neighbors for (shakespeare_char "
                        "vocab only). Requires {dataset}/meta.pkl in data/.")
    p.add_argument("--neighbors-topk", type=int, default=10)

    return p.parse_args()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval().to(device)
    return model, ckpt


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.startswith("cuda"):
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    # Paths
    fdir = args.features_dir
    ds = args.dataset
    sample_file = samples_path(fdir, ds)
    Z_path = os.path.join(fdir, f"{ds}_Z.npy")
    meta_path = os.path.join(fdir, f"{ds}_meta.json")
    for pth in (Z_path, meta_path):
        if not os.path.exists(pth):
            raise FileNotFoundError(pth)

    with open(meta_path) as f:
        stage1_meta = json.load(f)
    print(f"Stage 1 meta: N_valid={stage1_meta['N_valid']:,}, V={stage1_meta['V']}, "
          f"d={stage1_meta['d']}, window={stage1_meta['window']}, "
          f"averaging={stage1_meta.get('averaging', 'aitchison')}, "
          f"project_degenerate={stage1_meta.get('project_degenerate', False)}")

    # Z selection
    Z_stage1 = np.load(Z_path)
    if args.z_from == "stage1":
        Z = Z_stage1
        print(f"Using Z from Stage 1. Σ_w Z_w = {Z.sum():.6f}")
    else:
        Z = None
        print("Using Z_emp recomputed from the samples.")

    # Target tokens (row subset)
    target_tokens = None
    if args.target_tokens_file:
        target_tokens = np.load(args.target_tokens_file)
        print(f"Row-subsetting to {len(target_tokens)} tokens from file.")
    elif args.top_tokens > 0:
        order = np.argsort(-Z_stage1)
        target_tokens = order[:args.top_tokens]
        print(f"Row-subsetting to top-{args.top_tokens} tokens by Z.")

    if target_tokens is not None and args.use_ilr:
        print("Note: --top-tokens / --target-tokens-file forces --no-ilr "
              "(Ψ needs full-V rows).")
        args.use_ilr = False

    # Model → W
    model, ckpt = load_model(args.ckpt, device)
    W = model.lm_head.weight.detach().to(device).float()
    V, d = W.shape
    print(f"W: V={V}, d={d}")
    if (V, d) != (stage1_meta['V'], stage1_meta['d']):
        raise ValueError(
            f"Model V/d ({V},{d}) does not match Stage 1 meta "
            f"({stage1_meta['V']},{stage1_meta['d']}).")

    # Compute
    result = compute_embeddings(
        sample_file, W,
        Z=Z,
        k=args.k,
        eig_weight=args.eig_weight,
        use_ilr=args.use_ilr,
        center=args.center,
        target_tokens=target_tokens,
        batch_size=args.batch_size,
        device=device,
        eps=args.eps,
        verbose=True,
    )

    print(f"\nPMI shape: {result['pmi'].shape}")
    print(f"Embedding shape: {result['embedding'].shape}")
    print(f"Explained variance (first {args.k} SVs): "
          f"{result['explained_variance_ratio']:.4f}")
    print(f"Σ_w Z_w (empirical) = {result['Z_emp'].sum():.6f}")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_kwargs = {
        'embedding': result['embedding'],
        'PMI': result['pmi'].astype(np.float32),
        'Z': result['Z'].astype(np.float32),
        'Z_emp': result['Z_emp'].astype(np.float32),
        'S': result['S'].astype(np.float32),
    }
    if result['meta']['use_ilr']:
        save_kwargs['ilr_emb'] = result['ilr_emb'].astype(np.float32)
    if target_tokens is not None:
        save_kwargs['target_tokens'] = np.asarray(target_tokens, dtype=np.int64)
    meta = dict(result['meta'])
    meta['stage1_meta'] = stage1_meta
    meta['z_from'] = args.z_from
    meta['ckpt'] = args.ckpt
    save_kwargs['meta_json'] = np.array(json.dumps(meta))
    np.savez(args.out, **save_kwargs)
    print(f"Saved → {args.out}")

    # Optional nearest-neighbors sanity check (shakespeare_char style)
    if args.neighbors_for.strip():
        data_dir = os.path.join(PROJECT_ROOT, 'data', stage1_meta.get('dataset_dir', ds))
        meta_pkl = os.path.join(data_dir, 'meta.pkl')
        stoi, itos = None, None
        if os.path.exists(meta_pkl):
            with open(meta_pkl, 'rb') as f:
                pm = pickle.load(f)
            stoi = pm.get('stoi')
            itos = pm.get('itos')
        if stoi is None:
            print(f"Cannot load stoi/itos from {meta_pkl}; skipping neighbor print.")
        else:
            queries = args.neighbors_for.split()
            qids = []
            for q in queries:
                if q in stoi:
                    qids.append(stoi[q])
                else:
                    print(f"  token {q!r} not in vocab; skipping")
            if qids:
                print(f"\nNearest neighbors (cosine, top {args.neighbors_topk}):")
                emb_for_nn = result['embedding']
                if target_tokens is not None:
                    # Neighbor lookup on row-subset: translate ids
                    id_to_row = {int(t): i for i, t in enumerate(target_tokens)}
                    qids = [id_to_row[q] for q in qids if q in id_to_row]
                    itos_local = {i: itos[int(t)] for t, i in id_to_row.items()}
                    nearest_neighbors(emb_for_nn, qids,
                                      top_k=args.neighbors_topk, itos=itos_local)
                else:
                    nearest_neighbors(emb_for_nn, qids,
                                      top_k=args.neighbors_topk, itos=itos)


if __name__ == "__main__":
    main()
