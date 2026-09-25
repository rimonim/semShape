"""
Stage 2: global density estimation on sampled hidden states h.

Fits a normalizing flow (Zuko NSF) on the `{name}_h.npy` hidden-state samples
written by shape.extract.extract_features (no window or averaging='aitchison';
probability-space samples have no hidden-state representation), providing `log_density(h)` and `sample(n)` in the original h-space.

NSF's spline transformations are defined over [-5, 5], so we standardize
h to zero mean / unit variance before training. The change of variable
is absorbed into `log_density` / `sample` by storing (mean, std) alongside
the flow weights.

Outputs under `out_dir`:
  {name}_flow.pt       — state_dict for NSF + mean/std + config
"""

import json
import os

import numpy as np
import torch
from tqdm import tqdm
from zuko.flows import NSF


def _make_flow(d, transforms, hidden_features, bins):
    return NSF(
        features=d,
        transforms=transforms,
        bins=bins,
        hidden_features=hidden_features,
    )


class FlowDensity:
    """Wraps an NSF trained on standardized h.

    log_density / sample operate on the original (unstandardized) space by
    applying the linear change of variable x_std = (x - mu) / sigma, whose
    log-Jacobian is -Σ log σ_i (a constant).
    """

    def __init__(self, flow, mean, std):
        self.flow = flow
        self.mean = mean                          # (d,) tensor
        self.std = std                            # (d,) tensor
        self._log_sigma_sum = float(torch.log(std).sum().item())

    def to(self, device):
        self.flow = self.flow.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def log_density(self, h):
        """log p(h) in the original h-space. h: (..., d) tensor."""
        h_std = (h - self.mean) / self.std
        return self.flow().log_prob(h_std) - self._log_sigma_sum

    def sample(self, n):
        """Draw n samples in the original h-space."""
        with torch.no_grad():
            z_std = self.flow().sample((n,))
        return z_std * self.std + self.mean

    def save(self, path, extra_meta=None):
        payload = {
            'flow_state_dict': self.flow.state_dict(),
            'mean': self.mean.detach().cpu(),
            'std': self.std.detach().cpu(),
            'd': int(self.mean.shape[0]),
            'config': getattr(self, '_config', {}),
            'meta': extra_meta or {},
        }
        torch.save(payload, path)


def load_flow(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload['config']
    flow = _make_flow(
        d=payload['d'],
        transforms=cfg['transforms'],
        hidden_features=cfg['hidden_features'],
        bins=cfg['bins'],
    )
    flow.load_state_dict(payload['flow_state_dict'])
    flow.eval()
    fd = FlowDensity(flow, payload['mean'].to(device), payload['std'].to(device))
    fd._config = cfg
    return fd, payload.get('meta', {})


def _compute_standardization(h_mm, sample_n=200_000, seed=0):
    """Compute mean/std over a random sample of rows (or all rows if small)."""
    N = h_mm.shape[0]
    rng = np.random.default_rng(seed)
    if N <= sample_n:
        block = np.asarray(h_mm[:])
    else:
        idx = rng.choice(N, size=sample_n, replace=False)
        idx.sort()
        block = np.asarray(h_mm[idx])
    mean = block.mean(axis=0)
    std = block.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)           # guard degenerate dims
    return torch.from_numpy(mean.astype(np.float32)), \
           torch.from_numpy(std.astype(np.float32))


def fit_flow(
    h_path,
    *,
    out_path,
    transforms=8,
    hidden_features=(512, 512, 512),
    bins=8,
    epochs=5,
    batch_size=4096,
    lr=1e-3,
    weight_decay=0.0,
    val_frac=0.02,
    device='cuda',
    seed=1337,
    verbose=True,
    extra_meta=None,
):
    """Train an NSF on a hidden-state sample memmap; save to out_path. Returns FlowDensity."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    h_mm = np.load(h_path, mmap_mode='r')
    assert h_mm.ndim == 2, f"Expected 2D hidden states, got {h_mm.shape}"
    N, d = h_mm.shape
    if verbose:
        print(f"hidden states: {N:,} × {d} from {h_path}")

    mean, std = _compute_standardization(h_mm, seed=seed)
    if verbose:
        print(f"standardization: mean ∈ [{mean.min():.3f}, {mean.max():.3f}], "
              f"std ∈ [{std.min():.3f}, {std.max():.3f}]")

    flow = _make_flow(d, transforms, list(hidden_features), bins).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    if verbose:
        print(f"NSF: features={d}, transforms={transforms}, "
              f"hidden={list(hidden_features)}, bins={bins}, params={n_params:,}")

    mean_d = mean.to(device)
    std_d = std.to(device)

    # Held-out split (index-based; avoids copying data)
    n_val = max(1, int(N * val_frac)) if val_frac > 0 else 0
    perm = rng.permutation(N)
    val_idx = np.sort(perm[:n_val]) if n_val > 0 else None
    train_idx = np.sort(perm[n_val:])

    if val_idx is not None:
        # Keep val data on CPU; chunked through the flow in eval to bound VRAM.
        val_h_cpu = torch.from_numpy(np.asarray(h_mm[val_idx]).astype(np.float32))

    opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_nll': [], 'val_nll': []}
    for epoch in range(epochs):
        rng.shuffle(train_idx)
        n_train = len(train_idx)
        n_batches = (n_train + batch_size - 1) // batch_size
        pbar = tqdm(range(n_batches), disable=not verbose,
                    desc=f"epoch {epoch + 1}/{epochs}", unit="batch")
        running = 0.0
        running_n = 0
        for b in pbar:
            sel = train_idx[b * batch_size:(b + 1) * batch_size]
            # sel need not be sorted for fancy-indexing a memmap, but sorted reads
            # are faster. mmap reads work fine with int arrays.
            batch_np = np.asarray(h_mm[np.sort(sel)]).astype(np.float32)
            batch = torch.from_numpy(batch_np).to(device)
            batch_std = (batch - mean_d) / std_d

            nll = -flow().log_prob(batch_std).mean()
            opt.zero_grad(set_to_none=True)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()

            running += float(nll.item()) * batch.shape[0]
            running_n += batch.shape[0]
            if verbose and (b % 20 == 0):
                pbar.set_postfix({'nll_std': f"{running / max(running_n, 1):.3f}"})

        train_nll = running / max(running_n, 1)
        history['train_nll'].append(train_nll)

        if val_idx is not None:
            flow.eval()
            with torch.no_grad():
                val_sum = 0.0
                val_n = 0
                for vs in range(0, val_h_cpu.shape[0], batch_size):
                    vb = val_h_cpu[vs:vs + batch_size].to(device)
                    vb_std = (vb - mean_d) / std_d
                    val_sum += float(-flow().log_prob(vb_std).sum().item())
                    val_n += vb.shape[0]
                val_nll = val_sum / max(val_n, 1)
            flow.train()
            history['val_nll'].append(val_nll)
            if verbose:
                print(f"  epoch {epoch + 1}: train_nll(std)={train_nll:.3f}  "
                      f"val_nll(std)={val_nll:.3f}")

    flow.eval()
    fd = FlowDensity(flow.cpu(), mean.cpu(), std.cpu())
    fd._config = {
        'transforms': transforms,
        'hidden_features': list(hidden_features),
        'bins': bins,
    }
    meta = {
        'd': d,
        'N': int(N),
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'history': history,
        'h_path': os.path.abspath(h_path),
    }
    if extra_meta:
        meta.update(extra_meta)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fd.save(out_path, extra_meta=meta)
    if verbose:
        print(f"\nSaved flow to {out_path}")
        # log Jacobian correction: log p(h) = log p(h_std) - sum log sigma
        print(f"  log |det diag(σ)| = {fd._log_sigma_sum:.3f} "
              f"(subtracted when returning log_density in h-space)")

    # Also drop a small sidecar json for readability
    sidecar = out_path.replace('.pt', '_meta.json')
    try:
        with open(sidecar, 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    return fd.to('cpu')
