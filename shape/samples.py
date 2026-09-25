"""
Reading stored samples of the Global Semantic Manifold.

`shape.extract.extract_features` stores predictive states in one of two formats:

  {name}_h.npy      (N, d) hidden states; the predictive state is softmax(W h)
  {name}_probs.npy  (N, V) probability vectors (probability-space window averages)

Downstream estimators only need probability vectors, so `iter_sample_probs`
hides the difference: pass `W` and the format is inferred from the number of
columns (d → hidden states, V → probabilities).
"""

import os

import numpy as np
import torch
from tqdm import tqdm

SAMPLE_FORMATS = ('h', 'probs')


def open_samples(samples):
    """Return a 2-D array (memmapped if `samples` is a path)."""
    if isinstance(samples, (str, bytes, os.PathLike)):
        arr = np.load(samples, mmap_mode='r')
    else:
        arr = samples
    if arr.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {arr.shape}")
    return arr


def sample_format(samples, W=None):
    """
    Infer whether `samples` holds hidden states ('h') or probabilities ('probs').

    With `W` (V, d), a d-column array is 'h' and a V-column array is 'probs'.
    Without `W`, samples are assumed to be probabilities.
    """
    arr = open_samples(samples)
    if W is None:
        return 'probs'
    V, d = W.shape
    if arr.shape[1] == d:
        return 'h'
    if arr.shape[1] == V:
        return 'probs'
    raise ValueError(
        f"samples have {arr.shape[1]} columns; expected d={d} (hidden states) "
        f"or V={V} (probabilities) for W of shape {tuple(W.shape)}.")


def iter_sample_probs(samples, W=None, *, batch_size=2048, device='cuda',
                      verbose=True, desc='samples'):
    """
    Yield (B, V) fp32 probability tensors for consecutive row batches.

    Args:
        samples: path to a .npy memmap or an (N, d) / (N, V) ndarray.
        W: (V, d) unembedding matrix (model.lm_head.weight). Required for
            hidden-state samples; ignored for probability samples.
        batch_size: rows per device batch.
        device: torch device string.
        verbose: show tqdm progress bar.
        desc: progress-bar label.
    """
    arr = open_samples(samples)
    fmt = sample_format(arr, W)
    W_t = W.detach().to(device=device, dtype=torch.float32) if fmt == 'h' else None
    N = arr.shape[0]
    n_batches = (N + batch_size - 1) // batch_size
    pbar = tqdm(range(0, N, batch_size), total=n_batches,
                desc=desc, disable=not verbose, unit="batch")
    with torch.no_grad():
        for s in pbar:
            e = min(s + batch_size, N)
            x_np = np.ascontiguousarray(arr[s:e]).astype(np.float32, copy=False)
            x = torch.from_numpy(x_np).to(device=device, non_blocking=True)
            yield torch.softmax(x @ W_t.T, dim=-1) if fmt == 'h' else x


def samples_path(features_dir, dataset_name):
    """
    Locate `{dataset_name}_{h|probs}.npy` under `features_dir`.

    Falls back to the legacy `{dataset_name}_h_eff.npy` name written by earlier
    versions of extract_features (which projected out v_degen by default).
    """
    for fmt in SAMPLE_FORMATS:
        path = os.path.join(features_dir, f"{dataset_name}_{fmt}.npy")
        if os.path.exists(path):
            return path
    legacy = os.path.join(features_dir, f"{dataset_name}_h_eff.npy")
    if os.path.exists(legacy):
        print(f"Note: using legacy sample file {legacy}")
        return legacy
    raise FileNotFoundError(
        f"No samples for '{dataset_name}' in {features_dir} "
        f"(looked for _h.npy, _probs.npy, _h_eff.npy).")
