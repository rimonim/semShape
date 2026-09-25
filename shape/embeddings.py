"""
Stage 3 — expected-value layer: PMI matrix, ILR embeddings, SVD.

Computes static word embeddings from a trained GPT via the continuous pipeline:

    samples  →  moment matrix M  →  PMI  →  [ILR]  →  SVD  →  low-dim embedding

These are the continuous analogue of the discrete FCM+PMI+SVD pipeline in
build_fcm.py. At window=0 they should be numerically comparable up to the
predicted-token-vs-observed-token distinction (which agrees in the
well-trained limit).

The key quantity is the moment matrix

    M[t, w] = E_g[ p_t(h) · p_w(h) ]   ≈   (1/N) Σ_i p_t(h_i) · p_w(h_i)

from which the conditional and global expectations follow:

    μ_t^X_w = E_{g(·|t)}[softmax(Wh)_w] = M[t, w] / Z[t]
    μ^X_w   = Z[w]                                            (from Stage 1)

giving the symmetric PMI matrix

    PMI[t, w] = log μ_t^X_w - log Z[w]
               = log M[t, w] - log Z[t] - log Z[w]

The Monte-Carlo estimator uses the empirical corpus sample written by
shape.extract.extract_features — hidden states (pass W) or probability vectors
(probability-space window averages). This is the natural unbiased estimator and matches
build_fcm.py's implicit empirical-distribution sampling; we can also pass in
samples drawn from the Stage-2 flow if desired.

Scales: M is V×V, so the full moment computation is O(V²) memory on-device
and O(N · V²) flops for the accumulation. For small vocabularies this is
trivial; for COCA (V ≈ 150k) use `target_tokens=` to compute only a row-subset.
"""

import numpy as np
import torch
from tqdm import tqdm

from shape.geometry import compute_A, ilr_apply
from shape.samples import iter_sample_probs, open_samples, sample_format


def _open_h(h, W):
    """Open hidden-state samples; return (array, N, d). Rejects probability samples."""
    arr = open_samples(h)
    if sample_format(arr, W) != 'h':
        raise ValueError(
            f"This function needs hidden-state samples (N, d={W.shape[1]}), got "
            f"shape {arr.shape}. Probability-space window averages have no "
            f"hidden-state representation; use averaging='aitchison' or no window.")
    return arr, arr.shape[0], arr.shape[1]


def _moment_from_batches(prob_batches, V, target_tokens, device):
    """Accumulate M and Z_emp over an iterable of (B, V) probability batches."""
    if target_tokens is not None:
        target_tokens = np.asarray(target_tokens, dtype=np.int64)
        assert target_tokens.ndim == 1
        tgt_idx = torch.from_numpy(target_tokens).to(device)
        M = torch.zeros((target_tokens.shape[0], V), dtype=torch.float64, device=device)
    else:
        tgt_idx = None
        M = torch.zeros((V, V), dtype=torch.float64, device=device)
    Z_emp = torch.zeros(V, dtype=torch.float64, device=device)

    N = 0
    for p in prob_batches:                                                    # (B, V) fp32
        if tgt_idx is not None:
            p_sub = p.index_select(1, tgt_idx)                                # (B, |T|)
            M += (p_sub.T @ p).to(torch.float64)
        else:
            M += (p.T @ p).to(torch.float64)
        Z_emp += p.sum(dim=0).to(torch.float64)
        N += p.shape[0]

    M /= float(N)
    Z_emp /= float(N)
    return {
        'M': M.cpu().numpy(),
        'Z_emp': Z_emp.cpu().numpy(),
        'target_tokens': target_tokens,
        'N': int(N),
    }


def compute_moment_matrix(
    samples,
    W=None,
    *,
    target_tokens=None,
    batch_size=2048,
    device='cuda',
    verbose=True,
):
    """
    Accumulate the moment matrix and empirical Z over stored corpus samples.

        M[t, w]  = (1/N) Σ_i p_t(Y_i) · p_w(Y_i)
        Z_emp[w] = (1/N) Σ_i p_w(Y_i)

    where p(Y_i) = softmax(W h_i) for hidden-state samples, or the stored
    probability vector itself.

    Args:
        samples: path to .npy memmap or ndarray of (N, d) hidden states or
            (N, V) probability vectors.
        W: tensor of shape (V, d) — `model.lm_head.weight`. Required for
            hidden-state samples.
        target_tokens: optional array/list of token ids for which to compute
            rows of M. If None (default), computes the full V×V matrix. Use
            this for large V (e.g. COCA): M_sub is (|target_tokens|, V).
        batch_size: sample rows per device batch.
        device: torch device for the matmul + softmax.

    Returns:
        dict with keys:
          'M'      — (V, V) or (|target_tokens|, V) float64 ndarray
          'Z_emp'  — (V,) float64 ndarray (full, regardless of target_tokens)
          'target_tokens' — the array passed in (or None)
          'N'      — number of rows accumulated
    """
    arr = open_samples(samples)
    V = W.shape[0] if sample_format(arr, W) == 'h' else arr.shape[1]
    batches = iter_sample_probs(arr, W, batch_size=batch_size, device=device,
                                verbose=verbose, desc="moment")
    return _moment_from_batches(batches, V, target_tokens, device)


def compute_moment_matrix_streaming(
    model,
    data,
    weights_lookup,
    *,
    averaging='aitchison',
    target_tokens=None,
    block_size=None,
    min_context=32,
    batch_size=32,
    device='cuda',
    compute_dtype='bfloat16',
    verbose=True,
):
    """
    Streaming corpus pass that accumulates the moment matrix over
    window-averaged predictive states, without writing samples to disk.

    For each valid corpus position i, Y_i is the window-averaged predictive
    state from shape.extract.iter_window_states, and

        M[t, w] += p_t(Y_i) · p_w(Y_i)

    With averaging='probability', p(Y_i) = X̄_i = Σ_d α_d X_{i+d} / Σ_d α_d; with
    'aitchison', p(Y_i) = softmax(W h̄_i). The two differ by a Jensen-style gap
    whenever softmax is non-linear over the windowed h's.

    Args:
        model: GPT in eval mode, on `device`.
        data: np.memmap / np.ndarray of corpus tokens.
        weights_lookup: {offset_d: weight} as built by
            shape.windowing.build_weight_lookup. d=0 included by convention.
        averaging: 'aitchison' or 'probability'.
        target_tokens: optional row-subset (V,) or (|T|,) selector for M.
        block_size: forward-pass length; defaults to model.config.block_size.
        min_context: discards positions with < this many left-context tokens.
            Bumped up to the backward window if smaller.

    Returns:
        dict with keys:
          'M'      — (V, V) or (|target_tokens|, V) float64 ndarray
          'Z_emp'  — (V,) float64 ndarray, mean sampled distribution
          'target_tokens', 'N', 'meta'
    """
    from shape.extract import _resolve_window, iter_window_states

    L = block_size if block_size is not None else model.config.block_size
    win = _resolve_window(weights_lookup, min_context, L)
    batches = (probs for _, probs in iter_window_states(
        model, data, weights_lookup,
        averaging=averaging, block_size=L, min_context=min_context,
        batch_size=batch_size, device=device, compute_dtype=compute_dtype,
        verbose=verbose, desc="moment-stream"))
    out = _moment_from_batches(batches, model.config.vocab_size, target_tokens, device)
    out['meta'] = {
        'V': int(model.config.vocab_size),
        'L': int(L),
        'averaging': averaging,
        'forward_window': int(win['forward_window']),
        'backward_window': int(win['backward_window']),
        'min_context': int(win['min_context']),
        'offsets': win['offsets'],
        'offset_weights': [float(weights_lookup[o]) for o in win['offsets']],
        'total_weight': win['total_weight'],
        'compute_dtype': compute_dtype,
    }
    return out


def aitchison_token_embeddings(
    h,
    W,
    *,
    origin='aitchison',
    target_tokens=None,
    batch_size=2048,
    device='cuda',
    verbose=True,
):
    """
    Aitchison-centroid token embeddings (E[Y|t] path, no SVD).

    For a sample {h_i} and W = lm_head.weight, with p(h) = softmax(Wh):
        h_bar_t = (Σ_i p_t(h_i) · h_i) / (Σ_i p_t(h_i))           # (V, d)
        h_bar   = (1/N) Σ_i h_i                                    # (d,)
        A       = Ψ W                                              # (V-1, d)

    Returns embedding of shape (V, d):
        origin='aitchison' → e_t = h_bar_t - h_bar   (PMI⊥_Aitchison)
        origin='ilr'       → e_t = h_bar_t           (ILR_Aitchison)

    The embedding is naturally d-dimensional (= model hidden size, typically
    1024) because h lives in ℝ^d.  No SVD is required. Requires hidden-state
    samples (no window, or averaging='aitchison').

    Args:
        h: path to .npy memmap or (N, d) ndarray of hidden states.
        W: tensor of shape (V, d) — `model.lm_head.weight`.
        origin: 'aitchison' or 'ilr'.
        target_tokens: optional row-subset to return.
        batch_size: sample rows per device batch.

    Returns:
        dict with keys 'embedding', 'h_bar_t', 'h_bar', 'Z_emp', 'A',
        'origin', 'N', 'target_tokens'.
    """
    if origin not in ('aitchison', 'ilr'):
        raise ValueError(f"origin must be 'aitchison' or 'ilr', got {origin!r}")

    h_arr, N, d = _open_h(h, W)
    W_t = W.detach().to(device=device, dtype=torch.float32)
    V = W_t.shape[0]

    h_bar_t_acc = torch.zeros((V, d), dtype=torch.float64, device=device)
    Z_acc = torch.zeros(V, dtype=torch.float64, device=device)
    h_bar_acc = torch.zeros(d, dtype=torch.float64, device=device)

    n_batches = (N + batch_size - 1) // batch_size
    pbar = tqdm(range(0, N, batch_size), total=n_batches,
                desc="h_bar_t", disable=not verbose, unit="batch")
    with torch.no_grad():
        for s in pbar:
            e = min(s + batch_size, N)
            h_np = np.ascontiguousarray(h_arr[s:e]).astype(np.float32, copy=False)
            h = torch.from_numpy(h_np).to(device=device, non_blocking=True)  # (B, d)
            logits = h @ W_t.T
            p = torch.softmax(logits.float(), dim=-1)                        # (B, V) fp32
            h64 = h.to(torch.float64)
            p64 = p.to(torch.float64)
            h_bar_t_acc += p64.T @ h64
            Z_acc += p64.sum(dim=0)
            h_bar_acc += h64.sum(dim=0)

    Z_safe = Z_acc.clamp(min=1e-30)
    h_bar_t = h_bar_t_acc / Z_safe[:, None]      # (V, d)
    h_bar = h_bar_acc / float(N)                  # (d,)
    Z_emp = Z_acc / float(N)

    A = compute_A(W_t.to(torch.float64))          # (V-1, d)

    if origin == 'aitchison':
        delta = h_bar_t - h_bar[None, :]
    else:
        delta = h_bar_t

    embedding = delta.to(torch.float32).cpu().numpy()  # (V, d)

    if target_tokens is not None:
        target_tokens = np.asarray(target_tokens, dtype=np.int64)
        embedding = embedding[target_tokens]

    return {
        'embedding': embedding,
        'h_bar_t': h_bar_t.cpu().numpy(),
        'h_bar': h_bar.cpu().numpy(),
        'Z_emp': Z_emp.cpu().numpy(),
        'A': A.cpu().numpy(),
        'origin': origin,
        'N': int(N),
        'target_tokens': target_tokens,
    }


def compute_moment_matrix_dt(
    h,
    W,
    Z,
    *,
    target_tokens=None,
    batch_size=2048,
    device='cuda',
    verbose=True,
):
    """
    Moment matrix with D_t^+-weighting (semantic distinctiveness).

    Instead of weighting each corpus position i by p_t(h_i), weight by
    max(0, D_t(h_i)) where D_t(h_i) = W[t]·h_i − log_Z_of_h_i − log Z_t
    is the log-distinctiveness of context h_i for token t.

        M_D[t, w]  = (1/N) Σ_i max(0, D_t(h_i)) · p_w(h_i)
        Z_D[t]     = (1/N) Σ_i max(0, D_t(h_i))

    PMI_D[t, w] = log M_D[t,w] − log Z_D[t] − log Z[w]
    where Z[w] is the standard Stage-1 marginal (passed in as `Z`).

    Requires hidden-state samples (no window, or averaging='aitchison').

    Args:
        h: path to .npy memmap or (N, d) ndarray of hidden states.
        W: tensor of shape (V, d) — `model.lm_head.weight`.
        Z: (V,) array of marginal token probabilities from Stage 1 (window=0).
        target_tokens: optional row-subset selector for M_D.
        batch_size: sample rows per device batch.

    Returns:
        dict with keys 'M', 'Z_D', 'Z_emp', 'target_tokens', 'N'.
    """
    h_arr, N, d = _open_h(h, W)
    W_t = W.detach().to(device=device, dtype=torch.float32)
    V = W_t.shape[0]

    log_Z = torch.from_numpy(
        np.log(np.maximum(np.asarray(Z, dtype=np.float64), 1e-30)).astype(np.float32)
    ).to(device)  # (V,)

    if target_tokens is not None:
        target_tokens = np.asarray(target_tokens, dtype=np.int64)
        assert target_tokens.ndim == 1
        tgt_idx = torch.from_numpy(target_tokens).to(device)
        M = torch.zeros((target_tokens.shape[0], V), dtype=torch.float64, device=device)
        Z_D = torch.zeros(target_tokens.shape[0], dtype=torch.float64, device=device)
    else:
        tgt_idx = None
        M = torch.zeros((V, V), dtype=torch.float64, device=device)
        Z_D = torch.zeros(V, dtype=torch.float64, device=device)

    Z_emp = torch.zeros(V, dtype=torch.float64, device=device)

    n_batches = (N + batch_size - 1) // batch_size
    pbar = tqdm(range(0, N, batch_size), total=n_batches,
                desc='Dt-moment', disable=not verbose, unit='batch')
    with torch.no_grad():
        for s in pbar:
            e = min(s + batch_size, N)
            h_np = np.ascontiguousarray(h_arr[s:e]).astype(np.float32, copy=False)
            h = torch.from_numpy(h_np).to(device=device, non_blocking=True)   # (B, d)
            logits = h @ W_t.T                                                 # (B, V)
            log_Z_h = torch.logsumexp(logits, dim=-1, keepdim=True)           # (B, 1)
            D = logits - log_Z_h - log_Z[None, :]                             # (B, V)
            D_pos = D.clamp(min=0).to(torch.float64)                          # (B, V)
            p = torch.softmax(logits.float(), dim=-1).to(torch.float64)       # (B, V)

            if tgt_idx is not None:
                D_sub = D_pos.index_select(1, tgt_idx)                        # (B, |T|)
                M += (D_sub.T @ p).to(torch.float64)
                Z_D += D_sub.sum(dim=0)
            else:
                M += (D_pos.T @ p).to(torch.float64)
                Z_D += D_pos.sum(dim=0)

            Z_emp += p.sum(dim=0).to(torch.float64)

    M /= float(N)
    Z_D /= float(N)
    Z_emp /= float(N)

    return {
        'M': M.cpu().numpy(),
        'Z_D': Z_D.cpu().numpy(),
        'Z_emp': Z_emp.cpu().numpy(),
        'target_tokens': target_tokens,
        'N': int(N),
    }


def aitchison_token_embeddings_dt(
    h,
    W,
    Z,
    *,
    origin='ilr',
    target_tokens=None,
    batch_size=2048,
    device='cuda',
    verbose=True,
):
    """
    D_t^+-weighted Aitchison-centroid token embeddings.

    Replaces the p_t(h_i) importance weights in `aitchison_token_embeddings`
    with max(0, D_t(h_i)), concentrating the centroid on contexts where the
    token is semantically distinctive.

        h_bar_t^D = (Σ_i max(0, D_t(h_i)) · h_i) / Σ_i max(0, D_t(h_i))
        A = Ψ W   (V-1, d)

    Returns embedding of shape (V, d):
        origin='ilr'       → e_t = h_bar_t^D
        origin='aitchison' → e_t = h_bar_t^D − h_bar

    Requires hidden-state samples (no window, or averaging='aitchison').

    Args:
        Z: (V,) marginal token probabilities from Stage 1 (window=0).
        origin: 'ilr' or 'aitchison'.
    """
    if origin not in ('aitchison', 'ilr'):
        raise ValueError(f"origin must be 'aitchison' or 'ilr', got {origin!r}")

    h_arr, N, d = _open_h(h, W)
    W_t = W.detach().to(device=device, dtype=torch.float32)
    V = W_t.shape[0]

    log_Z = torch.from_numpy(
        np.log(np.maximum(np.asarray(Z, dtype=np.float64), 1e-30)).astype(np.float32)
    ).to(device)  # (V,)

    h_bar_t_acc = torch.zeros((V, d), dtype=torch.float64, device=device)
    D_acc = torch.zeros(V, dtype=torch.float64, device=device)
    h_bar_acc = torch.zeros(d, dtype=torch.float64, device=device)

    n_batches = (N + batch_size - 1) // batch_size
    pbar = tqdm(range(0, N, batch_size), total=n_batches,
                desc='Dt-hbar_t', disable=not verbose, unit='batch')
    with torch.no_grad():
        for s in pbar:
            e = min(s + batch_size, N)
            h_np = np.ascontiguousarray(h_arr[s:e]).astype(np.float32, copy=False)
            h = torch.from_numpy(h_np).to(device=device, non_blocking=True)  # (B, d)
            logits = h @ W_t.T                                                # (B, V)
            log_Z_h = torch.logsumexp(logits, dim=-1, keepdim=True)          # (B, 1)
            D = logits - log_Z_h - log_Z[None, :]                            # (B, V)
            D_pos = D.clamp(min=0).to(torch.float64)                         # (B, V)
            h64 = h.to(torch.float64)
            h_bar_t_acc += D_pos.T @ h64                                     # (V, d)
            D_acc += D_pos.sum(dim=0)                                        # (V,)
            h_bar_acc += h64.sum(dim=0)                                      # (d,)

    D_safe = D_acc.clamp(min=1e-30)
    h_bar_t = h_bar_t_acc / D_safe[:, None]    # (V, d)
    h_bar = h_bar_acc / float(N)               # (d,)
    Z_emp = D_acc / float(N)

    A = compute_A(W_t.to(torch.float64))       # (V-1, d)

    if origin == 'aitchison':
        delta = h_bar_t - h_bar[None, :]
    else:
        delta = h_bar_t

    embedding = delta.to(torch.float32).cpu().numpy()  # (V, d)

    if target_tokens is not None:
        target_tokens = np.asarray(target_tokens, dtype=np.int64)
        embedding = embedding[target_tokens]

    return {
        'embedding': embedding,
        'h_bar_t': h_bar_t.cpu().numpy(),
        'h_bar': h_bar.cpu().numpy(),
        'Z_emp': Z_emp.cpu().numpy(),
        'A': A.cpu().numpy(),
        'origin': origin,
        'N': int(N),
        'target_tokens': target_tokens,
    }


def pmi_matrix(
    samples,
    W=None,
    *,
    Z=None,
    target_tokens=None,
    eps=1e-30,
    batch_size=2048,
    device='cuda',
    verbose=True,
):
    """
    Compute the PMI matrix:

        PMI[t, w] = log M[t, w] − log Z[t] − log Z[w]

    Args:
        samples: path or ndarray (see compute_moment_matrix).
        W: (V, d) tensor. Required for hidden-state samples.
        Z: (V,) vector. If None, uses the Z_emp recomputed from the samples,
            which equals the `{name}_Z.npy` written by extract_features.
        target_tokens: row-subset selector (None = full V×V).
        eps: floor for log to avoid −inf on zero entries.

    Returns:
        dict with keys 'pmi', 'M', 'Z', 'Z_emp', 'target_tokens', 'N'.
    """
    out = compute_moment_matrix(
        samples, W,
        target_tokens=target_tokens,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )
    M = out['M']
    Z_emp = out['Z_emp']
    Z_used = np.asarray(Z, dtype=np.float64) if Z is not None else Z_emp

    log_Z = np.log(np.maximum(Z_used, eps))
    log_M = np.log(np.maximum(M, eps))
    if target_tokens is not None:
        log_Z_rows = log_Z[target_tokens]
        pmi = log_M - log_Z_rows[:, None] - log_Z[None, :]
    else:
        pmi = log_M - log_Z[:, None] - log_Z[None, :]
    return {
        'pmi': pmi,
        'M': M,
        'Z': Z_used,
        'Z_emp': Z_emp,
        'target_tokens': target_tokens,
        'N': out['N'],
    }


def ilr_embeddings(pmi):
    """
    Apply the ILR transform Ψ to each row of PMI.

    Per the README §"The ILR expected-value embedding":

        e_t = μ_t^Y − μ^Y = Ψ · PMI_t ∈ ℝ^(V−1)

    This removes the informativeness component (PMI's projection onto the
    all-ones direction), which is what discrete word-embedding post-processing
    approximates by ad-hoc first-PC removal (Mu & Viswanath, 2018).

    Args:
        pmi: (V, V) or (|T|, V) array.

    Returns:
        (V, V−1) array of the same dtype (numpy → numpy, torch → torch).
    """
    was_np = isinstance(pmi, np.ndarray)
    x = torch.from_numpy(pmi) if was_np else pmi
    y = ilr_apply(x)
    return y.numpy() if was_np else y


def svd_embeddings(
    X, k, *,
    eig_weight=0.5,
    center=False,
    solver='auto',
    oversample=10,
    n_iter=4,
    device=None,
    dtype=torch.float32,
):
    """
    Low-rank embeddings via (truncated) SVD, analogous to word2vec / GloVe.

    X = U diag(S) Vᵀ
    embedding = U[:, :k] · diag(S[:k])^α

    Args:
        X: (V, D) array. D=V for PMI, D=V−1 for ILR embeddings.
        k: embedding dimensionality.
        eig_weight: exponent α ∈ [0, 1]. α=0 → orthonormal U; α=0.5 matches
            the symmetric Levy-&-Goldberg form that tracks word2vec well;
            α=1 is the raw SVD coordinate.
        center: if True, subtract per-column mean before SVD.
        solver: 'auto' | 'torch_lowrank' | 'sklearn' | 'numpy'.
            'auto' picks 'torch_lowrank' when k < min(V, D)/2, else 'numpy'.
            'torch_lowrank' uses torch.svd_lowrank (randomized; works on GPU
                via `device='cuda'`). Far faster than full SVD on V ~ 10^4.
            'sklearn' uses sklearn.utils.extmath.randomized_svd (CPU-only).
            'numpy'  uses np.linalg.svd (full SVD; previous behavior).
        oversample, n_iter: randomized-SVD knobs (used by torch_lowrank /
            sklearn). Defaults match sklearn's TruncatedSVD.
        device: torch device for 'torch_lowrank'. None → match X if torch
            tensor, else CPU.
        dtype: torch compute dtype for 'torch_lowrank' (float32 is fast and
            usually sufficient; use float64 for tighter accuracy).

    Returns:
        dict with keys
          'embedding'   — (V, k) float32 array
          'S'           — (k,) float64 singular values (only the top-k for
                          truncated solvers; full spectrum for 'numpy')
          'explained_variance_ratio' — top-k variance fraction (float).
                          For truncated solvers this uses ||X||_F² as the
                          denominator (computed cheaply without forming the
                          full SVD).
    """
    # Resolve solver
    V_, D_ = X.shape
    rank_max = min(V_, D_)
    if k > rank_max:
        raise ValueError(f"k={k} > min(V, D)={rank_max}")
    if solver == 'auto':
        solver = 'torch_lowrank' if k < rank_max // 2 else 'numpy'

    # Normalize input + optional centering
    if isinstance(X, torch.Tensor):
        if device is None:
            device = X.device
        X_t = X.detach().to(device=device, dtype=dtype)
    else:
        X_t = None  # numpy path may stay numpy

    if solver == 'numpy':
        if X_t is not None:
            X_np = X_t.cpu().numpy().astype(np.float64, copy=False)
        else:
            X_np = np.asarray(X, dtype=np.float64)
        if center:
            X_np = X_np - X_np.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(X_np, full_matrices=False)
        emb = U[:, :k] * (S[:k] ** eig_weight)
        total_var = float((S ** 2).sum())
        evr_k = float((S[:k] ** 2).sum() / total_var) if total_var > 0 else 0.0
        return {
            'embedding': emb.astype(np.float32),
            'S': S,
            'explained_variance_ratio': evr_k,
            'solver': 'numpy',
        }

    if solver == 'sklearn':
        try:
            from sklearn.utils.extmath import randomized_svd
        except ImportError as exc:
            raise ImportError("solver='sklearn' requires scikit-learn") from exc
        if X_t is not None:
            X_np = X_t.cpu().numpy().astype(np.float64, copy=False)
        else:
            X_np = np.asarray(X, dtype=np.float64)
        if center:
            X_np = X_np - X_np.mean(axis=0, keepdims=True)
        total_var = float((X_np ** 2).sum())
        U, S, _ = randomized_svd(
            X_np, n_components=k, n_oversamples=oversample, n_iter=n_iter,
            random_state=0,
        )
        emb = U * (S ** eig_weight)
        evr_k = float((S ** 2).sum() / total_var) if total_var > 0 else 0.0
        return {
            'embedding': emb.astype(np.float32),
            'S': S.astype(np.float64),
            'explained_variance_ratio': evr_k,
            'solver': 'sklearn',
        }

    if solver == 'torch_lowrank':
        # Build a torch tensor on the requested device.
        if X_t is None:
            X_np = np.asarray(X)
            X_t = torch.from_numpy(X_np).to(
                device=(device if device is not None else 'cpu'),
                dtype=dtype,
            )
        if center:
            X_t = X_t - X_t.mean(dim=0, keepdim=True)
        # ||X||_F² as denominator for explained variance.
        total_var = float((X_t.to(torch.float64) ** 2).sum().item())
        q = min(rank_max, k + oversample)
        U, S, _ = torch.svd_lowrank(X_t, q=q, niter=n_iter)
        U = U[:, :k]
        S = S[:k]
        emb = (U * (S ** eig_weight)).to(torch.float32).cpu().numpy()
        S_np = S.to(torch.float64).cpu().numpy()
        evr_k = float((S_np ** 2).sum() / total_var) if total_var > 0 else 0.0
        return {
            'embedding': emb,
            'S': S_np,
            'explained_variance_ratio': evr_k,
            'solver': 'torch_lowrank',
        }

    raise ValueError(f"unknown solver: {solver!r}")


def compute_embeddings(
    samples,
    W=None,
    *,
    Z=None,
    k=300,
    eig_weight=0.5,
    use_ilr=True,
    center=False,
    target_tokens=None,
    batch_size=2048,
    device='cuda',
    verbose=True,
    eps=1e-30,
    svd_solver='auto',
    svd_oversample=10,
    svd_n_iter=4,
):
    """
    End-to-end: samples → PMI → [ILR] → SVD → (V, k) embedding.

    `samples` and `W` are as in compute_moment_matrix.

    Args:
        use_ilr: if True, SVD is taken on Ψ·PMI (meaning-only, the
            README-recommended embedding); if False, on raw PMI.

    Returns:
        dict with everything: 'embedding', 'pmi', 'M', 'Z', 'Z_emp', 'S',
        'ilr_emb' (if use_ilr), 'explained_variance_ratio', 'meta'.
    """
    cp = pmi_matrix(
        samples, W,
        Z=Z,
        target_tokens=target_tokens,
        eps=eps,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )

    if use_ilr:
        if target_tokens is not None:
            raise NotImplementedError(
                "ilr_embeddings expects full V-vocabulary rows (Ψ operates "
                "along the vocab dimension); pass target_tokens=None or set "
                "use_ilr=False for row-subsetted mode."
            )
        X = ilr_embeddings(cp['pmi'])
    else:
        X = cp['pmi']

    svd = svd_embeddings(
        X, k=k, eig_weight=eig_weight, center=center,
        solver=svd_solver, oversample=svd_oversample, n_iter=svd_n_iter,
        device=device,
    )

    out = {
        'embedding': svd['embedding'],
        'pmi': cp['pmi'],
        'M': cp['M'],
        'Z': cp['Z'],
        'Z_emp': cp['Z_emp'],
        'S': svd['S'],
        'explained_variance_ratio': svd['explained_variance_ratio'],
        'meta': {
            'k': int(k),
            'eig_weight': float(eig_weight),
            'use_ilr': bool(use_ilr),
            'center': bool(center),
            'N': cp['N'],
            'V': int(cp['Z_emp'].shape[0]),
            'd': None if W is None else int(W.shape[1]),
            'target_tokens': None if target_tokens is None else list(map(int, target_tokens)),
        },
    }
    if use_ilr:
        out['ilr_emb'] = X
    return out


def nearest_neighbors(embedding, query_ids, *, top_k=10, itos=None, metric='cosine'):
    """
    Quick-look nearest neighbors in the embedding space.

    Args:
        embedding: (V, k) array.
        query_ids: iterable of token ids to query.
        top_k: number of neighbors per query.
        itos: optional {id: token_str} for pretty printing.
        metric: 'cosine' or 'euclidean'.

    Returns:
        dict {query_id: list of (neighbor_id, score)} where score is the
        chosen similarity (higher = closer for cosine; lower for euclidean).
    """
    E = np.asarray(embedding, dtype=np.float32)
    if metric == 'cosine':
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        Ehat = E / norms
        sims_matrix = Ehat @ Ehat.T
    elif metric == 'euclidean':
        # −||a−b||² as "similarity" so argsort-desc still gives nearest first
        d2 = ((E[:, None, :] - E[None, :, :]) ** 2).sum(axis=-1)
        sims_matrix = -d2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    out = {}
    for q in query_ids:
        q = int(q)
        sims = sims_matrix[q].copy()
        sims[q] = -np.inf
        order = np.argsort(-sims)[:top_k]
        out[q] = [(int(i), float(sims[i])) for i in order]

    if itos is not None:
        for q, neigh in out.items():
            q_tok = itos.get(q, f"<id={q}>")
            rendered = [f"{itos.get(i, f'<id={i}>')}({s:+.3f})" for i, s in neigh]
            print(f"  {q_tok!r}: {', '.join(rendered)}")
    return out
