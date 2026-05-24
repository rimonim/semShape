"""
Stage 3 — expected-value layer: PMI matrix, ILR embeddings, SVD.

Computes static word embeddings from a trained GPT via the continuous pipeline:

    h_eff  →  moment matrix M  →  PMI  →  [ILR]  →  SVD  →  low-dim embedding

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

The Monte-Carlo estimator uses the empirical h_eff sample (the actual corpus
positions from Stage 1). This is the natural unbiased estimator and matches
build_fcm.py's implicit empirical-distribution sampling; we can also pass in
samples drawn from the Stage-2 flow if desired.

Scales: M is V×V, so the full moment computation is O(V²) memory on-device
and O(N · V²) flops for the accumulation. For small vocabularies this is
trivial; for COCA (V ≈ 150k) use `target_tokens=` to compute only a row-subset.
"""

from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from shape.geometry import compute_A, ilr_apply


def _open_h_eff(h_eff):
    """Accept either a path (→ mmap) or an ndarray; return (array, N, d)."""
    if isinstance(h_eff, str):
        arr = np.load(h_eff, mmap_mode='r')
    else:
        arr = h_eff
    assert arr.ndim == 2, f"h_eff must be 2D, got shape {arr.shape}"
    return arr, arr.shape[0], arr.shape[1]


def compute_moment_matrix(
    h_eff,
    W,
    *,
    target_tokens=None,
    batch_size=2048,
    device='cuda',
    verbose=True,
):
    """
    Accumulate the moment matrix and empirical Z over the h_eff sample.

        M[t, w]  = (1/N) Σ_i p_t(h_i) · p_w(h_i)        where p(h) = softmax(Wh)
        Z_emp[w] = (1/N) Σ_i p_w(h_i)

    Args:
        h_eff: path to .npy memmap or (N, d) ndarray of hidden states.
        W: tensor of shape (V, d) — `model.lm_head.weight`.
        target_tokens: optional array/list of token ids for which to compute
            rows of M. If None (default), computes the full V×V matrix. Use
            this for large V (e.g. COCA): M_sub is (|target_tokens|, V).
        batch_size: rows of h_eff per device batch.
        device: torch device for the matmul + softmax.

    Returns:
        dict with keys:
          'M'      — (V, V) or (|target_tokens|, V) float64 ndarray
          'Z_emp'  — (V,) float64 ndarray (full, regardless of target_tokens)
          'target_tokens' — the array passed in (or None)
          'N'      — number of rows accumulated
    """
    h_arr, N, d = _open_h_eff(h_eff)
    W_t = W.detach().to(device=device, dtype=torch.float32)
    V, d_w = W_t.shape
    assert d == d_w, f"h_eff has d={d}, W has d={d_w}"

    if target_tokens is not None:
        target_tokens = np.asarray(target_tokens, dtype=np.int64)
        assert target_tokens.ndim == 1
        tgt_idx = torch.from_numpy(target_tokens).to(device)
        M = torch.zeros((target_tokens.shape[0], V), dtype=torch.float64, device=device)
    else:
        tgt_idx = None
        M = torch.zeros((V, V), dtype=torch.float64, device=device)

    Z_emp = torch.zeros(V, dtype=torch.float64, device=device)

    n_batches = (N + batch_size - 1) // batch_size
    pbar = tqdm(range(0, N, batch_size),
                total=n_batches, desc="moment", disable=not verbose, unit="batch")
    with torch.no_grad():
        for s in pbar:
            e = min(s + batch_size, N)
            h_np = np.ascontiguousarray(h_arr[s:e]).astype(np.float32, copy=False)
            h = torch.from_numpy(h_np).to(device=device, non_blocking=True)   # (B, d)
            logits = h @ W_t.T                                                # (B, V)
            p = torch.softmax(logits.float(), dim=-1)                         # (B, V) fp32
            if tgt_idx is not None:
                p_sub = p.index_select(1, tgt_idx)                            # (B, |T|)
                M += (p_sub.T @ p).to(torch.float64)
            else:
                M += (p.T @ p).to(torch.float64)
            Z_emp += p.sum(dim=0).to(torch.float64)

    M /= float(N)
    Z_emp /= float(N)
    return {
        'M': M.cpu().numpy(),
        'Z_emp': Z_emp.cpu().numpy(),
        'target_tokens': target_tokens,
        'N': int(N),
    }


def compute_moment_matrix_prob_window(
    model,
    data,
    weights_lookup,
    *,
    target_tokens=None,
    block_size=None,
    min_context=32,
    batch_size=32,
    device='cuda',
    compute_dtype='bfloat16',
    verbose=True,
):
    """
    Streaming corpus pass that accumulates a moment matrix using
    *probability-space* (simplex) window averaging.

    For each valid corpus position i:
        X_bar_i = (Σ_d α_d · X_{i+d}) / (Σ_d α_d)
    where X_j = softmax(W h_j) is the per-position predictive distribution and
    α_d = weights_lookup[d]. The moment matrix is then

        M[t, w] += X_bar_i[t] · X_bar_i[w]

    which differs from h-space averaging (Variant H in extract.py) by a
    Jensen-style gap whenever softmax is non-linear over the windowed h's.

    Args:
        model: GPT in eval mode, on `device`.
        data: np.memmap / np.ndarray of corpus tokens.
        weights_lookup: {offset_d: weight} as built by
            shape.windowing.build_weight_lookup. d=0 included by convention.
        target_tokens: optional row-subset (V,) or (|T|,) selector for M.
        block_size: forward-pass length; defaults to model.config.block_size.
        min_context: discards positions with < this many left-context tokens.
            Bumped up to max |offset| if smaller.

    Returns:
        dict with keys:
          'M'      — (V, V) or (|target_tokens|, V) float64 ndarray
          'Z_emp'  — (V,) float64 ndarray, mean of X_bar
          'target_tokens', 'N', 'meta'
    """
    from shape.extract import _count_valid, _iter_chunks

    L = block_size if block_size is not None else model.config.block_size
    V = model.config.vocab_size
    T = len(data)

    if not weights_lookup:
        raise ValueError("weights_lookup is empty")
    offsets = sorted(weights_lookup.keys())
    total_weight = float(sum(weights_lookup.values()))
    if total_weight == 0:
        raise ValueError("weights sum to zero")
    # Forward / backward windows are tracked separately: forward_window restricts
    # the right edge of every chunk (need t+forward_window ≤ L-1), while
    # backward_window restricts the left edge (need t ≥ backward_window). Treating
    # max|offset| as both gives an empty valid range whenever the window is
    # backward- or forward-only at chunk-scale.
    forward_window  = max(max(offsets),  0)
    backward_window = max(-min(offsets), 0)
    if min_context < backward_window:
        min_context = backward_window
        if verbose:
            print(f"Note: min_context bumped to {min_context} (= backward window).")
    # Per-chunk valid range is [min_context, L - forward_window). Empty unless
    # min_context < L - forward_window, i.e. backward_window + forward_window < L.
    if min_context + forward_window >= L:
        raise ValueError(
            f"backward_window ({backward_window}) + forward_window "
            f"({forward_window}) >= block_size ({L}); no chunk can contain "
            f"a valid position. Reduce window_size or increase block_size."
        )

    if target_tokens is not None:
        target_tokens = np.asarray(target_tokens, dtype=np.int64)
        assert target_tokens.ndim == 1
        tgt_idx = torch.from_numpy(target_tokens).to(device)
        M = torch.zeros((target_tokens.shape[0], V), dtype=torch.float64, device=device)
    else:
        tgt_idx = None
        M = torch.zeros((V, V), dtype=torch.float64, device=device)
    Z_emp = torch.zeros(V, dtype=torch.float64, device=device)

    n_valid_total = _count_valid(T, L, forward_window, min_context)
    chunks_list = list(_iter_chunks(T, L, forward_window, min_context))
    n_chunks = len(chunks_list)
    n_batches = (n_chunks + batch_size - 1) // batch_size

    if verbose:
        print(f"Corpus length: {T:,} tokens; L={L}, "
              f"forward_window={forward_window}, backward_window={backward_window}, "
              f"min_context={min_context}")
        print(f"Valid positions: {n_valid_total:,}")
        print(f"Active offsets: {offsets}")
        print(f"Σ w = {total_weight:.4f}")

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                   'float16': torch.float16}
    ptdtype = ptdtype_map[compute_dtype]
    ctx = nullcontext() if device_type == 'cpu' else \
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    cursor = 0
    model.eval()
    with torch.no_grad():
        with ctx:
            pbar = tqdm(range(0, n_chunks, batch_size), total=n_batches,
                        desc="prob-moment", disable=not verbose, unit="batch")
            for batch_start in pbar:
                chunks = chunks_list[batch_start:batch_start + batch_size]
                B = len(chunks)
                input_np = np.zeros((B, L), dtype=np.int64)
                for i, (s, _, _) in enumerate(chunks):
                    input_np[i] = data[s:s + L].astype(np.int64)
                input_tensor = torch.from_numpy(input_np).to(device)
                logits, _, _ = model(input_tensor, return_hidden=True)
                X = torch.softmax(logits.float(), dim=-1)            # (B, L, V) fp32

                b_idx_list, t_local_list = [], []
                for i, (_, vstart, vend) in enumerate(chunks):
                    for t_local in range(vstart, vend):
                        b_idx_list.append(i)
                        t_local_list.append(t_local)
                n_valid_batch = len(b_idx_list)
                if n_valid_batch == 0:
                    continue
                b_idx = torch.tensor(b_idx_list, dtype=torch.long, device=device)
                t_local = torch.tensor(t_local_list, dtype=torch.long, device=device)

                X_bar = torch.zeros(n_valid_batch, V, dtype=torch.float32, device=device)
                for off in offsets:
                    w_off = float(weights_lookup[off])
                    X_at_off = X[b_idx, t_local + off, :]            # (n_valid_batch, V)
                    X_bar.add_(X_at_off, alpha=w_off)
                X_bar /= total_weight

                if tgt_idx is not None:
                    X_sub = X_bar.index_select(1, tgt_idx)           # (n_valid_batch, |T|)
                    M += (X_sub.T @ X_bar).to(torch.float64)
                else:
                    M += (X_bar.T @ X_bar).to(torch.float64)
                Z_emp += X_bar.sum(dim=0).to(torch.float64)
                cursor += n_valid_batch
                if verbose:
                    pbar.set_postfix({'positions': cursor})

    assert cursor == n_valid_total, f"cursor {cursor} != n_valid_total {n_valid_total}"
    M /= float(n_valid_total)
    Z_emp /= float(n_valid_total)

    return {
        'M': M.cpu().numpy(),
        'Z_emp': Z_emp.cpu().numpy(),
        'target_tokens': target_tokens,
        'N': int(n_valid_total),
        'meta': {
            'V': int(V),
            'L': int(L),
            'forward_window': int(forward_window),
            'backward_window': int(backward_window),
            'min_context': int(min_context),
            'offsets': offsets,
            'offset_weights': [float(weights_lookup[o]) for o in offsets],
            'total_weight': total_weight,
            'compute_dtype': compute_dtype,
        },
    }


def aitchison_token_embeddings(
    h_eff,
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
    1024) because h_eff lives in ℝ^d.  No SVD is required.

    Args:
        h_eff: path to .npy memmap or (N, d) ndarray of hidden states.
        W: tensor of shape (V, d) — `model.lm_head.weight`.
        origin: 'aitchison' or 'ilr'.
        target_tokens: optional row-subset to return.
        batch_size: rows of h_eff per device batch.

    Returns:
        dict with keys 'embedding', 'h_bar_t', 'h_bar', 'Z_emp', 'A',
        'origin', 'N', 'target_tokens'.
    """
    if origin not in ('aitchison', 'ilr'):
        raise ValueError(f"origin must be 'aitchison' or 'ilr', got {origin!r}")

    h_arr, N, d = _open_h_eff(h_eff)
    W_t = W.detach().to(device=device, dtype=torch.float32)
    V, d_w = W_t.shape
    assert d == d_w, f"h_eff has d={d}, W has d={d_w}"

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
    h_eff,
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

    Args:
        h_eff: path to .npy memmap or (N, d) ndarray of hidden states.
        W: tensor of shape (V, d) — `model.lm_head.weight`.
        Z: (V,) array of marginal token probabilities from Stage 1 (window=0).
        target_tokens: optional row-subset selector for M_D.
        batch_size: rows of h_eff per device batch.

    Returns:
        dict with keys 'M', 'Z_D', 'Z_emp', 'target_tokens', 'N'.
    """
    h_arr, N, d = _open_h_eff(h_eff)
    W_t = W.detach().to(device=device, dtype=torch.float32)
    V, d_w = W_t.shape
    assert d == d_w, f"h_eff has d={d}, W has d={d_w}"

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
    h_eff,
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

    Args:
        Z: (V,) marginal token probabilities from Stage 1 (window=0).
        origin: 'ilr' or 'aitchison'.
    """
    if origin not in ('aitchison', 'ilr'):
        raise ValueError(f"origin must be 'aitchison' or 'ilr', got {origin!r}")

    h_arr, N, d = _open_h_eff(h_eff)
    W_t = W.detach().to(device=device, dtype=torch.float32)
    V, d_w = W_t.shape
    assert d == d_w, f"h_eff has d={d}, W has d={d_w}"

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
    h_eff,
    W,
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
        h_eff: path or ndarray (see compute_moment_matrix).
        W: (V, d) tensor.
        Z: (V,) vector. If None, uses the Z_emp recomputed from h_eff. When
            h_eff was window-averaged in Stage 1, passing the stage-1 Z
            (computed from un-averaged h) is the framework-aligned choice,
            though the two agree at window=0.
        target_tokens: row-subset selector (None = full V×V).
        eps: floor for log to avoid −inf on zero entries.

    Returns:
        dict with keys 'pmi', 'M', 'Z', 'Z_emp', 'target_tokens', 'N'.
    """
    out = compute_moment_matrix(
        h_eff, W,
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
    h_eff,
    W,
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
    End-to-end: h_eff → PMI → [ILR] → SVD → (V, k) embedding.

    Args:
        use_ilr: if True, SVD is taken on Ψ·PMI (meaning-only, the
            README-recommended embedding); if False, on raw PMI.

    Returns:
        dict with everything: 'embedding', 'pmi', 'M', 'Z', 'Z_emp', 'S',
        'ilr_emb' (if use_ilr), 'explained_variance_ratio', 'meta'.
    """
    cp = pmi_matrix(
        h_eff, W,
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
            'V': int(W.shape[0]),
            'd': int(W.shape[1]),
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
