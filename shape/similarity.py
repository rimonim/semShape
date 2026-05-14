"""
Pairwise semantic similarity quantities between token pairs.

Implements three probability-theoretic quantities (working paper §3):

    expected_probability:  E_{Y|t1}[ p_{t2}(Y) ]
    expected_surprisal:   -E_{Y|t1}[ log p_{t2}(Y) ]
    kl_divergence:         E_{Y|t1}[ log(p_{t1}(Y) / p_{t2}(Y)) ]

All use importance-weighted sampling from the corpus:

    E_{Y|t1}[f(Y)] ≈ Σ_i p_{t1}(Y_i) f(Y_i) / Σ_i p_{t1}(Y_i)

Two modes:

    compute_pairwise_similarities           — h_eff-based (no windowing)
    compute_pairwise_similarities_prob_window — live corpus pass with
        probability-space window averaging (X_bar_i = Σ_d α_d X_{i+d} / Σ α_d)
"""

from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

_LOG_EPS = 1e-30

_VALID_QUANTITIES = frozenset(
    ('expected_probability', 'expected_surprisal', 'kl_divergence')
)


def _open_h_eff(h_eff):
    if isinstance(h_eff, (str, bytes)):
        arr = np.load(h_eff, mmap_mode='r')
    else:
        arr = h_eff
    assert arr.ndim == 2, f"h_eff must be 2D, got shape {arr.shape}"
    return arr, arr.shape[0], arr.shape[1]


def _accumulate_pair_quantities(X, t1_idx, t2_idx, unique_t1_idx,
                                ep_acc, es_acc, kl_acc, Z_acc, quantities):
    """
    Update running accumulators in-place given a batch of probability vectors.

    Args:
        X: (B, V) float32 tensor — probability distributions at B positions.
        t1_idx: (P,) long tensor — token ids for the "query" of each pair.
        t2_idx: (P,) long tensor — token ids for the "target" of each pair.
        unique_t1_idx: (U,) long tensor — sorted unique values in t1_idx.
        ep_acc, es_acc, kl_acc: (P,) float64 accumulators (mutated).
        Z_acc: (V,) float64 accumulator for importance-weight sums (mutated).
        quantities: set of str.
    """
    X1 = X[:, t1_idx]   # (B, P)
    X2 = X[:, t2_idx]   # (B, P)

    # Z_acc: accumulate per unique t1 token
    contrib = X.to(torch.float64)[:, unique_t1_idx].sum(dim=0)  # (U,)
    Z_acc.index_add_(0, unique_t1_idx, contrib)

    if 'expected_probability' in quantities:
        ep_acc.add_((X1 * X2).to(torch.float64).sum(dim=0))

    if 'expected_surprisal' in quantities or 'kl_divergence' in quantities:
        log_X2 = torch.log(X2.clamp(min=_LOG_EPS))
        if 'expected_surprisal' in quantities:
            es_acc.add_((X1 * (-log_X2)).to(torch.float64).sum(dim=0))
        if 'kl_divergence' in quantities:
            log_X1 = torch.log(X1.clamp(min=_LOG_EPS))
            kl_acc.add_((X1 * (log_X1 - log_X2)).to(torch.float64).sum(dim=0))


def _normalize_and_collect(ep_acc, es_acc, kl_acc, Z_acc, t1_ids_tensor, quantities):
    Z_for_pairs = Z_acc[t1_ids_tensor].clamp(min=_LOG_EPS)
    result = {}
    if 'expected_probability' in quantities:
        result['expected_probability'] = (ep_acc / Z_for_pairs).cpu().numpy()
    if 'expected_surprisal' in quantities:
        result['expected_surprisal'] = (es_acc / Z_for_pairs).cpu().numpy()
    if 'kl_divergence' in quantities:
        result['kl_divergence'] = (kl_acc / Z_for_pairs).cpu().numpy()
    return result


def compute_pairwise_similarities(
    h_eff,
    W,
    t1_ids,
    t2_ids,
    *,
    quantities=('expected_probability', 'expected_surprisal', 'kl_divergence'),
    batch_size=2048,
    device='cuda',
    verbose=True,
):
    """
    Compute pairwise similarity quantities from precomputed h_eff hidden states.

    For each pair (t1_ids[i], t2_ids[i]), computes importance-weighted
    expectations E_{Y|t1}[f(Y)] using the corpus hidden states as the
    sample of Y's, with importance weights p_{t1}(h_j) = softmax(W @ h_j)[t1].

    Args:
        h_eff: path to .npy memmap or (N, d) ndarray of hidden states.
        W: (V, d) tensor — model.lm_head.weight.
        t1_ids: (P,) array-like of token ids for the query token.
        t2_ids: (P,) array-like of token ids for the target token.
        quantities: iterable of strings from
            {'expected_probability', 'expected_surprisal', 'kl_divergence'}.
        batch_size: rows of h_eff per device batch.
        device: torch device string.
        verbose: show tqdm progress bar.

    Returns:
        dict mapping each requested quantity name to a float64 ndarray of
        shape (P,).
    """
    quantities = set(quantities)
    unknown = quantities - _VALID_QUANTITIES
    if unknown:
        raise ValueError(f"Unknown quantities: {unknown}")

    t1_ids = np.asarray(t1_ids, dtype=np.int64)
    t2_ids = np.asarray(t2_ids, dtype=np.int64)
    assert t1_ids.shape == t2_ids.shape and t1_ids.ndim == 1
    P = len(t1_ids)

    h_arr, N, d = _open_h_eff(h_eff)
    W_t = W.detach().to(device=device, dtype=torch.float32)
    V = W_t.shape[0]

    t1_idx = torch.from_numpy(t1_ids).to(device)
    t2_idx = torch.from_numpy(t2_ids).to(device)
    unique_t1_idx = t1_idx.unique()

    ep_acc = torch.zeros(P, dtype=torch.float64, device=device)
    es_acc = torch.zeros(P, dtype=torch.float64, device=device)
    kl_acc = torch.zeros(P, dtype=torch.float64, device=device)
    Z_acc  = torch.zeros(V, dtype=torch.float64, device=device)

    n_batches = (N + batch_size - 1) // batch_size
    pbar = tqdm(range(0, N, batch_size), total=n_batches,
                desc="similarity", disable=not verbose, unit="batch")
    with torch.no_grad():
        for s in pbar:
            e = min(s + batch_size, N)
            h_np = np.ascontiguousarray(h_arr[s:e]).astype(np.float32, copy=False)
            h = torch.from_numpy(h_np).to(device=device, non_blocking=True)
            X = torch.softmax((h @ W_t.T).float(), dim=-1)   # (B, V) fp32
            _accumulate_pair_quantities(
                X, t1_idx, t2_idx, unique_t1_idx,
                ep_acc, es_acc, kl_acc, Z_acc, quantities)

    return _normalize_and_collect(ep_acc, es_acc, kl_acc, Z_acc, t1_idx, quantities)


def compute_pairwise_similarities_prob_window(
    model,
    data,
    weights_lookup,
    t1_ids,
    t2_ids,
    *,
    quantities=('expected_probability', 'expected_surprisal', 'kl_divergence'),
    block_size=None,
    min_context=32,
    batch_size=32,
    device='cuda',
    compute_dtype='bfloat16',
    verbose=True,
):
    """
    Compute pairwise similarity quantities via a streaming corpus pass with
    probability-space window averaging.

    For each valid corpus position, computes
        X_bar_i = (Σ_d α_d X_{i+d}) / (Σ_d α_d)
    where X_j = softmax(W h_j), then uses X_bar_i as the sampled Y in the
    importance-weighted expectation E_{Y|t1}[f(Y)].

    Args:
        model: GPT in eval mode on `device`.
        data: np.memmap or ndarray of corpus tokens (uint16).
        weights_lookup: {offset_d: weight} dict from build_weight_lookup.
        t1_ids: (P,) array-like of token ids for the query token.
        t2_ids: (P,) array-like of token ids for the target token.
        quantities: iterable of strings from
            {'expected_probability', 'expected_surprisal', 'kl_divergence'}.
        block_size: forward-pass sequence length (defaults to model block_size).
        min_context: minimum left-context tokens per position.
        batch_size: sequences per forward pass.
        device: torch device string.
        compute_dtype: autocast dtype ('bfloat16', 'float16', 'float32').
        verbose: show tqdm progress and print corpus stats.

    Returns:
        dict mapping each requested quantity name to a float64 ndarray of
        shape (P,).
    """
    from shape.extract import _count_valid, _iter_chunks

    quantities = set(quantities)
    unknown = quantities - _VALID_QUANTITIES
    if unknown:
        raise ValueError(f"Unknown quantities: {unknown}")

    t1_ids = np.asarray(t1_ids, dtype=np.int64)
    t2_ids = np.asarray(t2_ids, dtype=np.int64)
    assert t1_ids.shape == t2_ids.shape and t1_ids.ndim == 1
    P = len(t1_ids)

    L = block_size if block_size is not None else model.config.block_size
    V = model.config.vocab_size
    T = len(data)

    if not weights_lookup:
        raise ValueError("weights_lookup is empty")
    offsets = sorted(weights_lookup.keys())
    total_weight = float(sum(weights_lookup.values()))
    if total_weight == 0:
        raise ValueError("weights sum to zero")
    forward_window  = max(max(offsets),  0)
    backward_window = max(-min(offsets), 0)
    if min_context < backward_window:
        min_context = backward_window
        if verbose:
            print(f"Note: min_context bumped to {min_context} (= backward window).")
    if min_context + forward_window >= L:
        raise ValueError(
            f"backward_window ({backward_window}) + forward_window "
            f"({forward_window}) >= block_size ({L}); no valid positions. "
            f"Reduce window size or increase block_size.")

    t1_idx = torch.from_numpy(t1_ids).to(device)
    t2_idx = torch.from_numpy(t2_ids).to(device)
    unique_t1_idx = t1_idx.unique()

    ep_acc = torch.zeros(P, dtype=torch.float64, device=device)
    es_acc = torch.zeros(P, dtype=torch.float64, device=device)
    kl_acc = torch.zeros(P, dtype=torch.float64, device=device)
    Z_acc  = torch.zeros(V, dtype=torch.float64, device=device)

    n_valid_total = _count_valid(T, L, forward_window, min_context)
    chunks_list = list(_iter_chunks(T, L, forward_window, min_context))
    n_chunks = len(chunks_list)
    n_batches = (n_chunks + batch_size - 1) // batch_size

    if verbose:
        print(f"Corpus: {T:,} tokens; L={L}, "
              f"forward_window={forward_window}, backward_window={backward_window}, "
              f"min_context={min_context}")
        print(f"Valid positions: {n_valid_total:,}  |  Active offsets: {offsets}")

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
               'float16': torch.float16}[compute_dtype]
    ctx = (nullcontext() if device_type == 'cpu'
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    cursor = 0
    model.eval()
    with torch.no_grad():
        with ctx:
            pbar = tqdm(range(0, n_chunks, batch_size), total=n_batches,
                        desc="sim-prob-window", disable=not verbose, unit="batch")
            for batch_start in pbar:
                chunks = chunks_list[batch_start:batch_start + batch_size]
                B = len(chunks)
                input_np = np.zeros((B, L), dtype=np.int64)
                for i, (s, _, _) in enumerate(chunks):
                    input_np[i] = data[s:s + L].astype(np.int64)
                input_tensor = torch.from_numpy(input_np).to(device)
                logits, _, _ = model(input_tensor, return_hidden=True)
                X = torch.softmax(logits.float(), dim=-1)   # (B, L, V) fp32

                b_idx_list, t_local_list = [], []
                for i, (_, vstart, vend) in enumerate(chunks):
                    for t_local in range(vstart, vend):
                        b_idx_list.append(i)
                        t_local_list.append(t_local)
                n_valid_batch = len(b_idx_list)
                if n_valid_batch == 0:
                    continue
                b_idx = torch.tensor(b_idx_list, dtype=torch.long, device=device)
                t_local_t = torch.tensor(t_local_list, dtype=torch.long, device=device)

                X_bar = torch.zeros(n_valid_batch, V, dtype=torch.float32, device=device)
                for off in offsets:
                    X_bar.add_(X[b_idx, t_local_t + off, :], alpha=float(weights_lookup[off]))
                X_bar /= total_weight

                _accumulate_pair_quantities(
                    X_bar, t1_idx, t2_idx, unique_t1_idx,
                    ep_acc, es_acc, kl_acc, Z_acc, quantities)
                cursor += n_valid_batch
                if verbose:
                    pbar.set_postfix({'positions': cursor})

    assert cursor == n_valid_total, f"cursor {cursor} != n_valid_total {n_valid_total}"
    return _normalize_and_collect(ep_acc, es_acc, kl_acc, Z_acc, t1_idx, quantities)
