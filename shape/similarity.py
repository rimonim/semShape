"""
Pairwise semantic similarity quantities between token pairs.

Implements three probability-theoretic quantities (working paper §3):

    expected_probability:  E_{Y|t1}[ p_{t2}(Y) ]
    expected_surprisal:   -E_{Y|t1}[ log p_{t2}(Y) ]
    kl_divergence:         E_{Y|t1}[ log(p_{t1}(Y) / p_{t2}(Y)) ]

All use importance-weighted sampling from the corpus:

    E_{Y|t1}[f(Y)] ≈ Σ_i p_{t1}(Y_i) f(Y_i) / Σ_i p_{t1}(Y_i)

Two entry points:

    compute_pairwise_similarities            — stored samples from
        shape.extract.extract_features, either hidden states (pass W) or
        probability vectors
    compute_pairwise_similarities_streaming  — live corpus pass with window
        averaging; nothing is written to disk
"""

import numpy as np
import torch

from shape.samples import iter_sample_probs, open_samples, sample_format

_LOG_EPS = 1e-30

_VALID_QUANTITIES = frozenset(
    ('expected_probability', 'expected_surprisal', 'kl_divergence')
)


def _check_pairs(t1_ids, t2_ids, quantities):
    quantities = set(quantities)
    unknown = quantities - _VALID_QUANTITIES
    if unknown:
        raise ValueError(f"Unknown quantities: {unknown}")
    t1_ids = np.asarray(t1_ids, dtype=np.int64)
    t2_ids = np.asarray(t2_ids, dtype=np.int64)
    assert t1_ids.shape == t2_ids.shape and t1_ids.ndim == 1
    return t1_ids, t2_ids, quantities


def _similarities_from_batches(prob_batches, t1_ids, t2_ids, V, quantities, device):
    """Run the importance-weighted estimator over an iterable of (B, V) batches."""
    P = len(t1_ids)
    t1_idx = torch.from_numpy(t1_ids).to(device)
    t2_idx = torch.from_numpy(t2_ids).to(device)

    ep_acc = torch.zeros(P, dtype=torch.float64, device=device)
    es_acc = torch.zeros(P, dtype=torch.float64, device=device)
    kl_acc = torch.zeros(P, dtype=torch.float64, device=device)
    Z_acc  = torch.zeros(V, dtype=torch.float64, device=device)

    for X in prob_batches:
        _accumulate_pair_quantities(
            X, t1_idx, t2_idx, ep_acc, es_acc, kl_acc, Z_acc, quantities)

    return _normalize_and_collect(ep_acc, es_acc, kl_acc, Z_acc, t1_idx, t2_idx, quantities)


def _accumulate_pair_quantities(X, t1_idx, t2_idx,
                                ep_acc, es_acc, kl_acc, Z_acc, quantities):
    """
    Update running accumulators in-place given a batch of probability vectors.

    Args:
        X: (B, V) float32 tensor — probability distributions at B positions.
        t1_idx: (P,) long tensor — token ids for the "query" of each pair.
        t2_idx: (P,) long tensor — token ids for the "target" of each pair.
        ep_acc, es_acc, kl_acc: (P,) float64 accumulators (mutated).
        Z_acc: (V,) float64 accumulator for importance-weight sums (mutated).
        quantities: set of str.
    """
    X1 = X[:, t1_idx]   # (B, P)
    X2 = X[:, t2_idx]   # (B, P)

    # Z_acc over the full vocabulary: the KL base-rate correction needs Z[t2]
    # as well as Z[t1].
    Z_acc.add_(X.to(torch.float64).sum(dim=0))

    if 'expected_probability' in quantities:
        ep_acc.add_((X1 * X2).to(torch.float64).sum(dim=0))

    if 'expected_surprisal' in quantities or 'kl_divergence' in quantities:
        log_X2 = torch.log(X2.clamp(min=_LOG_EPS))
        if 'expected_surprisal' in quantities:
            es_acc.add_((X1 * (-log_X2)).to(torch.float64).sum(dim=0))
        if 'kl_divergence' in quantities:
            log_X1 = torch.log(X1.clamp(min=_LOG_EPS))
            kl_acc.add_((X1 * (log_X1 - log_X2)).to(torch.float64).sum(dim=0))


def _normalize_and_collect(ep_acc, es_acc, kl_acc, Z_acc, t1_ids_tensor, t2_ids_tensor,
                           quantities):
    Z_t1 = Z_acc[t1_ids_tensor].clamp(min=_LOG_EPS)
    result = {}
    if 'expected_probability' in quantities:
        result['expected_probability'] = (ep_acc / Z_t1).cpu().numpy()
    if 'expected_surprisal' in quantities:
        result['expected_surprisal'] = (es_acc / Z_t1).cpu().numpy()
    if 'kl_divergence' in quantities:
        # E_{Y|t1}[log p(t1|Y)/p(t2|Y)] estimates KL(p_{t1}||p_{t2}) only up to a
        # base-rate correction: KL = estimate + log(Z[t2]/Z[t1]).
        Z_t2 = Z_acc[t2_ids_tensor].clamp(min=_LOG_EPS)
        kl = kl_acc / Z_t1 + (torch.log(Z_t2) - torch.log(Z_t1))
        result['kl_divergence'] = kl.cpu().numpy()
    return result


def compute_pairwise_similarities(
    samples,
    t1_ids,
    t2_ids,
    *,
    W=None,
    quantities=('expected_probability', 'expected_surprisal', 'kl_divergence'),
    batch_size=2048,
    device='cuda',
    verbose=True,
):
    """
    Compute pairwise similarity quantities from stored corpus samples.

    For each pair (t1_ids[i], t2_ids[i]), computes importance-weighted
    expectations E_{Y|t1}[f(Y)] using the stored predictive states as the
    sample of Y's, with importance weights p_{t1}(Y_j).

    Args:
        samples: path to a .npy memmap or an ndarray, either (N, d) hidden
            states or (N, V) probability vectors (e.g. `{name}_h.npy` or
            `{name}_probs.npy` from extract_features).
        t1_ids: (P,) array-like of token ids for the query token.
        t2_ids: (P,) array-like of token ids for the target token.
        W: (V, d) tensor — model.lm_head.weight. Required for hidden-state
            samples; the format is inferred from the number of columns.
        quantities: iterable of strings from
            {'expected_probability', 'expected_surprisal', 'kl_divergence'}.
        batch_size: sample rows per device batch.
        device: torch device string.
        verbose: show tqdm progress bar.

    Returns:
        dict mapping each requested quantity name to a float64 ndarray of
        shape (P,).
    """
    t1_ids, t2_ids, quantities = _check_pairs(t1_ids, t2_ids, quantities)
    arr = open_samples(samples)
    V = W.shape[0] if sample_format(arr, W) == 'h' else arr.shape[1]
    batches = iter_sample_probs(arr, W, batch_size=batch_size, device=device,
                                verbose=verbose, desc="similarity")
    return _similarities_from_batches(batches, t1_ids, t2_ids, V, quantities, device)


def compute_pairwise_similarities_streaming(
    model,
    data,
    weights_lookup,
    t1_ids,
    t2_ids,
    *,
    averaging='aitchison',
    quantities=('expected_probability', 'expected_surprisal', 'kl_divergence'),
    block_size=None,
    min_context=32,
    batch_size=32,
    device='cuda',
    compute_dtype='bfloat16',
    verbose=True,
):
    """
    Compute pairwise similarity quantities via a streaming corpus pass.

    Each valid corpus position contributes its window-averaged predictive state
    (see shape.extract.iter_window_states) as a sampled Y. Equivalent to
    extract_features followed by compute_pairwise_similarities, without
    writing samples to disk.

    Args:
        model: GPT in eval mode on `device`.
        data: np.memmap or ndarray of corpus tokens (uint16).
        weights_lookup: {offset_d: weight} dict from build_weight_lookup.
        t1_ids: (P,) array-like of token ids for the query token.
        t2_ids: (P,) array-like of token ids for the target token.
        averaging: 'aitchison' (average hidden states) or 'probability'
            (average distributions). Irrelevant for a single-offset window.
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
    from shape.extract import iter_window_states

    t1_ids, t2_ids, quantities = _check_pairs(t1_ids, t2_ids, quantities)
    batches = (probs for _, probs in iter_window_states(
        model, data, weights_lookup,
        averaging=averaging, block_size=block_size, min_context=min_context,
        batch_size=batch_size, device=device, compute_dtype=compute_dtype,
        verbose=verbose, desc="similarity-stream"))
    return _similarities_from_batches(batches, t1_ids, t2_ids,
                                      model.config.vocab_size, quantities, device)


def compute_token_marginals(
    samples,
    *,
    W=None,
    batch_size=2048,
    device='cuda',
    verbose=True,
):
    """
    Compute per-token marginal probability sums Z[t] = Σ_i p(t|Y_i) over
    stored corpus samples.

    Z is proportional to the corpus marginal probability of each token and is
    the denominator used by the importance-weighted estimator.  Saving Z
    separately lets you apply the base-rate correction
        KL(p_{t1} || p_{t2}) = raw_estimate + log(Z[t2] / Z[t1])
    to previously computed similarity outputs without rerunning the corpus pass.
    (extract_features also writes the normalized version, Z / N, as `{name}_Z.npy`.)

    Args:
        samples: path to a .npy memmap or an ndarray of (N, d) hidden states
            or (N, V) probability vectors.
        W: (V, d) tensor — model.lm_head.weight. Required for hidden states.
        batch_size: sample rows per device batch.
        device: torch device string.
        verbose: show tqdm progress bar.

    Returns:
        (V,) float64 ndarray of per-token marginal sums.
    """
    Z_acc = None
    for X in iter_sample_probs(samples, W, batch_size=batch_size, device=device,
                               verbose=verbose, desc="marginals"):
        if Z_acc is None:
            Z_acc = torch.zeros(X.shape[-1], dtype=torch.float64, device=device)
        Z_acc.add_(X.to(torch.float64).sum(dim=0))
    return Z_acc.cpu().numpy()
