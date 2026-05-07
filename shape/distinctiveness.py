"""
Distinctiveness field D_t(y) and expected D_t = D_KL(g(·|t) ‖ g(·)).

The distinctiveness field is

    D_t(y) = log[g(y|t) / g(y)]

— how much more (or less) likely the meaning region y is, given that the
token is t, compared to the language as a whole. Its expected value under
g(·|t) is the KL divergence

    D_t = D_KL(g(·|t) ‖ g(·)).

In h-space, g(h|t) ∝ p_t(h) · g(h), so

    D_t(h) = log p_t(h) − log Z_t
           = W[t]·h − log_Z_of_h − log Z_t.

This is exactly log(ω_i) for the unstabilized importance weights
ω_i = p_t(h_i)/Z_t computed in `shape.viz.token_weights`. D_t is invariant
under change of variable h → y (the log-Jacobian cancels in the ratio), so
the same per-sample scalar serves both `D_t(y)` for plotting in any 2D
projection and `D_t` (the KL).

Per-sample evaluation is O(d) using `viz_sample.log_Z_of_h`; vocabulary-wide
evaluation is one (B × d) @ (d × N) matmul per batch of B tokens.
"""

import numpy as np
import torch
from tqdm import tqdm


def distinctiveness_field(viz_sample, W, t, Z_t):
    """
    Per-sample D_t(h_i) = log p_t(h_i) − log Z_t for each cached h_i.

    Returns the *unstabilized* log-distinctiveness in nats — i.e. the
    constant log Z_t is included, so signs and zero-crossings are
    meaningful. Equivalent to log of `token_weights(..., stabilize=False)`.

    Args:
        viz_sample: a VizSample.
        W: tensor or array of shape (V, d) — `model.lm_head.weight`.
        t: integer token id.
        Z_t: marginal probability of token t (from Stage 1's Z vector).

    Returns:
        D: np.ndarray of shape (N,), float32. D_t(h_i) in nats.
    """
    if Z_t <= 0.0:
        raise ValueError(
            f"Z_t must be positive (got {Z_t}); token never appears as the "
            f"conditional target."
        )
    if isinstance(W, np.ndarray):
        W_t = np.asarray(W[t], dtype=np.float32)
    else:
        W_t = W[t].detach().cpu().float().numpy()
    H = np.asarray(viz_sample.H, dtype=np.float32)
    log_p_t = H @ W_t - viz_sample.log_Z_of_h            # log p_t(h_i)
    return (log_p_t - float(np.log(Z_t))).astype(np.float32)


def _kl_from_log_omega(log_omega):
    """D_t = Σ ω log ω / Σ ω with ω = exp(log_omega), via log-sum-exp."""
    lo = np.asarray(log_omega, dtype=np.float64)
    m = float(lo.max())
    e = np.exp(lo - m)
    sum_e = float(e.sum())
    if sum_e <= 0.0:
        return 0.0, 0.0, 0.0
    # Σ ω log ω = Σ exp(lo)·lo = exp(m) · Σ e·lo
    D_t = float((e * lo).sum() / sum_e)
    ess = float(sum_e * sum_e / (e * e).sum())          # m cancels in ratio
    sum_omega = float(np.exp(m) * sum_e)
    return D_t, ess, sum_omega


def expected_distinctiveness(viz_sample, W, t, Z_t):
    """
    D_t = D_KL(g(·|t) ‖ g(·)) by importance sampling on the cached H.

    Since H is a sample of g(·), we have

        D_t = E_{g(·|t)}[D_t(h)] ≈ Σ_i ω_i · log ω_i / Σ_i ω_i

    with ω_i = exp(D_t(h_i)) = p_t(h_i)/Z_t (true magnitudes — no
    stabilization). KL ≥ 0 in expectation; sample estimates can dip slightly
    below zero on small samples and are noisy when ESS is small.

    Returns:
        dict with keys
          - 'D_t': float — KL in nats
          - 'ess': float — Kish ESS of ω (same diagnostic as the plot helpers)
          - 'sum_omega': float — Σ ω, an unbiased Monte Carlo estimate of 1
            (since E_g[p_t/Z_t] = 1). Big departures from 1 mean the cached
            sample doesn't well cover g(·|t).
    """
    D = distinctiveness_field(viz_sample, W, t, Z_t)
    D_t, ess, sum_omega = _kl_from_log_omega(D)
    # sum_omega here = Σ ω, but the importance-sampling estimate of 1 is
    # (1/N) Σ ω; report the latter so the user can read it as "should be ≈ 1".
    n = float(D.shape[0])
    return {'D_t': D_t, 'ess': ess, 'sum_omega': sum_omega / n if n > 0 else 0.0}


def expected_distinctiveness_all(viz_sample, W, Z, *,
                                 batch_size=1024,
                                 device='cpu',
                                 target_tokens=None,
                                 verbose=False):
    """
    Vocabulary-wide D_t. Vectorizes `expected_distinctiveness` over all
    (or a subset of) tokens by batching B rows of W at a time.

    For each token t in `target_tokens`:
        D[t] = D_KL(g(·|t) ‖ g(·))     (importance sampling on cached H)
        ess[t] = Kish ESS of ω_t

    Tokens with Z[t] ≤ 0 (never observed in Stage 1) get D=NaN and ess=0.

    Args:
        viz_sample: a VizSample.
        W: (V, d) tensor or array — `model.lm_head.weight`.
        Z: (V,) array of marginal token probabilities.
        batch_size: number of token rows of W to process per batch. The
            transient memory is `B × N × 8` bytes for the float64 reduction.
        device: torch device for the matmul ('cpu' or e.g. 'cuda').
        target_tokens: optional iterable of token ids to evaluate; default
            (None) is the full vocabulary.
        verbose: if True, show a tqdm progress bar.

    Returns:
        dict with keys
          - 'D': (T,) float32 array — D_t in nats, indexed by `target_tokens`
          - 'ess': (T,) float32
          - 'token_ids': (T,) int64 — the ids these entries correspond to
    """
    H = torch.from_numpy(np.asarray(viz_sample.H, dtype=np.float32))\
            .to(device).float()
    log_Z_h = torch.from_numpy(
        np.asarray(viz_sample.log_Z_of_h, dtype=np.float32)
    ).to(device).float()
    if isinstance(W, np.ndarray):
        W_full = torch.from_numpy(W).to(device).float()
    else:
        W_full = W.detach().to(device).float()
    V, _ = W_full.shape

    if target_tokens is None:
        token_ids = np.arange(V, dtype=np.int64)
    else:
        token_ids = np.asarray(list(target_tokens), dtype=np.int64)

    Z_np = np.asarray(Z, dtype=np.float64)
    Z_sub = Z_np[token_ids]
    valid = Z_sub > 0

    D_out = np.full(len(token_ids), np.nan, dtype=np.float32)
    ess_out = np.zeros(len(token_ids), dtype=np.float32)

    iter_obj = range(0, len(token_ids), batch_size)
    if verbose:
        iter_obj = tqdm(iter_obj, desc='D_t', unit='batch')

    for start in iter_obj:
        end = min(start + batch_size, len(token_ids))
        ids_chunk = token_ids[start:end]
        v_chunk = valid[start:end]
        if not v_chunk.any():
            continue
        ids_v = ids_chunk[v_chunk]
        z_v = Z_sub[start:end][v_chunk]

        Wb = W_full[torch.from_numpy(ids_v).to(device)]        # (B, d)
        log_p = Wb @ H.T - log_Z_h[None, :]                    # (B, N)
        log_Zt = torch.from_numpy(np.log(z_v)).to(device).float()
        D = log_p - log_Zt[:, None]                            # (B, N)

        # Per-row reduce in float64: D_t = Σ exp(D)·D / Σ exp(D), ESS via Kish.
        D64 = D.double()
        m = D64.max(dim=1, keepdim=True).values
        e = torch.exp(D64 - m)
        sum_e = e.sum(dim=1)
        num = (e * D64).sum(dim=1)
        D_t_chunk = (num / sum_e).float().cpu().numpy()
        ess_chunk = (sum_e * sum_e / (e * e).sum(dim=1)).float().cpu().numpy()

        out_idx = np.flatnonzero(v_chunk) + start
        D_out[out_idx] = D_t_chunk
        ess_out[out_idx] = ess_chunk

    return {'D': D_out, 'ess': ess_out, 'token_ids': token_ids}
