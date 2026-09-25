"""
Polysemy analysis: mixture decomposition of the token-conditional density.

The README/working_paper specify

    g(y | t) = Σ_k π_{t,k} · g_k(y | t)

with each component a *sense* of t, characterized by

    π_{t,k} — sense weight (relative frequency)
    ν_{t,k} = E_{g_k}[Y]              — sense centroid in ILR space
    Σ_{t,k} = Cov_{g_k}[Y]            — within-sense covariance in ILR space

and the law-of-total-variance decomposition

    Cov[Y | t] = Σ_k π_k Σ_k  +  Σ_k π_k (ν_k − μ_t)(ν_k − μ_t)^T
                 \________ within ________/  \________ between ________/

This module fits that mixture with weighted EM on the cached `viz_sample.H`,
and exposes utilities for variance decomposition and corpus retrieval.

Two field choices for the fitting weights:

  field='token_conditional':  w_i = p_t(h_i) / Z_t
      Importance weights for g(·|t). Components are the README's g_k(·|t)
      directly, and the LoTV decomposition gives Cov[Y|t] exactly.

  field='distinctiveness':    w_i = p_t(h_i) / (Z_t · ĝ(h_i))
      = exp(D_t(h_i)) / ĝ(h_i). Pulls components toward regions of high
      distinctiveness D_t(y) = log[g(y|t)/g(y)] regardless of mass — useful
      when a token has a "common-context" mode that dominates g(·|t) but
      tells you little about the token's distinct senses. ĝ is the trained
      flow density (`shape.density.FlowDensity`); a small floor prevents
      blowups in low-density tails. The mixture no longer sums to g(·|t),
      so LoTV has the cov-of-the-mixture interpretation, not Cov[Y|t].

Y-space conversion uses Y = A·h with A = Ψ·W (ILR ∘ softmax is affine in
h with no constant, since Ψ annihilates the logsumexp shift). Sense
covariances in Y-space are kept implicit — for COCA-scale V we hold the
factor `A · basis` (V−1 × d_fit) and the d_fit × d_fit `Sigma_h`, and only
materialize `Sigma_Y` when the user explicitly asks for the dense form.
"""

from dataclasses import dataclass, field as _field
from typing import Optional

import numpy as np
import torch

from shape.geometry import compute_A
from shape.viz import (
    _weighted_centered_basis,
    effective_sample_size,
    token_weights,
)


# ─────────────────────────────────────────────────────────────────────────────
# Field weights
# ─────────────────────────────────────────────────────────────────────────────

def sense_weights(viz_sample, W, t, Z_t, *, field, flow=None,
                  log_g_floor=-30.0):
    """Per-sample weights used by the weighted EM step.

    Args:
        viz_sample: a VizSample (provides H and log_Z_of_h).
        W: (V, d) tensor or array — `model.lm_head.weight`.
        t: integer token id.
        Z_t: marginal probability of token t (Stage 1's Z[t]).
        field: 'token_conditional' or 'distinctiveness'.
        flow: a `shape.density.FlowDensity` — required iff
            field='distinctiveness'. Its `log_density(H)` is evaluated to
            divide out g(h_i).
        log_g_floor: clamp on log ĝ(h_i) before division. Samples in the
            extreme tails of ĝ would otherwise dominate the weights.

    Returns:
        np.ndarray (N,) float32 — un-normalized non-negative weights.
    """
    omega = token_weights(viz_sample, W, t, Z_t, stabilize=False)
    if field == 'token_conditional':
        return omega
    if field == 'distinctiveness':
        if flow is None:
            raise ValueError(
                "field='distinctiveness' requires a flow (FlowDensity from "
                "shape.density.fit_flow); the per-sample weight needs ĝ(h_i)."
            )
        H_t = torch.from_numpy(np.asarray(viz_sample.H, dtype=np.float32))
        with torch.no_grad():
            log_g = flow.log_density(H_t).detach().cpu().numpy()
        log_g = np.maximum(log_g.astype(np.float64), log_g_floor)
        # ω_i / ĝ(h_i) = exp(log ω - log ĝ). Stabilize against overflow via
        # max-subtraction; the GMM is invariant to a global weight scale.
        log_w = np.log(np.maximum(omega.astype(np.float64), 1e-300)) - log_g
        log_w -= float(log_w.max())
        return np.exp(log_w).astype(np.float32)
    raise ValueError(
        f"field must be 'token_conditional' or 'distinctiveness', got {field!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Weighted GMM (EM)
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_kmeans_pp_init(H, weights, K, rng):
    """Weighted k-means++ seeding. Returns (K, d_fit) initial means."""
    N, d = H.shape
    w = np.asarray(weights, dtype=np.float64)
    w_total = float(w.sum())
    if w_total <= 0:
        raise ValueError("All weights non-positive in k-means++ init.")
    p = w / w_total
    first = int(rng.choice(N, p=p))
    centers = [H[first].copy()]
    sq_dists = ((H - centers[0]) ** 2).sum(axis=1)            # (N,)
    for _ in range(1, K):
        prob = w * sq_dists
        s = prob.sum()
        if s <= 0:
            # All remaining samples coincide with chosen centers; pick uniformly.
            idx = int(rng.integers(0, N))
        else:
            idx = int(rng.choice(N, p=prob / s))
        centers.append(H[idx].copy())
        new_sq = ((H - centers[-1]) ** 2).sum(axis=1)
        sq_dists = np.minimum(sq_dists, new_sq)
    return np.asarray(centers, dtype=np.float64)


def _gaussian_log_prob(H, mu, Sigma_chol):
    """log N(H; mu, Sigma) for one component.

    H: (N, d); mu: (d,); Sigma_chol: (d, d) lower-triangular Cholesky factor.
    Returns (N,) float64.
    """
    from scipy.linalg import solve_triangular
    d = H.shape[1]
    diff = H - mu[None, :]                                    # (N, d)
    sol = solve_triangular(Sigma_chol, diff.T, lower=True)    # (d, N)
    sq_mahal = (sol ** 2).sum(axis=0)                         # (N,)
    log_det = 2.0 * np.log(np.diag(Sigma_chol)).sum()
    return -0.5 * (d * np.log(2 * np.pi) + log_det + sq_mahal)


def _safe_cholesky(Sigma, reg, max_tries=5):
    """Try to Cholesky-factor; on failure, inflate the diagonal and retry."""
    d = Sigma.shape[0]
    for k in range(max_tries):
        try:
            return np.linalg.cholesky(Sigma + (reg * (10.0 ** k)) * np.eye(d))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError(
        f"Cholesky failed after {max_tries} tries (Sigma may be degenerate)."
    )


def _em_one(H, weights, K, *, reg_covar, max_iter, tol, init_means):
    """One EM run with given initial means. Returns dict with fit + log-lik."""
    N, d = H.shape
    w = np.asarray(weights, dtype=np.float64)
    w_total = float(w.sum())

    # Init: shared diagonal covariance from weighted variance, equal mixing.
    means = init_means.astype(np.float64).copy()              # (K, d)
    # Weighted global covariance as a starting Σ for every component
    mu_global = (w[:, None] * H).sum(axis=0) / w_total
    diff_g = H - mu_global[None, :]
    Sigma_global = (diff_g * w[:, None]).T @ diff_g / w_total
    Sigma_global = Sigma_global + reg_covar * np.eye(d)
    covs = np.repeat(Sigma_global[None, :, :], K, axis=0)     # (K, d, d)
    pi = np.full(K, 1.0 / K, dtype=np.float64)

    prev_ll = -np.inf
    log_lik = -np.inf
    for it in range(max_iter):
        # E-step: log_resp[i, k] = log π_k + log N(h_i | μ_k, Σ_k)
        log_resp = np.empty((N, K), dtype=np.float64)
        for k in range(K):
            L = _safe_cholesky(covs[k], reg_covar)
            log_resp[:, k] = np.log(pi[k] + 1e-300) + _gaussian_log_prob(H, means[k], L)
        # log Σ_k exp(...) per row
        m = log_resp.max(axis=1, keepdims=True)
        log_norm = m.squeeze(1) + np.log(np.exp(log_resp - m).sum(axis=1))
        resp = np.exp(log_resp - log_norm[:, None])           # (N, K)
        # weighted log-likelihood = Σ_i w_i · log Σ_k π_k N_k
        log_lik = float((w * log_norm).sum())

        # M-step
        wr = w[:, None] * resp                                # (N, K)
        Nk = wr.sum(axis=0)                                   # (K,)
        # avoid divide-by-zero on collapsed components
        Nk_safe = np.maximum(Nk, 1e-12)
        pi = Nk / w_total
        means = (wr.T @ H) / Nk_safe[:, None]                 # (K, d)
        for k in range(K):
            diff = H - means[k]                               # (N, d)
            covs[k] = (diff * wr[:, k:k + 1]).T @ diff / Nk_safe[k] \
                      + reg_covar * np.eye(d)

        if abs(log_lik - prev_ll) < tol * max(1.0, abs(prev_ll)):
            break
        prev_ll = log_lik

    return {
        'pi': pi.astype(np.float64),
        'means': means.astype(np.float64),
        'covs': covs.astype(np.float64),
        'responsibilities': resp.astype(np.float64),
        'log_lik': log_lik,
        'n_iter': it + 1,
    }


def _bic(log_lik, K, d, ess):
    """BIC for a K-component full-cov GMM in d dims, with weighted ESS."""
    n_params = (K - 1) + K * d + K * d * (d + 1) / 2.0
    return -2.0 * log_lik + n_params * np.log(max(ess, 2.0))


def fit_weighted_gmm(
    H,
    weights,
    *,
    K,
    K_max=8,
    reg_covar=1e-4,
    n_init=3,
    max_iter=200,
    tol=1e-4,
    project_pcs=None,
    rng=None,
    verbose=False,
):
    """Weighted-EM Gaussian mixture on (H, weights).

    Returns a dict:
        pi:               (K,) component weights
        means:            (K, d_fit) means in fit space
        covs:             (K, d_fit, d_fit) full covariances in fit space
        responsibilities: (N, K) soft assignments r_{ik}
        log_lik:          weighted log-likelihood at convergence
        bic:              BIC at the chosen K
        basis:            (d, d_fit) projection used (eye(d) if project_pcs is None)
        center:           (d,) shift subtracted before projection (zeros if no PCA)
        ess:              Kish ESS of `weights`
        meta:             dict with K_used, K_candidates, n_iter, history

    The component order is arbitrary; callers that care about identity should
    sort by π_k (see `SenseDecomposition.sort_by_weight`).
    """
    if rng is None:
        rng = np.random.default_rng()
    H_full = np.asarray(H, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != H_full.shape[0]:
        raise ValueError(f"weights ({w.shape}) and H ({H_full.shape}) length mismatch.")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and non-negative.")
    if w.sum() <= 0:
        raise ValueError("weights sum to zero — cannot fit GMM.")

    ess = effective_sample_size(w)
    N, d_h = H_full.shape

    # ── Project to weighted-PCA subspace if requested or auto-default ────────
    if project_pcs is None and d_h > 16:
        # Conservative default to keep full-cov well-conditioned at COCA-scale d.
        project_pcs = max(2, min(20, int(ess // 10)))
    if project_pcs is not None and project_pcs < d_h:
        mu_w, V_full, _ = _weighted_centered_basis(
            H_full.astype(np.float32), w.astype(np.float32), center=True)
        d_fit = int(project_pcs)
        basis = V_full[:, :d_fit].astype(np.float64)          # (d, d_fit)
        center = mu_w.astype(np.float64)                      # (d,)
        H_fit = (H_full - center[None, :]) @ basis            # (N, d_fit)
    else:
        d_fit = d_h
        basis = np.eye(d_h, dtype=np.float64)
        center = np.zeros(d_h, dtype=np.float64)
        H_fit = H_full

    # ── K selection ─────────────────────────────────────────────────────────
    if K == 'auto':
        K_candidates = list(range(1, int(K_max) + 1))
    else:
        K_candidates = [int(K)]

    best = None
    history = []
    for K_try in K_candidates:
        best_for_K = None
        for trial in range(n_init):
            if K_try == 1:
                # Closed form; no init dependence.
                init = (w[:, None] * H_fit).sum(axis=0, keepdims=True) / w.sum()
            else:
                init = _weighted_kmeans_pp_init(H_fit, w, K_try, rng)
            try:
                fit = _em_one(H_fit, w, K_try,
                              reg_covar=reg_covar, max_iter=max_iter,
                              tol=tol, init_means=init)
            except (np.linalg.LinAlgError, ValueError) as e:
                if verbose:
                    print(f"  K={K_try} trial={trial}: {type(e).__name__}: {e}")
                continue
            if best_for_K is None or fit['log_lik'] > best_for_K['log_lik']:
                best_for_K = fit
        if best_for_K is None:
            history.append({'K': K_try, 'log_lik': None, 'bic': None})
            continue
        bic = _bic(best_for_K['log_lik'], K_try, d_fit, ess)
        best_for_K['bic'] = bic
        best_for_K['K'] = K_try
        history.append({'K': K_try, 'log_lik': best_for_K['log_lik'], 'bic': bic})
        if verbose:
            print(f"  K={K_try}: log_lik={best_for_K['log_lik']:.2f}, "
                  f"bic={bic:.2f}, n_iter={best_for_K['n_iter']}")
        if best is None or bic < best['bic']:
            best = best_for_K

    if best is None:
        raise RuntimeError("All EM trials failed across all K candidates.")

    return {
        'pi': best['pi'],
        'means': best['means'],
        'covs': best['covs'],
        'responsibilities': best['responsibilities'],
        'log_lik': best['log_lik'],
        'bic': best['bic'],
        'basis': basis,
        'center': center,
        'ess': float(ess),
        'meta': {
            'K_used': int(best['K']),
            'K_candidates': K_candidates,
            'n_iter': int(best['n_iter']),
            'd_fit': int(d_fit),
            'd_h': int(d_h),
            'history': history,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sense decomposition (high-level API)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SenseDecomposition:
    pi: np.ndarray                              # (K,) sense weights π_k
    nu_h: np.ndarray                            # (K, d) centroids in h-space
    Sigma_h_fit: np.ndarray                     # (K, d_fit, d_fit) cov in fit space
    basis: np.ndarray                           # (d, d_fit) PCA basis used to fit
    center: np.ndarray                          # (d,) PCA center
    nu_Y: Optional[np.ndarray] = None           # (K, V-1) centroids in ILR space
    A_basis: Optional[np.ndarray] = None        # (V-1, d_fit) = A · basis (low-rank Sigma_Y factor)
    responsibilities: np.ndarray = _field(default_factory=lambda: np.empty(0))
    field: str = 'token_conditional'
    weights: Optional[np.ndarray] = None        # (N,) the per-sample weights used at fit
    diagnostics: dict = _field(default_factory=dict)

    @property
    def K(self):
        return int(self.pi.shape[0])

    def sort_by_weight(self, descending=True):
        """Return a new SenseDecomposition with components sorted by π_k."""
        order = np.argsort(self.pi)
        if descending:
            order = order[::-1]
        return SenseDecomposition(
            pi=self.pi[order],
            nu_h=self.nu_h[order],
            Sigma_h_fit=self.Sigma_h_fit[order],
            basis=self.basis,
            center=self.center,
            nu_Y=None if self.nu_Y is None else self.nu_Y[order],
            A_basis=self.A_basis,
            responsibilities=(self.responsibilities[:, order]
                              if self.responsibilities.size else self.responsibilities),
            field=self.field,
            weights=self.weights,
            diagnostics=self.diagnostics,
        )

    def Sigma_Y(self, k):
        """Materialize Σ_k in ILR space. (V-1) × (V-1) — only call when V is small."""
        if self.A_basis is None:
            raise ValueError("nu_Y / Sigma_Y not computed (compute_Y was False).")
        Ab = self.A_basis                                     # (V-1, d_fit)
        return Ab @ self.Sigma_h_fit[k] @ Ab.T


def fit_senses(
    viz_sample,
    W,
    t,
    Z_t,
    *,
    field='token_conditional',
    flow=None,
    K='auto',
    K_max=8,
    project_pcs=None,
    compute_Y=True,
    A=None,
    rng=None,
    verbose=False,
    **gmm_kwargs,
):
    """Fit the polysemy mixture for a single token.

    Args:
        viz_sample: VizSample from `shape.viz.build_viz_sample`.
        W: (V, d) tensor or array — `model.lm_head.weight`.
        t: integer token id.
        Z_t: marginal probability of t (Stage 1's Z[t]).
        field: 'token_conditional' or 'distinctiveness'. See module docstring.
        flow: `FlowDensity`, required iff field='distinctiveness'.
        K: int or 'auto' (BIC over 1..K_max).
        K_max: upper bound when K='auto'.
        project_pcs: int or None. Default (None) auto-picks
            min(20, ⌊ESS/10⌋) when d > 16; else uses full d.
        compute_Y: if True, also return ν_Y = A · ν_h. A (= ΨW) is built via
            `geometry.compute_A` unless passed in.
        A: optional precomputed (V-1, d) tensor or array, to avoid rebuilding
            it across many calls in the same notebook.
        rng: numpy Generator.
        gmm_kwargs: forwarded to `fit_weighted_gmm` (reg_covar, n_init,
            max_iter, tol).

    Returns:
        SenseDecomposition.
    """
    weights = sense_weights(viz_sample, W, t, Z_t, field=field, flow=flow)
    fit = fit_weighted_gmm(
        viz_sample.H, weights,
        K=K, K_max=K_max, project_pcs=project_pcs,
        rng=rng, verbose=verbose, **gmm_kwargs,
    )

    # Lift means from fit-space back to h-space: nu_h = center + basis @ mean
    nu_h = (fit['center'][None, :] + fit['means'] @ fit['basis'].T).astype(np.float32)

    nu_Y = None
    A_basis = None
    if compute_Y:
        if A is None:
            A = compute_A(W if torch.is_tensor(W) else torch.from_numpy(W).float())
        if torch.is_tensor(A):
            A_np = A.detach().cpu().float().numpy()
        else:
            A_np = np.asarray(A, dtype=np.float32)
        # Center cancels in mean shift if we also include it: ν_Y = A · ν_h.
        # (No constant c because Ψ annihilates the logsumexp shift, and we
        # define Y = Ψ · log softmax(Wh) directly.)
        nu_Y = (nu_h.astype(np.float64) @ A_np.T.astype(np.float64)).astype(np.float32)
        # Low-rank factor for Sigma_Y: A · basis ∈ ℝ^{(V-1) × d_fit}.
        A_basis = (A_np.astype(np.float64) @ fit['basis'].astype(np.float64)).astype(np.float32)

    diagnostics = {
        'ess': fit['ess'],
        'log_lik': fit['log_lik'],
        'bic': fit['bic'],
        'n_iter': fit['meta']['n_iter'],
        'K_used': fit['meta']['K_used'],
        'K_candidates': fit['meta']['K_candidates'],
        'd_fit': fit['meta']['d_fit'],
        'd_h': fit['meta']['d_h'],
        'history': fit['meta']['history'],
        'token_id': int(t),
        'Z_t': float(Z_t),
        'field': field,
    }

    return SenseDecomposition(
        pi=fit['pi'].astype(np.float32),
        nu_h=nu_h,
        Sigma_h_fit=fit['covs'].astype(np.float32),
        basis=fit['basis'].astype(np.float32),
        center=fit['center'].astype(np.float32),
        nu_Y=nu_Y,
        A_basis=A_basis,
        responsibilities=fit['responsibilities'].astype(np.float32),
        field=field,
        weights=weights,
        diagnostics=diagnostics,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Variance decomposition
# ─────────────────────────────────────────────────────────────────────────────

def total_variance_decomposition(decomp, *, return_traces_only=True,
                                 in_Y_space=True):
    """Law-of-total-variance split for the fitted mixture.

    For 'token_conditional' fits, `total = Cov[Y|t]` exactly (the README's
    decomposition). For 'distinctiveness' fits, `total` is the covariance of
    the distinctiveness-landscape mixture rather than of g(·|t) — same
    arithmetic, different denotation.

    μ = Σ_k π_k ν_k is recovered from `decomp` itself; no external embedding
    input is needed.

    If `return_traces_only` (default), returns scalar traces — enough to
    quantify "within vs between" without materializing (V-1)² matrices.
    Trace identity tr(total) = tr(within) + tr(between) holds exactly.

    If `in_Y_space=True` (default) and `decomp.A_basis` is available, traces
    are computed in ILR space (the README's Σ_{t,k}); otherwise they use
    h-space. The trace ratio is unitless in either case.

    Returns dict with keys (traces always; matrices only if not traces_only):
        within_trace, between_trace, total_trace, within_fraction, between_fraction,
        within (optional), between (optional), total (optional),
        mu (the mixture-mean ν, in whichever space).
    """
    pi = decomp.pi.astype(np.float64)                         # (K,)
    Sigma_fit = decomp.Sigma_h_fit.astype(np.float64)         # (K, d_fit, d_fit)
    K, d_fit, _ = Sigma_fit.shape

    # μ in the requested space
    if in_Y_space:
        if decomp.nu_Y is None or decomp.A_basis is None:
            raise ValueError(
                "Y-space requested but decomp.nu_Y/A_basis is None "
                "(was compute_Y=False?). Pass in_Y_space=False or refit with "
                "compute_Y=True.")
        nu = decomp.nu_Y.astype(np.float64)                   # (K, V-1)
        # Σ_Y per component = (A·basis) · Sigma_fit · (A·basis)^T
        # but we never need it dense for the trace; trace(M Σ M^T) = trace(M^T M Σ).
        Ab = decomp.A_basis.astype(np.float64)                # (V-1, d_fit)
        AtA = Ab.T @ Ab                                        # (d_fit, d_fit)
    else:
        nu = decomp.nu_h.astype(np.float64)                    # (K, d)
        # Sigma in fit-space; the basis projection is unitary so the trace is
        # unchanged when going basis -> h (we just compare apples to apples
        # in fit space). For the within-trace, work directly in fit-space.
        AtA = None

    mu = (pi[:, None] * nu).sum(axis=0)                       # (V-1,) or (d,)

    # within: Σ_k π_k tr(Σ_k^Y) — for Y-space, tr(Σ_Y) = tr(AtA · Sigma_fit).
    if in_Y_space:
        within_trace = float(sum(pi[k] * np.trace(AtA @ Sigma_fit[k]) for k in range(K)))
    else:
        # If we projected to fit-space, the unprojected within would underestimate;
        # but for a rotation-only basis (PCA without throwing components away in
        # a meaningful sense) this is the trace of the projected within. Document
        # the asymmetry: callers who want full h-space can pass project_pcs=None.
        within_trace = float(sum(pi[k] * np.trace(Sigma_fit[k]) for k in range(K)))

    # between: Σ_k π_k ‖ν_k - μ‖²
    diffs = nu - mu[None, :]                                  # (K, dim)
    between_trace = float((pi * (diffs ** 2).sum(axis=1)).sum())

    total_trace = within_trace + between_trace
    out = {
        'within_trace': within_trace,
        'between_trace': between_trace,
        'total_trace': total_trace,
        'within_fraction': within_trace / total_trace if total_trace > 0 else float('nan'),
        'between_fraction': between_trace / total_trace if total_trace > 0 else float('nan'),
        'mu': mu.astype(np.float32),
    }

    if not return_traces_only:
        # Materialize full matrices. Only tractable for small V/d.
        if in_Y_space:
            within = sum(pi[k] * (Ab @ Sigma_fit[k] @ Ab.T) for k in range(K))
        else:
            within = sum(pi[k] * Sigma_fit[k] for k in range(K))
        between = sum(pi[k] * np.outer(diffs[k], diffs[k]) for k in range(K))
        out['within'] = np.asarray(within, dtype=np.float32)
        out['between'] = np.asarray(between, dtype=np.float32)
        out['total'] = (out['within'] + out['between']).astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sense interpretation: corpus exemplars
# ─────────────────────────────────────────────────────────────────────────────

def sense_examples(
    decomp,
    viz_sample,
    extract_meta,
    data,
    *,
    n_per_sense=10,
    context=8,
    decode=None,
    rank_by='weighted_responsibility',
    filter_next_token=False,
    token_id=None,
):
    """Top-N corpus instances per sense, with decoded surrounding text.

    Each sense k has soft assignments r_{ik} from the EM fit. We rank cached
    h samples by a per-sense score and decode `±context` tokens of corpus
    text around each chosen position.

    Args:
        decomp: SenseDecomposition.
        viz_sample: the same VizSample passed to `fit_senses` — for
            `viz_sample.row_index` mapping back to corpus positions.
        extract_meta: dict from `{name}_meta.json`.
        data: corpus memmap/array (uint16) used in Stage 1.
        n_per_sense: number of corpus examples per sense.
        context: number of tokens of context on each side of the focus token.
        decode: callable list[int] -> str (e.g. from `viz.load_decode`).
            Defaults to space-separated ids if None.
        rank_by: 'weighted_responsibility' (default), 'responsibility', or
            'distance'. See module docstring for trade-offs.
        filter_next_token: if True, restrict to positions whose next corpus
            token equals `token_id` (mirrors viz.sample_token_instances's
            empirical view). Off by default so rare tokens still surface
            enough examples; `decomp.diagnostics['token_id']` is used when
            `token_id` is None.
        token_id: token to filter on; only consulted when filter_next_token.

    Returns:
        pandas.DataFrame with columns
        sense_id, rank, score, corpus_pos, context_before, focus, context_after.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("sense_examples requires pandas.") from e
    from shape.extract import valid_positions

    if viz_sample.row_index is None or viz_sample.row_index.size == 0:
        raise ValueError(
            "viz_sample.row_index is empty — needed to map cached H rows "
            "to corpus positions. Rebuild with a recent build_viz_sample."
        )
    positions = valid_positions(extract_meta)
    if positions.size == 0:
        raise ValueError("extract_meta yielded no valid positions.")

    H_to_corpus = positions[viz_sample.row_index]              # (N,) corpus positions
    N = H_to_corpus.shape[0]
    K = decomp.K
    resp = np.asarray(decomp.responsibilities, dtype=np.float64)
    if resp.shape != (N, K):
        raise ValueError(
            f"responsibilities shape {resp.shape} doesn't match "
            f"(viz_sample.N, K)=({N}, {K})."
        )

    if filter_next_token:
        if token_id is None:
            token_id = decomp.diagnostics.get('token_id')
            if token_id is None:
                raise ValueError(
                    "filter_next_token=True but token_id not provided and "
                    "decomp.diagnostics has no 'token_id'."
                )
        T = len(data)
        target_pos = H_to_corpus + 1
        in_range = target_pos < T
        next_tok = np.full(N, -1, dtype=np.int64)
        next_tok[in_range] = np.asarray(data[target_pos[in_range]], dtype=np.int64)
        eligible = next_tok == int(token_id)
    else:
        eligible = np.ones(N, dtype=bool)

    if decode is None:
        decode = lambda ids: ' '.join(str(int(i)) for i in ids)

    if rank_by == 'distance':
        # Per-sense Euclidean distance from h_i (in fit space) to ν_k (in fit space).
        H = np.asarray(viz_sample.H, dtype=np.float64)
        H_fit = (H - decomp.center.astype(np.float64)[None, :]) \
            @ decomp.basis.astype(np.float64)                  # (N, d_fit)
        # means in fit-space: undo the lift used in fit_senses.
        # nu_h = center + basis @ mean  ⇒  mean = basis^T (nu_h - center) since basis is orthonormal.
        mean_fit = (decomp.nu_h.astype(np.float64) - decomp.center.astype(np.float64)[None, :]) \
            @ decomp.basis.astype(np.float64)                  # (K, d_fit)
        # score[i, k] = -‖h_i - μ_k‖ so that bigger = closer (matches the
        # "argmax score" convention of the other rank_by modes).
        score = -np.sqrt(((H_fit[:, None, :] - mean_fit[None, :, :]) ** 2).sum(axis=-1))
    elif rank_by == 'responsibility':
        score = resp                                            # (N, K)
    elif rank_by == 'weighted_responsibility':
        if decomp.weights is None:
            raise ValueError(
                "decomp.weights is None; cannot use rank_by='weighted_responsibility'. "
                "Refit, or pass rank_by='responsibility'/'distance'."
            )
        score = decomp.weights.astype(np.float64)[:, None] * resp
    else:
        raise ValueError(
            f"rank_by must be 'weighted_responsibility', 'responsibility', "
            f"or 'distance'; got {rank_by!r}"
        )

    rows = []
    T = len(data)
    eligible_idx = np.flatnonzero(eligible)
    if eligible_idx.size == 0:
        return pd.DataFrame(columns=['sense_id', 'rank', 'score', 'corpus_pos',
                                     'context_before', 'focus', 'context_after'])
    for k in range(K):
        sk = score[eligible_idx, k]
        order = np.argsort(sk)[::-1][:n_per_sense]
        chosen_local = eligible_idx[order]
        for r, i in enumerate(chosen_local):
            pos = int(H_to_corpus[i])
            focus = int(data[pos + 1]) if (pos + 1 < T) else -1
            before_start = max(0, pos + 1 - context)
            after_end = min(T, pos + 2 + context)
            ids_before = np.asarray(data[before_start:pos + 1], dtype=np.int64).tolist()
            ids_after = (np.asarray(data[pos + 2:after_end], dtype=np.int64).tolist()
                         if pos + 2 <= T else [])
            rows.append({
                'sense_id': int(k),
                'rank': int(r),
                'score': float(sk[order[r]]),
                'corpus_pos': pos,
                'context_before': decode(ids_before) if ids_before else '',
                'focus': decode([focus]) if focus >= 0 else '',
                'context_after': decode(ids_after) if ids_after else '',
            })
    return pd.DataFrame(rows)
