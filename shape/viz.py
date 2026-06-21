"""
Stage 2.5: polysemy visualization.

After Stage 1 writes the h_eff memmap, we materialize a working-size sample
H ∈ ℝ^{N×d} by uniformly subsampling h_eff rows, and cache it together with:
  - log_Z_of_h[i] = logsumexp(W h_i)   — per-sample full-vocab log-normalizer,
    so p_t(h_i) = exp(W[t]·h_i − log_Z_of_h[i]) is an O(d) query per token.
  - row_index ∈ ℝ^{N}                   — the h_eff memmap rows that H holds.

An uncentered SVD (V_basis, S, projections) is only required by the global
basis path; it is deferred by default and computed lazily on first use.
Pass `compute_svd=True` to `build_viz_sample` if you want it up-front.

Public plotting API (three functions):

  plot_global_density(viz_sample, pcs, ...)
      Heatmap of the unconditional empirical density g(·). pcs is a list of
      PC indices; all non-diagonal (i, j) pairs are shown in a matrix layout
      with columns sharing the x-axis and rows sharing the y-axis.

  plot_density(viz_sample, W, Z, tokens, *, ...)
      Filled-bands overlay of g(·|t) for one or more tokens. Single token →
      basis='token' (per-token centered PCA) by default; multiple tokens →
      basis='global'. Accepts token ids (int) or strings (str, requires stoi).
      Z is the full (V,) marginals array. examples can be a DataFrame, an int
      k (random corpus sample), or a list of phrase strings to search for.

  plot_distinctiveness(viz_sample, W, Z, token, *, ...)
      Diverging log-ratio heatmap D_t(y) = log[g(y|t)/g(y)] for a single
      token. Same Z-array and string-token conveniences as plot_density.

Both views support three bases:

  basis='token' (default for single token): weighted, centered PCA of the
    reweighted sample on the fly. Surfaces the principal directions of
    polysemy for that token; axes non-comparable across tokens.

  basis='global': cached global (uncentered) V_basis. Axes line up across
    plots — good for cross-token comparison.

  basis='contrast': user-specified token-pair contrasts. Each axis projects
    h onto W[pos] − W[neg], giving the log-odds log p(pos|h)/p(neg|h).
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class VizSample:
    H: np.ndarray                 # (N, d) float32 — h_eff rows in h_eff-space
    log_Z_of_h: np.ndarray        # (N,)   float32 — logsumexp(W h_i)
    V_basis: np.ndarray           # (d, d) float32 — right singular vectors (columns)
    S: np.ndarray                 # (d,)   float32 — singular values of H (uncentered)
    projections: np.ndarray       # (N, k_pc) float32 — H @ V_basis[:, :k_pc]
    row_index: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    meta: dict = field(default_factory=dict)

    @property
    def N(self):
        return self.H.shape[0]

    @property
    def d(self):
        return self.H.shape[1]

    @property
    def k_pc(self):
        return self.projections.shape[1]

    @property
    def has_svd(self):
        return self.V_basis.size > 0


@dataclass
class CorpusContext:
    """Bundles the three corpus-access objects used together for example extraction."""
    data: np.ndarray        # (T,) uint16 — corpus token ids
    h_eff: np.ndarray       # (N_valid, d) float32 — Stage 1 h_eff memmap
    extract_meta: dict      # loaded from {dataset}_meta.json


def _svd_and_project(H, k_pc, *, verbose=False):
    """Uncentered SVD of H (N×d), return (V_basis, S, projections[:, :k_pc])."""
    N, d = H.shape
    if verbose:
        print(f"Uncentered SVD of H ({N}×{d})...")
    _, S, Vt = np.linalg.svd(H.astype(np.float64), full_matrices=False)
    V_basis = Vt.T.astype(np.float32)
    S = S.astype(np.float32)
    k_pc = min(k_pc, d)
    projections = (H @ V_basis[:, :k_pc]).astype(np.float32)
    return V_basis, S, projections


def _ensure_svd(viz_sample, *, k_pc=None, verbose=False):
    """Compute and cache the uncentered SVD on `viz_sample.H` in-place if
    not yet done. Idempotent."""
    if viz_sample.has_svd:
        return
    if k_pc is None:
        k_pc = int(viz_sample.meta.get('k_pc_default',
                                       min(viz_sample.d, 16)))
    V_basis, S, projections = _svd_and_project(viz_sample.H, k_pc,
                                               verbose=verbose)
    viz_sample.V_basis = V_basis
    viz_sample.S = S
    viz_sample.projections = projections


def build_viz_sample(
    extract_dir,
    dataset_name,
    W,
    out_path,
    *,
    N=200_000,
    k_pc=None,
    compute_svd=False,
    logsumexp_batch_size=4096,
    device='cpu',
    seed=1337,
    verbose=True,
    extra_meta=None,
):
    """
    Build a working-size cache for visualization from the Stage 1 h_eff memmap.

    Uniformly subsamples N rows (without replacement) from
    `{extract_dir}/{dataset_name}_h_eff.npy` and computes log Z(h_i) for each.
    The uncentered SVD is only needed by `basis='global'` and
    `plot_global_density`; it is deferred by default and computed on first
    use (and mutated into the VizSample so subsequent calls hit the cache).

    Writes a single .npz at `out_path`.

    Args:
        extract_dir: directory containing Stage 1 outputs.
        dataset_name: Stage 1 dataset name; files
            `{dataset_name}_h_eff.npy` and `{dataset_name}_meta.json`
            are read from `extract_dir`.
        W: tensor of shape (V, d) — `model.lm_head.weight`.
        out_path: destination .npz path.
        N: number of h_eff rows to subsample. If None or >= n_valid, uses
            the full memmap (watch RAM: N × d × 4 bytes).
        k_pc: number of principal components to precompute projections for
              (default: min(d, 16)). Only consulted when `compute_svd=True`
              or when SVD is later triggered lazily.
        compute_svd: if True, do the SVD now. Leave False when you only plan
              to use `basis='token'` or `basis='contrast'` — both avoid the
              global basis entirely, and the SVD is ~O(Nd²).
        logsumexp_batch_size: chunk size for batched logsumexp over V
              (memory is `chunk × V × 4` bytes).
        seed: subsampling RNG seed.

    Returns:
        VizSample (also written to disk).
    """
    h_eff_path = os.path.join(extract_dir, f"{dataset_name}_h_eff.npy")
    h_eff_mm = np.load(h_eff_path, mmap_mode='r')
    n_valid, d = h_eff_mm.shape
    W_t = W.detach().to(device).float()
    V, d_W = W_t.shape
    if d_W != d:
        raise ValueError(
            f"W has d={d_W} but h_eff memmap has d={d} — mismatched checkpoints?"
        )

    if k_pc is None:
        k_pc = min(d, 16)
    k_pc = min(k_pc, d)

    if N is None or N >= n_valid:
        N = n_valid
        row_index = np.arange(n_valid, dtype=np.int64)
        if verbose:
            print(f"Using all {n_valid:,} h_eff rows (d={d}).")
    else:
        rng = np.random.default_rng(seed)
        row_index = rng.choice(n_valid, size=N, replace=False)
        row_index.sort()                             # memmap-friendly reads
        row_index = row_index.astype(np.int64)
        if verbose:
            print(f"Subsampling {N:,} of {n_valid:,} h_eff rows (d={d}, seed={seed}).")

    H = np.asarray(h_eff_mm[row_index], dtype=np.float32)

    if verbose:
        print(f"Computing log Z(h_i) via batched logsumexp over V={V}...")
    log_Z_of_h = np.empty(N, dtype=np.float32)
    H_t = torch.from_numpy(H).to(device)
    lz_iter = range(0, N, logsumexp_batch_size)
    if verbose:
        lz_iter = tqdm(lz_iter, desc="logsumexp", unit="batch")
    for start in lz_iter:
        end = min(start + logsumexp_batch_size, N)
        logits = H_t[start:end] @ W_t.T                     # (chunk, V)
        log_Z_of_h[start:end] = torch.logsumexp(logits, dim=-1).cpu().numpy()

    if compute_svd:
        V_basis, S, projections = _svd_and_project(H, k_pc, verbose=verbose)
    else:
        V_basis = np.empty((d, 0), dtype=np.float32)
        S = np.empty(0, dtype=np.float32)
        projections = np.empty((N, 0), dtype=np.float32)

    meta = {
        'N': int(N),
        'n_valid': int(n_valid),
        'd': int(d),
        'V': int(V),
        'k_pc_default': int(k_pc),
        'seed': int(seed),
        'extract_dir': os.fspath(extract_dir),
        'dataset_name': str(dataset_name),
        'source': 'h_eff_memmap',
        'svd_computed': bool(compute_svd),
    }
    if extra_meta:
        meta.update(extra_meta)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez(
        out_path,
        H=H,
        log_Z_of_h=log_Z_of_h,
        V_basis=V_basis,
        S=S,
        projections=projections,
        row_index=row_index,
        meta_json=np.array(json.dumps(meta)),
    )

    if verbose:
        size_mb = (H.nbytes + log_Z_of_h.nbytes + V_basis.nbytes
                   + S.nbytes + projections.nbytes + row_index.nbytes) / 1e6
        print(f"Saved viz sample → {out_path} ({size_mb:.1f} MB)")
        if compute_svd:
            total_var = float((S ** 2).sum())
            cum_var_k = float((S[:k_pc] ** 2).sum() / total_var) if total_var > 0 else 0.0
            print(f"  singular values (first 5): "
                  f"{np.array2string(S[:5], precision=3)}")
            print(f"  variance captured by first {k_pc} PCs: {cum_var_k:.3f}")
        else:
            print("  SVD deferred (pass compute_svd=True or call a basis='global' "
                  "plot to trigger lazy computation).")

    return VizSample(
        H=H,
        log_Z_of_h=log_Z_of_h,
        V_basis=V_basis,
        S=S,
        projections=projections,
        row_index=row_index,
        meta=meta,
    )


def load_viz_sample(path):
    """Load a cached VizSample from an .npz written by `build_viz_sample`."""
    arr = np.load(path, allow_pickle=False)
    meta = json.loads(str(arr['meta_json']))
    row_index = (arr['row_index'] if 'row_index' in arr.files
                 else np.empty(0, dtype=np.int64))
    return VizSample(
        H=arr['H'],
        log_Z_of_h=arr['log_Z_of_h'],
        V_basis=arr['V_basis'],
        S=arr['S'],
        projections=arr['projections'],
        row_index=row_index,
        meta=meta,
    )


def token_weights(viz_sample, W, t, Z_t, stabilize=True):
    """
    ω_i = p_t(h_i) / Z_t for each cached flow sample h_i, computed in log-space.

    log p_t(h_i) = W[t] · h_i − log_Z_of_h[i], which reuses the cached
    per-sample normalizer and keeps the query at O(N·d).

    Args:
        viz_sample: a VizSample.
        W: tensor or array of shape (V, d) — `model.lm_head.weight`.
        t: integer token id.
        Z_t: marginal probability of token t (from Stage 1's Z vector).
        stabilize: if True, subtract max(log ω) before exp — a positive constant
            that does not change plotted density shape. Set False for diagnostics.

    Returns:
        ω: np.ndarray of shape (N,), float32.
    """
    if Z_t <= 0.0:
        raise ValueError(f"Z_t must be positive (got {Z_t}); token never "
                         f"appears as the conditional target.")
    if isinstance(W, np.ndarray):
        W_t = torch.from_numpy(W[t]).float()
    else:
        W_t = W[t].detach().cpu().float()

    H = torch.from_numpy(viz_sample.H).float()
    log_num = (H @ W_t).numpy()                              # (N,) — W[t]·h_i
    log_omega = log_num - viz_sample.log_Z_of_h - float(np.log(Z_t))
    if stabilize:
        log_omega = log_omega - float(log_omega.max())
    return np.exp(log_omega).astype(np.float32)


def _weighted_centered_basis(H, weights, center=True, eps=1e-12):
    """
    Weighted PCA of the importance-reweighted sample.

    Diagonalizes the d×d weighted covariance
        C = Σ ω_i (h_i − μ)(h_i − μ)^T / Σ ω_i
    via `np.linalg.eigh`.

    Returns:
        mu: (d,) float32 — weighted mean, or zeros when `center=False`.
        V:  (d, d) float32 — eigenvectors in columns, descending by eigenvalue.
        eigvals_desc: (d,) float32 — eigenvalues in descending order.
    """
    w = np.asarray(weights, dtype=np.float64)
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= eps:
        raise ValueError("Weights sum to ~0; cannot form a token basis.")
    H64 = np.asarray(H, dtype=np.float64)
    if center:
        mu = (w[:, None] * H64).sum(axis=0) / w_sum
        Hc = H64 - mu
    else:
        mu = np.zeros(H64.shape[1], dtype=np.float64)
        Hc = H64
    C = (Hc * w[:, None]).T @ Hc / w_sum                     # (d, d)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    return (mu.astype(np.float32),
            eigvecs[:, order].astype(np.float32),
            eigvals[order].astype(np.float32))


def effective_sample_size(weights):
    """
    Kish ESS = (Σ ω)² / Σ ω². A rough indicator of how many of the cached
    samples are actually carrying the token-conditional plot.
    """
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    if s <= 0:
        return 0.0
    return float(s * s / (w * w).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (plotnine). Imported lazily so the rest of the module works in
# environments without plotnine installed.
# ─────────────────────────────────────────────────────────────────────────────

def _require_plotnine():
    try:
        import plotnine as pn
        import pandas as pd
        from scipy.stats import gaussian_kde
    except ImportError as e:
        raise ImportError(
            "Plotting requires plotnine, pandas, and scipy. "
            "Install with: pip install plotnine pandas scipy"
        ) from e
    return pn, pd, gaussian_kde


def _default_axis_labels(pairs, basis):
    """Axis labels for (x, y) PC pairs.

    pairs: list of (pi, pj) tuples.
    basis: 'token' or 'global' — determines the parenthetical suffix.
    """
    tag = {'token': 'token', 'global': 'global'}.get(basis, '')
    suffix = f" ({tag})" if tag else ''
    return [(f"PC{pi + 1}{suffix}", f"PC{pj + 1}{suffix}") for (pi, pj) in pairs]


def _weighted_kde_long_df(projections, pairs, weights=None,
                          n_grid=80, pad=0.05, bw_method=None,
                          pc_pair_labels=None):
    """
    Weighted 2D KDE on a regular grid for each PC pair.

    pairs: list of (xi, yi) index pairs into projections columns.
    Returns a DataFrame with columns: x, y, density, x_label, y_label.
    """
    _, pd, gaussian_kde = _require_plotnine()
    frames = []
    w_arr = None if weights is None else np.asarray(weights, dtype=np.float64)
    if w_arr is not None and w_arr.sum() <= 0:
        raise ValueError("All weights are zero/negative — cannot compute KDE.")

    for idx, (pi, pj) in enumerate(pairs):
        x = projections[:, pi].astype(np.float64)
        y = projections[:, pj].astype(np.float64)
        xr = float(x.max() - x.min()) or 1.0
        yr = float(y.max() - y.min()) or 1.0
        xs = np.linspace(x.min() - pad * xr, x.max() + pad * xr, n_grid)
        ys = np.linspace(y.min() - pad * yr, y.max() + pad * yr, n_grid)
        XX, YY = np.meshgrid(xs, ys)
        grid = np.vstack([XX.ravel(), YY.ravel()])
        xy = np.vstack([x, y])

        kde = gaussian_kde(xy, weights=w_arr, bw_method=bw_method)
        density = kde(grid).reshape(n_grid, n_grid)

        if pc_pair_labels is not None:
            x_label, y_label = pc_pair_labels[idx]
        else:
            x_label, y_label = f"PC{pi + 1}", f"PC{pj + 1}"
        frames.append(pd.DataFrame({
            'x': XX.ravel(),
            'y': YY.ravel(),
            'density': density.ravel(),
            'x_label': x_label,
            'y_label': y_label,
        }))
    return pd.concat(frames, ignore_index=True)


def _log_ratio_kde_long_df(projections, pairs, weights, *,
                           n_grid=80, pad=0.05, bw_method=None,
                           pc_pair_labels=None,
                           floor=1e-12):
    """Per-pair log[ĝ_t(y) / ĝ(y)] on a regular grid.

    pairs: list of (xi, yi) index pairs into projections columns.
    Returns columns x, y, log_ratio, x_label, y_label.
    """
    _, pd, gaussian_kde = _require_plotnine()
    w_arr = np.asarray(weights, dtype=np.float64)
    if w_arr.sum() <= 0:
        raise ValueError("All weights are zero/negative — cannot compute KDE.")
    frames = []
    for idx, (pi, pj) in enumerate(pairs):
        x = projections[:, pi].astype(np.float64)
        y = projections[:, pj].astype(np.float64)
        xr = float(x.max() - x.min()) or 1.0
        yr = float(y.max() - y.min()) or 1.0
        xs = np.linspace(x.min() - pad * xr, x.max() + pad * xr, n_grid)
        ys = np.linspace(y.min() - pad * yr, y.max() + pad * yr, n_grid)
        XX, YY = np.meshgrid(xs, ys)
        grid = np.vstack([XX.ravel(), YY.ravel()])
        xy = np.vstack([x, y])

        kde_g = gaussian_kde(xy, bw_method=bw_method)
        # Same bandwidth so smoothing Jacobian cancels in the log-ratio.
        kde_t = gaussian_kde(xy, weights=w_arr, bw_method=kde_g.factor)
        dens_g = kde_g(grid).reshape(n_grid, n_grid)
        dens_t = kde_t(grid).reshape(n_grid, n_grid)
        log_ratio = (np.log(np.maximum(dens_t, floor))
                     - np.log(np.maximum(dens_g, floor)))

        if pc_pair_labels is not None:
            x_label, y_label = pc_pair_labels[idx]
        else:
            x_label, y_label = f"PC{pi + 1}", f"PC{pj + 1}"
        frames.append(pd.DataFrame({
            'x': XX.ravel(),
            'y': YY.ravel(),
            'log_ratio': log_ratio.ravel(),
            'x_label': x_label,
            'y_label': y_label,
        }))
    return pd.concat(frames, ignore_index=True)


def _multi_token_kde_long_df(projections, pairs, token_specs,
                             *, n_grid=80, pad=0.05, bw_method=None,
                             pc_pair_labels=None):
    """Per-token weighted KDE on a shared grid for each PC pair.

    pairs: list of (xi, yi) index pairs into projections columns.
    token_specs: list of (name, weights-or-None). Densities are normalized
    to [0, 1] per (token, facet).
    Returns columns: x, y, density, x_label, y_label, token.
    """
    _, pd, gaussian_kde = _require_plotnine()
    frames = []
    for idx, (pi, pj) in enumerate(pairs):
        x = projections[:, pi].astype(np.float64)
        y = projections[:, pj].astype(np.float64)
        xr = float(x.max() - x.min()) or 1.0
        yr = float(y.max() - y.min()) or 1.0
        xs = np.linspace(x.min() - pad * xr, x.max() + pad * xr, n_grid)
        ys = np.linspace(y.min() - pad * yr, y.max() + pad * yr, n_grid)
        XX, YY = np.meshgrid(xs, ys)
        grid = np.vstack([XX.ravel(), YY.ravel()])
        xy = np.vstack([x, y])

        if pc_pair_labels is not None:
            x_label, y_label = pc_pair_labels[idx]
        else:
            x_label, y_label = f"PC{pi + 1}", f"PC{pj + 1}"

        for name, wts in token_specs:
            w_arr = None if wts is None else np.asarray(wts, dtype=np.float64)
            if w_arr is not None and w_arr.sum() <= 0:
                raise ValueError(
                    f"All weights are zero/negative for token {name!r} — "
                    f"cannot compute KDE."
                )
            kde = gaussian_kde(xy, weights=w_arr, bw_method=bw_method)
            density = kde(grid).reshape(n_grid, n_grid)
            mx = float(density.max())
            if mx > 0:
                density = density / mx
            frames.append(pd.DataFrame({
                'x': XX.ravel(),
                'y': YY.ravel(),
                'density': density.ravel(),
                'x_label': x_label,
                'y_label': y_label,
                'token': name,
            }))
    return pd.concat(frames, ignore_index=True)


def load_decode(dataset_name, data_dir='data', sep=' '):
    """
    Return a `decode(ids) -> str` for the tokenizer used to build `data/{dataset_name}`.

    Mirrors `sample.py`: prefers `meta.pkl` (char-level stoi/itos), then
    `meta.json` (HF tokenizers), then falls back to tiktoken `gpt2`.
    """
    import pickle
    meta_pkl = os.path.join(data_dir, dataset_name, 'meta.pkl')
    meta_json = os.path.join(data_dir, dataset_name, 'meta.json')
    if os.path.exists(meta_pkl):
        with open(meta_pkl, 'rb') as f:
            m = pickle.load(f)
        itos = {int(k): v for k, v in m['itos'].items()}
        return lambda ids: sep.join(itos[int(i)] for i in ids)
    if os.path.exists(meta_json):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(meta_json)
        return lambda ids: tok.decode([int(i) for i in ids],
                                      skip_special_tokens=False)
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    return lambda ids: enc.decode([int(i) for i in ids])


def sample_token_instances(
    viz_sample,
    extract_meta,
    data,
    h_eff,
    *,
    t=None,
    k=6,
    context=6,
    decode=None,
    rng=None,
):
    """
    Sample real corpus instances for overlay on the density plots.

    For a token plot `g(·|t)`, positions where the *next* token is `t` — those
    are the empirical analogues of the importance-reweighted density. For the
    global plot, `t=None` picks random valid positions.

    Each sampled instance is projected into the cached global PC basis
    (`V_basis[:, :k_pc]`) and returned in wide form with `pc1`…`pc_{k_pc}`
    columns for inspection. The raw `h_eff` vector is also stored on each row
    so plot functions can re-project it into any basis at plot time.

    Args:
        viz_sample: VizSample — provides `V_basis` and `k_pc`.
        extract_meta: dict from `{name}_meta.json`.
        data: corpus memmap/array (uint16) used in Stage 1.
        h_eff: `(N_valid, d)` memmap/array written by Stage 1.
        t: next-token id to filter on, or None for random valid positions.
        k: max number of instances to return.
        context: number of tokens (ending at the target) to decode into label.
        decode: `list[int] -> str`; defaults to space-separated ids.
        rng: numpy `Generator` (default `np.random.default_rng()`).

    Returns:
        pandas.DataFrame with columns {row_index, corpus_pos, label, h_eff,
        pc1, pc2, ..., pc_{k_pc}} (pc{k} only when SVD is cached).
    """
    _, pd, _ = _require_plotnine()
    if rng is None:
        rng = np.random.default_rng()
    from shape.extract import valid_positions
    positions = valid_positions(extract_meta)
    if len(positions) != h_eff.shape[0]:
        raise ValueError(
            f"N_valid mismatch: positions={len(positions)} but "
            f"h_eff has {h_eff.shape[0]} rows — extract_meta may not match "
            f"the h_eff file."
        )

    if t is None:
        candidates = np.arange(len(positions), dtype=np.int64)
    else:
        target_pos = positions + 1
        in_range = target_pos < len(data)
        next_tok = np.full(len(positions), -1, dtype=np.int64)
        next_tok[in_range] = np.asarray(data[target_pos[in_range]],
                                        dtype=np.int64)
        candidates = np.flatnonzero(next_tok == int(t))
        if len(candidates) == 0:
            raise ValueError(
                f"No valid positions whose next token is id={t} — did this "
                f"token appear in the corpus split used for extraction?"
            )

    k_actual = min(k, len(candidates))
    chosen = np.sort(rng.choice(candidates, size=k_actual, replace=False))

    h_sel = np.asarray(h_eff[chosen], dtype=np.float32)          # (k, d)
    if viz_sample.has_svd:
        V = viz_sample.V_basis[:, :viz_sample.k_pc]
        proj = h_sel @ V
    else:
        proj = None

    if decode is None:
        decode = lambda ids: ' '.join(str(int(i)) for i in ids)
    labels = []
    T = len(data)
    for idx in chosen:
        pos = int(positions[idx])
        end = min(pos + 2, T)
        start = max(0, end - context)
        ids = np.asarray(data[start:end], dtype=np.int64).tolist()
        labels.append(decode(ids))

    rows = []
    for j, idx in enumerate(chosen):
        row = {
            'row_index': int(idx),
            'corpus_pos': int(positions[idx]),
            'label': labels[j],
            'h_eff': np.asarray(h_sel[j], dtype=np.float32).copy(),
        }
        if proj is not None:
            for p in range(viz_sample.k_pc):
                row[f'pc{p + 1}'] = float(proj[j, p])
        rows.append(row)
    return pd.DataFrame(rows)


def _find_phrase_instances(phrase, viz_sample, corpus_ctx, stoi, decode, phrase_avg):
    """
    Search the corpus for a phrase and return an examples DataFrame row.

    phrase: whitespace-separated string of vocabulary words.
    phrase_avg: 1 = use first occurrence; N>1 = average up to N occurrences.

    The phrase is tokenized by splitting on whitespace and looking up each
    word in `stoi`. Returns a one-row DataFrame (or empty DataFrame if not
    found) compatible with `sample_token_instances` output.
    """
    _, pd, _ = _require_plotnine()
    if stoi is None:
        raise ValueError("stoi is required for phrase search examples.")
    words = phrase.split()
    try:
        phrase_ids = [int(stoi[w]) for w in words]
    except KeyError as missing:
        raise ValueError(
            f"Phrase word {missing} not found in stoi vocabulary. "
            f"Check tokenization — vocabulary uses word-level tokens."
        )
    n = len(phrase_ids)
    phrase_arr = np.array(phrase_ids, dtype=np.int64)

    from shape.extract import valid_positions
    positions = valid_positions(corpus_ctx.extract_meta)
    data_arr = np.asarray(corpus_ctx.data, dtype=np.int64)
    T = len(data_arr)

    # Candidate rows: next token matches phrase[-1]
    next_pos = positions + 1
    in_range = next_pos < T
    last_tok = np.where(in_range,
                        data_arr[np.minimum(next_pos, T - 1)],
                        np.int64(-1))
    candidates = np.flatnonzero(last_tok == phrase_ids[-1])

    matches = []
    for i in candidates:
        pos = int(positions[i])
        # phrase spans data[pos-n+2 : pos+2]
        start = pos - n + 2
        end = pos + 2
        if start < 0 or end > T:
            continue
        if np.array_equal(data_arr[start:end], phrase_arr):
            matches.append(int(i))

    if not matches:
        return pd.DataFrame()

    if phrase_avg == 1:
        i = matches[0]
        h = np.asarray(corpus_ctx.h_eff[i], dtype=np.float32)
        return pd.DataFrame([{
            'row_index': i,
            'corpus_pos': int(positions[i]),
            'label': phrase,
            'h_eff': h,
        }])
    else:
        selected = matches[:phrase_avg]
        h_vecs = np.stack([np.asarray(corpus_ctx.h_eff[i], dtype=np.float32)
                           for i in selected])
        h = h_vecs.mean(axis=0)
        return pd.DataFrame([{
            'row_index': -1,
            'corpus_pos': -1,
            'label': phrase,
            'h_eff': h,
        }])


def _resolve_examples(examples, ids, viz_sample, corpus_context, stoi, decode,
                      phrase_avg):
    """
    Normalize the `examples` parameter to a DataFrame or None.

    examples:
      None       → no examples
      DataFrame  → returned as-is
      int k      → randomly sample k instances of ids[0] from corpus
      list[str]  → phrase search; each phrase → one row in the output
    """
    _, pd, _ = _require_plotnine()
    if examples is None:
        return None
    if isinstance(examples, pd.DataFrame):
        return examples
    if isinstance(examples, int):
        if corpus_context is None:
            raise ValueError(
                "corpus_context is required when examples is an int (random sample)."
            )
        if len(ids) != 1:
            raise ValueError(
                "examples=k (random sampling) is only supported for single-token plots."
            )
        return sample_token_instances(
            viz_sample,
            corpus_context.extract_meta,
            corpus_context.data,
            corpus_context.h_eff,
            t=ids[0],
            k=examples,
            decode=decode,
        )
    if isinstance(examples, list) and all(isinstance(e, str) for e in examples):
        if corpus_context is None:
            raise ValueError(
                "corpus_context is required when examples is a list of phrases."
            )
        frames = []
        for phrase in examples:
            df = _find_phrase_instances(
                phrase, viz_sample, corpus_context, stoi, decode, phrase_avg
            )
            if len(df) > 0:
                frames.append(df)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)
    raise ValueError(
        f"examples must be None, a DataFrame, an int k, or a list of phrase strings; "
        f"got {type(examples).__name__!r}."
    )


def _examples_long_df(examples_df, pairs, V_basis=None, center=None,
                      pc_pair_labels=None):
    """Expand a wide-form examples DataFrame into long-form facet rows.

    pairs: list of (xi, yi) tuples indexing into the projection columns.
    Same k instances appear in every pair — only (x, y) differs per facet.
    Re-projects from the raw h_eff vector when V_basis is supplied.
    """
    _, pd, _ = _require_plotnine()
    has_h_eff = 'h_eff' in examples_df.columns
    use_reproject = has_h_eff and V_basis is not None
    if use_reproject:
        H = np.stack([np.asarray(h, dtype=np.float32)
                      for h in examples_df['h_eff'].values])
        if center is not None:
            H = H - np.asarray(center, dtype=np.float32)[None, :]
        proj = H @ np.asarray(V_basis, dtype=np.float32)          # (k, cols)

    parts = []
    base_cols = ['row_index', 'corpus_pos', 'label']
    for idx, (pi, pj) in enumerate(pairs):
        sub = examples_df[base_cols].copy()
        if use_reproject:
            sub['x'] = proj[:, pi]
            sub['y'] = proj[:, pj]
        else:
            px, py = f'pc{pi + 1}', f'pc{pj + 1}'
            if px not in examples_df.columns or py not in examples_df.columns:
                raise ValueError(
                    f"examples DataFrame missing projection columns "
                    f"{px}/{py} — was viz_sample.k_pc too small when it "
                    f"was built?"
                )
            sub['x'] = examples_df[px].values
            sub['y'] = examples_df[py].values
        if pc_pair_labels is not None:
            x_label, y_label = pc_pair_labels[idx]
        else:
            x_label, y_label = f"PC{pi + 1}", f"PC{pj + 1}"
        sub['x_label'] = x_label
        sub['y_label'] = y_label
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def _senses_dataframe(senses, V_basis, pairs, center=None, pc_pair_labels=None):
    """Project sense centroids onto the PC basis and return a long-form DF.

    pairs: list of (xi, yi) tuples.
    """
    _, pd, _ = _require_plotnine()
    S = np.asarray(senses, dtype=np.float32)
    if center is not None:
        S = S - np.asarray(center, dtype=np.float32)[None, :]
    centroid_proj = S @ np.asarray(V_basis, dtype=np.float32)     # (K, cols)
    rows = []
    for idx, (pi, pj) in enumerate(pairs):
        if pc_pair_labels is not None:
            x_label, y_label = pc_pair_labels[idx]
        else:
            x_label, y_label = f"PC{pi + 1}", f"PC{pj + 1}"
        for k_idx in range(centroid_proj.shape[0]):
            rows.append({
                'x': float(centroid_proj[k_idx, pi]),
                'y': float(centroid_proj[k_idx, pj]),
                'x_label': x_label,
                'y_label': y_label,
                'sense': f"s{k_idx}",
            })
    return pd.DataFrame(rows)


def _density_plot(df_grid, *, title, senses_df=None, examples_df=None):
    """Heatmap renderer used by plot_global_density."""
    pn, _, _ = _require_plotnine()
    p = (
        pn.ggplot(df_grid, pn.aes('x', 'y'))
        + pn.geom_raster(pn.aes(fill='density'))
        + pn.scale_fill_cmap(cmap_name='viridis')
        + pn.facet_grid('y_label ~ x_label', scales='free')
        + pn.coord_cartesian(expand=False)
        + pn.theme_minimal()
        + pn.labs(x=None, y=None, fill='density', title=title)
    )
    if senses_df is not None:
        p = p + pn.geom_point(senses_df, pn.aes('x', 'y'),
                              color='red', size=2.5, inherit_aes=False)
    if examples_df is not None:
        p = (
            p
            + pn.geom_point(examples_df, pn.aes('x', 'y'),
                            color='white', size=1.8, inherit_aes=False)
            + pn.geom_text(examples_df, pn.aes('x', 'y', label='label'),
                           color='white', size=7, ha='left', va='bottom',
                           inherit_aes=False)
        )
    return p


def _log_ratio_plot(df_grid, *, title, examples_df=None, clip=None):
    """Diverging log-ratio heatmap renderer used by plot_distinctiveness."""
    pn, _, _ = _require_plotnine()
    df = df_grid
    if clip is not None:
        lo, hi = clip
        df = df.copy()
        df['log_ratio'] = df['log_ratio'].clip(lower=lo, upper=hi)
    p = (
        pn.ggplot(df, pn.aes('x', 'y'))
        + pn.geom_raster(pn.aes(fill='log_ratio'))
        + pn.scale_fill_gradient2(low='#2166ac', mid='#f7f7f7',
                                  high='#b2182b', midpoint=0)
        + pn.facet_grid('y_label ~ x_label', scales='free')
        + pn.coord_cartesian(expand=False)
        + pn.theme_minimal()
        + pn.labs(x=None, y=None, fill='D_t(y)', title=title)
    )
    if examples_df is not None:
        p = (
            p
            + pn.geom_point(examples_df, pn.aes('x', 'y'),
                            color='black', size=1.8, inherit_aes=False)
            + pn.geom_text(examples_df, pn.aes('x', 'y', label='label'),
                           color='black', size=7, ha='left', va='bottom',
                           inherit_aes=False)
        )
    return p


def _filled_bands_plot(df, *, title, labels, colors, levels,
                       senses_df=None, examples_df=None):
    """
    Stacked translucent band renderer used by plot_density.

    df has columns: x, y, density (normalized to [0,1] per token per facet),
    x_label, y_label, token (Categorical ordered by labels).
    Bands are drawn in ascending threshold order so cores overlay halos.
    """
    pn, pd, _ = _require_plotnine()
    df = df.copy()
    df['token'] = pd.Categorical(df['token'], categories=list(labels), ordered=True)
    color_map = {name: c for name, c in zip(labels, colors)}

    p = (
        pn.ggplot(df, pn.aes('x', 'y'))
        + pn.facet_grid('y_label ~ x_label', scales='free')
        + pn.coord_cartesian(expand=False)
        + pn.theme_minimal()
        + pn.labs(x=None, y=None, title=title, fill='token')
        + pn.scale_fill_manual(values=color_map)
    )
    for level, alpha in sorted(levels, key=lambda la: la[0]):
        band = df[df['density'] > level]
        if len(band) == 0:
            continue
        p = p + pn.geom_tile(
            band, pn.aes('x', 'y', fill='token'),
            alpha=alpha, inherit_aes=False,
        )
    if senses_df is not None:
        p = p + pn.geom_point(senses_df, pn.aes('x', 'y'),
                              color='red', size=2.5, inherit_aes=False)
    if examples_df is not None:
        p = (
            p
            + pn.geom_point(examples_df, pn.aes('x', 'y'),
                            color='black', size=1.8, inherit_aes=False)
            + pn.geom_text(examples_df, pn.aes('x', 'y', label='label'),
                           color='black', size=7, ha='left', va='bottom',
                           inherit_aes=False)
        )
    return p


def plot_global_density(viz_sample, pcs=(0, 1, 2, 3),
                        basis='global',
                        W=None,
                        weights=None,
                        axes=None,
                        decode=None,
                        n_grid=80, bw_method=None,
                        title='Global density g(·)',
                        examples=None):
    """
    Faceted heatmap of the unconditional empirical density g(·).

    pcs: list of PC indices. All non-diagonal (i, j) combinations are shown
    in a matrix layout — columns share the x-axis PC, rows share the y-axis PC.

    basis: 'global' (default) uses the cached uncentered SVD. 'token' uses a
    weighted (or unweighted when weights=None) centered PCA — pass the omega
    array from token_weights() as `weights` to visualise g(·) on the same axes
    as a plot_density call for that token. 'contrast' uses token-pair log-odds
    axes; requires W and axes (same format as plot_density).

    W: (V, d) weight matrix — required for basis='contrast'.
    weights: (N,) importance weights — only consulted when basis='token';
    passing the omega from token_weights() places g(·) on the token-PCA axes.
    axes: list of ((pos, neg), (pos, neg)) contrast specs — required for
    basis='contrast'. Same format as plot_density.
    decode: optional list[int]→str for contrast axis labels.

    examples: optional DataFrame from `sample_token_instances` (pass t=None
    there for random corpus positions).
    """
    H_np = np.asarray(viz_sample.H, dtype=np.float32)
    projections, V_cols, center, pairs, pc_pair_labels = _dispatch_basis(
        H_np, weights, viz_sample, W,
        basis=basis, pcs=list(pcs), axes=axes, decode=decode,
    )
    df = _weighted_kde_long_df(projections, pairs, weights=None,
                               n_grid=n_grid, bw_method=bw_method,
                               pc_pair_labels=pc_pair_labels)
    ex_long = (_examples_long_df(examples, pairs, V_basis=V_cols, center=center,
                                 pc_pair_labels=pc_pair_labels)
               if examples is not None else None)
    return _density_plot(df, title=title, examples_df=ex_long)


def _contrast_basis(W, axes, decode=None):
    """
    Build a basis from token-pair contrasts. Each axis spec `(pos, neg)` maps
    to direction `W[pos] - W[neg]` in h-space; projecting h onto it yields
    the log-odds `log p(pos|h) / p(neg|h)` (the log-Z term cancels). Reused
    contrasts across facets are deduplicated.

    Returns:
        V_cols: (d, n_unique) basis matrix.
        facet_pairs: list of (xi, yi) column-index pairs into V_cols.
        labels: list of (x_label, y_label) tuples, one per facet.
    """
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().float().numpy()
    else:
        W_np = np.asarray(W, dtype=np.float32)
    V, d = W_np.shape

    seen = {}
    cols = []
    specs = []
    facet_pairs = []
    for axis_x, axis_y in axes:
        ix = _intern_contrast(axis_x, seen, cols, specs, W_np, V)
        iy = _intern_contrast(axis_y, seen, cols, specs, W_np, V)
        facet_pairs.append((ix, iy))

    V_cols = np.stack(cols, axis=1).astype(np.float32)            # (d, n_unique)
    labels = [
        (_contrast_label(specs[xi], decode),
         _contrast_label(specs[yi], decode))
        for (xi, yi) in facet_pairs
    ]
    return V_cols, facet_pairs, labels


def _intern_contrast(axis, seen, cols, specs, W_np, V):
    pos, neg = int(axis[0]), int(axis[1])
    if pos == neg:
        raise ValueError(f"contrast axis ({pos}, {neg}): pos and neg must differ.")
    if not (0 <= pos < V and 0 <= neg < V):
        raise ValueError(f"contrast axis ({pos}, {neg}): token id out of range [0, {V}).")
    key = (pos, neg)
    if key in seen:
        return seen[key]
    seen[key] = len(cols)
    cols.append(W_np[pos] - W_np[neg])
    specs.append(key)
    return seen[key]


def _contrast_label(spec, decode):
    pos, neg = spec
    if decode is None:
        return f"log p({pos})/p({neg})"
    return f"log p({decode([pos])!r})/p({decode([neg])!r})"


def _dispatch_basis(H_np, weights, viz_sample, W, *,
                    basis, pcs, axes, decode):
    """Project H_np onto the chosen basis and return rendering inputs.

    pcs: list of ints for basis='token'/'global', or ignored for basis='contrast'.
    Returns (projections, V_cols, center, pairs, pc_pair_labels):
      - projections: (N, k) array of H_np projected onto V_cols
      - V_cols:      (d, k) basis matrix
      - center:      (d,) shift subtracted before projection, or None
      - pairs:       list of (xi, yi) tuples indexing into projections columns
      - pc_pair_labels: list of (x_label, y_label), one per pair
    """
    if basis == 'token':
        if not pcs:
            raise ValueError("pcs must be non-empty.")
        max_pc = max(pcs) + 1
        if max_pc > H_np.shape[1]:
            raise ValueError(
                f"PC index {max_pc - 1} exceeds d={H_np.shape[1]}; the "
                f"token basis has at most d components."
            )
        pca_weights = (np.ones(H_np.shape[0], dtype=np.float32)
                       if weights is None else weights)
        mu, V_full, _ = _weighted_centered_basis(H_np, pca_weights, center=True)
        V_cols = V_full[:, :max_pc]
        projections = (H_np - mu[None, :]) @ V_cols
        pairs = [(i, j) for i in pcs for j in pcs if i != j]
        return projections, V_cols, mu, pairs, _default_axis_labels(pairs, 'token')

    if basis == 'global':
        _ensure_svd(viz_sample, verbose=False)
        _check_pcs(pcs, viz_sample.k_pc)
        max_pc = max(pcs) + 1
        V_cols = viz_sample.V_basis[:, :max_pc]
        projections = H_np @ V_cols
        pairs = [(i, j) for i in pcs for j in pcs if i != j]
        return projections, V_cols, None, pairs, _default_axis_labels(pairs, 'global')

    if basis == 'contrast':
        if axes is None:
            raise ValueError(
                "basis='contrast' requires axes=[((pos,neg),(pos,neg)),...]"
            )
        V_cols, facet_pairs, labels = _contrast_basis(W, axes, decode=decode)
        projections = H_np @ V_cols
        return projections, V_cols, None, facet_pairs, labels

    raise ValueError(
        f"basis must be 'token', 'global', or 'contrast', got {basis!r}"
    )


def _resolve_tokens(tokens_input, Z, stoi=None):
    """Normalize tokens_input to (ids: list[int], labels: list[str]).

    Accepts a single int/str or a list of either. String tokens require stoi.
    """
    if not isinstance(tokens_input, list):
        tokens_input = [tokens_input]
    V = len(Z)
    ids = []
    labels = []
    for t in tokens_input:
        if isinstance(t, str):
            if stoi is None:
                raise ValueError(
                    f"stoi is required to look up string token {t!r}."
                )
            if t not in stoi:
                raise ValueError(f"Token {t!r} not found in stoi vocabulary.")
            i = int(stoi[t])
            labels.append(t)
        elif isinstance(t, (int, np.integer)):
            i = int(t)
            labels.append(f"id={i}")
        else:
            raise ValueError(
                f"Token must be int or str, got {type(t).__name__!r}."
            )
        if not (0 <= i < V):
            raise ValueError(f"Token id {i} out of range [0, {V}).")
        ids.append(i)
    return ids, labels


def plot_density(viz_sample, W, Z, tokens, *,
                 stoi=None,
                 labels=None,
                 pcs=(0, 1, 2, 3),
                 basis=None,
                 axes=None,
                 decode=None,
                 senses=None,
                 title=None,
                 n_grid=80,
                 bw_method=None,
                 colors=None,
                 levels=((0.2, 0.15), (0.5, 0.25), (0.8, 0.4)),
                 examples=None,
                 corpus_context=None,
                 phrase_avg=1):
    """
    Filled-bands overlay of token-conditional density g(·|t) for one or more
    tokens. Each token is drawn as stacked translucent bands; darker core,
    lighter halo. Works for 1 to N tokens on a shared basis.

    pcs: list of PC indices. All non-diagonal (i, j) pairs are shown in a
    matrix layout (columns = x-axis PC, rows = y-axis PC). Default: [0,1,2,3].

    tokens: int | str | list[int | str]. String tokens require `stoi`.
    Z: full (V,) marginals array (no need to extract Z_t per token).

    basis: 'token' (default for single token) uses a weighted centered PCA —
    axes show the principal polysemy directions for that token, non-comparable
    across tokens. 'global' uses the cached uncentered SVD (shared axes across
    tokens). 'contrast' uses token-pair log-odds axes (see `axes`). For
    multiple tokens, basis defaults to 'global'.

    examples: None | int k (random corpus sample, single-token only) |
    list[str] (phrase search — first occurrence or average of phrase_avg
    occurrences) | DataFrame from sample_token_instances.
    corpus_context: CorpusContext, required when examples is int or list[str].
    phrase_avg: 1 = first corpus occurrence; N>1 = average N occurrences.

    senses: optional (K, d) array of sense centroids overlaid as red points
    (single-token only; silently ignored for multi-token).
    """
    Z_arr = np.asarray(Z)
    ids, auto_labels = _resolve_tokens(tokens, Z_arr, stoi=stoi)
    if labels is None:
        labels = auto_labels
    if len(labels) != len(ids):
        raise ValueError("len(labels) must equal len(tokens).")

    if basis is None:
        basis = 'token' if len(ids) == 1 else 'global'
    if basis == 'token' and len(ids) > 1:
        raise ValueError(
            "basis='token' is not supported for multiple tokens — each token "
            "would need its own basis. Use basis='global' or basis='contrast'."
        )

    if colors is None:
        _palette = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd',
                    '#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f']
        colors = _palette[:len(ids)]
    if len(colors) != len(ids):
        raise ValueError("len(colors) must equal len(tokens).")

    # Per-token weights and ESS
    weight_specs = []
    ess_tags = []
    for t, lbl in zip(ids, labels):
        omega = token_weights(viz_sample, W, t, float(Z_arr[t]))
        weight_specs.append((lbl, omega))
        ess_tags.append(f"{lbl}: ESS={effective_sample_size(omega):.0f}")

    H_np = np.asarray(viz_sample.H, dtype=np.float32)
    pcs_list = list(pcs)

    if basis == 'token':
        # Build basis from the single token's weights
        _, omega_0 = weight_specs[0]
        projections, V_cols, center, pairs, pc_pair_labels = _dispatch_basis(
            H_np, omega_0, viz_sample, W,
            basis='token', pcs=pcs_list, axes=axes, decode=decode,
        )
    else:
        projections, V_cols, center, pairs, pc_pair_labels = _dispatch_basis(
            H_np, None, viz_sample, W,
            basis=basis, pcs=pcs_list, axes=axes, decode=decode,
        )

    df = _multi_token_kde_long_df(
        projections, pairs, weight_specs,
        n_grid=n_grid, bw_method=bw_method, pc_pair_labels=pc_pair_labels,
    )

    ex_df = _resolve_examples(examples, ids, viz_sample, corpus_context,
                              stoi, decode, phrase_avg)
    ex_long = (_examples_long_df(ex_df, pairs, V_basis=V_cols, center=center,
                                 pc_pair_labels=pc_pair_labels)
               if ex_df is not None else None)

    senses_df = None
    if senses is not None and len(ids) == 1:
        senses_df = _senses_dataframe(senses, V_cols, pairs, center=center,
                                      pc_pair_labels=pc_pair_labels)

    default_title = 'g(·|t): ' + ',  '.join(ess_tags)
    return _filled_bands_plot(
        df,
        title=title or default_title,
        labels=labels,
        colors=colors,
        levels=levels,
        senses_df=senses_df,
        examples_df=ex_long,
    )


def plot_distinctiveness(viz_sample, W, Z, token, *,
                         stoi=None,
                         pcs=(0, 1, 2, 3),
                         basis='token',
                         axes=None,
                         decode=None,
                         title=None,
                         n_grid=80,
                         bw_method=None,
                         examples=None,
                         corpus_context=None,
                         phrase_avg=1,
                         clip=None):
    """
    Diverging log-ratio heatmap of D_t(y) = log[g(y|t) / g(y)].

    Red = regions distinctive of token t; blue = anti-distinctive; white = 0.
    Title reports the scalar D_t = D_KL(g(·|t) ‖ g(·)) and ESS.

    token: single int or str. Z is the full (V,) marginals array.
    pcs, basis, axes, decode, examples, corpus_context, phrase_avg: see
    plot_density. clip: optional (lo, hi) to bound the color scale (e.g. (-3,3)).
    """
    from shape.distinctiveness import expected_distinctiveness
    Z_arr = np.asarray(Z)
    ids, auto_labels = _resolve_tokens(token, Z_arr, stoi=stoi)
    if len(ids) != 1:
        raise ValueError("plot_distinctiveness accepts exactly one token.")
    t = ids[0]
    label = auto_labels[0]

    weights = token_weights(viz_sample, W, t, float(Z_arr[t]))
    summary = expected_distinctiveness(viz_sample, W, t, float(Z_arr[t]))
    default_title = (
        f'D_t(y) for {label}  '
        f'[D_t={summary["D_t"]:.2f} nats, '
        f'ESS={summary["ess"]:.0f}/{viz_sample.N}]'
    )

    H_np = np.asarray(viz_sample.H, dtype=np.float32)
    projections, V_cols, center, pairs, pc_pair_labels = _dispatch_basis(
        H_np, weights, viz_sample, W,
        basis=basis, pcs=list(pcs), axes=axes, decode=decode,
    )
    df = _log_ratio_kde_long_df(projections, pairs, weights,
                                n_grid=n_grid, bw_method=bw_method,
                                pc_pair_labels=pc_pair_labels)
    ex_df = _resolve_examples(examples, ids, viz_sample, corpus_context,
                              stoi, decode, phrase_avg)
    ex_long = (_examples_long_df(ex_df, pairs, V_basis=V_cols, center=center,
                                 pc_pair_labels=pc_pair_labels)
               if ex_df is not None else None)
    return _log_ratio_plot(df, title=title or default_title,
                           examples_df=ex_long, clip=clip)


def _plot_empirical_token_density(viz_sample, W, t, extract_meta, data, h_eff, *,
                                  pcs=(0, 1, 2, 3),
                                  basis='token',
                                  axes=None,
                                  decode=None,
                                  senses=None,
                                  token_label=None,
                                  title=None,
                                  n_grid=80,
                                  bw_method=None,
                                  max_samples=10_000,
                                  rng=None,
                                  examples=None):
    """Corpus-occurrence view: KDE over actual positions where next-token == t.

    Complementary diagnostic to plot_density (the model view). This function
    is private; prefer plot_density for general use.
    """
    from shape.extract import valid_positions
    if rng is None:
        rng = np.random.default_rng()
    positions = valid_positions(extract_meta)
    if len(positions) != h_eff.shape[0]:
        raise ValueError(
            f"N_valid mismatch: positions={len(positions)} but "
            f"h_eff has {h_eff.shape[0]} rows — extract_meta may not match "
            f"the h_eff file."
        )

    target_pos = positions + 1
    in_range = target_pos < len(data)
    next_tok = np.full(len(positions), -1, dtype=np.int64)
    next_tok[in_range] = np.asarray(data[target_pos[in_range]], dtype=np.int64)
    idx = np.flatnonzero(next_tok == int(t))
    n_total = len(idx)
    if n_total == 0:
        raise ValueError(
            f"No valid positions with next-token == id={t} in this split. "
            f"Use plot_density for the model view of rare/unseen tokens."
        )

    if n_total > max_samples:
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
    H_sub = np.asarray(h_eff[idx], dtype=np.float32)

    label = token_label if token_label is not None else f"id={t}"
    shown = len(idx)
    n_tag = f"n={shown}" if shown == n_total else f"n={shown}/{n_total}"
    default_title = f'g_emp(·|t) for {label}  [{n_tag}]'

    H_np = np.asarray(H_sub, dtype=np.float32)
    pcs_list = list(pcs)
    projections, V_cols, center, pairs, pc_pair_labels = _dispatch_basis(
        H_np, None, viz_sample, W,
        basis=basis, pcs=pcs_list, axes=axes, decode=decode,
    )
    token_specs = [(label, None)]
    df = _multi_token_kde_long_df(
        projections, pairs, token_specs,
        n_grid=n_grid, bw_method=bw_method, pc_pair_labels=pc_pair_labels,
    )
    senses_df = (_senses_dataframe(senses, V_cols, pairs, center=center,
                                   pc_pair_labels=pc_pair_labels)
                 if senses is not None else None)
    ex_long = (_examples_long_df(examples, pairs, V_basis=V_cols, center=center,
                                 pc_pair_labels=pc_pair_labels)
               if examples is not None else None)
    pn, pd, _ = _require_plotnine()
    colors = ['#1f77b4']
    levels = ((0.2, 0.15), (0.5, 0.25), (0.8, 0.4))
    return _filled_bands_plot(
        df,
        title=title or default_title,
        labels=[label],
        colors=colors,
        levels=levels,
        senses_df=senses_df,
        examples_df=ex_long,
    )


def suggest_contrast_axes(
    viz_sample,
    W,
    *,
    weights=None,
    k_pc=4,
    top_n=3,
    score='cosine',
    candidate_pool=200,
    restrict_to=None,
    decode=None,
    verbose=False,
):
    """
    Discover token-pair contrasts that line up with the principal directions
    of the cached sample — turning the guesswork of picking `basis='contrast'`
    axes into a search.

    Pipeline: build a (weighted, centered) PCA of `viz_sample.H` — the same
    decomposition `basis='token'` uses — then for each leading PC v_k, search
    W for a pair (pos, neg) whose direction W[pos] − W[neg] is most aligned
    with v_k. The resulting contrasts can be fed into `basis='contrast'`
    plots via `axes_from_suggestions`.

    Args:
        viz_sample: VizSample.
        W: (V, d) tensor or array — `model.lm_head.weight`.
        weights: (N,) optional non-negative importance weights for the PCA.
            None → unweighted (marginal basis). For token-specific axes,
            pass the output of `token_weights()`.
        k_pc: number of leading PCs to explain.
        top_n: how many candidate contrasts to return per PC, ranked by score.
        score: 'cosine' (default) or 'projection'.
        candidate_pool: for score='cosine', top/bottom tokens (by v_k
            projection) to enumerate exhaustively.
        restrict_to: optional iterable of token ids to consider.
        decode: optional `list[int] -> str`; with verbose=True, prints
            suggestions with decoded strings.
        verbose: print a per-PC summary.

    Returns:
        list[dict], one entry per PC with keys: 'pc', 'eigval', 'frac_var',
        'axes' — list of (pos_id, neg_id, score) triples, ranked.
    """
    H = np.asarray(viz_sample.H, dtype=np.float32)
    if weights is None:
        pca_weights = np.ones(H.shape[0], dtype=np.float64)
    else:
        pca_weights = np.asarray(weights, dtype=np.float64)
    mu, V_full, eigvals = _weighted_centered_basis(H, pca_weights, center=True)

    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().float().numpy()
    else:
        W_np = np.asarray(W, dtype=np.float32)
    V_total, d = W_np.shape
    if d != V_full.shape[0]:
        raise ValueError(
            f"W has d={d} but viz_sample.H has d={V_full.shape[0]} — "
            f"mismatched checkpoints?"
        )

    if restrict_to is not None:
        id_map = np.asarray(list(restrict_to), dtype=np.int64)
        if id_map.size < 2:
            raise ValueError("restrict_to must contain at least 2 token ids.")
        W_use = W_np[id_map]
    else:
        id_map = np.arange(V_total, dtype=np.int64)
        W_use = W_np

    norms_sq = (np.linalg.norm(W_use, axis=1).astype(np.float64)) ** 2

    k_pc_eff = min(int(k_pc), V_full.shape[1])
    total_var = float(eigvals.sum())
    results = []

    for k in range(k_pc_eff):
        v = V_full[:, k].astype(np.float64)
        r = W_use.astype(np.float64) @ v                    # (V_use,)

        if score == 'projection':
            order = np.argsort(r)
            picks = []
            for n in range(min(top_n, len(r) - 1)):
                pos_local = int(order[-(n + 1)])
                neg_local = int(order[n])
                s = float(r[pos_local] - r[neg_local])
                picks.append((int(id_map[pos_local]),
                              int(id_map[neg_local]), s))
        elif score == 'cosine':
            T = min(int(candidate_pool), len(r) // 2)
            if T < 1:
                raise ValueError(
                    "candidate_pool too small relative to vocabulary size.")
            order = np.argsort(r)
            top = order[-T:]
            bot = order[:T]
            W_top = W_use[top].astype(np.float64)
            W_bot = W_use[bot].astype(np.float64)
            cross = W_top @ W_bot.T                          # (T, T)
            d2 = norms_sq[top][:, None] + norms_sq[bot][None, :] - 2 * cross
            d2 = np.maximum(d2, 1e-20)
            num = r[top][:, None] - r[bot][None, :]          # (T, T)
            cos = num / np.sqrt(d2)
            flat = cos.ravel()
            n_take = min(top_n, flat.size)
            idx = np.argpartition(-flat, n_take - 1)[:n_take]
            idx = idx[np.argsort(-flat[idx])]
            picks = []
            for ii in idx:
                ti, bi = np.unravel_index(int(ii), cos.shape)
                picks.append((int(id_map[top[ti]]),
                              int(id_map[bot[bi]]),
                              float(cos[ti, bi])))
        else:
            raise ValueError(
                f"score must be 'cosine' or 'projection', got {score!r}")

        results.append({
            'pc': k,
            'eigval': float(eigvals[k]),
            'frac_var': (float(eigvals[k] / total_var)
                         if total_var > 0 else 0.0),
            'axes': picks,
        })

    if verbose:
        for entry in results:
            print(f"PC{entry['pc'] + 1}  "
                  f"(var={entry['eigval']:.3g}, "
                  f"frac={entry['frac_var']:.3f}):")
            for pos, neg, s in entry['axes']:
                if decode is not None:
                    pl, nl = decode([pos]), decode([neg])
                    print(f"  {pl!r}  −  {nl!r}     {score}={s:+.3f}")
                else:
                    print(f"  id={pos}  −  id={neg}     {score}={s:+.3f}")

    return results


def axes_from_suggestions(suggestions, pc_pairs=((0, 1), (2, 3), (4, 5)),
                          rank=0):
    """Convert `suggest_contrast_axes` output into the `axes` arg expected
    by `basis='contrast'` plots.

    Picks the `rank`-th-best contrast per PC (default: the top suggestion)
    and assembles one ((pos_x, neg_x), (pos_y, neg_y)) tuple per PC pair.

    Example:
        suggestions = suggest_contrast_axes(vs, W, weights=omega, k_pc=6)
        axes = axes_from_suggestions(suggestions,
                                     pc_pairs=((0, 1), (2, 3), (4, 5)))
        plot_density(vs, W, Z, 'bank', stoi=stoi,
                     basis='contrast', axes=axes, decode=decode)
    """
    axes = []
    for (pi, pj) in pc_pairs:
        if pi >= len(suggestions) or pj >= len(suggestions):
            raise ValueError(
                f"pc_pair ({pi}, {pj}) exceeds suggestions length "
                f"{len(suggestions)} — increase k_pc when calling "
                f"suggest_contrast_axes."
            )
        if (rank >= len(suggestions[pi]['axes']) or
                rank >= len(suggestions[pj]['axes'])):
            raise ValueError(
                f"rank={rank} exceeds the number of axes per PC; pass a "
                f"larger top_n to suggest_contrast_axes."
            )
        ax_x = suggestions[pi]['axes'][rank][:2]
        ax_y = suggestions[pj]['axes'][rank][:2]
        axes.append((tuple(ax_x), tuple(ax_y)))
    return axes


def _check_pcs(pcs, k_pc):
    """Raise if any PC index in pcs >= k_pc (the cached number of PCs)."""
    for pi in pcs:
        if pi >= k_pc:
            raise ValueError(
                f"PC index {pi} exceeds cached k_pc={k_pc}. "
                f"Re-cache with a larger k_pc, or request lower PCs."
            )
