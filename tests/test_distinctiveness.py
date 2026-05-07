"""Tests for shape.distinctiveness."""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape.distinctiveness import (
    distinctiveness_field,
    expected_distinctiveness,
    expected_distinctiveness_all,
)
from shape.viz import (
    VizSample,
    build_viz_sample,
    effective_sample_size,
    token_weights,
)


def _make_fake_extract(td, dataset='toy', d=4, n_valid=1200, V=12, seed=0,
                      min_context=4):
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 2, size=n_valid)
    mu1 = np.full(d, 1.2, dtype=np.float32)
    mu2 = np.full(d, -1.2, dtype=np.float32)
    H = np.where(
        comp[:, None] == 0,
        rng.standard_normal((n_valid, d)).astype(np.float32) * 0.4 + mu1,
        rng.standard_normal((n_valid, d)).astype(np.float32) * 0.4 + mu2,
    ).astype(np.float32)
    T = n_valid + min_context
    data = rng.integers(0, V, size=T, dtype=np.uint16)
    meta = {
        'corpus_length': T, 'block_size': T, 'window': 0,
        'min_context': min_context, 'N_valid': n_valid, 'd': d, 'V': V,
    }
    np.save(os.path.join(td, f'{dataset}_h_eff.npy'), H)
    with open(os.path.join(td, f'{dataset}_meta.json'), 'w') as f:
        json.dump(meta, f)
    return H, data, meta


def _build_vs(td, *, d=4, V=10, n_valid=1200, N=80, seed=0, W_scale=0.4):
    _make_fake_extract(td, d=d, n_valid=n_valid, V=V, seed=seed)
    torch.manual_seed(seed)
    W = torch.randn(V, d) * W_scale
    vs = build_viz_sample(
        td, 'toy', W, os.path.join(td, 'viz.npz'),
        N=N, k_pc=min(d, 4),
        logsumexp_batch_size=32, device='cpu', verbose=False,
    )
    return vs, W


def _empirical_Z(vs, W):
    """Marginal probabilities estimated on the cached H — matches what
    Stage 1 would write up to subsample noise. Used so Z_t is consistent
    with the importance-sampling reduction performed on this same H."""
    H_t = torch.from_numpy(vs.H).float()
    W_t = W.detach().float() if torch.is_tensor(W) else torch.from_numpy(W).float()
    p = torch.softmax(H_t @ W_t.T, dim=-1).numpy()         # (N, V)
    return p.mean(axis=0).astype(np.float64)


def test_distinctiveness_field_matches_token_weights():
    """exp(D_t(h)) ≈ token_weights(stabilize=False)."""
    with tempfile.TemporaryDirectory() as td:
        vs, W = _build_vs(td, seed=1)
        Z = _empirical_Z(vs, W)
        t = 3
        Z_t = float(Z[t])
        D = distinctiveness_field(vs, W, t, Z_t)
        omega = token_weights(vs, W, t, Z_t, stabilize=False)
        np.testing.assert_allclose(np.exp(D.astype(np.float64)),
                                   omega.astype(np.float64),
                                   rtol=1e-4, atol=1e-6)


def test_distinctiveness_field_rejects_nonpositive_Z():
    with tempfile.TemporaryDirectory() as td:
        vs, W = _build_vs(td, seed=2)
        try:
            distinctiveness_field(vs, W, 0, 0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for Z_t=0")


def test_expected_distinctiveness_nonnegative():
    """KL ≥ 0 — sample estimates should be non-negative within slack."""
    with tempfile.TemporaryDirectory() as td:
        vs, W = _build_vs(td, seed=3, N=200)
        Z = _empirical_Z(vs, W)
        for t in range(W.shape[0]):
            if Z[t] <= 0:
                continue
            out = expected_distinctiveness(vs, W, t, float(Z[t]))
            assert out['D_t'] >= -1e-6, f"D_t={out['D_t']} for t={t}"
            assert out['ess'] > 0
            assert out['ess'] <= vs.N + 1e-3


def test_expected_distinctiveness_zero_when_omega_is_constant():
    """If ω_i is constant across cached samples, D_t = 0 and ESS = N.

    Constructed directly: with all-zero W (logits ≡ 0), softmax is uniform 1/V,
    so log_Z_of_h = log V is constant and p_t(h) = 1/V. Setting Z_t = 1/V
    then gives ω_i ≡ 1 exactly, the analytic g(h|t) = g(h) case.
    """
    N, d, V = 200, 4, 8
    rng = np.random.default_rng(0)
    H = rng.standard_normal((N, d)).astype(np.float32)
    W = np.zeros((V, d), dtype=np.float32)
    log_Z_of_h = np.full(N, np.log(V), dtype=np.float32)
    vs = VizSample(
        H=H,
        log_Z_of_h=log_Z_of_h,
        V_basis=np.empty((d, 0), dtype=np.float32),
        S=np.empty(0, dtype=np.float32),
        projections=np.empty((N, 0), dtype=np.float32),
        meta={'N': N, 'd': d, 'V': V},
    )
    out = expected_distinctiveness(vs, W, t=2, Z_t=1.0 / V)
    assert abs(out['D_t']) < 1e-6, f"expected D_t≈0, got {out['D_t']}"
    assert abs(out['ess'] - N) < 1e-3
    assert abs(out['sum_omega'] - 1.0) < 1e-6


def test_expected_distinctiveness_all_matches_per_token():
    with tempfile.TemporaryDirectory() as td:
        vs, W = _build_vs(td, seed=5, N=120, V=10)
        Z = _empirical_Z(vs, W)
        out = expected_distinctiveness_all(vs, W, Z, batch_size=4,
                                           device='cpu', verbose=False)
        assert out['D'].shape == (W.shape[0],)
        assert out['ess'].shape == (W.shape[0],)
        assert out['token_ids'].shape == (W.shape[0],)
        for t in (0, 3, 5, 9):
            ref = expected_distinctiveness(vs, W, t, float(Z[t]))
            np.testing.assert_allclose(out['D'][t], ref['D_t'],
                                       rtol=1e-3, atol=1e-4)
            np.testing.assert_allclose(out['ess'][t], ref['ess'],
                                       rtol=1e-3, atol=1e-3)


def test_expected_distinctiveness_all_target_tokens_subset():
    with tempfile.TemporaryDirectory() as td:
        vs, W = _build_vs(td, seed=6, N=80, V=8)
        Z = _empirical_Z(vs, W)
        targets = [1, 4, 7]
        out = expected_distinctiveness_all(vs, W, Z,
                                           target_tokens=targets,
                                           batch_size=2, device='cpu')
        assert out['D'].shape == (3,)
        assert list(out['token_ids']) == targets
        for i, t in enumerate(targets):
            ref = expected_distinctiveness(vs, W, t, float(Z[t]))
            np.testing.assert_allclose(out['D'][i], ref['D_t'],
                                       rtol=1e-3, atol=1e-4)


def test_expected_distinctiveness_all_handles_zero_Z():
    """Tokens with Z=0 yield NaN D and 0 ESS without crashing."""
    with tempfile.TemporaryDirectory() as td:
        vs, W = _build_vs(td, seed=7, N=60, V=6)
        Z = _empirical_Z(vs, W)
        Z[0] = 0.0                                  # synthetic absent token
        out = expected_distinctiveness_all(vs, W, Z, batch_size=3, device='cpu')
        assert np.isnan(out['D'][0])
        assert out['ess'][0] == 0.0
        # Other tokens still produce finite values.
        assert np.all(np.isfinite(out['D'][1:]))


def test_ess_matches_effective_sample_size_helper():
    with tempfile.TemporaryDirectory() as td:
        vs, W = _build_vs(td, seed=8, N=100)
        Z = _empirical_Z(vs, W)
        t = 4
        omega = token_weights(vs, W, t, float(Z[t]), stabilize=False)
        ref_ess = effective_sample_size(omega)
        out = expected_distinctiveness(vs, W, t, float(Z[t]))
        np.testing.assert_allclose(out['ess'], ref_ess, rtol=1e-4, atol=1e-3)


def test_plot_token_distinctiveness_smoke():
    """Plot returns a plotnine.ggplot for each basis."""
    try:
        import plotnine  # noqa: F401
    except ImportError:
        return

    with tempfile.TemporaryDirectory() as td:
        vs, W = _build_vs(td, seed=9, N=120, V=8)
        Z = _empirical_Z(vs, W)
        t = 2
        Z_t = float(Z[t])

        from shape.viz import plot_token_distinctiveness
        import plotnine as pn

        pcs = ((0, 1), (2, 3))

        p_token = plot_token_distinctiveness(
            vs, W, t, Z_t, pcs=pcs, basis='token', n_grid=30)
        assert isinstance(p_token, pn.ggplot)

        p_global = plot_token_distinctiveness(
            vs, W, t, Z_t, pcs=pcs, basis='global', n_grid=30)
        assert isinstance(p_global, pn.ggplot)

        axes = (((1, 0), (3, 2)),)
        p_contrast = plot_token_distinctiveness(
            vs, W, t, Z_t, basis='contrast', axes=axes, n_grid=30)
        assert isinstance(p_contrast, pn.ggplot)

        # clip applies without crashing
        p_clipped = plot_token_distinctiveness(
            vs, W, t, Z_t, pcs=((0, 1),), basis='token',
            n_grid=24, clip=(-2, 2))
        assert isinstance(p_clipped, pn.ggplot)


if __name__ == '__main__':
    test_funcs = [g for n, g in globals().items()
                  if n.startswith('test_') and callable(g)]
    for fn in test_funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            raise
    print(f"\nAll {len(test_funcs)} tests passed.")
