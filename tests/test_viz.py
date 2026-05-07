"""Tests for shape.viz (Stage 2.5 visualization sample + importance weights)."""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape.viz import (
    VizSample,
    build_viz_sample,
    effective_sample_size,
    load_decode,
    load_viz_sample,
    sample_token_instances,
    token_weights,
)


def _make_fake_extract(td, dataset='toy', d=4, n_valid=1200, V=12, seed=0,
                      min_context=4):
    """Write Stage-1-shaped outputs under `td`: `{dataset}_h_eff.npy` and
    `{dataset}_meta.json`. Returns (H, data, meta).

    Uses a 2-mode gaussian-mixture h_eff so SVD has meaningful structure.
    """
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
        'corpus_length': T,
        'block_size': T,
        'window': 0,
        'min_context': min_context,
        'N_valid': n_valid,
        'd': d,
        'V': V,
    }
    np.save(os.path.join(td, f'{dataset}_h_eff.npy'), H)
    with open(os.path.join(td, f'{dataset}_meta.json'), 'w') as f:
        json.dump(meta, f)
    return H, data, meta


def test_cache_and_load_roundtrip():
    d = 4
    V = 12
    N = 300
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=1200, V=V, seed=0)
        torch.manual_seed(0)
        W = torch.randn(V, d) * 0.3

        out = os.path.join(td, 'viz.npz')
        vs = build_viz_sample(
            td, 'toy', W, out, N=N, k_pc=3,
            compute_svd=True,
            logsumexp_batch_size=64,
            device='cpu', verbose=False,
        )

        assert vs.H.shape == (N, d)
        assert vs.log_Z_of_h.shape == (N,)
        assert vs.V_basis.shape == (d, d)
        assert vs.S.shape == (d,)
        assert vs.projections.shape == (N, 3)
        assert vs.row_index.shape == (N,)
        assert vs.row_index.dtype == np.int64
        # Subsample without replacement — all indices unique, in [0, n_valid)
        assert len(np.unique(vs.row_index)) == N
        assert vs.row_index.min() >= 0 and vs.row_index.max() < 1200
        assert np.all(np.isfinite(vs.H))
        assert np.all(np.isfinite(vs.log_Z_of_h))

        vs2 = load_viz_sample(out)
        assert np.array_equal(vs.H, vs2.H)
        assert np.array_equal(vs.log_Z_of_h, vs2.log_Z_of_h)
        assert np.array_equal(vs.V_basis, vs2.V_basis)
        assert np.array_equal(vs.projections, vs2.projections)
        assert np.array_equal(vs.row_index, vs2.row_index)
        assert vs.meta['N'] == N and vs.meta['d'] == d
        assert vs.meta['source'] == 'h_eff_memmap'
        assert vs.meta['svd_computed'] is True


def test_svd_deferred_by_default_and_lazy_on_global_plot():
    try:
        import plotnine  # noqa: F401
    except ImportError:
        return

    d = 4
    V = 6
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=600, V=V, seed=20)
        W = torch.randn(V, d) * 0.3
        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=80, k_pc=3,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )
        # Default: SVD deferred, empty arrays for V_basis/S/projections.
        assert not vs.has_svd
        assert vs.V_basis.size == 0 and vs.projections.size == 0
        assert vs.meta['svd_computed'] is False

        # Calling plot_global_density triggers lazy computation.
        from shape.viz import plot_global_density
        plot_global_density(vs, pcs=((0, 1),))
        assert vs.has_svd
        assert vs.V_basis.shape == (d, d)


def test_build_uses_all_rows_when_N_exceeds_memmap():
    d = 4
    V = 8
    n_valid = 80
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=n_valid, V=V, seed=1)
        W = torch.randn(V, d) * 0.3
        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=None, k_pc=3,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )
        assert vs.H.shape == (n_valid, d)
        assert np.array_equal(vs.row_index, np.arange(n_valid, dtype=np.int64))


def test_build_rejects_dimension_mismatch():
    d = 4
    V = 8
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=60, V=V, seed=2)
        W_wrong = torch.randn(V, d + 1) * 0.3
        try:
            build_viz_sample(td, 'toy', W_wrong,
                             os.path.join(td, 'viz.npz'),
                             N=40, device='cpu', verbose=False)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for W/h_eff dim mismatch")


def test_log_Z_matches_direct_softmax():
    """log_Z_of_h[i] should equal logsumexp(W @ H[i]) computed directly."""
    d = 4
    V = 20
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=1200, V=V, seed=3)
        torch.manual_seed(1)
        W = torch.randn(V, d) * 0.5

        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=80, k_pc=3,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )

        H_t = torch.from_numpy(vs.H)
        logits = H_t @ W.T                                    # (N, V)
        expected = torch.logsumexp(logits, dim=-1).numpy()
        np.testing.assert_allclose(vs.log_Z_of_h, expected, atol=1e-4, rtol=1e-4)


def test_svd_basis_is_orthonormal_and_projections_consistent():
    d = 4
    V = 8
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=1200, V=V, seed=4)
        W = torch.randn(V, d) * 0.3

        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=150, k_pc=d, compute_svd=True,
            logsumexp_batch_size=64, device='cpu', verbose=False,
        )

        gram = vs.V_basis.T @ vs.V_basis
        np.testing.assert_allclose(gram, np.eye(d), atol=1e-4)

        expected_proj = vs.H @ vs.V_basis[:, :d]
        np.testing.assert_allclose(vs.projections, expected_proj, atol=1e-4)

        # Singular values are non-increasing
        assert np.all(np.diff(vs.S) <= 1e-6)


def test_token_weights_match_explicit_softmax():
    """ω_i up to a constant equals softmax(W H[i])_t / Z_t."""
    d = 4
    V = 10
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=1200, V=V, seed=5)
        torch.manual_seed(3)
        W = torch.randn(V, d) * 0.4

        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=60, k_pc=3,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )

        t = 3
        Z_t = 0.25
        # Explicit per-sample p_t(h) / Z_t
        H_t = torch.from_numpy(vs.H)
        logits = H_t @ W.T
        p = torch.softmax(logits, dim=-1)
        expected = (p[:, t] / Z_t).numpy()

        # Un-stabilized — should match exactly (up to float precision)
        omega = token_weights(vs, W, t, Z_t, stabilize=False)
        np.testing.assert_allclose(omega, expected, rtol=1e-4, atol=1e-5)

        # Stabilized differs only by a positive scalar (same shape)
        omega_stab = token_weights(vs, W, t, Z_t, stabilize=True)
        ratio = omega_stab / (expected + 1e-30)
        np.testing.assert_allclose(ratio, ratio[0] * np.ones_like(ratio),
                                   rtol=1e-3, atol=1e-4)


def test_token_weights_rejects_nonpositive_Z():
    vs = VizSample(
        H=np.zeros((5, 3), dtype=np.float32),
        log_Z_of_h=np.zeros(5, dtype=np.float32),
        V_basis=np.eye(3, dtype=np.float32),
        S=np.ones(3, dtype=np.float32),
        projections=np.zeros((5, 3), dtype=np.float32),
        meta={},
    )
    try:
        token_weights(vs, np.zeros((4, 3), dtype=np.float32), 0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for Z_t=0")


def test_ess_bounded_by_N():
    w = np.ones(100, dtype=np.float32)
    assert abs(effective_sample_size(w) - 100.0) < 1e-4
    # Spike-y weights → small ESS
    w2 = np.zeros(100, dtype=np.float32)
    w2[0] = 1.0
    assert effective_sample_size(w2) == 1.0


def test_plot_functions_build_ggplot():
    """Smoke-test that plotting returns a plotnine.ggplot — skipped if not installed."""
    try:
        import plotnine  # noqa: F401
    except ImportError:
        return  # silently skip

    d = 4
    V = 8
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=1200, V=V, seed=6)
        W = torch.randn(V, d) * 0.3
        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=120, k_pc=4,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )

        from shape.viz import plot_global_density, plot_token_density
        import plotnine as pn
        pcs = ((0, 1), (2, 3))

        g = plot_global_density(vs, pcs=pcs)
        assert isinstance(g, pn.ggplot)

        t = plot_token_density(vs, W, 2, 0.1, pcs=pcs, token_label='x')
        assert isinstance(t, pn.ggplot)

        # senses overlay
        senses = np.random.RandomState(0).randn(2, d).astype(np.float32)
        t2 = plot_token_density(vs, W, 2, 0.1, pcs=pcs, senses=senses)
        assert isinstance(t2, pn.ggplot)


def test_plot_empirical_token_density_smoke():
    try:
        import plotnine  # noqa: F401
    except ImportError:
        return

    d = 4
    V = 6
    with tempfile.TemporaryDirectory() as td:
        H, data, meta = _make_fake_extract(td, d=d, n_valid=400, V=V, seed=7)
        W = torch.randn(V, d) * 0.3
        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=80, k_pc=4,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )

        from shape.viz import plot_empirical_token_density
        import plotnine as pn
        pcs = ((0, 1), (2, 3))

        # Pick a t that definitely appears as next-token somewhere
        next_tok = data[meta['min_context'] + 1 : meta['corpus_length']]
        t_present = int(np.bincount(next_tok.astype(np.int64),
                                    minlength=V).argmax())

        p = plot_empirical_token_density(
            vs, W, t_present, meta, data, H,
            pcs=pcs, basis='token', max_samples=64,
            rng=np.random.default_rng(0),
        )
        assert isinstance(p, pn.ggplot)

        p_global = plot_empirical_token_density(
            vs, W, t_present, meta, data, H,
            pcs=pcs, basis='global', max_samples=64,
            rng=np.random.default_rng(0),
        )
        assert isinstance(p_global, pn.ggplot)


def test_plot_token_comparison_builds_ggplot():
    try:
        import plotnine  # noqa: F401
    except ImportError:
        return

    d = 4
    V = 6
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=400, V=V, seed=21)
        W = torch.randn(V, d) * 0.3
        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=80, k_pc=3,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )

        from shape.viz import plot_token_comparison
        import plotnine as pn

        # Synthesize a plausible Z (uniform) and a couple of tokens.
        Z = np.full(V, 1.0 / V, dtype=np.float32)
        p = plot_token_comparison(
            vs, W, Z, tokens=[1, 3],
            labels=['a', 'b'],
            pcs=((0, 1), (1, 2)),
            basis='global',
            n_grid=30,
        )
        assert isinstance(p, pn.ggplot)
        # Global basis was absent before this call — should be lazy-cached now.
        assert vs.has_svd

        # basis='token' is rejected.
        try:
            plot_token_comparison(vs, W, Z, tokens=[1, 3],
                                  basis='token', n_grid=30)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for basis='token'")

        # Single token is rejected.
        try:
            plot_token_comparison(vs, W, Z, tokens=[1], n_grid=30)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for single token")


def test_plot_empirical_token_density_raises_when_token_absent():
    try:
        import plotnine  # noqa: F401
    except ImportError:
        return

    d = 4
    V = 6
    with tempfile.TemporaryDirectory() as td:
        H, data, meta = _make_fake_extract(td, d=d, n_valid=200, V=V, seed=8)
        W = torch.randn(V, d) * 0.3
        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=40, k_pc=3,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )
        from shape.viz import plot_empirical_token_density

        # Force an id that can't appear as a next-token (V is small, so use V+5)
        absent = V + 5
        try:
            plot_empirical_token_density(
                vs, W, absent, meta, data, H,
                pcs=((0, 1),), max_samples=32,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for absent token id")


def test_sample_token_instances_finds_next_token_positions():
    """Sampled corpus_pos+1 should equal the requested token id; projections match V_basis."""
    d = 4
    V = 8
    N_viz = 80
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=1200, V=V, seed=9)
        W = torch.randn(V, d) * 0.3
        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=N_viz, k_pc=4, compute_svd=True,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )

        T = 400
        rng_data = np.random.default_rng(0)
        data = rng_data.integers(0, V, size=T, dtype=np.int64).astype(np.uint16)
        min_ctx = 4
        meta = {'corpus_length': T, 'block_size': T,
                'window': 0, 'min_context': min_ctx}
        N_valid = T - min_ctx
        h_eff = rng_data.standard_normal((N_valid, d)).astype(np.float32)

        t_target = 3
        df = sample_token_instances(
            vs, meta, data, h_eff,
            t=t_target, k=5, context=6,
            decode=lambda ids: ','.join(str(int(i)) for i in ids),
            rng=np.random.default_rng(1),
        )

        # Wide form: one row per instance; projection columns for every PC
        k_returned = len(df)
        assert k_returned <= 5
        assert k_returned == df['row_index'].nunique()
        for p in range(vs.k_pc):
            assert f'pc{p + 1}' in df.columns

        # Every sampled corpus position has next-token == t_target
        for pos in df['corpus_pos']:
            assert int(data[pos + 1]) == t_target

        # Label ends with the target token (comma-separated), length == context
        for _, row in df.iterrows():
            ids = [int(x) for x in row['label'].split(',')]
            assert ids[-1] == t_target
            assert len(ids) == 6

        # Projections: pc1 should equal h_eff[row_index] @ V_basis[:, 0]
        V_basis = vs.V_basis[:, :vs.k_pc]
        for _, row in df.iterrows():
            expected = float(h_eff[int(row['row_index'])] @ V_basis[:, 0])
            np.testing.assert_allclose(row['pc1'], expected, atol=1e-5)


def test_sample_token_instances_random_when_w_is_none():
    d = 4
    V = 6
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=1200, V=V, seed=10)
        W = torch.randn(V, d) * 0.3
        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=60, k_pc=3,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )

        T = 200
        data = np.random.default_rng(0).integers(0, V, size=T, dtype=np.int64).astype(np.uint16)
        min_ctx = 4
        meta = {'corpus_length': T, 'block_size': T,
                'window': 0, 'min_context': min_ctx}
        h_eff = np.zeros((T - min_ctx, d), dtype=np.float32)

        df = sample_token_instances(
            vs, meta, data, h_eff,
            t=None, k=4, context=3,
            rng=np.random.default_rng(7),
        )
        assert len(df) == 4
        assert (df['corpus_pos'] >= min_ctx).all()
        assert (df['corpus_pos'] < T).all()


def test_sample_token_instances_rejects_mismatched_h_eff():
    d = 4
    V = 6
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=1200, V=V, seed=11)
        W = torch.randn(V, d) * 0.3
        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=40, k_pc=3,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )
        T = 100
        meta = {'corpus_length': T, 'block_size': T,
                'window': 0, 'min_context': 4}
        data = np.zeros(T, dtype=np.uint16)
        h_eff_wrong = np.zeros((T, d), dtype=np.float32)   # should be T-4
        try:
            sample_token_instances(vs, meta, data, h_eff_wrong,
                                   t=None, k=2)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for size mismatch")


def test_load_decode_char_meta_pickle():
    import pickle
    with tempfile.TemporaryDirectory() as td:
        ds = 'toy'
        os.makedirs(os.path.join(td, ds))
        stoi = {'a': 0, 'b': 1, 'c': 2}
        itos = {v: k for k, v in stoi.items()}
        with open(os.path.join(td, ds, 'meta.pkl'), 'wb') as f:
            pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': 3}, f)
        decode = load_decode(ds, data_dir=td)
        assert decode([0, 1, 2, 0]) == 'abca'


def test_plot_functions_accept_examples_overlay():
    try:
        import plotnine  # noqa: F401
    except ImportError:
        return

    d = 4
    V = 6
    with tempfile.TemporaryDirectory() as td:
        _make_fake_extract(td, d=d, n_valid=1200, V=V, seed=12)
        W = torch.randn(V, d) * 0.3
        vs = build_viz_sample(
            td, 'toy', W, os.path.join(td, 'viz.npz'),
            N=60, k_pc=4,
            logsumexp_batch_size=32, device='cpu', verbose=False,
        )
        T = 200
        data = np.random.default_rng(0).integers(0, V, size=T, dtype=np.int64).astype(np.uint16)
        meta = {'corpus_length': T, 'block_size': T,
                'window': 0, 'min_context': 4}
        h_eff = np.random.default_rng(0).standard_normal((T - 4, d)).astype(np.float32)

        pcs = ((0, 1), (2, 3))
        ex = sample_token_instances(vs, meta, data, h_eff,
                                    t=None, k=3, context=4,
                                    rng=np.random.default_rng(0))

        from shape.viz import plot_global_density, plot_token_density
        import plotnine as pn

        g = plot_global_density(vs, pcs=pcs, examples=ex)
        assert isinstance(g, pn.ggplot)

        t = plot_token_density(vs, W, 2, 0.1, pcs=pcs,
                               token_label='x', examples=ex)
        assert isinstance(t, pn.ggplot)


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
