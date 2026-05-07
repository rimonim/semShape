"""Tests for shape.polysemy."""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape.polysemy import (
    SenseDecomposition,
    fit_senses,
    fit_weighted_gmm,
    sense_examples,
    sense_weights,
    total_variance_decomposition,
)
from shape.viz import VizSample, build_viz_sample, token_weights


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _two_mode_h(n_per, d, mu1, mu2, sigma=0.4, seed=0):
    """Sample n_per from each of two isotropic Gaussians at mu1, mu2."""
    rng = np.random.default_rng(seed)
    H1 = rng.standard_normal((n_per, d)).astype(np.float32) * sigma + mu1
    H2 = rng.standard_normal((n_per, d)).astype(np.float32) * sigma + mu2
    H = np.concatenate([H1, H2], axis=0).astype(np.float32)
    rng.shuffle(H)
    return H


def _build_synth_extract(td, *, dataset='toy', d=6, V=10, n_per=900,
                         mu1=None, mu2=None, sigma=0.4, seed=0,
                         min_context=4):
    """Write a fake extract dir with a 2-mode h_eff so build_viz_sample loads."""
    if mu1 is None:
        mu1 = np.full(d, 1.5, dtype=np.float32)
    if mu2 is None:
        mu2 = np.full(d, -1.5, dtype=np.float32)
    H = _two_mode_h(n_per, d, mu1, mu2, sigma=sigma, seed=seed)
    n_valid = H.shape[0]
    T = n_valid + min_context
    rng = np.random.default_rng(seed)
    data = rng.integers(0, V, size=T, dtype=np.uint16)
    meta = {
        'corpus_length': T, 'block_size': T, 'window': 0,
        'min_context': min_context, 'N_valid': n_valid, 'd': d, 'V': V,
    }
    np.save(os.path.join(td, f'{dataset}_h_eff.npy'), H)
    with open(os.path.join(td, f'{dataset}_meta.json'), 'w') as f:
        json.dump(meta, f)
    return H, data, meta


def _viz_sample_from(td, dataset, W, *, N=None, k_pc=4):
    return build_viz_sample(
        td, dataset, W, os.path.join(td, 'viz.npz'),
        N=N, k_pc=k_pc,
        logsumexp_batch_size=64, device='cpu', verbose=False,
    )


def _empirical_Z(vs, W):
    H_t = torch.from_numpy(vs.H).float()
    W_t = W.detach().float() if torch.is_tensor(W) else torch.from_numpy(W).float()
    p = torch.softmax(H_t @ W_t.T, dim=-1).numpy()
    return p.mean(axis=0).astype(np.float64)


class _ConstFlow:
    """Stub FlowDensity: log_density(H) returns a constant per row."""
    def __init__(self, value=0.0):
        self.value = float(value)

    def log_density(self, H):
        if torch.is_tensor(H):
            return torch.full((H.shape[0],), self.value, dtype=torch.float32)
        return np.full((H.shape[0],), self.value, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_fit_weighted_gmm_recovers_two_modes_uniform_weights():
    """With uniform weights, GMM on a clean 2-mode mixture recovers π≈0.5
    and means close to the true mode locations."""
    d = 6
    mu1 = np.full(d, 1.8, dtype=np.float32)
    mu2 = np.full(d, -1.8, dtype=np.float32)
    H = _two_mode_h(800, d, mu1, mu2, sigma=0.35, seed=0)
    w = np.ones(H.shape[0], dtype=np.float32)

    fit = fit_weighted_gmm(H, w, K=2, project_pcs=None,
                           rng=np.random.default_rng(7))
    pi = fit['pi']
    means = fit['means']
    # Sort components by mean[0] for stable identification (mu1[0]>0, mu2[0]<0)
    order = np.argsort(means[:, 0])[::-1]
    pi = pi[order]
    means = means[order]
    np.testing.assert_allclose(pi, [0.5, 0.5], atol=0.08)
    np.testing.assert_allclose(means[0], mu1, atol=0.2)
    np.testing.assert_allclose(means[1], mu2, atol=0.2)


def test_fit_senses_token_conditional_yields_two_centroids():
    """A token whose W-row points along the first mode should produce a
    decomposition where one ν_h sits near that mode."""
    with tempfile.TemporaryDirectory() as td:
        d = 6
        mu1 = np.full(d, 1.8, dtype=np.float32)
        mu2 = np.full(d, -1.8, dtype=np.float32)
        _build_synth_extract(td, d=d, V=8, n_per=700,
                             mu1=mu1, mu2=mu2, seed=1)
        # W[t]·h is large when h ≈ mu1 (positive), so ω_i emphasizes mode 1
        # but mode 2 still gets some weight: tweak W so both modes contribute.
        torch.manual_seed(1)
        W = torch.randn(8, d) * 0.3
        # Make token 3 a "split" token: its W-row is near zero so weights
        # are roughly uniform over both modes.
        W[3] = torch.zeros(d)
        vs = _viz_sample_from(td, 'toy', W, k_pc=min(d, 4))
        Z = _empirical_Z(vs, W)
        t = 3

        decomp = fit_senses(vs, W, t, float(Z[t]),
                            field='token_conditional', K=2,
                            rng=np.random.default_rng(11),
                            project_pcs=None)
        decomp = decomp.sort_by_weight()
        assert decomp.K == 2
        # Each mode should be claimed by one centroid.
        nu = decomp.nu_h
        d_to_mu1 = np.linalg.norm(nu - mu1[None, :], axis=1)
        d_to_mu2 = np.linalg.norm(nu - mu2[None, :], axis=1)
        # Pair components to modes: each mode is closest to a different component
        assert (np.argmin(d_to_mu1) != np.argmin(d_to_mu2))
        assert d_to_mu1.min() < 0.5
        assert d_to_mu2.min() < 0.5


def test_total_variance_decomposition_trace_identity():
    """tr(within) + tr(between) == tr(total) by construction."""
    with tempfile.TemporaryDirectory() as td:
        d = 6
        _build_synth_extract(td, d=d, V=8, n_per=600, seed=2)
        torch.manual_seed(2)
        W = torch.randn(8, d) * 0.4
        vs = _viz_sample_from(td, 'toy', W, k_pc=4)
        Z = _empirical_Z(vs, W)
        t = 1
        decomp = fit_senses(vs, W, t, float(Z[t]),
                            field='token_conditional', K=2,
                            rng=np.random.default_rng(3),
                            project_pcs=None)
        # Y-space
        out = total_variance_decomposition(decomp, in_Y_space=True)
        assert abs(out['within_trace'] + out['between_trace']
                   - out['total_trace']) < 1e-5
        assert 0.0 <= out['within_fraction'] <= 1.0
        assert 0.0 <= out['between_fraction'] <= 1.0
        # h-space
        out_h = total_variance_decomposition(decomp, in_Y_space=False)
        assert abs(out_h['within_trace'] + out_h['between_trace']
                   - out_h['total_trace']) < 1e-5


def test_K_auto_picks_two_for_clean_bimodal():
    """BIC should choose K=2 (allow ±1 slack on small samples)."""
    with tempfile.TemporaryDirectory() as td:
        d = 6
        mu1 = np.full(d, 2.0, dtype=np.float32)
        mu2 = np.full(d, -2.0, dtype=np.float32)
        _build_synth_extract(td, d=d, V=6, n_per=900,
                             mu1=mu1, mu2=mu2, sigma=0.3, seed=4)
        torch.manual_seed(4)
        W = torch.randn(6, d) * 0.2
        W[2] = torch.zeros(d)
        vs = _viz_sample_from(td, 'toy', W, k_pc=4)
        Z = _empirical_Z(vs, W)
        t = 2
        decomp = fit_senses(vs, W, t, float(Z[t]),
                            field='token_conditional', K='auto', K_max=4,
                            rng=np.random.default_rng(0),
                            project_pcs=None)
        assert decomp.K in (2, 3), f"expected K∈{{2,3}}, got {decomp.K}"


def test_distinctiveness_field_requires_flow():
    """field='distinctiveness' without flow → ValueError."""
    with tempfile.TemporaryDirectory() as td:
        d = 4
        _build_synth_extract(td, d=d, V=6, n_per=300, seed=5)
        torch.manual_seed(5)
        W = torch.randn(6, d) * 0.3
        vs = _viz_sample_from(td, 'toy', W, k_pc=2)
        Z = _empirical_Z(vs, W)
        try:
            sense_weights(vs, W, 0, float(Z[0]), field='distinctiveness',
                          flow=None)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError when flow is missing")


def test_distinctiveness_with_const_flow_matches_token_conditional():
    """Constant log ĝ → distinctiveness weights ∝ token-conditional weights;
    fits should agree (mixtures are scale-invariant in the weights)."""
    with tempfile.TemporaryDirectory() as td:
        d = 4
        _build_synth_extract(td, d=d, V=6, n_per=400, seed=6)
        torch.manual_seed(6)
        W = torch.randn(6, d) * 0.3
        W[2] = torch.zeros(d)
        vs = _viz_sample_from(td, 'toy', W, k_pc=2)
        Z = _empirical_Z(vs, W)
        t = 2
        Z_t = float(Z[t])

        w_tok = sense_weights(vs, W, t, Z_t, field='token_conditional')
        w_dis = sense_weights(vs, W, t, Z_t, field='distinctiveness',
                              flow=_ConstFlow(0.0))
        # Both must be non-negative, finite, and proportional to each other.
        assert np.all(np.isfinite(w_tok))
        assert np.all(np.isfinite(w_dis))
        assert (w_tok > 0).any() and (w_dis > 0).any()
        ratio = w_dis[w_tok > 1e-30] / w_tok[w_tok > 1e-30]
        np.testing.assert_allclose(ratio, ratio.mean(), rtol=1e-4)


def test_distinctiveness_field_shifts_centroids_when_g_varies():
    """A flow that's high at one mode and low at the other should pull the
    distinctiveness fit's centroids away from the high-g mode."""
    with tempfile.TemporaryDirectory() as td:
        d = 4
        mu1 = np.full(d, 1.8, dtype=np.float32)
        mu2 = np.full(d, -1.8, dtype=np.float32)
        _build_synth_extract(td, d=d, V=6, n_per=600,
                             mu1=mu1, mu2=mu2, seed=7)
        torch.manual_seed(7)
        # Token 2 weights both modes evenly
        W = torch.randn(6, d) * 0.3
        W[2] = torch.zeros(d)
        vs = _viz_sample_from(td, 'toy', W, k_pc=2)
        Z = _empirical_Z(vs, W)
        t = 2
        Z_t = float(Z[t])

        # A flow that says ĝ is much higher near mu1 than mu2:
        # log_density(h) = -‖h - mu1‖² (so mode 1 has higher density)
        class _BiasedFlow:
            def log_density(self, H):
                if torch.is_tensor(H):
                    diff = H - torch.from_numpy(mu1).to(H.dtype)
                    return -(diff ** 2).sum(dim=-1)
                diff = H - mu1[None, :]
                return -(diff ** 2).sum(axis=-1)

        decomp_tc = fit_senses(vs, W, t, Z_t,
                               field='token_conditional', K=1,
                               rng=np.random.default_rng(0),
                               project_pcs=None)
        decomp_di = fit_senses(vs, W, t, Z_t,
                               field='distinctiveness', K=1, flow=_BiasedFlow(),
                               rng=np.random.default_rng(0),
                               project_pcs=None)
        # Token-conditional centroid sits near (mu1+mu2)/2 ≈ 0; distinctiveness
        # centroid should be biased *away* from mu1 (i.e. closer to mu2).
        center_tc = decomp_tc.nu_h[0]
        center_di = decomp_di.nu_h[0]
        d_tc_to_mu1 = float(np.linalg.norm(center_tc - mu1))
        d_di_to_mu1 = float(np.linalg.norm(center_di - mu1))
        assert d_di_to_mu1 > d_tc_to_mu1 + 0.1, (
            f"distinctiveness centroid not pushed away from mu1: "
            f"d_tc={d_tc_to_mu1:.2f}, d_di={d_di_to_mu1:.2f}"
        )


def test_sense_examples_returns_expected_shape_and_ranking():
    """sense_examples returns K*n_per_sense rows and assigns the right
    h-vectors to the right sense."""
    with tempfile.TemporaryDirectory() as td:
        d = 4
        mu1 = np.full(d, 1.8, dtype=np.float32)
        mu2 = np.full(d, -1.8, dtype=np.float32)
        H, data, meta = _build_synth_extract(
            td, d=d, V=6, n_per=300, mu1=mu1, mu2=mu2, sigma=0.3, seed=8)
        torch.manual_seed(8)
        W = torch.randn(6, d) * 0.3
        W[2] = torch.zeros(d)
        vs = _viz_sample_from(td, 'toy', W, k_pc=2)
        Z = _empirical_Z(vs, W)
        t = 2
        decomp = fit_senses(vs, W, t, float(Z[t]),
                            field='token_conditional', K=2,
                            rng=np.random.default_rng(0),
                            project_pcs=None)

        df = sense_examples(decomp, vs, meta, data,
                            n_per_sense=5, context=3,
                            rank_by='weighted_responsibility')
        assert len(df) == 2 * 5
        assert set(df.columns) == {
            'sense_id', 'rank', 'score', 'corpus_pos',
            'context_before', 'focus', 'context_after',
        }
        # Each sense's top examples should have h_eff vectors closer to its
        # own centroid than to the other sense's centroid.
        H_to_corpus = np.asarray(meta['N_valid'])  # noqa: not used; clarity
        for k in range(2):
            sub = df[df['sense_id'] == k]
            # Find each row's h_eff via corpus_pos → row in original H
            # (since we built data with min_context=4 and N_valid corresponds
            # to positions [4, 4+N_valid), the h_eff row index = pos - 4)
            row_ids = (sub['corpus_pos'].values - meta['min_context']).astype(int)
            chosen_h = vs.H[vs.row_index.searchsorted(row_ids)] \
                if vs.row_index.size > 0 else H[row_ids]
            d_own = np.linalg.norm(chosen_h - decomp.nu_h[k][None, :], axis=1)
            d_other = np.linalg.norm(
                chosen_h - decomp.nu_h[1 - k][None, :], axis=1)
            # Majority of the top examples should be closer to the own centroid
            assert (d_own < d_other).mean() >= 0.6


def test_sense_examples_filter_next_token_restricts_rows():
    """filter_next_token=True restricts rows to those where data[pos+1] == t."""
    with tempfile.TemporaryDirectory() as td:
        d = 4
        H, data, meta = _build_synth_extract(td, d=d, V=5, n_per=200, seed=9)
        torch.manual_seed(9)
        W = torch.randn(5, d) * 0.3
        vs = _viz_sample_from(td, 'toy', W, k_pc=2)
        Z = _empirical_Z(vs, W)
        # Pick a token that actually appears as a "next token" enough times
        positions_count = np.bincount(data[meta['min_context'] + 1:],
                                      minlength=meta['V'])
        t = int(np.argmax(positions_count))
        decomp = fit_senses(vs, W, t, float(Z[t]),
                            field='token_conditional', K=2,
                            rng=np.random.default_rng(0),
                            project_pcs=None)
        df = sense_examples(decomp, vs, meta, data,
                            n_per_sense=4, context=2,
                            rank_by='responsibility',
                            filter_next_token=True)
        # Every selected row must have data[corpus_pos + 1] == t.
        for _, r in df.iterrows():
            assert int(data[int(r['corpus_pos']) + 1]) == t


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
