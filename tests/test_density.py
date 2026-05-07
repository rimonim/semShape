"""Tests for shape.density (Stage 2 flow fitting)."""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape.density import FlowDensity, fit_flow, load_flow


def _make_toy_h_eff(N, d, seed=0, path=None):
    """Mixture of 2 gaussians in R^d, saved as .npy."""
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 2, size=N)
    mu1 = np.full(d, 1.5, dtype=np.float32)
    mu2 = np.full(d, -1.5, dtype=np.float32)
    X = np.where(
        comp[:, None] == 0,
        rng.standard_normal((N, d)).astype(np.float32) * 0.5 + mu1,
        rng.standard_normal((N, d)).astype(np.float32) * 0.5 + mu2,
    ).astype(np.float32)
    np.save(path, X)
    return X


def test_fit_flow_roundtrip_and_samples_finite():
    """Train a tiny flow and verify log_density is finite; save/load round-trip."""
    d = 4
    N = 2000
    with tempfile.TemporaryDirectory() as td:
        h_path = os.path.join(td, 'h.npy')
        X = _make_toy_h_eff(N, d, seed=0, path=h_path)

        flow_path = os.path.join(td, 'flow.pt')
        fd = fit_flow(
            h_path,
            out_path=flow_path,
            transforms=3,
            hidden_features=(32, 32),
            bins=6,
            epochs=2,
            batch_size=256,
            lr=5e-3,
            val_frac=0.05,
            device='cpu',
            verbose=False,
        )

        # log_density finite on training points
        X_t = torch.from_numpy(X[:50])
        lp = fd.log_density(X_t)
        assert lp.shape == (50,)
        assert torch.isfinite(lp).all(), "log_density has non-finite values"

        # samples are finite and have roughly the right shape
        s = fd.sample(100)
        assert s.shape == (100, d)
        assert torch.isfinite(s).all()

        # Round-trip via save/load: log_density identical
        fd2, meta = load_flow(flow_path, device='cpu')
        lp2 = fd2.log_density(X_t)
        assert torch.allclose(lp, lp2, atol=1e-6), \
            f"load mismatch: max |diff| = {(lp - lp2).abs().max().item():.2e}"


def test_flow_log_density_higher_on_modes_than_between():
    """After training on 2-mode data, log_density should be higher near modes
    than at the origin (between modes)."""
    d = 4
    N = 4000
    with tempfile.TemporaryDirectory() as td:
        h_path = os.path.join(td, 'h.npy')
        _make_toy_h_eff(N, d, seed=1, path=h_path)

        flow_path = os.path.join(td, 'flow.pt')
        fd = fit_flow(
            h_path,
            out_path=flow_path,
            transforms=4,
            hidden_features=(64, 64),
            bins=8,
            epochs=8,
            batch_size=512,
            lr=3e-3,
            val_frac=0.0,
            device='cpu',
            verbose=False,
        )

        mode1 = torch.full((1, d), 1.5)
        mode2 = torch.full((1, d), -1.5)
        origin = torch.zeros(1, d)

        lp_mode1 = fd.log_density(mode1).item()
        lp_mode2 = fd.log_density(mode2).item()
        lp_origin = fd.log_density(origin).item()

        assert lp_mode1 > lp_origin, \
            f"mode1 ({lp_mode1:.3f}) should beat origin ({lp_origin:.3f})"
        assert lp_mode2 > lp_origin, \
            f"mode2 ({lp_mode2:.3f}) should beat origin ({lp_origin:.3f})"


def test_flow_change_of_variable_is_consistent():
    """Verify that log_density = flow.log_prob(standardized) - Σ log σ."""
    d = 3
    N = 1000
    with tempfile.TemporaryDirectory() as td:
        h_path = os.path.join(td, 'h.npy')
        X = _make_toy_h_eff(N, d, seed=2, path=h_path)

        flow_path = os.path.join(td, 'flow.pt')
        fd = fit_flow(
            h_path,
            out_path=flow_path,
            transforms=3,
            hidden_features=(32,),
            bins=6,
            epochs=1,
            batch_size=256,
            lr=1e-3,
            val_frac=0.0,
            device='cpu',
            verbose=False,
        )

        h = torch.from_numpy(X[:20])
        h_std = (h - fd.mean) / fd.std
        flow_lp = fd.flow().log_prob(h_std)
        expected = flow_lp - float(torch.log(fd.std).sum())
        got = fd.log_density(h)
        assert torch.allclose(got, expected, atol=1e-6)


if __name__ == '__main__':
    test_funcs = [g for n, g in globals().items() if n.startswith('test_') and callable(g)]
    for fn in test_funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            raise
    print(f"\nAll {len(test_funcs)} tests passed.")
