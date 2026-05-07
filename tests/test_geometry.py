"""Tests for shape.geometry."""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape.geometry import (
    compute_A,
    degenerate_direction,
    ilr,
    ilr_apply,
    ilr_apply_T,
    ilr_basis_dense,
    project_h,
)


def test_psi_row_sums_zero():
    """Ψ · 1 = 0 by construction."""
    for V in (3, 10, 50):
        Psi = ilr_basis_dense(V)
        row_sums = Psi.sum(dim=1)
        assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-12)


def test_psi_orthonormal_rows():
    """Ψ Ψᵀ = I."""
    for V in (3, 10, 50):
        Psi = ilr_basis_dense(V)
        gram = Psi @ Psi.T
        eye = torch.eye(V - 1, dtype=torch.float64)
        assert torch.allclose(gram, eye, atol=1e-12)


def test_ilr_apply_matches_dense():
    """Sparse ilr_apply equals dense Ψv."""
    torch.manual_seed(0)
    for V in (3, 17, 100):
        Psi = ilr_basis_dense(V)
        v = torch.randn(V, dtype=torch.float64)
        expected = Psi @ v
        got = ilr_apply(v)
        assert torch.allclose(got, expected, atol=1e-12)


def test_ilr_apply_batched():
    """ilr_apply works along last dim of batched inputs."""
    torch.manual_seed(0)
    V = 17
    Psi = ilr_basis_dense(V)
    v = torch.randn(4, 5, V, dtype=torch.float64)
    expected = torch.einsum('ij,...j->...i', Psi, v)
    got = ilr_apply(v)
    assert torch.allclose(got, expected, atol=1e-12)


def test_ilr_apply_T_matches_dense():
    """Sparse ilr_apply_T equals dense Ψᵀ y."""
    torch.manual_seed(1)
    for V in (3, 17, 100):
        Psi = ilr_basis_dense(V)
        y = torch.randn(V - 1, dtype=torch.float64)
        expected = Psi.T @ y
        got = ilr_apply_T(y)
        assert torch.allclose(got, expected, atol=1e-12)


def test_ilr_T_then_ilr_is_identity():
    """Ψ Ψᵀ = I ⇒ ilr_apply(ilr_apply_T(y)) = y."""
    torch.manual_seed(2)
    V = 50
    y = torch.randn(V - 1, dtype=torch.float64)
    assert torch.allclose(ilr_apply(ilr_apply_T(y)), y, atol=1e-12)


def test_compute_A_matches_dense():
    """compute_A equals Ψ W computed densely."""
    torch.manual_seed(3)
    for V, d in ((3, 2), (17, 8), (100, 32)):
        W = torch.randn(V, d, dtype=torch.float64)
        Psi = ilr_basis_dense(V)
        expected = Psi @ W
        got = compute_A(W)
        assert torch.allclose(got, expected, atol=1e-12)


def test_ilr_softmax_identity():
    """ilr(softmax(Wh)) == A h, for any W, any h (bias=False model)."""
    torch.manual_seed(4)
    for V, d in ((10, 4), (100, 32), (500, 64)):
        W = torch.randn(V, d, dtype=torch.float64) * 0.5
        h = torch.randn(d, dtype=torch.float64)
        z = W @ h
        p = F.softmax(z, dim=-1)
        lhs = ilr(p)
        A = compute_A(W)
        rhs = A @ h
        assert torch.allclose(lhs, rhs, atol=1e-10), \
            f"V={V}, d={d}: max abs diff {(lhs - rhs).abs().max().item():.2e}"


def test_ilr_softmax_identity_batched():
    """ILR-softmax identity, batched over h."""
    torch.manual_seed(5)
    V, d = 50, 16
    W = torch.randn(V, d, dtype=torch.float64) * 0.5
    h = torch.randn(3, 7, d, dtype=torch.float64)
    z = h @ W.T
    p = F.softmax(z, dim=-1)
    lhs = ilr(p)
    A = compute_A(W)
    rhs = h @ A.T
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_ilr_softmax_identity_fp32_tolerance():
    """ILR-softmax identity holds to fp32 tolerance on realistic-scale W."""
    torch.manual_seed(6)
    V, d = 500, 64
    W = torch.randn(V, d, dtype=torch.float32) * 0.1
    h = torch.randn(d, dtype=torch.float32)
    p = F.softmax(W @ h, dim=-1)
    lhs = ilr(p)
    A = compute_A(W)
    rhs = A @ h
    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-3)


def test_degenerate_direction_when_one_in_colW():
    """
    When 1 ∈ col(W) exactly, the degenerate direction v satisfies Wv ∝ 1,
    and A v should be zero up to fp precision.
    """
    torch.manual_seed(7)
    V, d = 100, 16
    W = torch.randn(V, d, dtype=torch.float64) * 0.3
    # Force 1 ∈ col(W): set first column to 1/sqrt(V).
    W[:, 0] = 1.0 / math.sqrt(V)
    v = degenerate_direction(W)
    # Unit norm
    assert torch.isclose(torch.linalg.norm(v), torch.tensor(1.0, dtype=v.dtype))
    # Wv should be proportional to 1
    Wv = W @ v
    mean_Wv = Wv.mean()
    assert torch.allclose(Wv, mean_Wv.expand_as(Wv), atol=1e-10), \
        f"Wv deviation from constant: {(Wv - mean_Wv).abs().max().item():.2e}"
    # A v ≈ 0
    A = compute_A(W)
    Av = A @ v
    assert torch.allclose(Av, torch.zeros_like(Av), atol=1e-10), \
        f"||Av|| = {torch.linalg.norm(Av).item():.2e}"


def test_project_h_orthogonalizes():
    """After project_h, the result is orthogonal to v_degen."""
    torch.manual_seed(8)
    d = 32
    v = torch.randn(d, dtype=torch.float64)
    v = v / torch.linalg.norm(v)
    h = torch.randn(5, d, dtype=torch.float64)
    h_proj = project_h(h, v)
    # h_proj · v should be ~0
    dots = h_proj @ v
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-12)
    # Projection preserves component orthogonal to v
    h_orthog = h - (h @ v).unsqueeze(-1) * v
    assert torch.allclose(h_proj, h_orthog, atol=1e-12)


if __name__ == '__main__':
    # Run all tests
    test_funcs = [g for n, g in globals().items() if n.startswith('test_') and callable(g)]
    for fn in test_funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            raise
    print(f"\nAll {len(test_funcs)} tests passed.")
