"""
ILR geometry helpers for the continuous semantic framework.

The ILR basis Ψ ∈ ℝ^((V-1)×V) used here is the Helmert sequential-binary-partition:

    Ψ_{i,j} = +1/sqrt(i(i+1))   if j ≤ i       (1-indexed i = 1..V-1, j = 1..V)
             = -sqrt(i/(i+1))    if j = i+1
             = 0                  otherwise

Applied to log(p):

    y_i = sqrt(i/(i+1)) · log( GM(p_1..p_i) / p_{i+1} )

Properties:
  - Ψ · 1 = 0 (rows sum to zero; ILR annihilates the softmax constant-shift direction).
  - Ψ Ψᵀ = I   (orthonormal rows).

For vocabularies with V ~ 10^5, we never materialize Ψ densely. All operations
reduce to cumulative sums, giving O(V·d) time and no intermediate (V-1)×V storage.
"""

import math

import torch


def ilr_apply(v):
    """
    Apply Ψ to v ∈ ℝ^V along the last dimension.

    Args:
        v: tensor of shape (..., V).

    Returns:
        tensor of shape (..., V-1).
    """
    V = v.shape[-1]
    i = torch.arange(1, V, dtype=v.dtype, device=v.device)
    coef_a = torch.sqrt(1.0 / (i * (i + 1)))  # (V-1,)
    coef_b = torch.sqrt(i / (i + 1))           # (V-1,)
    cumv = torch.cumsum(v, dim=-1)             # (..., V)
    return coef_a * cumv[..., :V - 1] - coef_b * v[..., 1:V]


def ilr_apply_T(y):
    """
    Apply Ψᵀ to y ∈ ℝ^(V-1) along the last dimension.

    Ψᵀ maps ILR coordinates back to CLR coordinates (which sum to zero in ℝ^V).
    Writing the transpose of the formula above and re-using a suffix-sum:

        (Ψᵀ y)_j = (Σ_{i ≥ j} coef_a_i · y_i) − coef_b_{j-1} · y_{j-1}

    where the second term is present only for j ≥ 2 (1-indexed).

    Args:
        y: tensor of shape (..., V-1).

    Returns:
        tensor of shape (..., V).
    """
    Vm1 = y.shape[-1]
    V = Vm1 + 1
    i = torch.arange(1, V, dtype=y.dtype, device=y.device)
    coef_a = torch.sqrt(1.0 / (i * (i + 1)))  # (V-1,)
    coef_b = torch.sqrt(i / (i + 1))           # (V-1,)
    # Suffix sum of coef_a * y over the last dim
    ay = coef_a * y                            # (..., V-1)
    suffix = torch.flip(torch.cumsum(torch.flip(ay, dims=(-1,)), dim=-1), dims=(-1,))
    # out_j for j=1..V
    out = torch.zeros(*y.shape[:-1], V, dtype=y.dtype, device=y.device)
    # j=1..V-1 get the suffix from position j (0-indexed j-1) minus (coef_b at j-1) * y_{j-1}
    # Column 0 (j=1) of output: suffix[...,0], no subtracted term.
    out[..., 0] = suffix[..., 0]
    # Columns 1..V-1 (j=2..V): suffix at position j-1 (0-indexed) minus coef_b[j-2] * y[j-2]
    # For the last column (j=V, 0-indexed V-1): suffix at V-1 doesn't exist (empty), so it's -coef_b[V-2]*y[V-2].
    out[..., 1:V - 1] = suffix[..., 1:V - 1] - coef_b[:V - 2] * y[..., :V - 2]
    out[..., V - 1] = -coef_b[V - 2] * y[..., V - 2]
    return out


def ilr(p, eps=1e-30):
    """
    ILR transform of a probability vector on the simplex.

    Args:
        p: tensor of shape (..., V) with nonneg entries summing to 1 along last dim.
        eps: clamp for numerical stability.

    Returns:
        tensor of shape (..., V-1).
    """
    log_p = torch.log(p.clamp(min=eps))
    return ilr_apply(log_p)


def ilr_basis_dense(V, dtype=torch.float64, device=None):
    """
    Build Ψ ∈ ℝ^((V-1)×V) densely. For small V only (testing / small vocabs).
    Memory scales as O(V²); do not use for V > a few thousand.
    """
    Psi = torch.zeros(V - 1, V, dtype=dtype, device=device)
    for i in range(1, V):
        coef_a = math.sqrt(1.0 / (i * (i + 1)))
        coef_b = math.sqrt(i / (i + 1))
        Psi[i - 1, :i] = coef_a
        Psi[i - 1, i] = -coef_b
    return Psi


def compute_A(W):
    """
    Compute A = Ψ W ∈ ℝ^((V-1)×d) via the cumsum identity.

    For row i (1-indexed i = 1..V-1):
        A_i = (1/sqrt(i(i+1))) · Σ_{j=1..i} W_j  −  sqrt(i/(i+1)) · W_{i+1}

    Runs in O(V·d) time; avoids materializing Ψ.

    Args:
        W: tensor of shape (V, d) — the output projection `lm_head.weight`.

    Returns:
        A: tensor of shape (V-1, d).
    """
    V, d = W.shape
    i = torch.arange(1, V, dtype=W.dtype, device=W.device)
    coef_a = torch.sqrt(1.0 / (i * (i + 1))).unsqueeze(1)  # (V-1, 1)
    coef_b = torch.sqrt(i / (i + 1)).unsqueeze(1)           # (V-1, 1)
    cumW = torch.cumsum(W, dim=0)                           # (V, d)
    return coef_a * cumW[:V - 1] - coef_b * W[1:V]


def degenerate_direction(W):
    """
    Unit direction v ∈ ℝ^d such that Wv is as close to 1 ∈ ℝ^V as possible
    in the least-squares sense. This is the h-space direction annihilated by
    the ILR transform: ΨW v = Ψ·(Wv) ∝ Ψ·1 = 0.

    For full-column-rank W, v = W⁺·1 / ||W⁺·1||.

    Args:
        W: tensor of shape (V, d).

    Returns:
        v: tensor of shape (d,), unit norm.
    """
    V, d = W.shape
    ones = torch.ones(V, 1, dtype=W.dtype, device=W.device)
    # lstsq solves min_v ||W v − 1||²
    solution = torch.linalg.lstsq(W, ones).solution  # (d, 1)
    v = solution.squeeze(-1)
    norm = torch.linalg.norm(v)
    if norm == 0:
        raise RuntimeError("degenerate_direction: zero solution; W may be singular.")
    return v / norm


def project_h(h, v_degen):
    """
    Remove the component of h along v_degen (a unit vector).

    Args:
        h: tensor of shape (..., d).
        v_degen: tensor of shape (d,), assumed unit norm.

    Returns:
        h with the v_degen component subtracted out: h - (h·v_degen) v_degen.
    """
    coeff = (h * v_degen).sum(dim=-1, keepdim=True)
    return h - coeff * v_degen
