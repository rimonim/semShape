"""Tests for shape.embeddings.aitchison_token_embeddings."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape.embeddings import aitchison_token_embeddings, ilr_embeddings, pmi_matrix
from shape.geometry import compute_A


def test_origin_difference_is_constant_row():
    """origin='aitchison' and origin='ilr' should differ exactly by the
    constant row (A @ h_bar)^T."""
    rng = np.random.default_rng(0)
    V, d, N = 12, 5, 200
    W = torch.from_numpy(rng.standard_normal((V, d)).astype(np.float64))
    h = rng.standard_normal((N, d)).astype(np.float32)

    out_ait = aitchison_token_embeddings(
        h, W, origin='aitchison', batch_size=64, device='cpu', verbose=False,
    )
    out_ilr = aitchison_token_embeddings(
        h, W, origin='ilr', batch_size=64, device='cpu', verbose=False,
    )

    A = out_ait['A']
    h_bar = out_ait['h_bar']
    expected_row = A @ h_bar              # (V-1,)
    diff = out_ilr['embedding'] - out_ait['embedding']    # (V, V-1)

    np.testing.assert_allclose(
        diff,
        np.broadcast_to(expected_row.astype(np.float32), diff.shape),
        rtol=1e-4, atol=1e-5,
    )


def test_aitchison_matches_pmi_perp_at_zero_spread():
    """At zero token-conditional spread (h | t a delta distribution), the
    Aitchison-centroid embedding A · (h_bar_t - h_bar) should agree with the
    classic PMI⊥ row Ψ · log E[X | t] up to the global mean shift, which
    becomes a Jensen-zero identity here."""
    rng = np.random.default_rng(1)
    V, d = 5, 5
    # Square invertible W so that W h_t = α e_t is achievable exactly.
    # This makes softmax(W h_t) sharply peaked at t, i.e. h | t is a delta.
    W = rng.standard_normal((V, d))
    while abs(np.linalg.det(W)) < 1e-3:
        W = rng.standard_normal((V, d))
    W_t = torch.from_numpy(W).double()

    Wp = np.linalg.inv(W)
    one_hots = np.eye(V) * 50.0       # large so softmax(Wh) ≈ e_t
    h_per_token = (Wp @ one_hots).T   # (V, d): W @ h_per_token[t] = 50 · e_t

    # Replicate to N samples (5 per token).
    reps = 5
    h = np.tile(h_per_token, (reps, 1)).astype(np.float32)
    # Sanity: softmax(W h_t) should be ~ one-hot at t.
    logits = h.astype(np.float64) @ W.T
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    # Each row's argmax should equal t = (i mod V)
    argmax = p.argmax(axis=1)
    expected = np.tile(np.arange(V), reps)
    assert np.array_equal(argmax, expected), "synthetic peaks not aligned"

    # 1) Aitchison embedding (no SVD, no sample noise — h | t is delta).
    out_ait = aitchison_token_embeddings(
        h, W_t, origin='aitchison',
        batch_size=64, device='cpu', verbose=False,
    )
    aitchison_emb = out_ait['embedding']                  # (V, V-1)

    # 2) Analytical PMI⊥ at zero spread:
    #    Ψ · log E[X | t] = Ψ · log softmax(W h_t)  (since X | t is delta on h_t)
    #                    = Ψ · (W h_t − logsumexp(W h_t) · 1)  = Ψ W h_t   (Ψ·1=0)
    #    Subtracting the global Ψ · log E[X] → at zero spread reduces to
    #    Ψ W (h_t − h_bar) = A (h_t − h_bar). Same as our 'aitchison' embedding.
    A = compute_A(W_t)                                    # (V-1, d)
    h_bar = h.astype(np.float64).mean(axis=0)         # (d,)
    expected_emb = (h_per_token - h_bar) @ A.numpy().T    # (V, V-1)

    np.testing.assert_allclose(aitchison_emb, expected_emb.astype(np.float32),
                               rtol=1e-3, atol=1e-4)

    # 3) Cross-check: the existing PMI⊥ pipeline (pmi_matrix → ilr_embeddings)
    #    should land on the same row directions (cosine ≈ 1) at zero spread.
    cp = pmi_matrix(h, W_t, batch_size=64, device='cpu', verbose=False)
    pmi_perp = ilr_embeddings(cp['pmi'])                  # (V, V-1)
    # Cosine similarity per row.
    a = aitchison_emb / (np.linalg.norm(aitchison_emb, axis=1, keepdims=True) + 1e-12)
    b = pmi_perp / (np.linalg.norm(pmi_perp, axis=1, keepdims=True) + 1e-12)
    cos = (a * b).sum(axis=1)
    assert np.all(cos > 0.99), f"PMI⊥ vs Aitchison cosine min = {cos.min():.4f}"
