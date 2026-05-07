"""Tests for shape.embeddings.compute_moment_matrix_prob_window."""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from shape.embeddings import (
    compute_moment_matrix,
    compute_moment_matrix_prob_window,
)
from shape.extract import extract_features


def make_toy_model(V=8, d=4, L=32, n_layer=2, n_head=2, seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(block_size=L, vocab_size=V, n_layer=n_layer, n_head=n_head,
                    n_embd=d, bias=False)
    return GPT(cfg).eval()


def test_prob_window_window_zero_matches_compute_moment_matrix():
    """weights_lookup={0: 1.0} should reproduce compute_moment_matrix on the
    h_eff extracted at window=0 (modulo FP rounding)."""
    V, d, L, T = 8, 4, 32, 300
    model = make_toy_model(V=V, d=d, L=L)
    data = np.random.RandomState(42).randint(0, V, size=T, dtype=np.uint16)

    with tempfile.TemporaryDirectory() as td:
        extract_features(
            model, data,
            out_dir=td, dataset_name='toy',
            block_size=L, window=0, min_context=4,
            project_degenerate=False, batch_size=4,
            device='cpu', compute_dtype='float32', verbose=False,
        )
        h_eff = np.load(os.path.join(td, 'toy_h_eff.npy'), mmap_mode='r')

        ref = compute_moment_matrix(
            h_eff, model.lm_head.weight,
            batch_size=64, device='cpu', verbose=False,
        )

    out = compute_moment_matrix_prob_window(
        model, data, weights_lookup={0: 1.0},
        block_size=L, min_context=4, batch_size=4,
        device='cpu', compute_dtype='float32', verbose=False,
    )

    assert ref['N'] == out['N']
    np.testing.assert_allclose(out['M'], ref['M'], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(out['Z_emp'], ref['Z_emp'], rtol=1e-5, atol=1e-7)


def test_prob_window_differs_from_h_window():
    """With a non-trivial window, prob-space averaging gives a different M than
    h-space averaging (Jensen-style discrepancy)."""
    V, d, L, T = 8, 4, 32, 400
    model = make_toy_model(V=V, d=d, L=L, seed=1)
    data = np.random.RandomState(0).randint(0, V, size=T, dtype=np.uint16)

    # Matches build_weight_lookup(window_size=1, decay_type='exponential',
    # alpha=ln(2), include_target=True, direction='symmetric').
    import math
    alpha = math.log(2.0)
    weights_lookup = {-1: 0.5, 0: 1.0, 1: 0.5}

    prob_out = compute_moment_matrix_prob_window(
        model, data, weights_lookup=weights_lookup,
        block_size=L, min_context=4, batch_size=4,
        device='cpu', compute_dtype='float32', verbose=False,
    )

    with tempfile.TemporaryDirectory() as td:
        extract_features(
            model, data,
            out_dir=td, dataset_name='toy',
            block_size=L, window=1,
            weights='exponential', weights_alpha=alpha,
            direction='symmetric', include_target=True,
            min_context=4, project_degenerate=False,
            batch_size=4, device='cpu',
            compute_dtype='float32', verbose=False,
        )
        h_eff = np.load(os.path.join(td, 'toy_h_eff.npy'), mmap_mode='r')
        h_out = compute_moment_matrix(
            h_eff, model.lm_head.weight,
            batch_size=64, device='cpu', verbose=False,
        )

    assert prob_out['N'] == h_out['N']
    diff = np.max(np.abs(prob_out['M'] - h_out['M']))
    assert diff > 1e-6, (
        f"prob-space and h-space M's should differ under non-trivial windowing; "
        f"got max|ΔM| = {diff:.2e}"
    )
