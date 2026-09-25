"""Tests for shape.extract (Stage 1 corpus pass)."""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from shape.extract import _count_valid, _iter_chunks, extract_features, valid_positions


def make_toy_model(V=32, d=16, L=64, n_layer=2, n_head=2, seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(block_size=L, vocab_size=V, n_layer=n_layer, n_head=n_head,
                    n_embd=d, bias=False)
    m = GPT(cfg)
    m.eval()
    return m


def test_iter_chunks_tiles_corpus():
    """Every position in [min_context, T-window) is covered exactly once."""
    T, L, window, min_context = 1000, 64, 4, 8
    covered = []
    for s, vs, ve in _iter_chunks(T, L, window, min_context):
        for local in range(vs, ve):
            covered.append(s + local)
    expected = list(range(min_context, T - window))
    assert covered == expected, \
        f"coverage mismatch: got {len(covered)} positions, expected {len(expected)}"


def test_iter_chunks_no_window():
    """With window=0, coverage is [min_context, T)."""
    T, L, min_context = 500, 32, 4
    covered = []
    for s, vs, ve in _iter_chunks(T, L, 0, min_context):
        for local in range(vs, ve):
            covered.append(s + local)
    assert covered == list(range(min_context, T))


def test_valid_positions_matches_iter_chunks():
    """valid_positions(meta) should enumerate the same corpus indices as _iter_chunks."""
    for (T, L, window, min_ctx) in [(500, 32, 0, 4), (1000, 64, 4, 8), (10000, 128, 2, 16)]:
        meta = {'corpus_length': T, 'block_size': L,
                'window': window, 'min_context': min_ctx}
        got = valid_positions(meta)
        expected = np.array(
            [s + local
             for (s, vs, ve) in _iter_chunks(T, L, window, min_ctx)
             for local in range(vs, ve)],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(got, expected)
        assert got.dtype == np.int64
        # Every corpus index in [min_ctx, T-window) appears exactly once.
        assert got.tolist() == list(range(min_ctx, T - window))


def test_count_valid_matches_iter():
    for (T, L, window, min_ctx) in [(500, 32, 0, 4), (1000, 64, 4, 8), (10000, 128, 2, 16)]:
        n = _count_valid(T, L, window, min_ctx)
        actual = sum(1 for _ in _iter_chunks(T, L, window, min_ctx) for _ in range(_[1], _[2]))
        assert n == actual, f"_count_valid disagrees at T={T}, L={L}: {n} vs {actual}"


def test_extract_end_to_end_window_zero():
    """End-to-end smoke test with window=0 and project_degenerate=False."""
    V, d, L = 16, 8, 32
    T = 300
    model = make_toy_model(V=V, d=d, L=L)
    data = np.random.RandomState(42).randint(0, V, size=T, dtype=np.uint16)

    with tempfile.TemporaryDirectory() as td:
        result = extract_features(
            model, data,
            out_dir=td, dataset_name='toy',
            block_size=L, window=0, min_context=4,
            project_degenerate=False, batch_size=3,
            device='cpu', compute_dtype='float32', verbose=False,
        )

        Z = result['Z']
        n_valid = result['n_valid']
        assert Z.shape == (V,)
        # Σ_w Z_w should be close to 1
        assert abs(Z.sum() - 1.0) < 1e-4, f"Z sum = {Z.sum():.6f}"
        # Expected n_valid = T - min_context
        assert n_valid == T - 4, f"n_valid = {n_valid}"

        # h memmap check
        h = np.load(os.path.join(td, 'toy_h.npy'), mmap_mode='r')
        assert h.shape == (n_valid, d)
        # No NaN/Inf
        assert np.isfinite(h).all()

        # Meta file
        with open(os.path.join(td, 'toy_meta.json')) as f:
            meta = json.load(f)
        assert meta['N_valid'] == n_valid
        assert meta['V'] == V and meta['d'] == d
        assert abs(meta['Z_sum_check'] - 1.0) < 1e-4


def test_extract_projection_removes_degenerate_direction():
    """When project_degenerate=True, h has zero component along v_degen."""
    V, d, L = 16, 8, 32
    T = 200
    model = make_toy_model(V=V, d=d, L=L, seed=1)
    data = np.random.RandomState(7).randint(0, V, size=T, dtype=np.uint16)

    with tempfile.TemporaryDirectory() as td:
        extract_features(
            model, data,
            out_dir=td, dataset_name='toy',
            block_size=L, window=0, min_context=4,
            project_degenerate=True, batch_size=4,
            device='cpu', compute_dtype='float32', verbose=False,
        )
        v_degen = np.load(os.path.join(td, 'toy_v_degen.npy'))
        h = np.load(os.path.join(td, 'toy_h.npy'), mmap_mode='r')
        # Every row should have ~zero dot product with v_degen
        dots = h @ v_degen
        assert np.abs(dots).max() < 1e-4, f"max |h · v_degen| = {np.abs(dots).max():.2e}"


def test_extract_window_averaging_reduces_to_single_position_at_zero():
    """With window=0, h[i] should equal h_t at global position (min_context + i)
    when project_degenerate=False."""
    V, d, L = 16, 8, 32
    T = 100
    model = make_toy_model(V=V, d=d, L=L, seed=2)
    data = np.random.RandomState(11).randint(0, V, size=T, dtype=np.uint16)

    with tempfile.TemporaryDirectory() as td:
        extract_features(
            model, data,
            out_dir=td, dataset_name='toy',
            block_size=L, window=0, min_context=4,
            project_degenerate=False, batch_size=2,
            device='cpu', compute_dtype='float32', verbose=False,
        )
        h = np.load(os.path.join(td, 'toy_h.npy'), mmap_mode='r')

        # Recompute h at global position 4 (first valid) directly
        # First chunk covers [0, L); valid local start = 4; global = 4.
        # Build a chunk and compute h[4] via the model
        context = torch.from_numpy(data[:L].astype(np.int64)).unsqueeze(0)
        _, _, h_direct = model(context, return_hidden=True)
        h0 = h_direct[0, 4, :].detach().numpy()
        assert np.allclose(h[0], h0, atol=1e-5), \
            f"h[0] mismatch: max abs diff {np.abs(h[0] - h0).max():.2e}"


def test_extract_window_averaging_nonzero():
    """Window averaging with window=2, symmetric, include_target, no decay (uniform):
    h[i] should equal arithmetic mean of h at 5 neighboring positions."""
    V, d, L = 16, 8, 32
    T = 100
    model = make_toy_model(V=V, d=d, L=L, seed=3)
    data = np.random.RandomState(13).randint(0, V, size=T, dtype=np.uint16)

    with tempfile.TemporaryDirectory() as td:
        extract_features(
            model, data,
            out_dir=td, dataset_name='toy',
            block_size=L, window=2, weights='none',
            include_target=True, direction='symmetric', min_context=4,
            project_degenerate=False, batch_size=4,
            device='cpu', compute_dtype='float32', verbose=False,
        )
        h = np.load(os.path.join(td, 'toy_h.npy'), mmap_mode='r')

        # First valid position is local 4 (min_context) in chunk 0.
        # With window=2, offsets are {-2,-1,0,1,2} with weight=1 each, total_weight=5.
        # h[0] = mean(h[2..6])
        context = torch.from_numpy(data[:L].astype(np.int64)).unsqueeze(0)
        _, _, h_direct = model(context, return_hidden=True)
        expected = h_direct[0, 2:7, :].mean(dim=0).detach().numpy()
        assert np.allclose(h[0], expected, atol=1e-5), \
            f"max diff {np.abs(h[0] - expected).max():.2e}"


def test_extract_save_states_false_requires_subsample():
    """save_states=False without save_subsample should raise."""
    V, d, L = 16, 8, 32
    T = 100
    model = make_toy_model(V=V, d=d, L=L)
    data = np.random.RandomState(0).randint(0, V, size=T, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        try:
            extract_features(
                model, data,
                out_dir=td, dataset_name='toy',
                block_size=L, window=0, min_context=4,
                project_degenerate=False, batch_size=4,
                device='cpu', compute_dtype='float32',
                save_states=False, save_subsample=0, verbose=False,
            )
        except ValueError as e:
            assert 'save_subsample' in str(e)
            return
        raise AssertionError("Expected ValueError for save_states=False + save_subsample=0")


def test_extract_save_states_false_with_subsample():
    """save_states=False + save_subsample>0 writes subsample only; no full h."""
    V, d, L = 16, 8, 32
    T = 200
    model = make_toy_model(V=V, d=d, L=L, seed=4)
    data = np.random.RandomState(17).randint(0, V, size=T, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        result = extract_features(
            model, data,
            out_dir=td, dataset_name='toy',
            block_size=L, window=0, min_context=4,
            project_degenerate=False, batch_size=4,
            device='cpu', compute_dtype='float32',
            save_states=False, save_subsample=50, verbose=False,
        )
        # No full h file
        assert not os.path.exists(os.path.join(td, 'toy_h.npy'))
        # Subsample file exists at requested size (T - min_context = 196 > 50)
        sub = np.load(os.path.join(td, 'toy_subsample.npy'), mmap_mode='r')
        assert sub.shape == (50, d)
        assert np.isfinite(sub).all()
        # Z_w still written correctly
        assert abs(result['Z'].sum() - 1.0) < 1e-4


def test_extract_subsample_truncates_when_undersampled():
    """If valid positions < save_subsample, file is truncated to actual count."""
    V, d, L = 16, 8, 32
    T = 50   # only 50 - min_context = 46 valid positions
    model = make_toy_model(V=V, d=d, L=L, seed=5)
    data = np.random.RandomState(19).randint(0, V, size=T, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        extract_features(
            model, data,
            out_dir=td, dataset_name='toy',
            block_size=L, window=0, min_context=4,
            project_degenerate=False, batch_size=4,
            device='cpu', compute_dtype='float32',
            save_states=False, save_subsample=1000, verbose=False,
        )
        sub = np.load(os.path.join(td, 'toy_subsample.npy'), mmap_mode='r')
        assert sub.shape == (T - 4, d), f"expected {T - 4} rows, got {sub.shape[0]}"


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
