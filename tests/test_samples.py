"""Tests for averaging modes in shape.extract and the format-agnostic sample readers."""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from shape.embeddings import compute_moment_matrix, compute_moment_matrix_streaming
from shape.extract import extract_features
from shape.samples import samples_path
from shape.similarity import (
    compute_pairwise_similarities,
    compute_pairwise_similarities_streaming,
    compute_token_marginals,
)


def make_toy_model(V=16, d=8, L=32, n_layer=2, n_head=2, seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(block_size=L, vocab_size=V, n_layer=n_layer, n_head=n_head,
                    n_embd=d, bias=False)
    return GPT(cfg).eval()


def _extract(model, data, td, **kw):
    kw = {'block_size': model.config.block_size, 'min_context': 4, 'batch_size': 4,
          'device': 'cpu', 'compute_dtype': 'float32', 'verbose': False, **kw}
    return extract_features(model, data, out_dir=td, dataset_name='toy', **kw)


def test_probability_averaging_writes_probs_matching_manual_window():
    """Single chunk (T == L): X̄_t = Σ_d w_d softmax(logits)_{t+d} / Σ w, stored float16."""
    V, L = 16, 32
    model = make_toy_model(V=V, L=L)
    data = np.random.RandomState(0).randint(0, V, size=L, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        res = _extract(model, data, td, window=2, weights='exponential',
                       weights_alpha=0.69, direction='forward', averaging='probability')
        assert res['format'] == 'probs'
        probs = np.load(os.path.join(td, 'toy_probs.npy'))
        Z = np.load(os.path.join(td, 'toy_Z.npy'))

    assert probs.dtype == np.float16
    w = {0: 1.0, 1: np.exp(-0.69), 2: np.exp(-0.69 * 2)}
    with torch.no_grad():
        logits, _, _ = model(torch.from_numpy(data.astype(np.int64))[None], return_hidden=True)
    X = torch.softmax(logits[0].float(), dim=-1).numpy()
    positions = range(4, L - 2)
    expected = np.stack([sum(w[o] * X[t + o] for o in w) / sum(w.values())
                         for t in positions])
    assert probs.shape == expected.shape
    np.testing.assert_allclose(probs.astype(np.float32), expected, atol=1e-3)
    # Z is the mean of the sampled (window-averaged) distributions
    np.testing.assert_allclose(Z, probs.astype(np.float64).mean(axis=0), atol=1e-3)


def test_no_window_probability_averaging_stores_hidden_states():
    """With window=0 both geometries coincide; the compact h format is used."""
    model = make_toy_model()
    data = np.random.RandomState(1).randint(0, 16, size=200, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        res = _extract(model, data, td, window=0, averaging='probability')
        assert res['format'] == 'h'
        assert os.path.exists(os.path.join(td, 'toy_h.npy'))
        assert not os.path.exists(os.path.join(td, 'toy_probs.npy'))


def test_aitchison_z_is_mean_softmax_of_averaged_states():
    model = make_toy_model()
    W = model.lm_head.weight.detach()
    data = np.random.RandomState(2).randint(0, 16, size=200, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        res = _extract(model, data, td, window=2, weights='harmonic')
        assert res['format'] == 'h'
        h = np.load(os.path.join(td, 'toy_h.npy'))
        Z = np.load(os.path.join(td, 'toy_Z.npy'))
    expected = torch.softmax(torch.from_numpy(h) @ W.T, dim=-1).mean(0).numpy()
    np.testing.assert_allclose(Z, expected, rtol=1e-5, atol=1e-7)


def test_project_degenerate_rejected_for_probs():
    model = make_toy_model()
    data = np.random.RandomState(3).randint(0, 16, size=200, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        try:
            _extract(model, data, td, window=2, averaging='probability',
                     project_degenerate=True)
        except ValueError:
            return
    raise AssertionError("expected ValueError for project_degenerate + probability")


def test_similarities_agree_across_formats_and_streaming():
    """h + W, softmax(h Wᵀ) stored as probs, and the streaming pass all agree."""
    model = make_toy_model()
    W = model.lm_head.weight.detach()
    data = np.random.RandomState(4).randint(0, 16, size=300, dtype=np.uint16)
    t1, t2 = np.array([0, 3, 5, 5]), np.array([1, 3, 2, 9])
    kw = dict(batch_size=64, device='cpu', verbose=False)
    with tempfile.TemporaryDirectory() as td:
        _extract(model, data, td, window=0)
        h = np.load(os.path.join(td, 'toy_h.npy'))
    probs = torch.softmax(torch.from_numpy(h) @ W.T, dim=-1).numpy()

    from_h = compute_pairwise_similarities(h, t1, t2, W=W, **kw)
    from_p = compute_pairwise_similarities(probs, t1, t2, **kw)
    stream = compute_pairwise_similarities_streaming(
        model, data, {0: 1.0}, t1, t2, min_context=4, batch_size=4,
        device='cpu', compute_dtype='float32', verbose=False)
    for q in from_h:
        np.testing.assert_allclose(from_p[q], from_h[q], rtol=1e-5)
        np.testing.assert_allclose(stream[q], from_h[q], rtol=1e-4)

    np.testing.assert_allclose(compute_token_marginals(h, W=W, **kw),
                               compute_token_marginals(probs, **kw), rtol=1e-5)


def test_moment_matrix_from_probs_file_matches_streaming():
    model = make_toy_model()
    data = np.random.RandomState(5).randint(0, 16, size=300, dtype=np.uint16)
    lookup = {-2: 0.25, -1: 0.5, 0: 1.0}
    with tempfile.TemporaryDirectory() as td:
        _extract(model, data, td, window=2, weights='exponential',
                 weights_alpha=np.log(2), direction='backward', averaging='probability')
        stored = compute_moment_matrix(os.path.join(td, 'toy_probs.npy'),
                                       batch_size=64, device='cpu', verbose=False)
    stream = compute_moment_matrix_streaming(
        model, data, lookup, averaging='probability', min_context=4, batch_size=4,
        device='cpu', compute_dtype='float32', verbose=False)
    assert stored['N'] == stream['N']
    # stored probs are float16
    np.testing.assert_allclose(stored['M'], stream['M'], rtol=1e-2, atol=1e-5)


def test_samples_path_prefers_new_names_and_falls_back_to_legacy():
    with tempfile.TemporaryDirectory() as td:
        legacy = os.path.join(td, 'toy_h_eff.npy')
        np.save(legacy, np.zeros((2, 3), dtype=np.float32))
        assert samples_path(td, 'toy') == legacy
        new = os.path.join(td, 'toy_h.npy')
        np.save(new, np.zeros((2, 3), dtype=np.float32))
        assert samples_path(td, 'toy') == new


def test_kl_independent_of_other_pairs_in_call():
    """KL's base-rate correction reads Z[t2]; it must not depend on t2 also being a t1."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(16), size=500).astype(np.float32)
    kw = dict(quantities=('kl_divergence',), batch_size=64, device='cpu', verbose=False)
    alone = compute_pairwise_similarities(probs, [0], [1], **kw)['kl_divergence']
    with_rev = compute_pairwise_similarities(probs, [0, 1], [1, 0], **kw)['kl_divergence']
    np.testing.assert_allclose(alone[0], with_rev[0], rtol=1e-10)
    assert abs(alone[0]) < 10
