import csv
import io
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from shape.embeddings_benchmarks import (
    _lookup_word,
    evaluate,
    load_benchmark,
    run_benchmarks,
)


# ---------------------------------------------------------------------------
# _lookup_word
# ---------------------------------------------------------------------------

@pytest.fixture
def small_embedding():
    rng = np.random.default_rng(0)
    return rng.standard_normal((5, 8)).astype(np.float32)


def test_lookup_word_exact(small_embedding):
    stoi = {"cat": 0, "dog": 1}
    result = _lookup_word("cat", small_embedding, stoi)
    np.testing.assert_array_equal(result, small_embedding[0])


def test_lookup_word_space_prefix(small_embedding):
    stoi = {" cat": 0, " dog": 1}
    result = _lookup_word("cat", small_embedding, stoi)
    np.testing.assert_array_equal(result, small_embedding[0])


def test_lookup_word_lowercase(small_embedding):
    stoi = {"cat": 0}
    result = _lookup_word("Cat", small_embedding, stoi)
    np.testing.assert_array_equal(result, small_embedding[0])


def test_lookup_word_space_prefix_lowercase(small_embedding):
    stoi = {" cat": 0}
    result = _lookup_word("Cat", small_embedding, stoi)
    np.testing.assert_array_equal(result, small_embedding[0])


def test_lookup_word_oov(small_embedding):
    stoi = {"cat": 0}
    assert _lookup_word("dragon", small_embedding, stoi) is None


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _make_eval_fixtures():
    """5 words in 4-d space; cosine sims are perfectly rank-correlated with human scores."""
    rng = np.random.default_rng(42)
    n = 10
    # Random unit vectors
    E = rng.standard_normal((20, 4)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    stoi = {f"w{i}": i for i in range(20)}

    pairs = [(f"w{i}", f"w{i+10}") for i in range(n)]
    # Cosine sims for these pairs
    sims = np.array([(E[i] * E[i + 10]).sum() for i in range(n)])
    # Human scores = monotone function of cosine sims → perfect Spearman
    human = sims + rng.standard_normal(n) * 1e-9  # tiny noise keeps ordering

    bm = pd.DataFrame({
        "word1": [p[0] for p in pairs],
        "word2": [p[1] for p in pairs],
        "similarity": human,
    })
    return E, stoi, bm, sims


def test_evaluate_perfect_correlation():
    E, stoi, bm, _ = _make_eval_fixtures()
    result = evaluate(E, stoi, bm)
    assert result["spearman_r"] == pytest.approx(1.0, abs=1e-6)
    assert result["n_pairs"] == 10
    assert result["n_valid"] == 10


def test_evaluate_oov_skipped():
    E, stoi, bm, _ = _make_eval_fixtures()
    # Remove word "w0" from stoi so the first pair is OOV
    stoi_partial = {k: v for k, v in stoi.items() if k != "w0"}
    result = evaluate(E, stoi_partial, bm)
    assert result["n_valid"] == 9
    assert result["n_pairs"] == 10


def test_evaluate_dot_product():
    E, stoi, bm, _ = _make_eval_fixtures()
    result = evaluate(E, stoi, bm, method="dot")
    assert -1.0 <= result["spearman_r"] <= 1.0


def test_evaluate_unknown_method():
    E, stoi, bm, _ = _make_eval_fixtures()
    with pytest.raises(ValueError, match="Unknown method"):
        evaluate(E, stoi, bm, method="manhattan")


def test_evaluate_all_oov_returns_nan():
    E = np.eye(4, dtype=np.float32)
    stoi = {"a": 0}
    bm = pd.DataFrame({"word1": ["x", "y"], "word2": ["z", "q"], "similarity": [1.0, 2.0]})
    result = evaluate(E, stoi, bm)
    assert result["n_valid"] == 0
    assert result["spearman_r"] != result["spearman_r"]  # nan


# ---------------------------------------------------------------------------
# load_benchmark — local ws353
# ---------------------------------------------------------------------------

def test_load_benchmark_local_ws353(tmp_path):
    csv_file = tmp_path / "ws353.csv"
    rows = [
        {"Word 1": "car",  "Word 2": "automobile", "Human (Mean)": 8.94},
        {"Word 1": "gem",  "Word 2": "jewel",       "Human (Mean)": 8.96},
        {"Word 1": "noon", "Word 2": "string",      "Human (Mean)": 0.08},
    ]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Word 1", "Word 2", "Human (Mean)"])
        writer.writeheader()
        writer.writerows(rows)

    df = load_benchmark("ws353", path=str(csv_file))
    assert list(df.columns) == ["word1", "word2", "similarity"]
    assert len(df) == 3
    assert df["word1"].tolist() == ["car", "gem", "noon"]
    assert df["similarity"].dtype == float


def test_load_benchmark_local_requires_path():
    with pytest.raises(ValueError, match="requires a local file path"):
        load_benchmark("ws353")


def test_load_benchmark_unknown_name():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load_benchmark("nonexistent_benchmark_xyz")


# ---------------------------------------------------------------------------
# run_benchmarks — output schema (mocked load)
# ---------------------------------------------------------------------------

def test_run_benchmarks_output_schema():
    rng = np.random.default_rng(7)
    E = rng.standard_normal((10, 4)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    stoi = {f"w{i}": i for i in range(10)}

    fake_bm = pd.DataFrame({
        "word1": ["w0", "w1", "w2"],
        "word2": ["w5", "w6", "w7"],
        "similarity": [0.9, 0.5, 0.1],
    })

    with patch("shape.embeddings_benchmarks.load_all_benchmarks", return_value={"mc30": fake_bm, "rg65": fake_bm}):
        results = run_benchmarks(E, stoi)

    assert set(results.columns) == {"benchmark", "spearman_r", "n_pairs", "benchmark_size", "title", "task_type"}
    assert len(results) == 2
    assert set(results["benchmark"]) == {"mc30", "rg65"}
    assert (results["benchmark_size"] <= results["n_pairs"]).all()


def test_run_benchmarks_filter_by_name():
    rng = np.random.default_rng(8)
    E = rng.standard_normal((10, 4)).astype(np.float32)
    stoi = {f"w{i}": i for i in range(10)}

    fake_bm = pd.DataFrame({
        "word1": ["w0"], "word2": ["w5"], "similarity": [0.9],
    })

    with patch("shape.embeddings_benchmarks.load_all_benchmarks", return_value={"mc30": fake_bm, "rg65": fake_bm}):
        results = run_benchmarks(E, stoi, benchmarks=["mc30"])

    assert list(results["benchmark"]) == ["mc30"]
