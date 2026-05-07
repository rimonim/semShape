"""
Psycholinguistic benchmark evaluation for static word embeddings.

Evaluates (V, k) embedding matrices against 15 standard benchmarks spanning
word similarity/relatedness judgements, free association norms, and semantic
priming (lexical decision RT). Reports Spearman correlations.

Quick start::

    import numpy as np, pickle
    from shape.embeddings_benchmarks import run_benchmarks

    data = np.load("embeddings/my_emb.npz")
    with open("data/my_dataset/meta.pkl", "rb") as f:
        stoi = pickle.load(f)["stoi"]

    results = run_benchmarks(data["embedding"], stoi)
    print(results.sort_values("spearman_r", ascending=False).to_string(index=False))

Vocabulary lookup tries four candidates per word (exact, BPE-space-prefix,
lowercased variants) so it works with both character-level and BPE tokenizers.
OOV pairs are silently excluded; ``benchmark_size`` in the output reflects how
many pairs were actually evaluated.
"""

import io
import json
import logging
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

_VECTO = "https://raw.githubusercontent.com/vecto-ai/word-benchmarks/refs/heads/master/word-similarity/monolingual/en"

BENCHMARK_URLS: dict[str, str | None] = {
    "ws353":    None,  # local only
    "mc30":     f"{_VECTO}/mc-30.csv",
    "rg65":     f"{_VECTO}/rg-65.csv",
    "rw":       f"{_VECTO}/rw.csv",
    "semeval17": f"{_VECTO}/semeval17.csv",
    "simlex999": f"{_VECTO}/simlex999.csv",
    "yp130":    f"{_VECTO}/yp-130.csv",
    "mturk287": f"{_VECTO}/mturk-287.csv",
    "mturk771": f"{_VECTO}/mturk-771.csv",
    "verb143":  f"{_VECTO}/verb-143.csv",
    "eat":      "https://raw.githubusercontent.com/dariusk/ea-thesaurus/refs/heads/master/ea-thesaurus.json",
    "swow":     None,  # local only
    "usf":      None,  # local only
    "spaml":    None,  # local only
    "spp":      None,  # local only
}

BENCHMARK_METADATA: list[dict] = [
    {"name": "ws353",    "title": "WordSim353",                     "source": "Finkelstein et al. (2001)",        "task_type": "Relatedness Judgement"},
    {"name": "mc30",     "title": "Miller and Charles (1991)",       "source": "Miller and Charles (1991)",        "task_type": "Similarity Judgement"},
    {"name": "rg65",     "title": "Rubenstein and Goodenough (1965)","source": "Rubenstein and Goodenough (1965)", "task_type": "Similarity Judgement"},
    {"name": "rw",       "title": "Rare Word",                       "source": "Luong et al. (2013)",             "task_type": "Relatedness Judgement"},
    {"name": "semeval17","title": "SemEval 2017 Task 2",             "source": "Camacho-Collados et al. (2017)",   "task_type": "Similarity Judgement"},
    {"name": "simlex999","title": "SimLex-999",                      "source": "Hill et al. (2014)",              "task_type": "Similarity Judgement"},
    {"name": "yp130",    "title": "Yang and Powers (2006)",          "source": "Yang and Powers (2006)",          "task_type": "Similarity Judgement"},
    {"name": "mturk287", "title": "MTurk-287",                       "source": "Radinsky et al. (2011)",          "task_type": "Relatedness Judgement"},
    {"name": "mturk771", "title": "MTurk-771",                       "source": "Halawi et al. (2012)",            "task_type": "Relatedness Judgement"},
    {"name": "verb143",  "title": "Verb-143",                        "source": "Yang and Powers (2007)",          "task_type": "Similarity Judgement"},
    {"name": "eat",      "title": "Edinburgh Associative Thesaurus", "source": "Kiss et al. (1973)",              "task_type": "Free Association"},
    {"name": "swow",     "title": "Small World of Words",            "source": "De Deyne et al. (2019)",          "task_type": "Free Association"},
    {"name": "usf",      "title": "USF Norms",                       "source": "Nelson et al. (2004)",            "task_type": "Free Association"},
    {"name": "spaml",    "title": "SPAML",                           "source": "Buchanan et al. (2025)",          "task_type": "Semantic Priming (Lexical Decision)"},
    {"name": "spp",      "title": "Semantic Priming Project",        "source": "Hutchison et al. (2013)",         "task_type": "Semantic Priming (Lexical Decision)"},
]

_META_BY_NAME = {m["name"]: m for m in BENCHMARK_METADATA}
_DEFAULT_CACHE = Path.home() / ".cache" / "semshape" / "benchmarks"


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def _fetch_cached(url: str, cache_dir: Path) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / url.split("/")[-1]
    if fname.exists():
        return fname.read_bytes()
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    fname.write_bytes(data)
    return data


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------

def _load_vecto_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    # vecto-ai CSVs have columns: word1, word2, similarity (or score)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("word1", "word_1"):
            rename[c] = "word1"
        elif cl in ("word2", "word_2"):
            rename[c] = "word2"
        elif cl in ("similarity", "score", "human (mean)", "sim"):
            rename[c] = "similarity"
    df = df.rename(columns=rename)[["word1", "word2", "similarity"]]
    df["word1"] = df["word1"].str.lower()
    df["word2"] = df["word2"].str.lower()
    return df.dropna()


def _load_eat(raw: bytes) -> pd.DataFrame:
    eat = json.loads(raw)
    rows = []
    for cue, responses in eat.items():
        # Each value is either a plain dict {response: count} or
        # a list of single-key dicts [{response: count}, ...] (dariusk/ea-thesaurus).
        if isinstance(responses, list):
            responses = {k: v for item in responses for k, v in item.items()}
        try:
            counts = {r: int(c) for r, c in responses.items()}
        except (AttributeError, ValueError):
            continue
        total = sum(counts.values())
        if total == 0:
            continue
        for response, count in counts.items():
            if " " in response:
                continue
            rows.append({
                "word1": cue.lower(),
                "word2": response.lower(),
                "similarity": count / total,
            })
    return pd.DataFrame(rows, columns=["word1", "word2", "similarity"])


def _load_ws353(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("word 1", "word1"):
            rename[c] = "word1"
        elif cl in ("word 2", "word2"):
            rename[c] = "word2"
        elif cl in ("human (mean)", "similarity", "score"):
            rename[c] = "similarity"
    df = df.rename(columns=rename)[["word1", "word2", "similarity"]]
    df["word1"] = df["word1"].str.lower()
    df["word2"] = df["word2"].str.lower()
    return df.dropna()


def _load_swow(path: str) -> pd.DataFrame:
    # Sniff the delimiter; SWOW files are sometimes tab-separated despite a
    # .csv extension.
    import csv
    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        sample = fh.read(4096)
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=',\t;').delimiter
    except csv.Error:
        sep = '\t' if path.endswith('.tsv') else ','
    df = pd.read_csv(path, sep=sep, on_bad_lines='skip', engine='python')
    df.columns = [c.strip() for c in df.columns]
    # Expected columns: cue, response, R1.Strength (and R1 count)
    df = df.rename(columns={"cue": "word1", "response": "word2", "R1.Strength": "similarity"})
    if "R1" in df.columns:
        df = df[df["R1"] >= 1]
    df = df[~df["word2"].str.contains(" ", na=True)]
    df["word1"] = df["word1"].str.lower()
    df["word2"] = df["word2"].str.lower()
    return df[["word1", "word2", "similarity"]].dropna()


def _load_usf(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "cues":
            rename[c] = "word1"
        elif cl == "targets":
            rename[c] = "word2"
        elif cl == "forward strength":
            rename[c] = "similarity"
    df = df.rename(columns=rename)[["word1", "word2", "similarity"]]
    df["word1"] = df["word1"].str.lower()
    df["word2"] = df["word2"].str.lower()
    return df.dropna()


def _load_spaml(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "keep_target" in df.columns:
        df = df[df["keep_target"] == "keep"]
    if "keep_participant" in df.columns:
        df = df[df["keep_participant"] == "keep"]
    df = df.rename(columns={"cue_word": "word1", "target_word": "word2", "target_Z_RT": "similarity"})
    df = (
        df.groupby(["word1", "word2"], as_index=False)["similarity"]
        .mean()
    )
    df["similarity"] = -df["similarity"]  # lower RT = more similar
    return df[["word1", "word2", "similarity"]].dropna()


def _load_spp(path: str) -> pd.DataFrame:
    # Check extension. If .xlsx, try Excel
    if path.endswith(".xlsx"):
        try:
            df = pd.read_excel(path)
    else:
        import csv as _csv
        with open(path, newline='', encoding='utf-8', errors='replace') as fh:
            sample = fh.read(4096)
        try:
            sep = _csv.Sniffer().sniff(sample, delimiters=',\t;').delimiter
        except _csv.Error:
            sep = '\t' if path.endswith('.tsv') else ','
        df = pd.read_csv(path, sep=sep, on_bad_lines='skip', engine='python')
    df.columns = [c.strip() for c in df.columns]
    if "keep" in df.columns:
        df = df[df["keep"] == 1]
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "prime":
            rename[c] = "word1"
        elif cl == "target":
            rename[c] = "word2"
        elif cl == "target.rt":
            rename[c] = "similarity"
    df = df.rename(columns=rename)
    df = (
        df.groupby(["word1", "word2"], as_index=False)["similarity"]
        .mean()
    )
    df["similarity"] = -df["similarity"]  # lower RT = more similar
    df["word1"] = df["word1"].str.lower()
    return df[["word1", "word2", "similarity"]].dropna()


def load_benchmark(name: str, *, path: str | None = None, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Load a single benchmark dataset as a DataFrame with columns word1, word2, similarity.

    Args:
        name: Benchmark name, one of the keys in BENCHMARK_URLS.
        path: Required for local-only benchmarks (ws353, swow, usf, spaml, spp).
        cache_dir: Directory for caching downloaded files. Defaults to
            ~/.cache/semshape/benchmarks/.

    Returns:
        DataFrame with columns word1, word2, similarity.
    """
    if name not in BENCHMARK_URLS:
        raise ValueError(f"Unknown benchmark {name!r}. Available: {sorted(BENCHMARK_URLS)}")

    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
    url = BENCHMARK_URLS[name]

    if url is None:
        if path is None:
            raise ValueError(f"Benchmark {name!r} requires a local file path; pass path=...")
        loaders = {
            "ws353": _load_ws353,
            "swow":  _load_swow,
            "usf":   _load_usf,
            "spaml": _load_spaml,
            "spp":   _load_spp,
        }
        return loaders[name](path)

    raw = _fetch_cached(url, cache)
    if name == "eat":
        return _load_eat(raw)
    return _load_vecto_csv(raw)


def load_all_benchmarks(
    paths: dict[str, str] | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all available benchmarks.

    Args:
        paths: Dict mapping local-only benchmark names to file paths, e.g.
            {"ws353": "~/data/wordsim353crowd.csv", "swow": "~/data/swow.csv"}.
        cache_dir: Cache directory for downloaded benchmarks.

    Returns:
        Dict mapping benchmark name → DataFrame.
    """
    paths = paths or {}
    out = {}
    for name in BENCHMARK_URLS:
        url = BENCHMARK_URLS[name]
        if url is None and name not in paths:
            logger.warning("Skipping %r: no path provided (local-only benchmark).", name)
            continue
        try:
            out[name] = load_benchmark(name, path=paths.get(name), cache_dir=cache_dir)
        except Exception as exc:
            logger.warning("Failed to load benchmark %r: %s", name, exc)
    return out


# ---------------------------------------------------------------------------
# Vocabulary lookup
# ---------------------------------------------------------------------------

def _lookup_word(word: str, embedding: np.ndarray, stoi: dict[str, int]) -> np.ndarray | None:
    """Return the embedding row for word, or None if OOV.

    Tries four candidates in order: exact, BPE space-prefix, lowercased,
    lowercased BPE space-prefix. This covers both character-level and BPE
    tokenizers without requiring the tokenizer itself.
    """
    for candidate in (word, " " + word, word.lower(), " " + word.lower()):
        idx = stoi.get(candidate)
        if idx is not None:
            return embedding[idx]
    return None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    embedding: np.ndarray,
    stoi: dict[str, int],
    benchmark: pd.DataFrame,
    *,
    method: str = "cosine",
) -> dict:
    """Evaluate a single benchmark.

    Args:
        embedding: (V, k) float array.
        stoi: Dict mapping token strings to row indices into embedding.
        benchmark: DataFrame with columns word1, word2, similarity.
        method: 'cosine' (default) or 'dot'.

    Returns:
        Dict with keys spearman_r (float), n_pairs (int), n_valid (int).
    """
    words1 = benchmark["word1"].tolist()
    words2 = benchmark["word2"].tolist()
    human  = benchmark["similarity"].to_numpy(dtype=np.float64)
    n_pairs = len(words1)

    E = np.asarray(embedding, dtype=np.float32)

    vecs1, vecs2, mask = [], [], []
    for w1, w2 in zip(words1, words2):
        e1 = _lookup_word(w1, E, stoi)
        e2 = _lookup_word(w2, E, stoi)
        found = e1 is not None and e2 is not None
        mask.append(found)
        if found:
            vecs1.append(e1)
            vecs2.append(e2)

    mask = np.array(mask)
    n_valid = int(mask.sum())
    if n_valid < 2:
        return {"spearman_r": float("nan"), "n_pairs": n_pairs, "n_valid": n_valid}

    A = np.stack(vecs1)  # (n_valid, k)
    B = np.stack(vecs2)

    if method == "cosine":
        na = np.linalg.norm(A, axis=1, keepdims=True)
        nb = np.linalg.norm(B, axis=1, keepdims=True)
        na = np.where(na < 1e-12, 1.0, na)
        nb = np.where(nb < 1e-12, 1.0, nb)
        sims = (A / na * (B / nb)).sum(axis=1)
    elif method == "dot":
        sims = (A * B).sum(axis=1)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'cosine' or 'dot'.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = spearmanr(sims, human[mask])

    return {"spearman_r": float(r), "n_pairs": n_pairs, "n_valid": n_valid}


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_benchmarks(
    embedding: np.ndarray,
    stoi: dict[str, int],
    *,
    paths: dict[str, str] | None = None,
    benchmarks: list[str] | None = None,
    method: str = "cosine",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run embedding evaluation across benchmark datasets.

    Args:
        embedding: (V, k) float array of static word embeddings.
        stoi: Dict mapping token strings to row indices, e.g. from meta.pkl.
        paths: Local file paths for local-only benchmarks (ws353, swow, usf,
            spaml, spp). Omit keys to skip those benchmarks.
        benchmarks: List of benchmark names to run. Defaults to all available
            (i.e., all downloadable + any local ones with paths provided).
        method: Similarity metric passed to evaluate().
        cache_dir: Directory for caching downloaded benchmark files.

    Returns:
        DataFrame with columns:
            benchmark    — short name
            spearman_r   — Spearman correlation with human judgements
            n_pairs      — total pairs in the raw benchmark
            benchmark_size — pairs actually evaluated (both words in vocab)
            title        — full benchmark name
            task_type    — e.g. "Similarity Judgement", "Free Association"
    """
    loaded = load_all_benchmarks(paths=paths, cache_dir=cache_dir)
    if benchmarks is not None:
        loaded = {k: v for k, v in loaded.items() if k in benchmarks}

    rows = []
    for name, bdf in loaded.items():
        result = evaluate(embedding, stoi, bdf, method=method)
        meta = _META_BY_NAME[name]
        rows.append({
            "benchmark":      name,
            "spearman_r":     result["spearman_r"],
            "n_pairs":        result["n_pairs"],
            "benchmark_size": result["n_valid"],
            "title":          meta["title"],
            "task_type":      meta["task_type"],
        })

    return pd.DataFrame(rows, columns=["benchmark", "spearman_r", "n_pairs", "benchmark_size", "title", "task_type"])
