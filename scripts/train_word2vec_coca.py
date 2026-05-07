"""
Train word2vec (skip-gram, negative sampling, 300D) on COCA text files.

Reads a folder of text files where each line is a document. Lines start with
a document id token (e.g., "@@4000003") that is ignored. Any sequence of ten
"@" tokens ("@ @ @ @ @ @ @ @ @ @") is removed as redacted text.

Outputs an embedding aligned to the COCA stoi from a provided meta.pkl.
Tokens not learned by gensim (e.g., below min_count) are dropped and the
output meta.pkl is pruned to the learned tokens.

Usage:
    python scripts/train_word2vec_coca.py \
        --text-dir data/coca/text \
        --coca-meta data/coca/meta.pkl \
        --out-emb embeddings/word2vec_coca.npz \
        --out-meta embeddings/word2vec_coca_meta.pkl

Requires: gensim.
"""

import argparse
import os
import pickle
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text-dir", required=True,
                   help="Folder with COCA text files; each line is a document.")
    p.add_argument("--coca-meta", required=True,
                   help="Path to data/coca/meta.pkl with 'stoi' and 'itos'.")
    p.add_argument("--out-emb", required=True,
                   help="Output .npz with 'embedding' (V_kept, 300) fp32.")
    p.add_argument("--out-meta", required=True,
                   help="Output meta.pkl with stoi pruned to learned tokens.")
    p.add_argument("--vector-size", type=int, default=300)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--negative", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def iter_text_files(text_dir):
    for root, _, files in os.walk(text_dir):
        for name in sorted(files):
            if name.startswith("."):
                continue
            yield os.path.join(root, name)


def clean_tokens(tokens):
    cleaned = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "@" and i + 9 < len(tokens) and all(t == "@" for t in tokens[i:i + 10]):
            i += 10
            continue
        cleaned.append(tokens[i])
        i += 1
    return cleaned


def stream_sentences_from_text(text_dir):
    """Yield lists-of-words from COCA text files (one document per line)."""
    for path in iter_text_files(text_dir):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                tokens = tokens[1:]  # drop document id
                tokens = clean_tokens(tokens)
                if tokens:
                    yield tokens


def main():
    args = parse_args()

    try:
        from gensim.models import Word2Vec
    except ImportError:
        sys.exit("gensim is required. Install with: pip install gensim")

    with open(args.coca_meta, "rb") as f:
        coca_meta = pickle.load(f)
    stoi = coca_meta["stoi"]
    itos = coca_meta["itos"]
    print(f"COCA vocab: {len(stoi):,}")

    print(f"Reading text files from {args.text_dir} ...")
    sentences = list(stream_sentences_from_text(args.text_dir))
    n_sent = len(sentences)
    n_words = sum(len(s) for s in sentences)
    print(f"{n_sent:,} documents, {n_words:,} word tokens")

    print("Training Word2Vec (skip-gram, negative sampling) ...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        negative=args.negative,
        sg=1,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )
    learned = set(model.wv.key_to_index.keys())
    print(f"Learned vocab: {len(learned):,}")

    items = sorted(stoi.items(), key=lambda kv_: kv_[1])
    kept_words, kept_vecs = [], []
    for word, _ in items:
        if word in learned:
            kept_words.append(word)
            kept_vecs.append(model.wv[word])

    n_kept = len(kept_words)
    print(f"Aligned to COCA stoi: {n_kept:,} kept, "
          f"{len(stoi) - n_kept:,} dropped (below min_count or unseen)")

    embedding = np.asarray(kept_vecs, dtype=np.float32)        # (n_kept, vector_size)
    new_stoi = {w: i for i, w in enumerate(kept_words)}
    new_itos = {i: w for w, i in new_stoi.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out_emb)) or ".", exist_ok=True)
    np.savez(args.out_emb, embedding=embedding)
    with open(args.out_meta, "wb") as f:
        pickle.dump({'stoi': new_stoi, 'itos': new_itos, 'vocab_size': n_kept}, f)
    print(f"Wrote {args.out_emb}  (shape {embedding.shape})")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
