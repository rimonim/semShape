"""
Load Google-News pretrained word2vec, subset to the COCA vocabulary, and
write outputs compatible with scripts/run_embeddings_benchmarks.py.

Download instructions:
    The 'GoogleNews-vectors-negative300.bin.gz' file (~1.6 GB) is hosted on
    Kaggle and various mirrors. A common source:
        https://huggingface.co/fse/word2vec-google-news-300
    Place the file locally and pass its path via --w2v.

Outputs:
    --out-emb    .npz file with 'embedding' (V_kept × 300, fp32).
    --out-meta   pickle file with {'stoi', 'itos', 'vocab_size'} keyed only on
                 tokens for which a Google-News vector exists. Row order in
                 'embedding' matches the new (compacted) stoi.

Usage:
    python scripts/load_word2vec_googlenews.py \
        --w2v ~/Downloads/GoogleNews-vectors-negative300.bin.gz \
        --coca-meta data/coca/meta.pkl \
        --out-emb embeddings/word2vec_googlenews_coca.npz \
        --out-meta embeddings/word2vec_googlenews_coca_meta.pkl

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
    p.add_argument("--w2v", required=True,
                   help="Path to GoogleNews-vectors-negative300.bin(.gz).")
    p.add_argument("--coca-meta", required=True,
                   help="Path to data/coca/meta.pkl with the COCA stoi.")
    p.add_argument("--out-emb", required=True,
                   help="Output .npz path for the (V_kept, 300) embedding.")
    p.add_argument("--out-meta", required=True,
                   help="Output meta.pkl path (stoi/itos restricted to kept tokens).")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from gensim.models import KeyedVectors
    except ImportError:
        sys.exit("gensim is required. Install with: pip install gensim")

    with open(args.coca_meta, "rb") as f:
        coca_meta = pickle.load(f)
    coca_stoi = coca_meta["stoi"]
    print(f"COCA vocab: {len(coca_stoi):,}")

    print(f"Loading Google-News vectors from {args.w2v} ...")
    kv = KeyedVectors.load_word2vec_format(args.w2v, binary=True)
    print(f"Google-News vocab: {len(kv):,}")

    # Iterate COCA stoi in id order; keep only tokens with a Google-News vector.
    items = sorted(coca_stoi.items(), key=lambda kv_: kv_[1])
    kept_words, kept_vecs = [], []
    for word, _ in items:
        if word in kv:
            kept_words.append(word)
            kept_vecs.append(kv[word])

    n_kept, n_drop = len(kept_words), len(coca_stoi) - len(kept_words)
    print(f"Overlap: {n_kept:,} kept, {n_drop:,} dropped")

    embedding = np.asarray(kept_vecs, dtype=np.float32)        # (n_kept, 300)
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
