"""
Sample the Global Semantic Manifold: run a trained GPT over a binary-token
corpus and record one predictive state per valid corpus position.

A predictive state is a distribution over the vocabulary, optionally averaged
over a weighted context window {d: α_d}. Two averaging geometries are supported:

  - 'aitchison' (default): average hidden states, h̄_t = Σ_d α_d h_{t+d} / Σ_d α_d.
    Because the unembedding is linear, softmax(W h̄_t) is the normalized
    weighted geometric mean of the per-position distributions (the Aitchison
    centroid on the simplex). States are stored compactly as h̄_t ∈ ℝ^d.
  - 'probability': average distributions, X̄_t = Σ_d α_d softmax(W h_{t+d}) / Σ_d α_d,
    i.e. the probability of each token occurring within the window. X̄_t is not
    softmax of any single hidden state, so states are stored as full
    V-dimensional probability vectors (float16) — much larger on disk.

With no window (the default), both geometries coincide and states are stored as
the final hidden states h_t, from which the predictive state is recovered
exactly as softmax(W h_t).

Efficiency: processes the corpus in overlapping chunks of length L = block_size,
striding by L − forward_window − min_context. Each forward pass yields h and
logits at every position of B chunks simultaneously. Each valid corpus position
appears in exactly one chunk's "valid range" [min_context, L − forward_window).

Outputs of `extract_features` under `out_dir`:
  {name}_h.npy          — (N_valid, d) float32 hidden states (no window or
                          'aitchison' averaging), or
  {name}_probs.npy      — (N_valid, V) float16 distributions ('probability'
                          averaging with a non-trivial window)
  {name}_Z.npy          — (V,) mean sampled distribution, Σ_w Z_w ≈ 1
  {name}_meta.json      — configuration snapshot
  {name}_subsample.npy  — (save_subsample, d), only if save_subsample > 0
  {name}_v_degen.npy    — (d,), only if project_degenerate (experimental)
"""

import json
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from tqdm import tqdm

from shape.geometry import degenerate_direction, project_h
from shape.windowing import build_weight_lookup

AVERAGING_MODES = ('aitchison', 'probability')


def _iter_chunks(T, L, window, min_context):
    """
    Yield (chunk_start, valid_local_start, valid_local_end) triples covering every
    corpus position in [min_context, T − window) exactly once.

    Caller may assume min_context >= backward window (enforced by _resolve_window).
    """
    if T < L:
        raise ValueError(f"Corpus length {T} < block_size {L}")

    covered_end = min_context  # next global position awaiting coverage
    while covered_end < T - window:
        # Place chunk so that `covered_end` sits at local position min_context,
        # but don't run past the corpus end.
        s = max(0, covered_end - min_context)
        s = min(s, T - L)
        valid_start_local = max(min_context, covered_end - s)
        valid_end_global = min(s + L - window, T - window)
        valid_end_local = valid_end_global - s
        if valid_start_local >= valid_end_local:
            break
        yield (s, valid_start_local, valid_end_local)
        covered_end = s + valid_end_local


def _count_valid(T, L, window, min_context):
    return sum(end - start for (_, start, end) in _iter_chunks(T, L, window, min_context))


def valid_positions(extract_meta):
    """
    Map sample memmap row indices to corpus indices.

    The chunk walk in `_iter_chunks` is deterministic from
    (corpus_length, block_size, forward_window, min_context), so row `i` of
    `{name}_h.npy` / `{name}_probs.npy` corresponds to corpus position
    `valid_positions(meta)[i]`. Accepts the dict loaded from `{name}_meta.json`.
    """
    T = int(extract_meta['corpus_length'])
    L = int(extract_meta['block_size'])
    forward_window = int(extract_meta.get('forward_window', extract_meta.get('window')))
    min_context = int(extract_meta['min_context'])
    parts = [np.arange(s + vstart, s + vend, dtype=np.int64)
             for (s, vstart, vend) in _iter_chunks(T, L, forward_window, min_context)]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _resolve_window(weights_lookup, min_context, L, verbose=False):
    """
    Validate a {offset: weight} window and derive the chunking parameters.

    forward_window restricts the right edge of every chunk (need
    t + forward_window ≤ L − 1), while backward_window restricts the left edge
    (need t ≥ backward_window), so min_context is bumped up to backward_window.

    Returns:
        dict with keys offsets, total_weight, forward_window, backward_window,
        min_context, trivial (True iff the window is the target position alone).
    """
    if not weights_lookup:
        raise ValueError("weights_lookup is empty")
    offsets = sorted(weights_lookup.keys())
    total_weight = float(sum(weights_lookup.values()))
    if total_weight == 0:
        raise ValueError("Window weights sum to zero; refusing to run.")
    forward_window = max(max(offsets), 0)
    backward_window = max(-min(offsets), 0)
    if min_context < backward_window:
        min_context = backward_window
        if verbose:
            print(f"Note: min_context bumped to {min_context} (= backward window).")
    if min_context + forward_window >= L:
        raise ValueError(
            f"min_context ({min_context}) + forward_window ({forward_window}) "
            f">= block_size ({L}); no chunk can contain a valid position. "
            f"Reduce the window or increase block_size.")
    return {
        'offsets': offsets,
        'total_weight': total_weight,
        'forward_window': forward_window,
        'backward_window': backward_window,
        'min_context': min_context,
        'trivial': offsets == [0],
    }


def iter_window_states(
    model,
    data,
    weights_lookup,
    *,
    averaging='aitchison',
    block_size=None,
    min_context=32,
    batch_size=32,
    device='cuda',
    compute_dtype='bfloat16',
    verbose=True,
    desc='corpus',
):
    """
    Stream window-averaged predictive states over every valid corpus position.

    Yields one `(states, probs)` pair per batch of chunks, in corpus order:
      - trivial window ({0: w}): states = h_t (n, d), probs = softmax(W h_t);
      - 'aitchison':             states = h̄_t (n, d), probs = softmax(W h̄_t);
      - 'probability':           states = probs = X̄_t (n, V).
    Both are fp32 tensors on `device`.

    Call `_resolve_window` with the same arguments beforehand if the number of
    valid positions or the effective min_context is needed up front.

    Args:
        model: GPT in eval mode on `device`.
        data: np.memmap or ndarray of corpus tokens (uint16).
        weights_lookup: {offset_d: weight} dict from build_weight_lookup.
        averaging: 'aitchison' or 'probability' (see module docstring).
        block_size: forward-pass sequence length (defaults to model block_size).
        min_context: minimum left-context tokens per position; bumped up to the
            backward window if smaller.
        batch_size: sequences per forward pass.
        device: torch device string.
        compute_dtype: autocast dtype ('bfloat16', 'float16', 'float32').
        verbose: show tqdm progress and print corpus stats.
        desc: progress-bar label.
    """
    if averaging not in AVERAGING_MODES:
        raise ValueError(f"averaging must be one of {AVERAGING_MODES}, got {averaging!r}")

    L = block_size if block_size is not None else model.config.block_size
    assert L <= model.config.block_size, \
        f"block_size {L} > model.config.block_size {model.config.block_size}"
    T = len(data)

    win = _resolve_window(weights_lookup, min_context, L, verbose=verbose)
    offsets, total_weight = win['offsets'], win['total_weight']
    forward_window, min_context = win['forward_window'], win['min_context']
    # A single-offset window is a rescaling of one position: no averaging needed.
    trivial = win['trivial']
    prob_mode = averaging == 'probability' and not trivial

    chunks_list = list(_iter_chunks(T, L, forward_window, min_context))
    n_chunks = len(chunks_list)
    n_batches = (n_chunks + batch_size - 1) // batch_size

    if verbose:
        n_valid = sum(end - start for (_, start, end) in chunks_list)
        print(f"Corpus: {T:,} tokens; L={L}, forward_window={forward_window}, "
              f"backward_window={win['backward_window']}, min_context={min_context}")
        print(f"Valid positions: {n_valid:,}  |  offsets: {offsets[0]}..{offsets[-1]}, "
              f"Σw={total_weight:.4f}  |  averaging: "
              f"{'none' if trivial else averaging}")

    W = None
    if averaging == 'aitchison' and not trivial:
        W = model.lm_head.weight.detach().float()

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
               'float16': torch.float16}[compute_dtype]
    ctx = (nullcontext() if device_type == 'cpu'
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    model.eval()
    pbar = tqdm(range(0, n_chunks, batch_size), total=n_batches,
                desc=desc, disable=not verbose, unit="batch")
    for batch_start in pbar:
        chunks = chunks_list[batch_start:batch_start + batch_size]
        B = len(chunks)
        input_np = np.zeros((B, L), dtype=np.int64)
        for i, (s, _, _) in enumerate(chunks):
            input_np[i] = data[s:s + L].astype(np.int64)
        input_tensor = torch.from_numpy(input_np).to(device)

        # Autocast and no_grad are thread-local, so they are scoped to the
        # forward pass rather than held open across `yield`.
        with torch.no_grad(), ctx:
            logits, _, h = model(input_tensor, return_hidden=True)
        # logits: (B, L, V), h: (B, L, d) — autocast dtype

        b_idx_list, t_local_list = [], []
        for i, (_, vstart, vend) in enumerate(chunks):
            for t_local in range(vstart, vend):
                b_idx_list.append(i)
                t_local_list.append(t_local)
        n_pos = len(b_idx_list)
        if n_pos == 0:
            continue
        b_idx = torch.tensor(b_idx_list, dtype=torch.long, device=device)
        t_local = torch.tensor(t_local_list, dtype=torch.long, device=device)

        # t_local + off is guaranteed in [0, L) by the valid-range construction
        if prob_mode:
            X = torch.softmax(logits.float(), dim=-1)          # (B, L, V) fp32
            X_bar = torch.zeros(n_pos, X.shape[-1], dtype=torch.float32,
                                device=device)
            for off in offsets:
                X_bar.add_(X[b_idx, t_local + off, :],
                           alpha=float(weights_lookup[off]))
            X_bar /= total_weight
            states = probs = X_bar
        elif trivial:
            states = h[b_idx, t_local, :].float()               # (n_pos, d)
            probs = F.softmax(logits[b_idx, t_local, :].float(), dim=-1)
        else:
            h_bar = torch.zeros(n_pos, h.shape[-1], dtype=torch.float32,
                                device=device)
            for off in offsets:
                h_bar += float(weights_lookup[off]) * h[b_idx, t_local + off, :].float()
            h_bar /= total_weight
            states = h_bar
            probs = F.softmax(h_bar @ W.T, dim=-1)

        yield states, probs


def extract_features(
    model,
    data,
    *,
    out_dir,
    dataset_name,
    block_size=None,
    window=0,
    weights='linear',
    weights_alpha=1.0,
    direction='symmetric',
    include_target=True,
    tokens_per_minute=None,
    averaging='aitchison',
    min_context=32,
    project_degenerate=False,
    batch_size=32,
    device='cuda',
    compute_dtype='bfloat16',
    save_states=True,
    save_subsample=0,
    seed=1337,
    verbose=True,
    checkpoint_path=None,
):
    """
    Sample predictive states from the corpus and write them to disk.

    Args:
        model: a GPT instance in eval mode, already on `device`.
        data: np.memmap or np.ndarray of uint16/int tokens.
        out_dir: output directory (created if absent).
        dataset_name: filename prefix for outputs.
        block_size: forward-pass sequence length. Defaults to model.config.block_size.
        window: half-window size for averaging (0 = single position).
        weights: decay name (see shape.windowing.calculate_weight).
        weights_alpha: decay rate for exponential/power decay.
        direction: 'symmetric', 'forward' or 'backward'.
        include_target: whether d=0 is included in the averaging window.
        tokens_per_minute: if set, token distances are converted to minutes
            before applying the decay (see shape.windowing).
        averaging: 'aitchison' (average hidden states; compact d-dim storage) or
            'probability' (average distributions; V-dim float16 storage).
            Irrelevant when window=0.
        min_context: discards positions with fewer than this many left-context
            tokens. Bumped up to the backward window if smaller.
        project_degenerate: experimental. If True, subtract off each hidden
            state's component along v_degen (see shape.geometry.degenerate_direction).
            This changes softmax(W h) unless W v_degen is exactly constant, and
            is only available for hidden-state outputs.
        save_states: if False, skip the full sample memmap (e.g. at COCA-train
            scale) and rely on save_subsample instead.
        save_subsample: reservoir-sample this many hidden states (0 = skip).
            Stored on disk as a memmap (safe at 10M+ positions). Hidden-state
            outputs only.

    Returns:
        dict with keys {'Z', 'n_valid', 'd', 'V', 'format', 'samples_path',
        'subsample_path', 'meta'}.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not save_states and save_subsample <= 0:
        raise ValueError(
            "save_states=False requires save_subsample>0 — otherwise no samples "
            "would be written."
        )

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    L = block_size if block_size is not None else model.config.block_size
    V = model.config.vocab_size
    d = model.config.n_embd
    T = len(data)

    if window == 0:
        weights_lookup = {0: 1.0}
    else:
        weights_lookup = build_weight_lookup(
            window_size=window, decay_type=weights, alpha=weights_alpha,
            include_target=include_target, direction=direction,
            tokens_per_minute=tokens_per_minute,
        )
    win = _resolve_window(weights_lookup, min_context, L, verbose=verbose)
    offsets, total_weight = win['offsets'], win['total_weight']
    forward_window, min_context = win['forward_window'], win['min_context']

    fmt = 'probs' if (averaging == 'probability' and not win['trivial']) else 'h'
    width = V if fmt == 'probs' else d
    if fmt == 'probs' and project_degenerate:
        raise ValueError("project_degenerate applies to hidden states only; "
                         "it cannot be combined with averaging='probability'.")
    if fmt == 'probs' and save_subsample > 0:
        raise ValueError("save_subsample stores hidden states; it is unavailable "
                         "with averaging='probability'.")

    v_degen_tensor = None
    if project_degenerate:
        if verbose:
            print("Computing degenerate direction v_degen (lstsq of W v ≈ 1)...")
        W_cpu = model.lm_head.weight.detach().cpu().float()
        v = degenerate_direction(W_cpu)
        np.save(os.path.join(out_dir, f"{dataset_name}_v_degen.npy"), v.numpy())
        v_degen_tensor = v.to(device)

    n_valid_total = _count_valid(T, L, forward_window, min_context)

    samples_path = os.path.join(out_dir, f"{dataset_name}_{fmt}.npy")
    mm_dtype = np.float16 if fmt == 'probs' else np.float32
    samples_mm = None
    if save_states:
        samples_mm = open_memmap(samples_path, mode='w+', dtype=mm_dtype,
                                 shape=(n_valid_total, width))
        if verbose:
            print(f"Allocated {samples_path}: "
                  f"{samples_mm.size * samples_mm.dtype.itemsize / 1e9:.2f} GB")
    elif verbose:
        print(f"save_states=False → skipping full sample memmap "
              f"(would be {n_valid_total * width * np.dtype(mm_dtype).itemsize / 1e9:.2f} GB)")

    # Z accumulator (fp64 for numerical stability of sum)
    Z_sum = torch.zeros(V, dtype=torch.float64, device=device)

    # Optional reservoir sample (backed by disk memmap; safe at 10M+ positions)
    subsample = None
    subsample_path = os.path.join(out_dir, f"{dataset_name}_subsample.npy")
    n_seen_for_reservoir = 0
    if save_subsample > 0:
        subsample = open_memmap(subsample_path, mode='w+', dtype=np.float32,
                                shape=(save_subsample, d))
        if verbose:
            print(f"Allocated {subsample_path}: "
                  f"{subsample.size * subsample.dtype.itemsize / 1e9:.2f} GB "
                  f"(reservoir size {save_subsample:,})")

    W = model.lm_head.weight.detach().float() if v_degen_tensor is not None else None

    cursor = 0
    for states, probs in iter_window_states(
            model, data, weights_lookup,
            averaging=averaging, block_size=L, min_context=min_context,
            batch_size=batch_size, device=device, compute_dtype=compute_dtype,
            verbose=verbose, desc="extract"):
        n_batch = states.shape[0]

        if v_degen_tensor is not None:
            states = project_h(states, v_degen_tensor)
            probs = F.softmax(states @ W.T, dim=-1)

        Z_sum += probs.sum(dim=0).to(torch.float64)

        if fmt == 'probs':
            states_np = states.to(torch.float16).cpu().numpy()
        else:
            states_np = states.cpu().numpy()
        if samples_mm is not None:
            samples_mm[cursor:cursor + n_batch] = states_np

        if save_subsample > 0:
            for i in range(n_batch):
                if n_seen_for_reservoir < save_subsample:
                    subsample[n_seen_for_reservoir] = states_np[i]
                else:
                    j = int(rng.integers(0, n_seen_for_reservoir + 1))
                    if j < save_subsample:
                        subsample[j] = states_np[i]
                n_seen_for_reservoir += 1

        cursor += n_batch

    assert cursor == n_valid_total, f"cursor {cursor} != n_valid_total {n_valid_total}"

    if samples_mm is not None:
        samples_mm.flush()
        del samples_mm

    Z = (Z_sum / n_valid_total).cpu().numpy().astype(np.float32)
    z_sum_check = float(Z.sum())
    np.save(os.path.join(out_dir, f"{dataset_name}_Z.npy"), Z)

    if save_subsample > 0:
        # subsample is a memmap; truncate if fewer valid positions than requested
        subsample.flush()
        n_in_reservoir = min(n_seen_for_reservoir, save_subsample)
        if n_in_reservoir < save_subsample:
            # rewrite as a smaller memmap
            tight = open_memmap(subsample_path, mode='w+', dtype=np.float32,
                                shape=(n_in_reservoir, d))
            tight[:] = subsample[:n_in_reservoir]
            tight.flush()
            del tight
        del subsample

    meta = {
        'dataset': dataset_name,
        'checkpoint_path': checkpoint_path,
        'format': fmt,
        'averaging': averaging,
        'N_valid': n_valid_total,
        'shape': [n_valid_total, width],
        'dtype': np.dtype(mm_dtype).name,
        'd': d,
        'V': V,
        'block_size': L,
        'corpus_length': T,
        'window': window,
        'forward_window': forward_window,
        'backward_window': win['backward_window'],
        'weights': weights,
        'weights_alpha': weights_alpha,
        'direction': direction,
        'include_target': include_target,
        'tokens_per_minute': tokens_per_minute,
        'min_context': min_context,
        'project_degenerate': project_degenerate,
        'offsets': offsets,
        'offset_weights': [float(weights_lookup[o]) for o in offsets],
        'total_weight': total_weight,
        'Z_sum_check': z_sum_check,
        'save_states': save_states,
        'save_subsample': save_subsample,
        'n_subsample_actual': int(min(n_seen_for_reservoir, save_subsample)) if save_subsample > 0 else 0,
    }
    with open(os.path.join(out_dir, f"{dataset_name}_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print("\nExtraction complete.")
        if save_states:
            print(f"  samples: {n_valid_total:,} × {width} {meta['dtype']} → {samples_path}")
        if save_subsample > 0:
            n_in = min(n_seen_for_reservoir, save_subsample)
            print(f"  subsample: {n_in:,} × {d} float32 → {subsample_path}")
        print(f"  Σ_w Z_w = {z_sum_check:.6f} (should be ≈ 1)")

    return {
        'Z': Z,
        'n_valid': n_valid_total,
        'd': d,
        'V': V,
        'format': fmt,
        'samples_path': samples_path if save_states else None,
        'subsample_path': subsample_path if save_subsample > 0 else None,
        'meta': meta,
    }
