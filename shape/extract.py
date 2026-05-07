"""
Stage 1: corpus pass.

Extract per-position effective hidden states h_eff and marginal probabilities Z_w
from a trained GPT over a binary-token corpus.

Pipeline for each valid corpus position t:
  - h̄_t = (Σ_d w_d · h_{t+d}) / Σ_d w_d   (Variant H: Aitchison-natural geometric
                                            mean of simplex points; reduces to h_t
                                            when window=0)
  - h_eff_t = project_h(h̄_t, v_degen) if project_degenerate else h̄_t
  - Z_w   += softmax(W · h_t)_w   (un-averaged; matches the reweighting definition
                                   of g(y|w) at the single-position level)

Efficiency: processes the corpus in overlapping chunks of length L = block_size,
striding by L − window − min_context. Each forward pass yields h and logits at
every position of B chunks simultaneously. Each valid corpus position appears in
exactly one chunk's "valid range" [min_context, L − window).

Outputs under `out_dir`:
  {name}_Z.npy          — shape (V,), normalized so Σ_w Z_w ≈ 1
  {name}_v_degen.npy    — shape (d,), only if project_degenerate
  {name}_h_eff.npy      — shape (N_valid, d), float32, memmapped
  {name}_meta.json      — configuration snapshot
  {name}_subsample.npy  — shape (save_subsample, d), only if save_subsample > 0
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


def _iter_chunks(T, L, window, min_context):
    """
    Yield (chunk_start, valid_local_start, valid_local_end) triples covering every
    corpus position in [min_context, T − window) exactly once.

    Caller may assume min_context >= window (enforced by extract_features).
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
    Map h_eff memmap row indices to corpus indices.

    The chunk walk in `_iter_chunks` is deterministic from
    (corpus_length, block_size, window, min_context), so row `i` of
    `{name}_h_eff.npy` corresponds to corpus position `valid_positions(meta)[i]`.
    Accepts the dict loaded from `{name}_meta.json`.
    """
    T = int(extract_meta['corpus_length'])
    L = int(extract_meta['block_size'])
    forward_window = int(extract_meta.get('forward_window', extract_meta['window']))
    min_context = int(extract_meta['min_context'])
    parts = [np.arange(s + vstart, s + vend, dtype=np.int64)
             for (s, vstart, vend) in _iter_chunks(T, L, forward_window, min_context)]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


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
    min_context=32,
    project_degenerate=True,
    batch_size=32,
    device='cuda',
    compute_dtype='bfloat16',
    save_h_eff=True,
    save_subsample=0,
    seed=1337,
    verbose=True,
    checkpoint_path=None,
):
    """
    Run the Stage 1 corpus pass.

    Args:
        model: a GPT instance in eval mode, already on `device`.
        data: np.memmap or np.ndarray of uint16/int tokens.
        out_dir: output directory (created if absent).
        dataset_name: filename prefix for outputs.
        block_size: forward-pass sequence length. Defaults to model.config.block_size.
        window: half-window size for averaging (0 means h_eff_t = h_t only).
        weights: decay name (see shape.windowing.calculate_weight).
        include_target: whether d=0 is included in the averaging window.
        min_context: discards positions with fewer than this many left-context tokens.
            Bumped up to `window` if smaller.
        project_degenerate: if True, subtract off h's component along v_degen.
        save_h_eff: if True, write the full (N_valid, d) h_eff memmap. Set False
            at COCA-train scale (~TB) and rely on save_subsample instead.
        save_subsample: reservoir-sample this many h_eff vectors (0 = skip).
            Stored on disk as a memmap (safe at 10M+ positions).

    Returns:
        dict with keys {'Z', 'n_valid', 'd', 'V', 'subsample', 'meta'}.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not save_h_eff and save_subsample <= 0:
        raise ValueError(
            "save_h_eff=False requires save_subsample>0 — otherwise no h output "
            "would be written and Stage 2 would have nothing to train on."
        )

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    L = block_size if block_size is not None else model.config.block_size
    assert L <= model.config.block_size, \
        f"block_size {L} > model.config.block_size {model.config.block_size}"

    V = model.config.vocab_size
    d = model.config.n_embd
    T = len(data)

    if min_context < window:
        min_context = window
        if verbose:
            print(f"Note: min_context bumped to {min_context} (= window).")

    # Build weight lookup for window averaging
    if window == 0:
        weights_lookup = {0: 1.0}
    else:
        weights_lookup = build_weight_lookup(
            window_size=window, decay_type=weights, alpha=weights_alpha,
            include_target=include_target, direction=direction,
        )
    total_weight = sum(weights_lookup.values())
    if total_weight == 0:
        raise ValueError("Window weights sum to zero; refusing to run.")
    offsets = sorted(weights_lookup.keys())
    # forward_window / backward_window from actual offsets, not the raw `window`
    # parameter. A backward-only window has forward_window=0; _iter_chunks must
    # use forward_window so the right-edge valid range is [min_context, L), not
    # the empty [min_context, L-window).
    forward_window  = max(max(offsets), 0)
    backward_window = max(-min(offsets), 0)

    # Degenerate direction
    v_degen_tensor = None
    if project_degenerate:
        if verbose:
            print("Computing degenerate direction v_degen (lstsq of W v ≈ 1)...")
        W_cpu = model.lm_head.weight.detach().cpu().float()
        v = degenerate_direction(W_cpu)
        np.save(os.path.join(out_dir, f"{dataset_name}_v_degen.npy"), v.numpy())
        v_degen_tensor = v.to(device)

    n_valid_total = _count_valid(T, L, forward_window, min_context)
    if verbose:
        print(f"Corpus length: {T:,} tokens")
        print(f"block_size L: {L}, forward_window: {forward_window}, "
              f"backward_window: {backward_window}, min_context: {min_context}")
        print(f"Valid positions: {n_valid_total:,}")
        print(f"d = {d}, V = {V}")
        print(f"Active offsets: {offsets}")
        print(f"Σ w = {total_weight:.4f}")

    # Allocate memmapped h_eff array (optional at COCA-train scale)
    h_eff_path = os.path.join(out_dir, f"{dataset_name}_h_eff.npy")
    h_eff_mm = None
    if save_h_eff:
        h_eff_mm = open_memmap(h_eff_path, mode='w+', dtype=np.float32,
                               shape=(n_valid_total, d))
        if verbose:
            print(f"Allocated {h_eff_path}: "
                  f"{h_eff_mm.size * h_eff_mm.dtype.itemsize / 1e9:.2f} GB")
    elif verbose:
        print(f"save_h_eff=False → skipping full h_eff memmap "
              f"(would be {n_valid_total * d * 4 / 1e9:.2f} GB)")

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

    # Compute context
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                   'float16': torch.float16}
    ptdtype = ptdtype_map[compute_dtype]
    ctx = nullcontext() if device_type == 'cpu' else \
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Count batches for progress bar
    chunks_list = list(_iter_chunks(T, L, forward_window, min_context))
    n_chunks = len(chunks_list)
    n_batches = (n_chunks + batch_size - 1) // batch_size

    cursor = 0
    model.eval()

    with torch.no_grad():
        with ctx:
            pbar = tqdm(range(0, n_chunks, batch_size),
                        total=n_batches, desc="extract", disable=not verbose,
                        unit="batch")
            for batch_start in pbar:
                chunks = chunks_list[batch_start:batch_start + batch_size]
                B = len(chunks)

                # Prepare input tensor
                input_np = np.zeros((B, L), dtype=np.int64)
                for i, (s, _, _) in enumerate(chunks):
                    input_np[i] = data[s:s + L].astype(np.int64)
                input_tensor = torch.from_numpy(input_np).to(device)

                # Forward pass with hidden states
                logits, _, h = model(input_tensor, return_hidden=True)
                # logits: (B, L, V) — bfloat16 under autocast
                # h:      (B, L, d) — also autocast dtype

                # Collect (batch_idx, local_t) for every valid position in this batch
                b_idx_list, t_local_list = [], []
                for i, (_, vstart, vend) in enumerate(chunks):
                    for t_local in range(vstart, vend):
                        b_idx_list.append(i)
                        t_local_list.append(t_local)
                n_valid_batch = len(b_idx_list)

                b_idx = torch.tensor(b_idx_list, dtype=torch.long, device=device)
                t_local = torch.tensor(t_local_list, dtype=torch.long, device=device)

                # Z_w accumulation (un-averaged positions)
                valid_logits = logits[b_idx, t_local, :]                  # (n_valid_batch, V)
                valid_probs = F.softmax(valid_logits.float(), dim=-1)     # fp32
                Z_sum += valid_probs.sum(dim=0).to(torch.float64)

                # Window-averaged h̄
                h_bar = torch.zeros(n_valid_batch, d, dtype=torch.float32, device=device)
                for off in offsets:
                    w_off = float(weights_lookup[off])
                    # Note: t_local + off is guaranteed in [0, L) by the valid-range construction
                    h_at_off = h[b_idx, t_local + off, :].float()         # (n_valid_batch, d)
                    h_bar += w_off * h_at_off
                h_bar /= float(total_weight)

                # Project out degenerate direction
                if v_degen_tensor is not None:
                    h_bar = project_h(h_bar, v_degen_tensor)

                # Write batch to h_eff memmap (if enabled)
                h_bar_np = h_bar.cpu().numpy()
                if h_eff_mm is not None:
                    h_eff_mm[cursor:cursor + n_valid_batch] = h_bar_np

                # Reservoir sampling
                if save_subsample > 0:
                    for i in range(n_valid_batch):
                        if n_seen_for_reservoir < save_subsample:
                            subsample[n_seen_for_reservoir] = h_bar_np[i]
                        else:
                            j = int(rng.integers(0, n_seen_for_reservoir + 1))
                            if j < save_subsample:
                                subsample[j] = h_bar_np[i]
                        n_seen_for_reservoir += 1

                cursor += n_valid_batch
                if verbose:
                    pbar.set_postfix({'positions': cursor,
                                      'Σ_w Z_w': f"{(Z_sum.sum() / max(cursor, 1)).item():.4f}"})

    assert cursor == n_valid_total, f"cursor {cursor} != n_valid_total {n_valid_total}"

    if h_eff_mm is not None:
        h_eff_mm.flush()
        del h_eff_mm

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
        'N_valid': n_valid_total,
        'd': d,
        'V': V,
        'block_size': L,
        'corpus_length': T,
        'window': window,
        'forward_window': forward_window,
        'backward_window': backward_window,
        'weights': weights,
        'weights_alpha': weights_alpha,
        'direction': direction,
        'include_target': include_target,
        'min_context': min_context,
        'project_degenerate': project_degenerate,
        'offsets': offsets,
        'offset_weights': [float(weights_lookup[o]) for o in offsets],
        'total_weight': float(total_weight),
        'Z_sum_check': z_sum_check,
        'save_h_eff': save_h_eff,
        'save_subsample': save_subsample,
        'n_subsample_actual': int(min(n_seen_for_reservoir, save_subsample)) if save_subsample > 0 else 0,
    }
    with open(os.path.join(out_dir, f"{dataset_name}_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print("\nStage 1 complete.")
        if save_h_eff:
            print(f"  h_eff: {n_valid_total:,} × {d} float32 → {h_eff_path}")
        if save_subsample > 0:
            n_in = min(n_seen_for_reservoir, save_subsample)
            print(f"  subsample: {n_in:,} × {d} float32 → {subsample_path}")
        print(f"  Σ_w Z_w = {z_sum_check:.6f} (should be ≈ 1)")

    return {
        'Z': Z,
        'n_valid': n_valid_total,
        'd': d,
        'V': V,
        'subsample_path': subsample_path if save_subsample > 0 else None,
        'h_eff_path': h_eff_path if save_h_eff else None,
        'meta': meta,
    }
