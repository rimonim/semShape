"""
(legacy script)
Build a Feature Co-occurrence Matrix (FCM) using a trained nanoGPT model.

Instead of counting actual co-occurrences, this computes the model's predicted
probability distribution at each position and accumulates these probabilities
into an FCM. The result is a dense vocab_size x vocab_size matrix where each
element [i, j] represents the weighted sum of probabilities that token j appears
in the context window around token i.

Parameters match those from the wordembeddings R package's fcm() function.
"""
import os
import math
import pickle
import time
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm

import torch
import torch.nn.functional as F

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default configuration (can be overridden via command line or config file)
# -----------------------------------------------------------------------------

# Model loading
init_from = 'resume'  # 'resume' (from out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'

# Data
data_dir = ''  # path to data directory containing train.bin (auto-detected if empty)
split = 'train'  # 'train' or 'val'

# Context window parameters (matching wordembeddings R package)
window = 5  # window size on each side of target
weights = 'linear'  # 'linear', 'harmonic', 'exponential', 'power', 'none', or comma-separated vector
weights_alpha = 1.0  # parameter for exponential/power decay
direction = 'symmetric'  # 'symmetric', 'forward', 'backward', or numeric ratio
include_target = False  # include target word at distance 0

# Vocabulary filtering
vocab_size_limit = None  # limit to top N most frequent types (None = no limit)
vocab_coverage = None  # limit to types covering this proportion of tokens
min_count = 1  # minimum frequency threshold

# Processing
batch_size = 64  # number of positions to process in parallel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False  # use PyTorch 2.0 compilation
seed = 1337
verbose = True  # print detailed progress information
log_interval = 10000  # print stats every N positions

# Output
output_file = 'fcm.npy'  # output file for the FCM matrix
save_vocab = True  # also save vocabulary mapping

exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------


def parse_weights(weights_str, window_size, include_target):
    """Parse weights parameter: either a decay function name or a custom vector."""
    if weights_str in ('linear', 'harmonic', 'exponential', 'power', 'none'):
        return weights_str, None
    
    # Try to parse as comma-separated vector
    try:
        vec = [float(x.strip()) for x in weights_str.split(',')]
        expected_len = window_size + 1 if include_target else window_size
        expected_len_asym = (2 * window_size + 1) if include_target else (2 * window_size)
        
        if len(vec) == expected_len:
            # Symmetric weights
            return 'vector', np.array(vec)
        elif len(vec) == expected_len_asym:
            # Asymmetric weights
            return 'vector_asym', np.array(vec)
        else:
            raise ValueError(
                f"Weight vector length {len(vec)} doesn't match window={window_size}, "
                f"include_target={include_target}. Expected {expected_len} or {expected_len_asym}."
            )
    except ValueError as e:
        if 'could not convert' in str(e):
            raise ValueError(f"Unknown weights type: {weights_str}")
        raise


def parse_direction(direction_str):
    """Parse direction parameter: string or numeric ratio."""
    if direction_str == 'symmetric':
        return 1.0, 1.0  # forward_weight, backward_weight
    elif direction_str == 'forward':
        return 1.0, 0.0
    elif direction_str == 'backward':
        return 0.0, 1.0
    else:
        # Try to parse as numeric ratio
        try:
            ratio = float(direction_str)
            if ratio < 0:
                raise ValueError("Direction ratio must be non-negative")
            return ratio, 1.0  # forward = ratio, backward = 1.0
        except ValueError:
            raise ValueError(
                f"direction must be 'symmetric', 'forward', 'backward', or a numeric ratio, got: {direction_str}"
            )


def calculate_weight(dist, window_size, decay_type, alpha=1.0):
    """
    Calculate weight for a given distance using the specified decay function.
    Matches the formulas in fcm.cpp from the wordembeddings R package.
    
    Args:
        dist: Distance from target (1 = immediate neighbor)
        window_size: Window size
        decay_type: 'linear', 'harmonic', 'exponential', 'power', or 'none'
        alpha: Parameter for exponential/power decay
    
    Returns:
        Weight value (0 if dist > window_size)
    """
    if dist > window_size:
        return 0.0
    
    if decay_type == 'linear':
        return max(0.0, (window_size - dist + 1.0) / window_size)
    elif decay_type == 'harmonic':
        return 1.0 / dist if dist > 0 else 1.0
    elif decay_type == 'exponential':
        return math.exp(-alpha * dist)
    elif decay_type == 'power':
        return dist ** (-alpha) if dist > 0 else 1.0
    else:  # 'none'
        return 1.0


def get_target_weight(window_size, decay_type, alpha=1.0):
    """
    Get weight for the target word itself (distance 0).
    Matches special handling in fcm.cpp.
    """
    if decay_type in ('harmonic', 'power'):
        return 1.0  # Avoid division by zero
    elif decay_type == 'linear':
        return (window_size + 1.0) / window_size
    else:
        return calculate_weight(0.0, window_size, decay_type, alpha)


def build_weight_lookup(window_size, decay_type, alpha, include_target, 
                        custom_weights, forward_weight, backward_weight):
    """
    Pre-compute weights for all offsets in [-window, window].
    
    Returns:
        dict mapping offset d -> weight
        (offset is from target's perspective: d < 0 means context is before target)
    """
    weights_lookup = {}
    
    for d in range(-window_size, window_size + 1):
        if d == 0:
            if include_target:
                if custom_weights is not None:
                    # For custom weights, index 0 (or middle for asymmetric) is the target
                    if len(custom_weights) == window_size + 1:
                        w = custom_weights[0]
                    else:  # asymmetric: 2*window + 1
                        w = custom_weights[window_size]
                else:
                    w = get_target_weight(window_size, decay_type, alpha)
                weights_lookup[0] = w
            continue
        
        abs_d = abs(d)
        
        if custom_weights is not None:
            if len(custom_weights) == window_size + 1:
                # Symmetric weights: [0, 1, 2, ..., W] or [1, 2, ..., W]
                if include_target:
                    idx = abs_d
                else:
                    idx = abs_d - 1
                w = custom_weights[idx] if idx < len(custom_weights) else 0.0
            else:
                # Asymmetric weights: [-W, ..., -1, 0?, 1, ..., W]
                if include_target:
                    idx = d + window_size
                else:
                    # [-W, ..., -1, 1, ..., W] without 0
                    if d < 0:
                        idx = d + window_size
                    else:
                        idx = d + window_size - 1
                w = custom_weights[idx] if 0 <= idx < len(custom_weights) else 0.0
        else:
            w = calculate_weight(abs_d, window_size, decay_type, alpha)
        
        # Apply direction weights
        if d < 0:  # backward context (context before target)
            w *= backward_weight
        else:  # forward context (context after target)
            w *= forward_weight
        
        if w > 0:
            weights_lookup[d] = w
    
    return weights_lookup


def compute_token_frequencies(data):
    """Compute frequency of each token in the corpus."""
    unique, counts = np.unique(data, return_counts=True)
    freq_dict = dict(zip(unique, counts))
    return freq_dict


def get_vocab_mask(vocab_size, token_freqs, vocab_size_limit=None, 
                   vocab_coverage=None, min_count=1):
    """
    Create a boolean mask indicating which tokens to include in the FCM.
    
    Returns:
        numpy array of shape (vocab_size,) with True for included tokens
    """
    mask = np.ones(vocab_size, dtype=bool)
    
    # Sort tokens by frequency
    sorted_tokens = sorted(token_freqs.items(), key=lambda x: -x[1])
    total_count = sum(token_freqs.values())
    
    # Apply vocab_size_limit
    if vocab_size_limit is not None:
        top_tokens = set(t for t, _ in sorted_tokens[:vocab_size_limit])
        for i in range(vocab_size):
            if i not in top_tokens:
                mask[i] = False
    
    # Apply vocab_coverage
    if vocab_coverage is not None:
        cumsum = 0
        coverage_tokens = set()
        for token, count in sorted_tokens:
            cumsum += count
            coverage_tokens.add(token)
            if cumsum / total_count >= vocab_coverage:
                break
        for i in range(vocab_size):
            if i not in coverage_tokens:
                mask[i] = False
    
    # Apply min_count
    if min_count > 1:
        for i in range(vocab_size):
            if token_freqs.get(i, 0) < min_count:
                mask[i] = False
    
    return mask


def main():
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup device and dtype
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Load model
    print("Loading model...")
    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    else:
        raise ValueError(f"Unknown init_from: {init_from}")
    
    model.eval()
    model.to(device)
    if compile_model:
        model = torch.compile(model)
    
    model_vocab_size = model.config.vocab_size
    block_size = model.config.block_size
    print(f"Model vocab size: {model_vocab_size}, block size: {block_size}")
    
    # Determine data directory
    global data_dir
    if not data_dir:
        if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
            data_dir = os.path.join('data', checkpoint['config']['dataset'])
        else:
            raise ValueError("data_dir not specified and couldn't auto-detect from checkpoint")
    
    # Load data
    data_file = os.path.join(data_dir, f'{split}.bin')
    print(f"Loading data from {data_file}...")
    data = np.memmap(data_file, dtype=np.uint16, mode='r')
    corpus_len = len(data)
    print(f"Corpus length: {corpus_len:,} tokens")
    
    # Parse parameters
    weights_type, custom_weights = parse_weights(weights, window, include_target)
    forward_weight, backward_weight = parse_direction(direction)
    
    print("\nFCM Parameters:")
    print(f"  Window: {window}")
    print(f"  Weights: {weights_type}" + (f" (alpha={weights_alpha})" if weights_type in ('exponential', 'power') else ""))
    print(f"  Direction: forward={forward_weight}, backward={backward_weight}")
    print(f"  Include target: {include_target}")
    
    # Build weight lookup
    weights_lookup = build_weight_lookup(
        window, weights_type if custom_weights is None else 'custom', 
        weights_alpha, include_target, custom_weights,
        forward_weight, backward_weight
    )
    print(f"  Active offsets: {sorted(weights_lookup.keys())}")
    
    # Compute token frequencies and vocabulary mask
    print("\nComputing token frequencies...")
    token_freqs = compute_token_frequencies(data)
    vocab_mask = get_vocab_mask(
        model_vocab_size, token_freqs, 
        vocab_size_limit, vocab_coverage, min_count
    )
    n_active_tokens = vocab_mask.sum()
    print(f"Active vocabulary: {n_active_tokens:,} / {model_vocab_size:,} tokens")
    
    # Initialize FCM as dense matrix (will be large for big vocabularies!)
    print(f"\nInitializing FCM matrix ({model_vocab_size} x {model_vocab_size})...")
    fcm = np.zeros((model_vocab_size, model_vocab_size), dtype=np.float32)
    
    # Process corpus
    print(f"\nBuilding FCM (batch_size={batch_size})...")
    
    # We need at least (block_size - 1) context tokens before position t to get P(token_t | context)
    # For each position t, we compute P(next_token | tokens[0:t+1])
    # This distribution contributes to targets at positions t-W+1 through t+W
    
    # Progress tracking
    start_time = time.time()
    last_log_time = start_time
    last_log_pos = 0
    total_batches = (corpus_len + batch_size - 1) // batch_size
    
    if verbose:
        print(f"Total batches: {total_batches:,}")
        print(f"Estimated positions per batch: {batch_size}")
        print()
    
    with torch.no_grad():
        with ctx:
            # Process in batches of positions
            pbar = tqdm(range(0, corpus_len, batch_size), desc="Processing", 
                        disable=not verbose, unit="batch")
            for batch_start in pbar:
                batch_end = min(batch_start + batch_size, corpus_len)
                actual_batch_size = batch_end - batch_start
                
                # Prepare batch: for each position t, we need context tokens[max(0, t-block_size+1):t+1]
                # Then we get the distribution for position t (predicting what comes next)
                batch_contexts = []
                batch_positions = []
                
                for t in range(batch_start, batch_end):
                    # Context for predicting position t
                    # We feed tokens[0:t] and get logits for position t
                    ctx_start = max(0, t - block_size + 1)
                    context = data[ctx_start:t + 1].astype(np.int64)
                    batch_contexts.append(context)
                    batch_positions.append(t)
                
                # Pad contexts to same length for batching
                max_ctx_len = max(len(c) for c in batch_contexts)
                padded_contexts = np.zeros((actual_batch_size, max_ctx_len), dtype=np.int64)
                for i, ctx in enumerate(batch_contexts):
                    padded_contexts[i, -len(ctx):] = ctx  # right-align
                
                # Forward pass
                input_tensor = torch.from_numpy(padded_contexts).to(device)
                logits, _ = model(input_tensor)  # (batch, 1, vocab_size) - only last position
                probs = F.softmax(logits[:, -1, :], dim=-1).float().cpu().numpy()  # (batch, vocab_size)
                
                # Accumulate into FCM
                for i, t in enumerate(batch_positions):
                    # This probability distribution at position t contributes to all targets
                    # at positions t-W through t+W (where t is in their context window)
                    for d, weight in weights_lookup.items():
                        # d is the offset from target's perspective
                        # If d > 0, then t > t_target, meaning context position is after target
                        # If d < 0, then t < t_target, meaning context position is before target
                        t_target = t - d
                        
                        if t_target < 0 or t_target >= corpus_len:
                            continue
                        
                        target_id = int(data[t_target])
                        
                        # Skip if target not in active vocabulary
                        if not vocab_mask[target_id]:
                            continue
                        
                        # Add weighted probabilities to target's row
                        fcm[target_id, :] += weight * probs[i]
                
                # Verbose progress logging
                if verbose and (batch_end - last_log_pos >= log_interval or batch_end == corpus_len):
                    current_time = time.time()
                    elapsed = current_time - start_time
                    interval_elapsed = current_time - last_log_time
                    interval_tokens = batch_end - last_log_pos
                    
                    tokens_per_sec = interval_tokens / interval_elapsed if interval_elapsed > 0 else 0
                    overall_tokens_per_sec = batch_end / elapsed if elapsed > 0 else 0
                    
                    remaining_tokens = corpus_len - batch_end
                    eta_sec = remaining_tokens / overall_tokens_per_sec if overall_tokens_per_sec > 0 else 0
                    
                    # Update progress bar description with stats
                    pbar.set_postfix({
                        'tok/s': f'{tokens_per_sec:.0f}',
                        'FCM_sum': f'{fcm.sum():.2e}',
                        'ETA': f'{eta_sec/60:.1f}m'
                    })
                    
                    last_log_time = current_time
                    last_log_pos = batch_end
    
    # Print final timing stats
    total_time = time.time() - start_time
    if verbose:
        print("\nProcessing complete:")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Average speed: {corpus_len/total_time:.0f} tokens/sec")
    
    # Apply vocabulary mask to both dimensions
    if not vocab_mask.all():
        print("\nApplying vocabulary mask...")
        # Zero out rows and columns for excluded tokens
        fcm[~vocab_mask, :] = 0
        fcm[:, ~vocab_mask] = 0
    
    # Save results
    print(f"\nSaving FCM to {output_file}...")
    np.save(output_file, fcm)
    
    if save_vocab:
        vocab_file = output_file.replace('.npy', '_vocab.pkl')
        print(f"Saving vocabulary info to {vocab_file}...")
        
        # Try to load vocabulary mapping
        vocab_info = {
            'vocab_size': model_vocab_size,
            'vocab_mask': vocab_mask,
            'token_freqs': token_freqs,
        }
        
        meta_pickle_path = os.path.join(data_dir, 'meta.pkl')
        meta_json_path = os.path.join(data_dir, 'meta.json')
        
        if os.path.exists(meta_pickle_path):
            with open(meta_pickle_path, 'rb') as f:
                meta = pickle.load(f)
            vocab_info['stoi'] = meta.get('stoi')
            vocab_info['itos'] = meta.get('itos')
        elif os.path.exists(meta_json_path):
            vocab_info['tokenizer_path'] = meta_json_path
        
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocab_info, f)
    
    print("\nDone!")
    print(f"FCM shape: {fcm.shape}")
    print(f"FCM non-zero elements: {(fcm != 0).sum():,}")
    print(f"FCM sum: {fcm.sum():.2f}")


if __name__ == '__main__':
    main()
