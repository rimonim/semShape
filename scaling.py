# MIT License — Copyright (c) 2022 Andrej Karpathy
# Source: https://github.com/karpathy/nanochat
"""
Compute-optimal hyperparameter scaling, adapted from karpathy/nanochat.

Given a model depth (number of transformer layers), derives all other
hyperparameters for a compute-optimal training configuration.

Usage from a config file:
    from scaling import compute_optimal_config
    cfg = compute_optimal_config(depth=20)
    n_layer = cfg['n_layer']
    n_head = cfg['n_head']
    n_embd = cfg['n_embd']
    max_iters = cfg['max_iters']
    # ... etc

Or call compute_optimal_config() and unpack into globals:
    from scaling import compute_optimal_config
    globals().update(compute_optimal_config(depth=20))

Reference: karpathy/nanochat scripts/base_train.py
"""

import math


def compute_optimal_config(
    depth,
    vocab_size=None,
    block_size=1024,
    head_dim=64,
    aspect_ratio=64,
    target_param_data_ratio=20,
    batch_size=None,
    gradient_accumulation_steps=None,
    corpus_tokens=None,
    max_epochs=None,
):
    """
    Derive compute-optimal hyperparameters from model depth.

    Args:
        depth: Number of transformer layers (the single complexity dial).
        vocab_size: Vocabulary size. If None, not set in output.
        block_size: Sequence length (context window).
        head_dim: Dimension per attention head (default 64).
        aspect_ratio: Multiplier from depth to base model dimension (default 64).
            model_dim ~= depth * aspect_ratio, rounded to multiple of head_dim.
        target_param_data_ratio: Ratio of training tokens to model parameters.
            Chinchilla-optimal is ~20. Nanochat uses 20 by default.
        batch_size: Micro batch size. If None, auto-derived.
        gradient_accumulation_steps: If None, auto-derived.
        corpus_tokens: Number of tokens in the training corpus. If provided,
            the summary will report the implied number of epochs. If max_epochs
            is also set, max_iters will be capped accordingly.
        max_epochs: Maximum number of epochs over the corpus. Requires
            corpus_tokens. If the compute-optimal target_tokens would exceed
            max_epochs * corpus_tokens, max_iters is reduced to fit.

    Returns:
        Dict of training config variables suitable for unpacking into train.py globals.
    """
    # --- Model architecture ---
    base_dim = depth * aspect_ratio
    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = n_embd // head_dim
    n_layer = depth

    # --- Estimate parameter count (transformer blocks only, for scaling) ---
    # Each layer: attn (4 * n_embd^2) + MLP with SwiGLU (3 * n_embd * hidden_dim)
    hidden_dim = int(8 * n_embd / 3)
    hidden_dim = ((hidden_dim + 255) // 256) * 256
    params_per_layer = 4 * n_embd * n_embd + 3 * n_embd * hidden_dim
    n_scaling_params = n_layer * params_per_layer
    # Add embedding + unembedding (untied)
    if vocab_size is not None:
        n_scaling_params += 2 * vocab_size * n_embd

    # --- Training duration ---
    target_tokens = int(target_param_data_ratio * n_scaling_params)

    # --- Batch size scaling (Power Lines: B_opt ∝ D^0.383) ---
    # Reference values from nanochat (GPT-2 scale)
    B_REF = 524288  # reference total batch size in tokens
    D_REF = 10_000_000_000  # reference dataset size (10B tokens)

    if batch_size is None:
        batch_size = 12  # micro batch size

    tokens_per_micro = batch_size * block_size

    if gradient_accumulation_steps is None:
        # Derive optimal total batch size
        batch_size_ratio = target_tokens / D_REF
        predicted_batch_tokens = B_REF * batch_size_ratio ** 0.383
        # Round to nearest power of 2
        predicted_batch_tokens = 2 ** round(math.log2(max(predicted_batch_tokens, tokens_per_micro)))
        # Clamp to reasonable range
        predicted_batch_tokens = max(predicted_batch_tokens, tokens_per_micro)
        gradient_accumulation_steps = max(1, predicted_batch_tokens // tokens_per_micro)

    total_batch_tokens = gradient_accumulation_steps * tokens_per_micro
    max_iters = max(1, target_tokens // total_batch_tokens)

    # Cap to max_epochs if corpus size is known
    epochs = None
    if corpus_tokens is not None:
        if max_epochs is not None:
            max_iters_from_epochs = max(1, int(max_epochs * corpus_tokens / total_batch_tokens))
            max_iters = min(max_iters, max_iters_from_epochs)
        epochs = max_iters * total_batch_tokens / corpus_tokens

    # --- Learning rate scaling ---
    # LR scales as sqrt(B / B_ref) relative to reference
    batch_lr_scale = (total_batch_tokens / B_REF) ** 0.5
    # Base LR of 6e-4 at reference batch size
    learning_rate = 6e-4 * batch_lr_scale
    min_lr = learning_rate / 10

    # --- Weight decay scaling ---
    # Scales with sqrt(B/B_ref) * (D_ref/D)
    weight_decay = 0.1 * math.sqrt(total_batch_tokens / B_REF) * (D_REF / target_tokens)
    weight_decay = min(weight_decay, 0.5)  # clamp

    # --- Warmup and decay ---
    warmup_iters = max(100, min(2000, max_iters // 20))
    lr_decay_iters = max_iters  # decay over full training

    cfg = dict(
        # Model
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=False,
        dropout=0.0,
        # Training
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_iters=max_iters,
        # Optimizer
        use_muon=True,
        learning_rate=learning_rate,
        min_lr=min_lr,
        weight_decay=weight_decay,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        # LR schedule
        decay_lr=True,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
    )
    if vocab_size is not None:
        cfg['vocab_size'] = vocab_size

    # Print summary
    print(f"Compute-optimal config for depth={depth}:")
    print(f"  Architecture: {n_layer}L / {n_head}H / {n_embd}d (head_dim={head_dim})")
    print(f"  Estimated params: {n_scaling_params/1e6:.1f}M")
    print(f"  Target tokens: {target_tokens/1e6:.0f}M (ratio={target_param_data_ratio})")
    print(f"  Batch: {batch_size} x {block_size} x {gradient_accumulation_steps} = {total_batch_tokens:,} tokens/iter")
    iters_line = f"  Training: {max_iters:,} iters"
    if epochs is not None:
        iters_line += f" ({epochs:.1f} epochs over {corpus_tokens/1e6:.0f}M tokens)"
    print(iters_line)
    print(f"  LR: {learning_rate:.2e} -> {min_lr:.2e} (warmup={warmup_iters})")
    print(f"  Weight decay: {weight_decay:.4f}")

    return cfg
