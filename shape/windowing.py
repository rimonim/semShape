"""
Weight-decay functions for window averaging. Shared between the continuous
Variant-H pipeline (shape.extract) and the discrete FCM pipeline (discrete/build_fcm).

All decay functions return 1.0 at dist=0. Decay formulas for dist >= 1:
  linear:      (window_size - dist) / window_size  (reaches 0 at dist == window_size)
  harmonic:    1 / dist
  exponential: exp(-alpha * dist)
  power:       (1 + dist) ** (-alpha)              (Averell & Heathcote 2011 form)
  none:        1.0
"""

import math


def calculate_weight(dist, window_size, decay_type, alpha=1.0, tokens_per_minute=None):
    """
    Weight for a given |distance| using the named decay function.
    Returns 1.0 at dist=0 and 0.0 if dist > window_size.

    tokens_per_minute: if provided, converts token distance to minutes before
        applying power decay: d_minutes = dist / tokens_per_minute,
        weight = (1 + d_minutes) ** (-alpha). Ignored for other decay types.
    """
    if dist > window_size:
        return 0.0
    if dist == 0:
        return 1.0
    if decay_type == 'linear':
        return max(0.0, (window_size - dist) / window_size)
    elif decay_type == 'harmonic':
        return 1.0 / dist
    elif decay_type == 'exponential':
        return math.exp(-alpha * dist)
    elif decay_type == 'power':
        d_eff = dist / tokens_per_minute if tokens_per_minute is not None else dist
        return (1.0 + d_eff) ** (-alpha)
    elif decay_type == 'none':
        return 1.0
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")


def build_weight_lookup(window_size, decay_type, alpha=1.0,
                        include_target=True, direction='symmetric',
                        tokens_per_minute=None):
    """
    Build {offset d -> weight} for d in [-window_size, window_size].

    d > 0: position is after target; d < 0: before target; d = 0: target itself.

    direction:
      'symmetric' - both sides equally weighted
      'forward'   - only d > 0
      'backward'  - only d < 0

    tokens_per_minute: if provided, converts token distances to minutes before
        applying power decay (d_minutes = |d| / tokens_per_minute). Ignored
        for other decay types. Must be positive.

    Returns:
        dict mapping d -> weight (zero-weighted offsets omitted).
    """
    if tokens_per_minute is not None and tokens_per_minute <= 0:
        raise ValueError(f"tokens_per_minute must be positive, got {tokens_per_minute}")
    if direction == 'symmetric':
        forward_w, backward_w = 1.0, 1.0
    elif direction == 'forward':
        forward_w, backward_w = 1.0, 0.0
    elif direction == 'backward':
        forward_w, backward_w = 0.0, 1.0
    else:
        raise ValueError(f"Unknown direction: {direction}")

    lookup = {}
    for d in range(-window_size, window_size + 1):
        if d == 0:
            if include_target:
                lookup[0] = 1.0
            continue
        w = calculate_weight(abs(d), window_size, decay_type, alpha,
                             tokens_per_minute=tokens_per_minute)
        w *= forward_w if d > 0 else backward_w
        if w > 0:
            lookup[d] = w
    return lookup
