"""Continuous-framework package for the Shape of Language project."""

from shape.distinctiveness import (
    distinctiveness_field,
    expected_distinctiveness,
    expected_distinctiveness_all,
)
from shape.extract import extract_features, iter_window_states
from shape.similarity import (
    compute_pairwise_similarities,
    compute_pairwise_similarities_streaming,
    compute_token_marginals,
)

__all__ = [
    'distinctiveness_field',
    'expected_distinctiveness',
    'expected_distinctiveness_all',
    'compute_pairwise_similarities',
    'compute_pairwise_similarities_streaming',
    'compute_token_marginals',
    'extract_features',
    'iter_window_states',
]
