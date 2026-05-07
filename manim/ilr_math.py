"""
Math helpers for the semantic manifold Manim visualization.

Vocabulary tokens (simplex axes):
  index 0: "trouble"
  index 1: "pasta"
  index 2: "sense"

ILR convention (3-component Helmert sequential binary partition):
  y1 = (1/sqrt(2)) * ln(p0 / p1)           # trouble vs pasta
  y2 = sqrt(2/3)  * ln(sqrt(p0*p1) / p2)   # {trouble,pasta} vs sense
"""

import numpy as np
from scipy.stats import gaussian_kde

# ── Simplex embedding: vertices at the three Cartesian axis endpoints ────
# The simplex Δ² = {p : p₀+p₁+p₂=1, pᵢ≥0} is embedded so that each
# vertex coincides with one Cartesian axis endpoint:
#   trouble → +y (appears up on screen)
#   pasta   → +z (appears toward camera / forward-left)
#   sense   → +x (appears to the right)
# This ensures the probability axes ARE the simplex edges, eliminating any
# mismatch between drawn axes and the simplex geometry.
AXIS_LEN = 2.2
VERTICES_3D = np.array([
    [0.0,      AXIS_LEN, 0.0     ],   # trouble: +y
    [0.0,      0.0,      AXIS_LEN],   # pasta:   +z
    [AXIS_LEN, 0.0,      0.0     ],   # sense:   +x
])


def simplex_to_3d(p):
    """Barycentric simplex coords (trouble, pasta, sense) → 3D point."""
    p = np.asarray(p, dtype=float)
    return p @ VERTICES_3D


# ── ILR transform (3-component) ──────────────────────────────────────────

def ilr(p):
    """Map 3-simplex point → R^2 via ILR."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1)
    y1 = (1.0 / np.sqrt(2.0)) * np.log(p[0] / p[1])
    y2 = np.sqrt(2.0 / 3.0) * np.log(np.sqrt(p[0] * p[1]) / p[2])
    return np.array([y1, y2])


def ilr_inverse(y):
    """Map R^2 → interior of 3-simplex via inverse ILR."""
    y1, y2 = float(y[0]), float(y[1])
    # CLR coordinates (sum to zero by construction of ILR basis)
    c0 =  y1 / np.sqrt(2.0) + y2 / np.sqrt(6.0)
    c1 = -y1 / np.sqrt(2.0) + y2 / np.sqrt(6.0)
    c2 = -2.0 * y2 / np.sqrt(6.0)
    z = np.array([c0, c1, c2])
    p = np.exp(z)
    return p / p.sum()


# ── Example points ───────────────────────────────────────────────────────

EXAMPLE_POINTS_P = {
    "stop_making":  np.array([0.56, 0.07, 0.37]),
    "lunch_making": np.array([0.04, 0.63, 0.33]),
    "cooking":      np.array([0.05, 0.72, 0.23]),
}

EXAMPLE_LABELS = {
    "stop_making":  ('"…stop  ',   'making"'),
    "lunch_making": ('"…I am  ',   'making"'),
    "cooking":      ('"…she is  ', 'cooking"'),
}

EXAMPLE_COLORS_HEX = {
    "stop_making":  "#EF4444",
    "lunch_making": "#F59E0B",
    "cooking":      "#34D399",
}

EXAMPLE_POINTS_ILR = {k: ilr(v) for k, v in EXAMPLE_POINTS_P.items()}
EXAMPLE_POINTS_3D  = {k: simplex_to_3d(v) for k, v in EXAMPLE_POINTS_P.items()}


# ── Background points: 5-Gaussian mixture in ILR space ──────────────────

def generate_background_points(n=200, seed=42):
    """
    Returns (simplex_pts, ilr_pts) arrays of shape (N, 3) and (N, 2).

    Five clusters:
      A (40%): unrelated contexts — dominant cluster near ILR origin but
               offset toward lower ratio values
      B (20%): "stop making" / trouble sense
      C (20%): food sense ("making pasta", "cooking dinner")
      D (10%): "making sense" region
      E (10%): pasta–sense boundary
    """
    rng = np.random.default_rng(seed)

    ilr_stop   = EXAMPLE_POINTS_ILR["stop_making"]
    ilr_lunch  = EXAMPLE_POINTS_ILR["lunch_making"]

    clusters = [
        # A: dominant cluster — most contexts unrelated to all 3 tokens
        {"mean": np.array([-0.6, -0.8]),
         "cov":  np.array([[0.40, 0.06], [0.06, 0.35]]),
         "n": int(0.40 * n)},
        # B: "stop making" / hostile sense
        {"mean": ilr_stop + np.array([0.0, 0.0]),
         "cov":  np.array([[0.22, 0.04], [0.04, 0.18]]),
         "n": int(0.20 * n)},
        # C: food sense
        {"mean": ilr_lunch + np.array([0.0, 0.0]),
         "cov":  np.array([[0.15, 0.02], [0.02, 0.22]]),
         "n": int(0.20 * n)},
        # D: "making sense" region (high ILR2 = high p_sense relative to others)
        {"mean": np.array([0.0, 1.6]),
         "cov":  np.array([[0.18, 0.0], [0.0, 0.14]]),
         "n": int(0.10 * n)},
        # E: pasta–sense boundary (positive ILR1 and moderate ILR2)
        {"mean": np.array([1.2, 0.5]),
         "cov":  np.array([[0.12, 0.0], [0.0, 0.18]]),
         "n": int(0.10 * n)},
    ]

    ilr_pts_list = []
    for c in clusters:
        pts = rng.multivariate_normal(c["mean"], c["cov"], max(1, c["n"]))
        ilr_pts_list.append(pts)

    ilr_pts = np.clip(np.vstack(ilr_pts_list), -3.0, 3.0)
    simplex_pts = np.array([ilr_inverse(y) for y in ilr_pts])
    return simplex_pts, ilr_pts


# ── ILR gridlines ────────────────────────────────────────────────────────

def ilr_gridlines(n_lines=11, n_pts=80, y_range=2.5):
    """
    Compute ILR gridline curves on the simplex.

    Returns two lists of arrays:
      lines_y1_const: curves where ILR1 = const (vary ILR2)
      lines_y2_const: curves where ILR2 = const (vary ILR1)

    Each element is an array of shape (n_pts, 3) — 3D simplex positions.
    """
    vals = np.linspace(-y_range, y_range, n_lines)
    t    = np.linspace(-y_range, y_range, n_pts)

    lines_y1_const = []
    for y1 in vals:
        pts3d = np.array([simplex_to_3d(ilr_inverse([y1, y2])) for y2 in t])
        lines_y1_const.append(pts3d)

    lines_y2_const = []
    for y2 in vals:
        pts3d = np.array([simplex_to_3d(ilr_inverse([y1, y2])) for y1 in t])
        lines_y2_const.append(pts3d)

    return lines_y1_const, lines_y2_const


def ilr_gridlines_in_ilr_space(n_lines=11, y_range=2.5):
    """
    Straight gridlines in ILR space (for the warped end state).
    Returns two lists of (start, end) pairs in R^2 (z=0 plane).
    """
    vals = np.linspace(-y_range, y_range, n_lines)

    lines_y1_const = []
    for y1 in vals:
        lines_y1_const.append((np.array([y1, -y_range]), np.array([y1,  y_range])))

    lines_y2_const = []
    for y2 in vals:
        lines_y2_const.append((np.array([-y_range, y2]), np.array([ y_range, y2])))

    return lines_y1_const, lines_y2_const


# ── KDE surface ──────────────────────────────────────────────────────────

def make_kde(ilr_bg_pts, bw=0.55):
    """Fit a Gaussian KDE to ILR-space points (including example points)."""
    example_ilr = np.array(list(EXAMPLE_POINTS_ILR.values()))
    all_pts = np.vstack([ilr_bg_pts, example_ilr])
    return gaussian_kde(all_pts.T, bw_method=bw)


def kde_surface_grid(kde, x_range=(-3.5, 3.5), y_range=(-3.5, 3.5), n=60):
    """Evaluate KDE on a regular grid; returns X, Y, Z arrays."""
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(pts).reshape(X.shape)
    return X, Y, Z
