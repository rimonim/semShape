"""
Manim visualization: The Shape of Language on the Probability Simplex.

Run with:
  manim -pql semantic_manifold.py SimplexScene   # low-quality preview
  manim -pqh semantic_manifold.py SimplexScene   # high-quality render

Coordinate system
-----------------
The three probability axes coincide with the Cartesian axes:
  trouble → +y
  pasta   → +z
  sense   → +x

The simplex vertices are at the axis endpoints: (0,L,0), (0,0,L), (L,0,0)
where L = AXIS_LEN = 2.2.

The simplex plane has normal N = (1,1,1)/√3.  After the ILR warp the simplex
is unfolded INTO THE SAME PLANE using the two in-plane ILR basis vectors:
  E1 = (0,1,-1)/√2   (trouble vs pasta direction)
  E2 = (-2,1,1)/√6   ({trouble,pasta} vs sense direction)
ILR origin → SIMPLEX_CENTROID, so the camera stays face-on throughout the warp.

The KDE density surface rises along +N above the ILR plane.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from manim import *
from ilr_math import (
    AXIS_LEN,
    VERTICES_3D,
    simplex_to_3d,
    EXAMPLE_POINTS_P,
    EXAMPLE_POINTS_ILR,
    EXAMPLE_LABELS,
    EXAMPLE_COLORS_HEX,
    generate_background_points,
    ilr_gridlines,
    make_kde,
    kde_surface_grid,
)

# ── Color palette ────────────────────────────────────────────────────────
BG_COLOR       = "#1C1C2E"
SIMPLEX_FILL   = "#3B82F6"
SIMPLEX_EDGE   = "#93C5FD"
DOT_COLOR      = "#94A3B8"
GRID_COLOR     = "#60A5FA"
AXIS_COLOR     = "#E2E8F0"
KDE_LOW        = "#818CF8"
KDE_HIGH       = "#F472B6"

POINT_COLORS = {
    "stop_making":  ManimColor(EXAMPLE_COLORS_HEX["stop_making"]),
    "lunch_making": ManimColor(EXAMPLE_COLORS_HEX["lunch_making"]),
    "cooking":      ManimColor(EXAMPLE_COLORS_HEX["cooking"]),
}

# ── Layout constants ─────────────────────────────────────────────────────
SPHERE_RADIUS  = 0.08
DOT_RADIUS     = 0.035
ILR_SCALE      = 1.4   # scale ILR coords into Manim units

# ILR plane basis vectors (lie IN the simplex plane)
E1    = np.array([0.,  1., -1.]) / np.sqrt(2)   # trouble vs pasta
E2    = np.array([-2., 1.,  1.]) / np.sqrt(6)   # {trouble,pasta} vs sense
N_VEC = np.array([1.,  1.,  1.]) / np.sqrt(3)   # simplex normal / KDE height dir

# Gridline parameters
N_GRID_LINES = 41
GRID_Y_RANGE = 4.5
N_GRID_PTS   = 60

# Camera angles
PHI_INIT     = 105 * DEGREES
THETA_INIT   = -120 * DEGREES
PHI_FACEON   = 125.26 * DEGREES  # camera at −N: looks along +N at simplex
THETA_FACEON = -135 * DEGREES
PHI_BIRD     = 200 * DEGREES      # 3/4 view to reveal KDE height along N
THETA_BIRD   = -100 * DEGREES
ZOOM         = 1.3

# Simplex centroid in world space (ILR origin maps here)
SIMPLEX_CENTROID = np.array([AXIS_LEN / 3, AXIS_LEN / 3, AXIS_LEN / 3])


def v3(arr):
    """Convert array-like to np.ndarray float (3,)."""
    return np.array(arr, dtype=float)


def simplex_3d(p):
    """Simplex barycentric coords → 3D world position."""
    return v3(simplex_to_3d(p))


def ilr_to_scene(y):
    """Map 2D ILR coords → 3D position in the simplex plane.

    ILR origin (0,0) → SIMPLEX_CENTROID.
    Basis vectors E1, E2 span the same plane as the simplex, so the
    camera angle does not change between the simplex and ILR views.
    """
    return SIMPLEX_CENTROID + float(y[0]) * ILR_SCALE * E1 \
                            + float(y[1]) * ILR_SCALE * E2


# ── Simplex surface ───────────────────────────────────────────────────────

def make_simplex_surface(opacity=0.35):
    def param_func(u, v):
        w = max(1e-6, 1.0 - u - v)
        p = np.array([u, v, w]); p /= p.sum()
        return simplex_3d(p)

    surf = Surface(
        lambda u, v: param_func(u, v),
        u_range=[0.01, 0.98],
        v_range=[0.01, 0.97],
        resolution=(20, 20),
        fill_color=SIMPLEX_FILL,
        fill_opacity=opacity,
        stroke_width=0,
    )
    surf.set_fill(SIMPLEX_FILL, opacity=opacity)
    return surf


def make_simplex_edges():
    v0, v1, v2 = [v3(vx) for vx in VERTICES_3D]
    return VGroup(
        Line3D(v0, v1, color=SIMPLEX_EDGE, thickness=0.015),
        Line3D(v1, v2, color=SIMPLEX_EDGE, thickness=0.015),
        Line3D(v2, v0, color=SIMPLEX_EDGE, thickness=0.015),
    )


# ── Example spheres + annotation boxes ───────────────────────────────────

def make_example_sphere(key, pos3d):
    color = POINT_COLORS[key]
    sphere = Sphere(radius=SPHERE_RADIUS, color=color)
    sphere.move_to(pos3d)
    sphere.set_color(color)
    return sphere


def make_annotation(key):
    """Single-line label: prefix in white, final token in point color."""
    prefix, token = EXAMPLE_LABELS[key]
    color = POINT_COLORS[key]
    text = VGroup(
        Text(prefix, font_size=20, color=WHITE),
        Text(token,  font_size=20, color=color),
    ).arrange(RIGHT, buff=0.1)
    box = SurroundingRectangle(
        text, buff=0.10, color=color,
        fill_color=BG_COLOR, fill_opacity=0.85,
        corner_radius=0.08, stroke_width=1.5,
    )
    return VGroup(box, text)


# ── Background dots ───────────────────────────────────────────────────────

def make_background_dots(simplex_pts):
    dots = VGroup()
    for p in simplex_pts:
        d = Dot3D(point=simplex_3d(p), radius=DOT_RADIUS,
                  color=DOT_COLOR, fill_opacity=0.28)
        dots.add(d)
    return dots


# ── ILR gridlines ─────────────────────────────────────────────────────────

def _grid_opacity(i, n):
    """Fade from center (high) to edges (low) for an 'infinite' look."""
    t = abs(i - (n - 1) / 2.0) / ((n - 1) / 2.0)
    return max(0.12, 0.55 * (1.0 - 0.72 * t))


def make_gridlines_on_simplex():
    """Curved ILR preimage gridlines on the simplex surface."""
    lines_y1, lines_y2 = ilr_gridlines(
        n_lines=N_GRID_LINES, n_pts=N_GRID_PTS, y_range=GRID_Y_RANGE)
    all_lines = VGroup()
    for i, pts3d in enumerate(lines_y1):
        op = _grid_opacity(i, N_GRID_LINES)
        curve = VMobject(stroke_color=GRID_COLOR, stroke_width=1.2,
                         stroke_opacity=op)
        curve.set_points_smoothly([v3(p) for p in pts3d])
        all_lines.add(curve)
    for i, pts3d in enumerate(lines_y2):
        op = _grid_opacity(i, N_GRID_LINES)
        curve = VMobject(stroke_color=GRID_COLOR, stroke_width=1.2,
                         stroke_opacity=op)
        curve.set_points_smoothly([v3(p) for p in pts3d])
        all_lines.add(curve)
    return all_lines


def make_gridlines_in_ilr_space():
    """Straight ILR gridlines in the simplex plane (post-warp).

    Uses the same VMobject type and point count as make_gridlines_on_simplex
    so that Transform interpolates smoothly between the two.
    """
    vals = np.linspace(-GRID_Y_RANGE, GRID_Y_RANGE, N_GRID_LINES)
    t    = np.linspace(-GRID_Y_RANGE, GRID_Y_RANGE, N_GRID_PTS)
    all_lines = VGroup()
    # y1 = const lines (vary y2) — must match order of lines_y1 in source
    for i, y1 in enumerate(vals):
        op = _grid_opacity(i, N_GRID_LINES)
        pts = [v3(ilr_to_scene([y1, y2])) for y2 in t]
        curve = VMobject(stroke_color=GRID_COLOR, stroke_width=1.2,
                         stroke_opacity=op)
        curve.set_points_smoothly(pts)
        all_lines.add(curve)
    # y2 = const lines (vary y1) — must match order of lines_y2 in source
    for i, y2 in enumerate(vals):
        op = _grid_opacity(i, N_GRID_LINES)
        pts = [v3(ilr_to_scene([y1, y2])) for y1 in t]
        curve = VMobject(stroke_color=GRID_COLOR, stroke_width=1.2,
                         stroke_opacity=op)
        curve.set_points_smoothly(pts)
        all_lines.add(curve)
    return all_lines


# ── KDE surface ───────────────────────────────────────────────────────────

def make_kde_surface(ilr_bg_pts, opacity=0.8):
    kde = make_kde(ilr_bg_pts)
    _, _, Z_grid = kde_surface_grid(kde, x_range=(-4, 4),
                                         y_range=(-4, 4), n=80)
    z_max = float(Z_grid.max())
    HEIGHT_SCALE = -1.8  # max height in world units along N_VEC

    def surface_func(u, v):
        density = float(kde([[u], [v]])[0])
        height = density / z_max * HEIGHT_SCALE
        base = ilr_to_scene([u, v])
        return base + height * N_VEC

    surf = Surface(
        surface_func,
        u_range=[-4, 4],
        v_range=[-4, 4],
        resolution=(30, 30),
        fill_opacity=opacity,
        stroke_width=0,
    )
    low_color  = ManimColor(KDE_LOW)
    high_color = ManimColor(KDE_HIGH)
    for face in surf.family_members_with_points():
        center = v3(face.get_center())
        h = float(np.dot(center - SIMPLEX_CENTROID, N_VEC))
        t_val = np.clip(h / HEIGHT_SCALE, 0.0, 1.0)
        face.set_fill(interpolate_color(low_color, high_color, t_val),
                      opacity=opacity)
    return surf


# ════════════════════════════════════════════════════════════════════════
# Main Scene
# ════════════════════════════════════════════════════════════════════════

class SimplexScene(ThreeDScene):

    def construct(self):
        self.camera.background_color = BG_COLOR
        simplex_bg_pts, ilr_bg_pts = generate_background_points(n=200)

        self._scene1_setup(simplex_bg_pts)
        self._scene2_rotate_faceon()
        self._scene3_gridlines()
        self._scene4_warp(simplex_bg_pts, ilr_bg_pts)
        self._scene5_tilt_for_kde()
        self._scene6_kde_surface(ilr_bg_pts)

    # ── Scene 1 ──────────────────────────────────────────────────────────

    def _scene1_setup(self, simplex_bg_pts):
        self.set_camera_orientation(phi=PHI_INIT, theta=THETA_INIT, zoom=ZOOM)

        # 1. Title card
        title = Text("The Shape of Language", font_size=48, color=WHITE)
        subtitle = Text("Probability Distributions on the Simplex",
                        font_size=28, color=ManimColor(AXIS_COLOR))
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(FadeIn(title_group), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(title_group), run_time=0.8)

        # 2. Axes: Line3D to each simplex vertex, endpoint caps (no arrowheads)
        v_trouble = v3(VERTICES_3D[0])
        v_pasta   = v3(VERTICES_3D[1])
        v_sense   = v3(VERTICES_3D[2])

        axes = VGroup(
            Line3D(ORIGIN, v_trouble, color=AXIS_COLOR, thickness=0.012),
            Line3D(ORIGIN, v_pasta,   color=AXIS_COLOR, thickness=0.012),
            Line3D(ORIGIN, v_sense,   color=AXIS_COLOR, thickness=0.012),
        )
        caps = VGroup(
            Sphere(radius=0.05, color=AXIS_COLOR).move_to(v_trouble),
            Sphere(radius=0.05, color=AXIS_COLOR).move_to(v_pasta),
            Sphere(radius=0.05, color=AXIS_COLOR).move_to(v_sense),
        )
        self.play(Create(axes), FadeIn(caps), run_time=1.0)
        self._axes = axes
        self._caps = caps

        # Axis labels beyond the endpoint caps
        F = 1.22
        lbl_trouble = Text('P("trouble")', font_size=22, color=AXIS_COLOR)
        lbl_pasta   = Text('P("pasta")',   font_size=22, color=AXIS_COLOR)
        lbl_sense   = Text('P("sense")',   font_size=22, color=AXIS_COLOR)
        self.add_fixed_orientation_mobjects(lbl_trouble, lbl_pasta, lbl_sense)
        lbl_trouble.move_to(v3([0.0,          AXIS_LEN * F,  0.0          ]))
        lbl_pasta.move_to(  v3([-0.3,         0.0,           AXIS_LEN * F ]))
        lbl_sense.move_to(  v3([AXIS_LEN * F, 0.0,           0.0          ]))
        self.play(FadeIn(lbl_trouble), FadeIn(lbl_pasta), FadeIn(lbl_sense),
                  run_time=0.8)
        self._axis_labels = VGroup(lbl_trouble, lbl_pasta, lbl_sense)

        # 3. Example spheres + single-line annotation (before simplex appears)
        keys = ["stop_making", "lunch_making", "cooking"]
        self._example_spheres = {}
        self._annotations = {}

        for key in keys:
            pos = simplex_3d(EXAMPLE_POINTS_P[key])
            sphere = make_example_sphere(key, pos)
            ann = make_annotation(key)
            self.add_fixed_orientation_mobjects(ann)
            ann.move_to(pos + v3([0.6, 0.35, 0.0]))

            self.play(FadeIn(sphere, scale=0.3), FadeIn(ann), run_time=1)
            self.wait(0.75)
            self._example_spheres[key] = sphere
            self._annotations[key] = ann

        # 4. Simplex triangle fades in
        simplex_surf  = make_simplex_surface(opacity=0.32)
        simplex_edges = make_simplex_edges()
        self.play(FadeIn(simplex_surf), Create(simplex_edges), run_time=1.2)
        self._simplex_surf  = simplex_surf
        self._simplex_edges = simplex_edges

        # 5. Background dots scatter in staggered burst
        bg_dots = make_background_dots(simplex_bg_pts)
        self._bg_dots = bg_dots
        anims = [FadeIn(d, run_time=0.015) for d in bg_dots]
        self.play(LaggedStart(*anims, lag_ratio=0.012), run_time=1.2)

        # 6. Brief orbit to show 3D structure
        self.wait(0.5)
        self.begin_ambient_camera_rotation(rate=-0.12)
        self.wait(3.5)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)

    # ── Scene 2 ──────────────────────────────────────────────────────────

    def _scene2_rotate_faceon(self):
        sub = Text("Each point is a probability distribution over the vocabulary",
                   font_size=24, color=ManimColor(AXIS_COLOR))
        self.add_fixed_in_frame_mobjects(sub)
        sub.to_edge(DOWN)
        self.play(FadeIn(sub), run_time=0.6)

        self.move_camera(phi=PHI_FACEON, theta=THETA_FACEON,
                         frame_center=SIMPLEX_CENTROID,
                         zoom=ZOOM,
                         run_time=2.5, rate_func=smooth)
        self.wait(1.5)
        self.play(FadeOut(sub), run_time=0.5)

    # ── Scene 3 ──────────────────────────────────────────────────────────

    def _scene3_gridlines(self):
        gridlines = make_gridlines_on_simplex()
        self._gridlines = gridlines
        self.play(Create(gridlines), run_time=1.8)
        self.wait(1.5)

    # ── Scene 4 ──────────────────────────────────────────────────────────

    def _scene4_warp(self, _simplex_bg_pts, ilr_bg_pts):
        sub = Text("The ILR transform maps the simplex to flat Euclidean space",
                   font_size=24, color=ManimColor(AXIS_COLOR))
        self.add_fixed_in_frame_mobjects(sub)
        sub.to_edge(DOWN)
        self.play(FadeIn(sub), run_time=0.6)
        self.wait(0.4)

        # ── Target objects in simplex-plane ILR space ─────────────────

        # Flat ILR surface spanning the simplex plane
        big_r = GRID_Y_RANGE
        target_surf = Surface(
            lambda u, v: v3(ilr_to_scene([u * big_r, v * big_r])),
            u_range=[-1, 1],
            v_range=[-1, 1],
            resolution=(20, 20),
            fill_color=SIMPLEX_FILL,
            fill_opacity=0.08,
            stroke_width=0,
        )

        # Straight gridlines in simplex plane (same VMobject format as source)
        target_gridlines = make_gridlines_in_ilr_space()

        # Example spheres at ILR positions in the simplex plane
        target_spheres = {}
        for key in self._example_spheres:
            y = EXAMPLE_POINTS_ILR[key]
            pos = v3(ilr_to_scene(y))
            target_spheres[key] = make_example_sphere(key, pos)

        # Background dots at ILR positions in the simplex plane
        target_bg_dots = VGroup()
        for y in ilr_bg_pts:
            target_bg_dots.add(
                Dot3D(point=v3(ilr_to_scene(y)), radius=DOT_RADIUS,
                      color=DOT_COLOR, fill_opacity=0.28))

        # ── Animate warp (camera stays face-on) ───────────────────────
        anim_list = [
            Transform(self._simplex_surf,  target_surf),
            Transform(self._gridlines,     target_gridlines),
            Transform(self._bg_dots,       target_bg_dots),
            FadeOut(self._simplex_edges),
            FadeOut(self._axes),
            FadeOut(self._caps),
            FadeOut(self._axis_labels),
        ]
        for key in self._example_spheres:
            anim_list.append(Transform(self._example_spheres[key],
                                       target_spheres[key]))
        for ann in self._annotations.values():
            anim_list.append(FadeOut(ann))

        self.play(*anim_list, run_time=3.5, rate_func=smooth)
        self.wait(1.0)
        self.play(FadeOut(sub), run_time=0.5)

    # ── Scene 5 ──────────────────────────────────────────────────────────

    def _scene5_tilt_for_kde(self):
        """Tilt camera from face-on to a 3/4 view so KDE height is visible."""
        self.move_camera(phi=PHI_BIRD, theta=THETA_BIRD, gamma=-30 * DEGREES,
                         frame_center=SIMPLEX_CENTROID,
                         zoom=ZOOM,
                         run_time=2.0, rate_func=smooth)
        self.wait(1.0)

    # ── Scene 6 ──────────────────────────────────────────────────────────

    def _scene6_kde_surface(self, ilr_bg_pts):
        sub = Text("Global semantic density  g(y)",
                   font_size=26, color=ManimColor(AXIS_COLOR))
        self.add_fixed_in_frame_mobjects(sub)
        sub.to_edge(DOWN)

        kde_surface = make_kde_surface(ilr_bg_pts, opacity=0.55)

        self.play(FadeIn(sub), Create(kde_surface), run_time=2.5)
        self.wait(3.0)

        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(5.0)
        self.stop_ambient_camera_rotation()
        self.wait(1.0)
