"""Point-cloud colouring for the three display modes.

All three modes colour the *same* point set (the object's canonical-frame point
cloud with a per-point part index), so the geometry / NOCS / NPCS toggle only
swaps colours -- never geometry:

  * ``geometry`` -- a fixed distinct colour per part, so links are separable;
  * ``nocs``     -- Normalised Object Coordinate Space: the whole object mapped
    into the unit cube, colour = coordinate;
  * ``npcs``     -- Normalised Part Coordinate Space: each part mapped into its
    own unit cube, colour = coordinate.

Colours are ``uint8`` ``(N, 3)`` arrays, ready for ``add_point_cloud``.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .normalization import normalize_from_points

MODES = ("geometry", "nocs", "npcs")

# Distinct, colour-blind-friendly-ish palette for per-part "geometry" colouring.
_PALETTE = np.array(
    [
        [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
        [255, 127, 0], [255, 214, 0], [166, 86, 40], [247, 129, 191],
        [153, 153, 153], [26, 188, 156], [52, 152, 219], [155, 89, 182],
    ],
    dtype=np.uint8,
)


def _to_rgb(coords01: np.ndarray) -> np.ndarray:
    """Map coordinates already in ``[0, 1]`` to ``uint8`` RGB."""
    return np.clip(coords01 * 255.0, 0, 255).astype(np.uint8)


def geometry_colors(part_ids: np.ndarray) -> np.ndarray:
    return _PALETTE[np.asarray(part_ids, dtype=int) % len(_PALETTE)]


def nocs_colors(points_canonical: np.ndarray) -> np.ndarray:
    """Global NOCS colours using diagonal scaling and cube-centering."""
    pts = np.asarray(points_canonical, dtype=float)
    if len(pts) == 0:
        return np.zeros((0, 3), np.uint8)
    return _to_rgb(normalize_from_points(pts))


def npcs_colors(points_canonical: np.ndarray, part_ids: np.ndarray) -> np.ndarray:
    """Per-part NPCS colours using diagonal scaling and cube-centering."""
    pts = np.asarray(points_canonical, dtype=float)
    part_ids = np.asarray(part_ids, dtype=int)
    colors = np.zeros((len(pts), 3), np.uint8)
    for pid in np.unique(part_ids):
        mask = part_ids == pid
        if not np.any(mask):
            continue
        colors[mask] = _to_rgb(normalize_from_points(pts[mask]))
    return colors


def colors_for_mode(
    mode: str,
    points_canonical: np.ndarray,
    part_ids: np.ndarray,
) -> np.ndarray:
    """Dispatch to the colouring for ``mode`` (one of :data:`MODES`)."""
    if mode == "geometry":
        return geometry_colors(part_ids)
    if mode == "nocs":
        return nocs_colors(points_canonical)
    if mode == "npcs":
        return npcs_colors(points_canonical, part_ids)
    raise ValueError(f"unknown colouring mode: {mode!r}")
