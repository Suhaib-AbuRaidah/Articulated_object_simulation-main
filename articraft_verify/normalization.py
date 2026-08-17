"""NOCS/NPCS normalization matching Wang et al. (CVPR 2019).

The tight axis-aligned bounding box is uniformly scaled so its diagonal has
length one, then translated so its center is at ``(0.5, 0.5, 0.5)``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def factor_and_corners(points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return diagonal scale and tight bounding-box corners for ``points``."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError("NOCS normalization needs a non-empty (N, 3) point set")

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    # Keep the verifier usable for point- or line-degenerate preview geometry.
    factor = 1.0 / diagonal if diagonal > 1e-12 else 1.0
    return factor, bbox_min, bbox_max


def normalize(
    points: np.ndarray,
    factor: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    """Map points to centered, unit-bounding-box-diagonal NOCS coordinates."""
    pts = np.asarray(points, dtype=float)
    bbox_center = 0.5 * (
        np.asarray(bbox_min, dtype=float) + np.asarray(bbox_max, dtype=float)
    )
    return 0.5 + float(factor) * (pts - bbox_center)


def normalize_from_points(points: np.ndarray) -> np.ndarray:
    """Fit the paper-compatible normalization to ``points`` and apply it."""
    factor, bbox_min, bbox_max = factor_and_corners(points)
    return normalize(points, factor, bbox_min, bbox_max)


def npcs_to_nocs_similarity(
    part_factor: float,
    part_bbox_min: np.ndarray,
    part_bbox_max: np.ndarray,
    object_factor: float,
    object_bbox_min: np.ndarray,
    object_bbox_max: np.ndarray,
    rotation: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Return ``(scale, translation)`` for ``NOCS = scale * R @ NPCS + t``."""
    part_center = 0.5 * (
        np.asarray(part_bbox_min, dtype=float)
        + np.asarray(part_bbox_max, dtype=float)
    )
    object_center = 0.5 * (
        np.asarray(object_bbox_min, dtype=float)
        + np.asarray(object_bbox_max, dtype=float)
    )
    rotation = np.asarray(rotation, dtype=float)
    scale = float(object_factor / part_factor)
    cube_center = np.full(3, 0.5, dtype=float)
    translation = (
        cube_center
        - scale * (rotation @ cube_center)
        + float(object_factor) * (rotation @ part_center - object_center)
    )
    return scale, translation
