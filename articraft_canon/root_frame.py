"""Stage 3 -- PER-OBJECT ROOT FRAME.

With articulation fixed at the canonical extended rest pose (Stage 2), we give
each object a repeatable root frame and scale:

  * **orientation**: PCA of the rest point cloud yields three principal axes.
    We disambiguate them into an ``up``/``front``/``right`` triad by scoring the
    six signed axis candidates (``±e0, ±e1, ±e2``) against world references
    (``+Z`` for up, ``+X`` for front), then complete a right-handed frame.
  * **symmetry**: a discrete symmetry group is detected by self-Chamfer -- we
    apply a menu of candidate rotations and keep those that map the object onto
    itself within a scale-relative tolerance.
  * **scale**: normalised so the bounding-box diagonal becomes 1.

The frame is returned as an object->canonical rotation ``R`` (so that
``R @ up = +Z`` etc.), a centroid, a scale, and the symmetry group (rotation
matrices expressed in the *canonical* frame).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import geometry as geo
from .parse import ObjectModel

logger = logging.getLogger(__name__)

_WORLD_UP = np.array([0.0, 0.0, 1.0])
_WORLD_FRONT = np.array([1.0, 0.0, 0.0])


@dataclass
class RootFrame:
    """Canonical root frame for one object (all in the object base frame)."""

    centroid: np.ndarray               # (3,) rest-cloud centroid
    rotation: np.ndarray               # (3,3) object -> canonical orientation
    scale: float                       # multiply to reach unit bbox diagonal
    symmetry_group: List[np.ndarray] = field(default_factory=list)  # in canonical frame
    points_canonical: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def order(self) -> int:
        return max(1, len(self.symmetry_group))


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #
def _orient_from_pca(axes: np.ndarray) -> np.ndarray:
    """Turn 3 PCA axes into an object->canonical rotation matrix.

    Chooses, among the six signed axis candidates, the one closest to world
    ``+Z`` as ``up`` and the closest remaining to world ``+X`` as ``front``,
    then builds a right-handed orthonormal frame.  The returned ``R`` maps the
    object's (up, front, right) onto (Z, X, Y) so canonical +Z is "up".
    """
    candidates = []
    for k in range(3):
        candidates.append(axes[:, k])
        candidates.append(-axes[:, k])

    # up = candidate most aligned with world up.
    up = max(candidates, key=lambda c: float(np.dot(c, _WORLD_UP)))

    # front = remaining candidate (not (anti)parallel to up) most aligned +X.
    def not_parallel(c):
        return abs(float(np.dot(c, up))) < 0.9

    front = max(
        (c for c in candidates if not_parallel(c)),
        key=lambda c: float(np.dot(c, _WORLD_FRONT)),
    )
    # Orthonormalise front against up, then right = up x front.
    front = front - np.dot(front, up) * up
    front /= np.linalg.norm(front) + 1e-15
    right = np.cross(up, front)

    # Rows map object-frame vectors onto canonical axes: X<-front, Y<-right, Z<-up.
    R = np.vstack([front, right, up])
    # Guard against numerical left-handedness.
    if np.linalg.det(R) < 0:
        R[1] = -R[1]
    return R


# --------------------------------------------------------------------------- #
# Symmetry detection
# --------------------------------------------------------------------------- #
def _candidate_symmetries() -> List[np.ndarray]:
    """Discrete rotation menu tested for self-symmetry (in the canonical frame).

    Covers 2/3/4-fold rotations about each canonical axis -- enough for the
    box-, cylinder- and yoke-like parts in this dataset.  Identity is excluded
    (it is trivially a symmetry and is added back by the caller).
    """
    rots: List[np.ndarray] = []
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        for fold in (2, 3, 4):
            for k in range(1, fold):
                angle = 2.0 * np.pi * k / fold
                rots.append(geo.rotation_about_axis(axis, angle)[:3, :3])
    return rots


def _detect_symmetry(
    points_canonical: np.ndarray, tol_ratio: float, max_points: int
) -> List[np.ndarray]:
    """Return the discrete symmetry group (incl. identity) of the cloud.

    A candidate rotation R is a symmetry if the Chamfer distance between the
    cloud and ``R @ cloud`` is below ``tol_ratio`` times the bbox diagonal.
    """
    group = [np.eye(3)]
    if len(points_canonical) < 8:
        return group
    cloud = geo.farthest_point_subsample(points_canonical, max_points, seed=1)
    diag = geo.bbox_diagonal(cloud) + 1e-12
    tol = tol_ratio * diag
    seen = [np.eye(3)]
    for R in _candidate_symmetries():
        rotated = cloud @ R.T
        if geo.chamfer_distance(cloud, rotated) <= tol:
            if not any(np.allclose(R, s, atol=1e-6) for s in seen):
                group.append(R)
                seen.append(R)
    return group


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def compute_root_frame(
    model: ObjectModel,
    *,
    n_points: int = 4096,
    symmetry_tol_ratio: float = 0.02,
    symmetry_max_points: int = 1024,
) -> RootFrame:
    """Run Stage 3 on ``model`` (assumes Stage 2 already applied)."""
    cloud = model.object_points()  # canonical rest pose (q = 0)
    if cloud.shape[0] == 0:
        logger.warning("empty cloud for %s; using identity root frame", model.object_id)
        return RootFrame(np.zeros(3), np.eye(3), 1.0, [np.eye(3)])

    cloud = geo.farthest_point_subsample(cloud, n_points, seed=2)
    centroid, axes, _ = geo.pca_axes(cloud)
    R = _orient_from_pca(axes)

    # Express the cloud in the canonical, centroid-origin frame.
    canon = (cloud - centroid) @ R.T
    scale = 1.0 / (geo.bbox_diagonal(canon) + 1e-12)

    symmetry = _detect_symmetry(canon * scale, symmetry_tol_ratio, symmetry_max_points)

    return RootFrame(
        centroid=centroid,
        rotation=R,
        scale=scale,
        symmetry_group=symmetry,
        points_canonical=canon * scale,
    )
