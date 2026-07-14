"""Shared analytic geometry helpers for the canonicalization pipeline.

Everything in here is deliberately dependency-light (numpy + scipy only) and
side-effect free, so each stage can be reasoned about and unit-tested in
isolation.  The helpers fall into four groups:

  * SE(3) construction / manipulation (rotations, translations, axis-angle);
  * forward kinematics for a serial chain;
  * point-cloud statistics (PCA, bounding box, unit-cube normalisation);
  * point-cloud comparison (symmetric Chamfer distance, Umeyama alignment).

Conventions
-----------
* Transforms are 4x4 homogeneous matrices, column-vector convention:
  ``p_parent = T @ p_child`` where ``p`` is ``[x, y, z, 1]``.
* Directions are length-3 unit vectors unless noted.
* Angles are radians.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# SE(3) primitives
# --------------------------------------------------------------------------- #
def identity() -> np.ndarray:
    """Return a fresh 4x4 identity transform."""
    return np.eye(4, dtype=float)


def translation(t: Sequence[float]) -> np.ndarray:
    """Homogeneous transform for a pure translation ``t`` (length 3)."""
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def rotation_about_axis(axis: Sequence[float], angle: float) -> np.ndarray:
    """Homogeneous rotation of ``angle`` radians about a (not-necessarily unit)
    ``axis`` through the origin, via Rodrigues' formula."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(4, dtype=float)
    a = a / n
    x, y, z = a
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    return T


def prismatic_offset(axis: Sequence[float], q: float) -> np.ndarray:
    """Homogeneous transform for sliding distance ``q`` along a unit ``axis``."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(4, dtype=float)
    return translation((a / n) * q)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an ``(N, 3)`` array of points."""
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


def rotate_vectors(T: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Apply only the rotation part of ``T`` to ``(N, 3)`` (or ``(3,)``) vectors."""
    return np.asarray(vecs, dtype=float) @ T[:3, :3].T


def signed_angle_about_axis(
    source: np.ndarray, target: np.ndarray, axis: np.ndarray
) -> float:
    """Signed angle (radians) that rotates ``source`` toward ``target`` about
    ``axis``, measuring only the components perpendicular to ``axis``.

    This is the closed-form solution to "spin ``source`` about ``axis`` until it
    is as parallel as possible to ``target``".  Components parallel to the axis
    are unreachable by an axis rotation and are therefore ignored.
    """
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-15)
    s = np.asarray(source, dtype=float)
    t = np.asarray(target, dtype=float)
    s_perp = s - np.dot(s, a) * a
    t_perp = t - np.dot(t, a) * a
    if np.linalg.norm(s_perp) < 1e-9 or np.linalg.norm(t_perp) < 1e-9:
        # One of the directions is (anti)parallel to the axis: nothing to align.
        return 0.0
    cross = np.cross(s_perp, t_perp)
    sin_term = np.dot(cross, a)
    cos_term = np.dot(s_perp, t_perp)
    return float(np.arctan2(sin_term, cos_term))


# --------------------------------------------------------------------------- #
# Forward kinematics for a serial chain
# --------------------------------------------------------------------------- #
def joint_motion(joint_type: str, axis: np.ndarray, q: float) -> np.ndarray:
    """Local motion transform contributed by a joint at value ``q``."""
    if joint_type == "revolute" or joint_type == "continuous":
        return rotation_about_axis(axis, q)
    if joint_type == "prismatic":
        return prismatic_offset(axis, q)
    return np.eye(4, dtype=float)  # fixed / unsupported -> no motion


def link_world_transforms(
    base_link: str,
    joints: Sequence["object"],
    cfg: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """Compute world transforms of every link for a serial chain.

    ``joints`` must be ordered base-outward and expose ``.parent``, ``.child``,
    ``.origin`` (4x4 at q=0), ``.type`` and ``.axis``.  ``cfg`` maps joint name
    to value (missing joints default to 0).
    """
    transforms: Dict[str, np.ndarray] = {base_link: np.eye(4, dtype=float)}
    for j in joints:
        parent_T = transforms[j.parent]
        q = float(cfg.get(j.name, 0.0))
        local = j.origin @ joint_motion(j.type, j.axis, q)
        transforms[j.child] = parent_T @ local
    return transforms


# --------------------------------------------------------------------------- #
# Point-cloud statistics
# --------------------------------------------------------------------------- #
def pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(centroid, eigvecs, eigvals)`` of the point covariance.

    ``eigvecs`` columns are unit axes sorted by *descending* eigenvalue, so
    column 0 is the long axis.  ``eigvals`` are the matching variances.
    """
    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered / max(len(pts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(eigvals)[::-1]
    return centroid, eigvecs[:, order], eigvals[order]


def bbox_diagonal(points: np.ndarray) -> float:
    """Length of the axis-aligned bounding-box diagonal of ``points``."""
    pts = np.asarray(points, dtype=float)
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def unit_cube_normalization(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(bbox_min, bbox_max, factor)`` mapping ``points`` into the unit
    cube ``[0, 1]^3`` via ``(p - bbox_min) * factor`` where
    ``factor = 1 / max_extent``.  A single isotropic factor preserves shape.
    """
    pts = np.asarray(points, dtype=float)
    bmin = pts.min(axis=0)
    bmax = pts.max(axis=0)
    extent = float(np.max(bmax - bmin))
    factor = 1.0 / extent if extent > 1e-12 else 1.0
    return bmin, bmax, factor


# --------------------------------------------------------------------------- #
# Point-cloud comparison
# --------------------------------------------------------------------------- #
def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean Chamfer distance between two point clouds.

    ``mean_{x in a} min_{y in b} ||x - y|| + mean_{y in b} min_{x in a} ||y - x||``
    computed with KD-trees.  Symmetric and scale-sensitive; callers usually feed
    scale-normalised clouds so the value is comparable across objects.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    d_ab, _ = tree_b.query(a)
    d_ba, _ = tree_a.query(b)
    return float(d_ab.mean() + d_ba.mean())


def umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, float]:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Implements Umeyama (1991) for *ordered* corresponding point sets.  Returns
    ``(T, rms_residual)`` where ``T`` is 4x4 (rotation, optional uniform scale,
    translation) such that ``T @ src ~= dst``.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = dst_c.T @ src_c / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:  # reflection guard
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / n
        scale = float(np.trace(np.diag(D) @ S) / (var_src + 1e-15))
    else:
        scale = 1.0
    t = mu_dst - scale * R @ mu_src
    T = np.eye(4, dtype=float)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    aligned = transform_points(T, src)
    rms = float(np.sqrt(((aligned - dst) ** 2).sum(axis=1).mean()))
    return T, rms


def farthest_point_subsample(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Uniform random subsample of ``k`` points (cheap stand-in for FPS).

    Chamfer/medoid computations are quadratic in the number of objects, so we
    down-sample clouds to keep them tractable.  A plain random draw is adequate
    because Chamfer already averages over the whole cloud.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) <= k:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=k, replace=False)
    return pts[idx]
