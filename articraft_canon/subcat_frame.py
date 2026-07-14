"""Stage 4 -- SUB-CATEGORY FRAME CONSISTENCY.

After Stage 3 every object has its *own* PCA-derived root frame.  PCA sign/axis
choices are locally sensible but not consistent across a sub-category (two lamps
can end up mirrored or rolled by 90 degrees).  This stage picks one reference
object per sub-category and rigidly re-aligns everyone else to it.

Pipeline
--------
1. Represent each object by its Stage-3 canonical, unit-scaled rest cloud.
2. Pick a **reference** = the medoid (object minimising summed pairwise Chamfer
   distance), unless the caller overrides it with an exemplar id.
3. For every object, find the rigid transform (:func:`geometry.umeyama`, no
   scale -- articulation and scale are already fixed) that best maps its cloud
   onto the reference.  Correspondences come from nearest-neighbour matching.
   For **symmetric** objects we try each symmetry-group element and keep the one
   with the smallest residual.
4. Fold that rigid transform into the object's root rotation so the whole
   sub-category shares a single canonical frame.

Because clouds are unordered, we obtain correspondences by nearest-neighbour
matching (ICP-style) and iterate a couple of times; the objects are already
coarsely aligned by Stage 3, so this converges quickly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from . import geometry as geo
from .root_frame import RootFrame

logger = logging.getLogger(__name__)


@dataclass
class Alignment:
    """Result of aligning one object's root frame to the reference."""

    object_id: str
    is_reference: bool
    transform: np.ndarray              # (4,4) rigid, applied in canonical frame
    symmetry_index: int                # which symmetry element was chosen
    residual: float                    # RMS point residual after alignment
    predicted_chamfer: float           # Chamfer to reference after alignment


# --------------------------------------------------------------------------- #
# Correspondence + rigid fit (ICP-style, few iterations)
# --------------------------------------------------------------------------- #
def _icp_rigid(
    src: np.ndarray, dst: np.ndarray, iters: int = 6
) -> tuple[np.ndarray, float]:
    """Rigid transform mapping ``src`` onto ``dst`` via nearest-neighbour ICP.

    Returns ``(T, rms)``.  Both clouds are assumed pre-centred and unit-scaled,
    and coarsely aligned by Stage 3, so a handful of iterations suffice.
    """
    tree = cKDTree(dst)
    T_total = np.eye(4)
    moving = src.copy()
    rms = float("inf")
    for _ in range(iters):
        _, idx = tree.query(moving)
        T_step, rms = geo.umeyama(moving, dst[idx], with_scale=False)
        moving = geo.transform_points(T_step, moving)
        T_total = T_step @ T_total
    return T_total, rms


# --------------------------------------------------------------------------- #
# Reference selection
# --------------------------------------------------------------------------- #
def _pairwise_chamfer(clouds: List[np.ndarray], subsample: int) -> np.ndarray:
    """Symmetric pairwise Chamfer matrix over a list of clouds."""
    reduced = [geo.farthest_point_subsample(c, subsample, seed=3) for c in clouds]
    n = len(reduced)
    M = np.zeros((n, n))
    for i in range(n):
        for k in range(i + 1, n):
            d = geo.chamfer_distance(reduced[i], reduced[k])
            M[i, k] = M[k, i] = d
    return M


def choose_reference(
    object_ids: List[str],
    clouds: List[np.ndarray],
    *,
    exemplar: Optional[str] = None,
    subsample: int = 512,
) -> int:
    """Index of the reference object: exemplar override, else Chamfer medoid."""
    if exemplar is not None and exemplar in object_ids:
        return object_ids.index(exemplar)
    if len(clouds) == 1:
        return 0
    M = _pairwise_chamfer(clouds, subsample)
    return int(np.argmin(M.sum(axis=1)))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def align_sub_category(
    object_ids: List[str],
    frames: List[RootFrame],
    *,
    exemplar: Optional[str] = None,
    align_subsample: int = 1024,
) -> tuple[int, List[Alignment]]:
    """Align every object's root frame to a sub-category reference (Stage 4).

    Returns ``(reference_index, alignments)``.  Each :class:`Alignment` carries
    a rigid ``transform`` to be folded into that object's root rotation.
    """
    clouds = [f.points_canonical for f in frames]
    ref_idx = choose_reference(object_ids, clouds, exemplar=exemplar)
    ref_cloud = geo.farthest_point_subsample(clouds[ref_idx], align_subsample, seed=4)

    alignments: List[Alignment] = []
    for i, (oid, frame) in enumerate(zip(object_ids, frames)):
        if i == ref_idx:
            alignments.append(
                Alignment(oid, True, np.eye(4), 0, 0.0, 0.0)
            )
            continue

        src = geo.farthest_point_subsample(clouds[i], align_subsample, seed=5)

        # Try every symmetry element; keep the lowest-residual alignment.
        best: Optional[Alignment] = None
        best_T = np.eye(4)
        group = frame.symmetry_group or [np.eye(3)]
        for s_idx, S in enumerate(group):
            S4 = np.eye(4)
            S4[:3, :3] = S
            src_s = geo.transform_points(S4, src)
            T, rms = _icp_rigid(src_s, ref_cloud)
            T_full = T @ S4  # apply symmetry then rigid alignment
            aligned = geo.transform_points(T_full, src)
            cham = geo.chamfer_distance(aligned, ref_cloud)
            if best is None or cham < best.predicted_chamfer:
                best = Alignment(oid, False, T_full, s_idx, rms, cham)
                best_T = T_full
        alignments.append(best)
        logger.debug("aligned %s: sym=%d rms=%.4f chamfer=%.4f",
                     oid, best.symmetry_index, best.residual, best.predicted_chamfer)

    return ref_idx, alignments


def fold_into_root(frame: RootFrame, alignment: Alignment) -> RootFrame:
    """Compose the sub-category alignment rotation into an object's root frame.

    The alignment is a rigid transform in the canonical frame; its rotation is
    pre-multiplied into ``frame.rotation`` so the object's canonical orientation
    now matches the shared sub-category frame.  (The small residual translation
    is absorbed by re-centring and is not baked into the rotation.)
    """
    R_align = alignment.transform[:3, :3]
    new_rotation = R_align @ frame.rotation
    new_points = geo.transform_points(alignment.transform, frame.points_canonical)
    # Rotate the symmetry group into the new frame: R_align S R_align^T.
    new_group = [R_align @ S @ R_align.T for S in frame.symmetry_group]
    return RootFrame(
        centroid=frame.centroid,
        rotation=new_rotation,
        scale=frame.scale,
        symmetry_group=new_group,
        points_canonical=new_points,
    )
