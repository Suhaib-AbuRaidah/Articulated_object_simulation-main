"""Stage 5 -- OPTION B FRAMES + NOCS / NPCS.

Produces the per-object supervision sidecar.  Two decisions define "Option B":

  * **Link frames share the object's canonical orientation.**  At ``q = 0`` every
    part frame has identity rotation *relative to the object frame* -- we do NOT
    rotate part frames to follow their joint axes.  A part is therefore fully
    described by its origin (a translation) plus explicit joint parameters.
  * **Joint geometry is stored as explicit parameters** in the canonical frame:
    axis, type, limits and pivot point.

Normalised coordinates:

  * **NOCS** -- Normalised Object Coordinate Space: the whole object at rest,
    mapped into the unit cube ``[0,1]^3`` by a single isotropic factor.
  * **NPCS** -- Normalised Part Coordinate Space: each part mapped into its own
    unit cube.
  * **NPCS -> NOCS** -- because parts share the object orientation (Option B),
    the map from a part's unit cube to the object's unit cube is a pure
    ``scale * x + translation`` (no rotation), stored per part.

The canonical transform mapping a base-frame point ``p`` into canonical object
coordinates is::

    c = scale * R_root @ (p - centroid) + align_translation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from . import geometry as geo
from .parse import JointSpec, ObjectModel
from .root_frame import RootFrame

logger = logging.getLogger(__name__)


def bake_base_aligned_link_frames(model: ObjectModel) -> None:
    """Re-orient every link frame in ``model.urdf`` to match the base frame.

    After Stage 2 each link frame carries the accumulated joint rotations, so in
    the emitted URDF the link frames are *not* axis-aligned with the base.  This
    rewrites the URDF so that at ``q = 0`` every link frame has the **base's
    orientation**, merely translated to the link's origin -- the "Option B"
    convention, made visible in the URDF itself (and in any viewer).

    The rewrite is kinematically and geometrically identical to the input; it
    only changes how each link's frame is *parameterised*:

      * each joint origin becomes a **pure translation** ``p_child - p_parent``;
      * each joint axis is expressed in the new (base-aligned) child frame,
        ``axis <- R_child @ axis``, so the physical joint axis is unchanged;
      * each link's visual/collision origins are pre-rotated by ``R_child`` so the
        geometry stays in the same world pose;
      * each link's **inertial origin is neutralised to identity**.  PyBullet
        frames every link by its inertial (CoM) frame, so a rotated inertial
        origin would place the geometry relative to a *different* frame than the
        base-aligned link frame and the parts would visibly detach at their
        joints under motion.  This pipeline is purely kinematic (fixed base, zero
        gravity, direct joint resets), so the inertial pose is unused and safe to
        zero; doing so makes PyBullet's link frame coincide with the base-aligned
        URDF link frame.

    ``R_child`` / ``p_child`` are the link's world rotation/position at ``q = 0``
    computed from the (already baked) joint origins.  Call once, at write time.
    """
    urdf = model.urdf
    ujoints = {j.name: j for j in urdf.robot.joints}
    ulinks = {l.name: l for l in urdf.robot.links}

    # World transforms at q=0 from the current (baked) origins, base -> tip.
    world: Dict[str, np.ndarray] = {model.base_link: np.eye(4)}
    for j in model.joints:
        uj = ujoints[j.name]
        origin = np.asarray(uj.origin, dtype=float) if uj.origin is not None else np.eye(4)
        world[j.child] = world[j.parent] @ origin
    R = {ln: world[ln][:3, :3].copy() for ln in world}
    p = {ln: world[ln][:3, 3].copy() for ln in world}

    # Rewrite joints: pure-translation origins, axes expressed in the new frame.
    for j in model.joints:
        uj = ujoints[j.name]
        new_origin = np.eye(4)
        new_origin[:3, 3] = p[j.child] - p[j.parent]
        uj.origin = new_origin
        if uj.axis is not None and j.child in R:
            uj.axis = R[j.child] @ np.asarray(uj.axis, dtype=float)

    # Rewrite each link's geometry origins so world placement is preserved, and
    # neutralise its inertial origin so PyBullet frames the link by the URDF
    # (base-aligned) frame rather than a rotated CoM frame.
    for ln in model.chain:
        link = ulinks.get(ln)
        if link is None:
            continue

        # Neutralise the inertial pose for *every* chain link (unused kinematically;
        # a non-identity inertial frame would misplace the geometry in PyBullet).
        inertial = getattr(link, "inertial", None)
        if inertial is not None and getattr(inertial, "origin", None) is not None:
            inertial.origin = np.eye(4)

        Rc = R.get(ln)
        if Rc is None or np.allclose(Rc, np.eye(3), atol=1e-9):
            continue  # base link (and any already-aligned link) needs no rotation
        Rc4 = np.eye(4)
        Rc4[:3, :3] = Rc
        for group in ("visuals", "collisions"):
            for item in getattr(link, group, None) or []:
                V = np.asarray(item.origin, dtype=float) if item.origin is not None else np.eye(4)
                item.origin = Rc4 @ V


@dataclass
class CanonicalTransform:
    """Base-frame -> canonical-object-frame similarity transform."""

    centroid: np.ndarray       # (3,)
    rotation: np.ndarray       # (3,3) R_root (post sub-category alignment)
    scale: float               # isotropic
    translation: np.ndarray    # (3,) sub-category alignment translation

    def apply(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        return self.scale * (pts - self.centroid) @ self.rotation.T + self.translation

    def apply_direction(self, vec: np.ndarray) -> np.ndarray:
        v = self.rotation @ np.asarray(vec, dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v


def _canonical_transform(frame: RootFrame, align_translation: np.ndarray) -> CanonicalTransform:
    return CanonicalTransform(
        centroid=frame.centroid,
        rotation=frame.rotation,
        scale=frame.scale,
        translation=np.asarray(align_translation, dtype=float),
    )


def build_sidecar(
    model: ObjectModel,
    frame: RootFrame,
    align_translation: np.ndarray,
) -> Dict:
    """Assemble the full NOCS/NPCS + frame sidecar dictionary for ``model``.

    Assumes Stages 2-4 have been applied (model is at its canonical rest pose and
    ``frame`` already carries the sub-category-aligned rotation/symmetry).
    """
    ct = _canonical_transform(frame, align_translation)
    world = geo.link_world_transforms(model.base_link, model.joints, {})  # q = 0

    # ------------------------------------------------------------------ #
    # Per-part canonical point clouds and the whole-object cloud.
    # ------------------------------------------------------------------ #
    part_points_canon: Dict[str, np.ndarray] = {}
    for name in model.chain:
        link = model.links.get(name)
        if link is None or link.points.size == 0 or name not in world:
            continue
        world_pts = geo.transform_points(world[name], link.points)
        part_points_canon[name] = ct.apply(world_pts)

    if not part_points_canon:
        obj_cloud = np.zeros((1, 3))
    else:
        obj_cloud = np.concatenate(list(part_points_canon.values()), axis=0)

    obj_bmin, obj_bmax, obj_factor = geo.unit_cube_normalization(obj_cloud)

    # ------------------------------------------------------------------ #
    # Map link -> its (single) inbound joint for per-part joint parameters.
    # ------------------------------------------------------------------ #
    inbound: Dict[str, JointSpec] = {j.child: j for j in model.joints}

    parts: List[Dict] = []
    for name in model.chain:
        if name not in part_points_canon:
            continue
        pmin, pmax, pfactor = geo.unit_cube_normalization(part_points_canon[name])

        # Option B: part frame origin in canonical coords, identity orientation.
        origin_world = world[name][:3, 3]
        origin_canon = ct.apply(origin_world[None, :])[0]

        entry: Dict = {
            "link": name,
            "frame_origin_canonical": origin_canon.tolist(),
            "frame_rotation_canonical": np.eye(3).tolist(),  # Option B: identity
            "npcs_bbox_min": pmin.tolist(),
            "npcs_bbox_max": pmax.tolist(),
            "npcs_factor": float(pfactor),
            # NPCS -> NOCS is a pure scale + translation (shared orientation).
            "npcs_to_nocs_scale": float(obj_factor / pfactor),
            "npcs_to_nocs_translation": ((pmin - obj_bmin) * obj_factor).tolist(),
        }

        j = inbound.get(name)
        if j is not None and j.is_moving:
            axis_world = geo.rotate_vectors(world[name], j.axis)
            pivot_world = world[name][:3, 3]  # joint sits at the child frame origin
            entry["joint"] = {
                "name": j.name,
                "type": j.type,
                "axis_canonical": ct.apply_direction(axis_world).tolist(),
                "pivot_canonical": ct.apply(pivot_world[None, :])[0].tolist(),
                "limit_lower": float(j.lower),
                "limit_upper": float(j.upper),
                # Provenance: the original URDF limits, and whether Stage 2 had to
                # widen this joint's range to admit the straight canonical pose.
                "limit_lower_original": float(j.orig_lower),
                "limit_upper_original": float(j.orig_upper),
                "limit_widened": bool(j.widened),
            }
        else:
            entry["joint"] = None  # base link or fixed attachment
        parts.append(entry)

    sidecar = {
        "object_id": model.object_id,
        "category": model.category,
        "sub_category": model.sub_category,
        "split": model.split,
        "base_link": model.base_link,
        "chain": model.chain,
        "canonical_fallback": model.canonical_fallback,
        "limits_widened": model.canonical_widened,
        "canonical_transform": {
            "centroid": ct.centroid.tolist(),
            "rotation": ct.rotation.tolist(),
            "scale": float(ct.scale),
            "align_translation": ct.translation.tolist(),
            "note": "canonical = scale * R @ (p - centroid) + align_translation",
        },
        "symmetry_group_canonical": [S.tolist() for S in frame.symmetry_group],
        "symmetry_order": frame.order,
        "nocs": {
            "bbox_min": obj_bmin.tolist(),
            "bbox_max": obj_bmax.tolist(),
            "factor": float(obj_factor),
            "note": "nocs = (canonical - bbox_min) * factor  -> unit cube",
        },
        "parts": parts,
    }
    return sidecar
