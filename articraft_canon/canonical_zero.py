"""Stage 2 -- CANONICAL ZERO (fully extended).

Goal: redefine each object's ``q = 0`` state so that the kinematic chain is
*fully extended* (straight), giving every object in a sub-category the same
articulation baseline for NOCS/NPCS supervision.

Method (base-outward, closed form)
----------------------------------
For every link we define a **canonical direction** (unit vector in the link's
local frame):

  * the vector from the link's own frame origin to the *next* joint
    ("joint-to-next-joint"), if the link has a downstream joint;
  * otherwise the link mesh's PCA long axis (the tip link).

Walking base -> tip, for each joint we solve in closed form the value ``q*``
that makes the child link's canonical direction parallel to its parent's
canonical direction:

  * **revolute**: the signed angle about the joint axis that aligns the two
    directions in the plane perpendicular to the axis
    (:func:`geometry.signed_angle_about_axis`);
  * **prismatic**: pure translation cannot rotate a direction, so "extended"
    means travelling to the joint limit that maximises reach along the parent
    direction.

We then **bake** ``q*`` into the kinematics so the *new* zero is the extended
pose::

    new_origin = old_origin @ joint_motion(axis, q*)
    new_limits = [lower - q*, upper - q*]

If the required ``q*`` is unreachable (outside the joint limits) or the extended
pose self-collides, we fall back to the per-joint **mid-limit** state for the
whole object and flag it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import geometry as geo
from .parse import JointSpec, ObjectModel

logger = logging.getLogger(__name__)


# Per-category canonical "extension" axis (in the base frame).  Where a whole
# category is mounted along one consistent axis, the per-object mount vector
# (origin -> first joint) is an unreliable reference -- a small off-axis offset
# tilts the straightened chain (e.g. lamp _0008's arm ended up 44 deg off
# vertical).  Forcing the category axis makes every object's chain extend the
# same way.  Only categories with a strong single-axis consensus are listed;
# mixed-mount categories (cctv_mast, laptop_stand, wall_mounts) are omitted and
# fall back to their own mount vector.  Measured over the dataset:
#   +Z consensus 94-100%: lamp, telescope, cartesian, dish, robotic arm, spotlight
#   +X consensus 74%:     telescoping slide (mounted horizontally)
_CATEGORY_UP_AXIS = {
    "articulated_task_lamp": np.array([0.0, 0.0, 1.0]),
    "astronomical_telescope_on_tripod": np.array([0.0, 0.0, 1.0]),
    "cartesian_stage": np.array([0.0, 0.0, 1.0]),
    "parabolic_dish_on_azimuth_elevation_mount": np.array([0.0, 0.0, 1.0]),
    "robotic_arm": np.array([0.0, 0.0, 1.0]),
    "spotlight_and_ringlight": np.array([0.0, 0.0, 1.0]),
    "telescoping_slide": np.array([1.0, 0.0, 0.0]),
}


@dataclass
class CanonicalResult:
    """Outcome of Stage 2 for one object."""

    q_star: Dict[str, float]           # canonical value baked per joint
    fallback: bool                     # True if mid-limit fallback was used
    reason: str = ""                   # why the fallback happened (if any)
    collided: bool = False
    widened: bool = False              # True if joint limits were extended
    widened_joints: List[str] = None   # names of joints whose limits grew


# --------------------------------------------------------------------------- #
# Canonical directions
# --------------------------------------------------------------------------- #
def _link_canonical_direction(model: ObjectModel, link_name: str) -> np.ndarray:
    """Canonical direction for ``link_name`` in its own local frame.

    The direction represents the axis along which the link's body extends,
    pointing **outward** (toward the next link / away from the base):

      * **base (root) link** -- if its category has a forced canonical axis
        (:data:`_CATEGORY_UP_AXIS`), that axis; otherwise the vector to the first
        joint (``origin -> first joint``).  Forcing the category axis keeps every
        object in a category extending the same way, instead of tilting with each
        object's mount offset;
      * **interior link** -- the vector to the next joint (``joint-to-next-joint``),
        which runs along the body;
      * **leaf (tip) link** -- no outbound joint, so the mesh PCA long axis,
        sign-flipped to point away from the inbound joint (toward the geometry
        centroid).  Without this the last link often ends up *anti*-parallel to
        its parent (folded back on the chain).

    Returns a unit vector; defaults to +X if the link is degenerate.
    """
    # Base link with a forced category axis: use it as the chain reference.
    if link_name == model.base_link and model.category in _CATEGORY_UP_AXIS:
        axis = _CATEGORY_UP_AXIS[model.category].astype(float)
        return axis / np.linalg.norm(axis)

    # Interior/base link: the vector to the next joint runs along the chain.
    downstream = next((j for j in model.joints if j.parent == link_name), None)
    if downstream is not None:
        v = np.asarray(downstream.origin[:3, 3], dtype=float)
        if np.linalg.norm(v) > 1e-6:
            return v / np.linalg.norm(v)

    # Leaf link: PCA long axis, oriented away from the inbound joint (origin).
    link = model.links.get(link_name)
    if link is not None and link.points.shape[0] >= 3:
        centroid, axes, _ = geo.pca_axes(link.points)
        long_axis = axes[:, 0]
        if np.dot(long_axis, centroid) < 0:
            long_axis = -long_axis
        return long_axis
    return np.array([1.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# Solving q* base-outward
# --------------------------------------------------------------------------- #
def _solve_extended_config(
    model: ObjectModel, first_moving_is_reference: bool = False
) -> Dict[str, float]:
    """Compute the extended ``q*`` for every moving joint, base -> tip.

    "Fully extended" means the kinematic chain is *straight*: each link's
    canonical direction is made collinear with its parent's, **starting from the
    base link** -- the first moving link is aligned to the base's own direction
    (base-outward, as specified).

    ``first_moving_is_reference`` (default False) affects only the first moving
    joint.  When True it is left at ``q* = 0`` so the first moving link, not the
    base, seeds the chain -- historically used to avoid the mid-limit fallbacks
    that base alignment provokes under fixed joint limits.  With the
    ``extend-limits`` policy that reachability concern is handled by widening
    limits, so the default now aligns from the base.
    """
    directions = {ln: _link_canonical_direction(model, ln) for ln in model.chain}

    # World transforms accumulated with the q* chosen so far (start all zero).
    world: Dict[str, np.ndarray] = {model.base_link: geo.identity()}
    q_star: Dict[str, float] = {}
    anchored = False  # have we passed the reference (first moving) joint yet?

    for j in model.joints:
        parent_world = world[j.parent]
        origin_world = parent_world @ j.origin  # child frame @ q=0, in base coords

        # Parent's canonical direction, in base coords.
        parent_dir = geo.rotate_vectors(parent_world, directions[j.parent])
        # Child's canonical direction @ q=0, in base coords (before this joint moves).
        child_dir0 = geo.rotate_vectors(origin_world, directions[j.child])

        # Leave the first moving joint as the chain reference (q* = 0).
        if j.is_moving and first_moving_is_reference and not anchored:
            anchored = True
            q_star[j.name] = 0.0
            world[j.child] = origin_world
            continue

        if j.type in ("revolute", "continuous"):
            axis_world = geo.rotate_vectors(origin_world, j.axis)
            q = geo.signed_angle_about_axis(child_dir0, parent_dir, axis_world)
            local = geo.rotation_about_axis(j.axis, q)
        elif j.type == "prismatic":
            axis_world = geo.rotate_vectors(origin_world, j.axis)
            # Extend along whichever limit increases reach along the parent dir.
            sign = np.dot(axis_world, parent_dir)
            q = j.upper if sign >= 0 else j.lower
            local = geo.prismatic_offset(j.axis, q)
        else:  # fixed
            q = 0.0
            local = geo.identity()

        q_star[j.name] = float(q)
        world[j.child] = origin_world @ local

    return q_star


# --------------------------------------------------------------------------- #
# Self-collision check
# --------------------------------------------------------------------------- #
def _self_collides_aabb(model: ObjectModel, cfg: Dict[str, float]) -> bool:
    """Cheap non-adjacent link collision test via AABB overlap.

    Fallback when python-fcl is unavailable.  Conservative and dependency-free,
    but false-positives on concentric/nested parts -- do not gate limit-widening
    on this.
    """
    world = geo.link_world_transforms(model.base_link, model.joints, cfg)
    boxes = []
    for name in model.chain:
        link = model.links.get(name)
        if link is None or link.points.size == 0 or name not in world:
            boxes.append(None)
            continue
        pts = geo.transform_points(world[name], link.points)
        boxes.append((pts.min(axis=0), pts.max(axis=0)))

    def overlap(a, b) -> bool:
        (amin, amax), (bmin, bmax) = a, b
        pad = 0.02  # so faces that merely kiss do not count as a collision
        return bool(np.all(amin < bmax - pad) and np.all(bmin < amax - pad))

    for i in range(len(boxes)):
        for k in range(i + 2, len(boxes)):  # skip self and immediate neighbour
            if boxes[i] and boxes[k] and overlap(boxes[i], boxes[k]):
                return True
    return False


def _self_collides_fcl(model: ObjectModel, cfg: Dict[str, float]) -> Optional[bool]:
    """Accurate non-adjacent mesh--mesh self-collision test via python-fcl.

    Returns True/False, or None if fcl (or link meshes) are unavailable so the
    caller can fall back to the AABB test.  Immediate chain neighbours are
    excluded because they legitimately meet at their shared joint.
    """
    try:
        from trimesh.collision import CollisionManager
        import fcl  # noqa: F401 - presence check; CollisionManager needs it
    except Exception:  # noqa: BLE001
        return None

    world = geo.link_world_transforms(model.base_link, model.joints, cfg)
    placed: List[str] = []
    managers: Dict[str, "CollisionManager"] = {}
    for name in model.chain:
        link = model.links.get(name)
        if link is None or link.mesh is None or name not in world:
            placed.append(None)
            continue
        mgr = CollisionManager()
        mesh = link.mesh.copy()
        mesh.apply_transform(world[name])
        mgr.add_object(name, mesh)
        managers[name] = mgr
        placed.append(name)

    names = [n for n in placed if n is not None]
    index = {n: i for i, n in enumerate(placed)}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            if abs(index[na] - index[nb]) < 2:  # adjacent in the chain
                continue
            if managers[na].in_collision_other(managers[nb]):
                return True
    return False


def _self_collides(model: ObjectModel, cfg: Dict[str, float]) -> bool:
    """Non-adjacent self-collision test at ``cfg`` (fcl if available, else AABB)."""
    result = _self_collides_fcl(model, cfg)
    if result is None:
        logger.warning("python-fcl unavailable; using coarse AABB collision test "
                       "for %s (may false-positive on nested parts)", model.object_id)
        return _self_collides_aabb(model, cfg)
    return result


# --------------------------------------------------------------------------- #
# Baking
# --------------------------------------------------------------------------- #
def _bake(model: ObjectModel, q: Dict[str, float], widen_to_zero: bool = False) -> List[str]:
    """Fold ``q`` into joint origins/limits so the new q=0 is that pose.

    Mutates both the in-memory :class:`JointSpec`s and the underlying yourdfpy
    joint objects (so a subsequent ``write_xml_file`` emits the canonical URDF).

    With ``widen_to_zero`` the (shifted) limits are extended just enough to admit
    ``q = 0`` -- used by the ``extend-limits`` policy so the straight canonical
    pose is a valid, reachable configuration.  Returns the list of joint names
    whose limits were actually widened.
    """
    urdf_joints = {jj.name: jj for jj in model.urdf.robot.joints}
    widened: List[str] = []
    for j in model.joints:
        qv = float(q.get(j.name, 0.0))
        if not j.is_moving:
            continue

        motion = geo.joint_motion(j.type, j.axis, qv)
        new_origin = j.origin @ motion
        new_lower = j.lower - qv
        new_upper = j.upper - qv

        if widen_to_zero:
            widened_lower = min(new_lower, 0.0)
            widened_upper = max(new_upper, 0.0)
            if widened_lower < new_lower - 1e-9 or widened_upper > new_upper + 1e-9:
                widened.append(j.name)
                j.widened = True
            new_lower, new_upper = widened_lower, widened_upper

        if abs(qv) < 1e-12 and not j.widened:
            continue

        # Update in-memory spec.
        j.origin = new_origin
        j.lower = new_lower
        j.upper = new_upper

        # Update the yourdfpy joint that will be serialised.
        uj = urdf_joints[j.name]
        uj.origin = new_origin
        if uj.limit is not None:
            if uj.limit.lower is not None:
                uj.limit.lower = float(new_lower)
            if uj.limit.upper is not None:
                uj.limit.upper = float(new_upper)
    return widened


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def canonicalize_zero(
    model: ObjectModel,
    *,
    check_collision: bool = True,
    first_moving_is_reference: bool = False,
    unreachable_policy: str = "mid-limit",
    apply: bool = True,
) -> CanonicalResult:
    """Run Stage 2 on ``model``.

    With ``apply=True`` the canonical q* is baked into the model (and its
    yourdfpy handle).  With ``apply=False`` nothing is mutated -- useful for
    dry-run reporting.  See :func:`_solve_extended_config` for
    ``first_moving_is_reference``.

    ``unreachable_policy`` controls what happens when the extended ``q*`` lands
    outside a joint's limits:

      * ``"mid-limit"`` (spec default): abandon the extended pose and fall back
        to the per-joint mid-limit state for the whole object;
      * ``"clamp"``: clamp each offending joint to its nearest limit (the most
        extended *reachable* pose) and keep the rest of the solution;
      * ``"extend-limits"``: **widen** the offending joint limits just enough to
        admit the straight pose -- but only if that pose is self-collision-free
        (checked with real mesh collision, mandatory for this policy); otherwise
        fall back to mid-limit.  Original limits are preserved for provenance.

    For the non-extend policies a self-collision in the extended pose falls back
    to mid-limit only when ``check_collision`` is enabled.
    """
    if not model.moving_joints:
        result = CanonicalResult(q_star={}, fallback=False, reason="no moving joints",
                                 widened_joints=[])
        model.canonical_q, model.canonical_fallback = {}, False
        model.canonical_widened = False
        return result

    q_star = _solve_extended_config(model, first_moving_is_reference)

    # Reachability: every extended q* must sit inside its joint limits.
    unreachable = [
        j.name for j in model.moving_joints
        if not (j.lower - 1e-6 <= q_star[j.name] <= j.upper + 1e-6)
    ]

    fallback = False
    collided = False
    widen = False
    reason = ""

    if unreachable and unreachable_policy == "clamp":
        limits = {j.name: (j.lower, j.upper) for j in model.moving_joints}
        for name in unreachable:
            lo, hi = limits[name]
            q_star[name] = float(np.clip(q_star[name], lo, hi))
        reason = f"clamped to limits (joints: {', '.join(unreachable)})"
        logger.info("CLAMP %s: %s", model.object_id, reason)

    elif unreachable and unreachable_policy == "extend-limits":
        # The straight pose becomes the canonical pose ONLY if it is collision
        # free.  This gate is mandatory (independent of --check-collision).
        if _self_collides(model, q_star):
            fallback, collided = True, True
            reason = ("straight pose self-collides; cannot widen (joints: "
                      f"{', '.join(unreachable)})")
        else:
            widen = True
            reason = f"widened limits to admit straight pose (joints: {', '.join(unreachable)})"
            logger.info("EXTEND %s: %s", model.object_id, reason)

    elif unreachable and unreachable_policy == "mid-limit":
        fallback = True
        reason = f"unreachable extended pose (joints: {', '.join(unreachable)})"

    # Optional self-collision fallback for the non-extend policies.
    if (not fallback and not widen and unreachable_policy != "extend-limits"
            and check_collision and _self_collides(model, q_star)):
        fallback, collided = True, True
        reason = "self-collision in extended pose"

    if fallback:
        # Per-joint mid-limit state for the whole object.
        q_star = {j.name: j.mid_limit for j in model.moving_joints}
        logger.info("FALLBACK %s -> mid-limit: %s", model.object_id, reason)

    widened_joints: List[str] = []
    if apply:
        widened_joints = _bake(model, q_star, widen_to_zero=widen)
        model.canonical_q = dict(q_star)
        model.canonical_fallback = fallback
        model.canonical_widened = bool(widened_joints)
    elif widen:
        widened_joints = list(unreachable)  # predicted, for dry-run reporting

    return CanonicalResult(q_star=q_star, fallback=fallback, reason=reason,
                           collided=collided, widened=widen,
                           widened_joints=widened_joints)
