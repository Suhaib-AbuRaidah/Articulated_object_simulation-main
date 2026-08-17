"""Stage 1 -- PARSE.

Load a single Articraft URDF and turn it into a plain, self-contained
:class:`ObjectModel`: the serial kinematic chain from the base outward, each
joint's type / axis / origin / limits, and a sampled point cloud per link in
that link's local frame.

Anything that is not a clean serial chain of single-DOF joints (branching,
multi-DOF, mimic couplings, unsupported joint types) is *logged and skipped* by
setting ``ObjectModel.skip_reason`` rather than raising -- the batch pipeline
relies on that to keep going. Callers may opt into fixed-only side branches;
the moving-joint skeleton must still be serial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh
import yourdfpy

from . import geometry as geo

logger = logging.getLogger(__name__)

# Joint types we can canonicalize.  Everything else -> skip.
# 'continuous' is a single-DOF revolute joint with no angle limit, so we treat
# it like 'revolute' (with a default +-pi range) rather than skipping it.
_MOVING_TYPES = {"revolute", "prismatic", "continuous"}
_SUPPORTED_TYPES = _MOVING_TYPES | {"fixed"}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class JointSpec:
    """A single URDF joint, normalised for the pipeline."""

    name: str
    type: str                      # 'revolute' | 'prismatic' | 'fixed'
    axis: np.ndarray               # (3,) unit axis in the child/joint frame @ q=0
    origin: np.ndarray             # (4,4) parent -> child frame @ q=0
    lower: float                   # current (working) limits; mutated by Stage 2
    upper: float
    parent: str
    child: str
    orig_lower: float = 0.0        # limits as parsed, for provenance
    orig_upper: float = 0.0
    widened: bool = False          # Stage 2 extended the range to admit q=0

    @property
    def is_moving(self) -> bool:
        return self.type in _MOVING_TYPES

    @property
    def mid_limit(self) -> float:
        return 0.5 * (self.lower + self.upper)


@dataclass
class LinkSpec:
    """A single URDF link plus a point cloud (and mesh) in its local frame."""

    name: str
    points: np.ndarray             # (N,3) surface samples in link-local coords
    has_geometry: bool = True
    mesh: Optional["trimesh.Trimesh"] = None  # combined visual mesh, for collision


@dataclass
class ObjectModel:
    """Everything downstream stages need about one object."""

    object_id: str                 # archive stem, e.g. rec_..._478a...
    source_archive: Path           # original .tar.gz (never modified)
    urdf_path: Path                # extracted model.urdf (working copy)
    category: str                  # top-level folder, e.g. robotic_arm
    sub_category: str              # leaf folder, e.g. serial_elbow_arm
    split: str                     # train | val | test
    base_link: str
    chain: List[str]               # link names in parent-before-child order
    joints: List[JointSpec]        # joints in parent-before-child order
    links: Dict[str, LinkSpec]
    urdf: yourdfpy.URDF            # live handle, used to write the canonical URDF
    skip_reason: Optional[str] = None
    fixed_branches_allowed: bool = False
    # Filled in by later stages:
    canonical_q: Dict[str, float] = field(default_factory=dict)
    canonical_fallback: bool = False
    canonical_widened: bool = False

    @property
    def moving_joints(self) -> List[JointSpec]:
        return [j for j in self.joints if j.is_moving]

    def object_points(self, cfg: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Whole-object point cloud in the base frame at configuration ``cfg``
        (defaults to the all-zero state)."""
        cfg = cfg or {}
        world = geo.link_world_transforms(self.base_link, self.joints, cfg)
        clouds = []
        for name, link in self.links.items():
            if link.points.size == 0 or name not in world:
                continue
            clouds.append(geo.transform_points(world[name], link.points))
        if not clouds:
            return np.zeros((0, 3))
        return np.concatenate(clouds, axis=0)


# --------------------------------------------------------------------------- #
# Mesh sampling
# --------------------------------------------------------------------------- #
def _visual_to_mesh(visual, urdf_dir: Path) -> Optional[trimesh.Trimesh]:
    """Convert one yourdfpy visual (primitive or mesh) into a trimesh in the
    link-local frame, honouring the visual's own origin transform."""
    g = visual.geometry
    mesh: Optional[trimesh.Trimesh] = None
    try:
        if g.box is not None:
            mesh = trimesh.creation.box(extents=np.asarray(g.box.size, dtype=float))
        elif g.cylinder is not None:
            mesh = trimesh.creation.cylinder(
                radius=float(g.cylinder.radius), height=float(g.cylinder.length)
            )
        elif g.sphere is not None:
            mesh = trimesh.creation.icosphere(radius=float(g.sphere.radius))
        elif g.mesh is not None:
            mesh_path = (urdf_dir / g.mesh.filename).resolve()
            loaded = trimesh.load(str(mesh_path), force="mesh")
            if isinstance(loaded, trimesh.Trimesh):
                mesh = loaded
            if mesh is not None and g.mesh.scale is not None:
                mesh.apply_scale(np.asarray(g.mesh.scale, dtype=float))
    except Exception as exc:  # noqa: BLE001 - geometry is best-effort
        logger.warning("failed to build visual geometry: %s", exc)
        return None
    if mesh is None or mesh.vertices.size == 0:
        return None
    if visual.origin is not None:
        mesh.apply_transform(visual.origin)
    return mesh


def _sample_link_points(
    link, urdf_dir: Path, n_points: int, seed: int
) -> tuple[np.ndarray, Optional[trimesh.Trimesh]]:
    """Surface-sample ``n_points`` from all of a link's visuals combined.

    Returns ``(points, combined_mesh)``; the combined mesh (in the link-local
    frame) is reused by the Stage 2 collision check.
    """
    meshes = [m for m in (_visual_to_mesh(v, urdf_dir) for v in link.visuals) if m]
    if not meshes:
        return np.zeros((0, 3)), None
    combined = trimesh.util.concatenate(meshes)
    area = float(getattr(combined, "area", 0.0))
    if area <= 0:
        return np.asarray(combined.vertices, dtype=float), combined
    rng = np.random.RandomState(seed)
    try:
        pts, _ = trimesh.sample.sample_surface(combined, n_points, seed=rng)
    except TypeError:  # older trimesh without the seed kwarg
        pts, _ = trimesh.sample.sample_surface(combined, n_points)
    return np.asarray(pts, dtype=float), combined


# --------------------------------------------------------------------------- #
# Chain extraction
# --------------------------------------------------------------------------- #
def _extract_serial_chain(
    urdf: yourdfpy.URDF,
    *,
    allow_fixed_branches: bool = False,
) -> tuple[List[str], List[str], Optional[str]]:
    """Return ``(link_chain, joint_name_chain, skip_reason)``.

    By default the chain follows parent->child from the base link and rejects
    every branch. With ``allow_fixed_branches``, the full link tree is returned
    in parent-before-child order, but no link may have multiple outgoing
    branches that contain non-fixed joints. Thus rigid visual/structural side
    branches are retained while the moving-joint skeleton remains serial.
    """
    child_links = {j.child for j in urdf.robot.joints}
    roots = [l.name for l in urdf.robot.links if l.name not in child_links]
    if len(roots) != 1:
        return [], [], f"expected 1 base link, found {len(roots)}"
    base = roots[0]

    children_of: Dict[str, List] = {}
    for j in urdf.robot.joints:
        children_of.setdefault(j.parent, []).append(j)

    if allow_fixed_branches:
        # URDF is expected to be a rooted tree: every non-root link has exactly
        # one inbound joint. Check this explicitly before traversing it.
        parent_counts: Dict[str, int] = {}
        for joint in urdf.robot.joints:
            parent_counts[joint.child] = parent_counts.get(joint.child, 0) + 1
        multiply_parented = sorted(
            name for name, count in parent_counts.items() if count > 1
        )
        if multiply_parented:
            return [], [], (
                "links with multiple parent joints: "
                + ", ".join(multiply_parented)
            )

        moving_cache: Dict[str, bool] = {}
        visiting_moving = set()

        def subtree_has_moving_joint(link_name: str) -> bool:
            cached = moving_cache.get(link_name)
            if cached is not None:
                return cached
            if link_name in visiting_moving:
                raise ValueError(f"cycle at link '{link_name}'")
            visiting_moving.add(link_name)
            result = any(
                joint.type != "fixed" or subtree_has_moving_joint(joint.child)
                for joint in children_of.get(link_name, [])
            )
            visiting_moving.remove(link_name)
            moving_cache[link_name] = result
            return result

        try:
            for parent, outgoing in children_of.items():
                moving_branches = [
                    joint
                    for joint in outgoing
                    if joint.type != "fixed"
                    or subtree_has_moving_joint(joint.child)
                ]
                if len(moving_branches) > 1:
                    return [], [], (
                        f"moving-joint branching at link '{parent}' "
                        f"({len(moving_branches)} moving branches)"
                    )
        except ValueError as exc:
            return [], [], str(exc)

        link_order: List[str] = []
        joint_order: List[str] = []
        visited = set()
        visiting = set()

        def walk(link_name: str) -> None:
            if link_name in visiting:
                raise ValueError(f"cycle at link '{link_name}'")
            if link_name in visited:
                return
            visiting.add(link_name)
            visited.add(link_name)
            link_order.append(link_name)
            outgoing = children_of.get(link_name, [])
            # Visit the articulated backbone first, followed by fixed-only
            # attachments in their original URDF order.
            ordered = sorted(
                enumerate(outgoing),
                key=lambda item: (
                    not (
                        item[1].type != "fixed"
                        or subtree_has_moving_joint(item[1].child)
                    ),
                    item[0],
                ),
            )
            for _, joint in ordered:
                joint_order.append(joint.name)
                walk(joint.child)
            visiting.remove(link_name)

        try:
            walk(base)
        except ValueError as exc:
            return [], [], str(exc)

        all_links = {link.name for link in urdf.robot.links}
        missing = sorted(all_links - visited)
        if missing:
            return [], [], "links unreachable from base: " + ", ".join(missing)
        return link_order, joint_order, None

    link_chain = [base]
    joint_chain: List[str] = []
    cursor = base
    while cursor in children_of:
        outgoing = children_of[cursor]
        if len(outgoing) > 1:
            return [], [], f"branching at link '{cursor}' ({len(outgoing)} children)"
        j = outgoing[0]
        joint_chain.append(j.name)
        link_chain.append(j.child)
        cursor = j.child
    return link_chain, joint_chain, None


def _joint_spec(j) -> JointSpec:
    axis = np.asarray(j.axis, dtype=float) if j.axis is not None else np.array([1.0, 0, 0])
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-12 else np.array([1.0, 0, 0])
    origin = np.asarray(j.origin, dtype=float) if j.origin is not None else np.eye(4)
    lower = float(j.limit.lower) if (j.limit and j.limit.lower is not None) else -np.pi
    upper = float(j.limit.upper) if (j.limit and j.limit.upper is not None) else np.pi
    return JointSpec(
        name=j.name, type=j.type, axis=axis, origin=origin.copy(),
        lower=lower, upper=upper, parent=j.parent, child=j.child,
        orig_lower=lower, orig_upper=upper,
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_object(
    urdf_path: Path,
    *,
    object_id: str,
    source_archive: Path,
    category: str,
    sub_category: str,
    split: str,
    n_points_per_link: int = 2048,
    seed: int = 0,
    allow_fixed_branches: bool = False,
) -> ObjectModel:
    """Parse ``urdf_path`` into an :class:`ObjectModel` (Stage 1).

    On any structural problem the returned model carries a ``skip_reason`` and
    empty chain data; callers should check it before running later stages.
    """
    urdf = yourdfpy.URDF.load(
        str(urdf_path),
        build_scene_graph=False,
        load_meshes=False,
        build_collision_scene_graph=False,
    )
    urdf_dir = Path(urdf_path).parent

    link_chain, joint_names, skip = _extract_serial_chain(
        urdf,
        allow_fixed_branches=allow_fixed_branches,
    )

    joint_map = {j.name: j for j in urdf.robot.joints}
    joints: List[JointSpec] = []
    if not skip:
        for jn in joint_names:
            jspec = _joint_spec(joint_map[jn])
            if jspec.type not in _SUPPORTED_TYPES:
                skip = f"unsupported joint type '{jspec.type}' on '{jspec.name}'"
                break
            if joint_map[jn].mimic is not None:
                skip = f"mimic joint '{jspec.name}' (multi-DOF coupling)"
                break
            joints.append(jspec)

    # Sample geometry per link regardless (cheap, and useful for QC even if the
    # object is skipped for articulation reasons).
    links: Dict[str, LinkSpec] = {}
    for i, link in enumerate(urdf.robot.links):
        pts, mesh = _sample_link_points(link, urdf_dir, n_points_per_link, seed + i)
        links[link.name] = LinkSpec(link.name, pts, has_geometry=pts.size > 0, mesh=mesh)

    model = ObjectModel(
        object_id=object_id,
        source_archive=source_archive,
        urdf_path=Path(urdf_path),
        category=category,
        sub_category=sub_category,
        split=split,
        base_link=link_chain[0] if link_chain else "",
        chain=link_chain,
        joints=joints,
        links=links,
        urdf=urdf,
        skip_reason=skip,
        fixed_branches_allowed=allow_fixed_branches,
    )
    if skip:
        logger.info("SKIP %s: %s", object_id, skip)
    return model
