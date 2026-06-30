#!/usr/bin/env python3
"""Generate two-pose point-cloud training samples for Particulate.

This is the two-state companion to ``generate_particulate_pointcloud_data.py``.
It consumes the objects stored in the ``fit`` folders of an Articraft-10K
category (``.tar.gz`` archives holding ``model.urdf`` + meshes) and, for every
generated scene, poses the articulated joints **randomly twice** -- once for a
``start`` state and once for an ``end`` state. Every field that the single-pose
generator emits is stored for both poses (suffixed ``_start`` / ``_end``):
point clouds, per-point segmentation labels, joint Plucker axes, joint ranges,
joint states/limits, closest-point targets, and so on.

The two poses share a single normalization frame (one center + scale computed
from the union of both poses' point clouds) so that the start->end articulation
is directly comparable in normalized coordinates.

Object staging is automatic: the ``fit`` archives ship without a
``bounding_box.json`` (which the S2U simulator needs to size/place the object),
so each archive is extracted into a staging object-set directory and a
``bounding_box.json`` is synthesized from the URDF's visual geometry.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple
import trimesh
import numpy as np

# Reuse all of the single-pose generator's helpers instead of duplicating them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_particulate_pointcloud_data as base

SampleGenerationError = base.SampleGenerationError
PART_COLORS = base.PART_COLORS


# --------------------------------------------------------------------------- #
# Object discovery and staging (Articraft-10K "fit" archives -> S2U layout).
# --------------------------------------------------------------------------- #
def collect_fit_dirs(
    articraft_root: Path,
    categories: Sequence[str],
    fit_dirs: Sequence[str],
    use_all: bool,
    dataset_type: str = "train"
) -> List[Path]:
    """Resolve the set of Articraft-10K ``fit`` directories to draw objects from."""

    resolved: List[Path] = []
    for explicit in fit_dirs:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise SampleGenerationError(f"--fit-dir does not exist: {path}")
        resolved.append(path)

    if use_all:
        resolved.extend(sorted(articraft_root.glob(f"**/sub_categories/*/fit/{dataset_type}")))

    for category in categories:
        matches = sorted(articraft_root.glob(f"**/sub_categories/{category}/fit/{dataset_type}"))
        if not matches:
            # Allow passing a top-level category folder name as well.
            matches = sorted(articraft_root.glob(f"{category}/sub_categories/*/fit/{dataset_type}"))
        if not matches:
            raise SampleGenerationError(
                f"No 'fit' folder found for category '{category}' under "
                f"{articraft_root}"
            )
        resolved.extend(matches)

    unique = sorted({path.resolve() for path in resolved})
    if not unique:
        raise SampleGenerationError(
            "No objects selected; pass --category, --fit-dir, or --all-categories"
        )
    return unique


def compute_urdf_aabb(
    urdf_path: Path,
    sim_src: Path,
    is_syn: bool = False,
    pos_rot: bool = True,
    num_pose_samples: int = 16,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize a pose-aware axis-aligned bounding box from visual geometry.

    The Articraft objects only carry visual geometry (no collision shapes), so
    PyBullet's collision-based ``getAABB`` is unusable. We walk every visual
    shape and transform its local extent into the world frame.

    Crucially, the box is the **union over many joint configurations** sampled
    from the same ranges the generator uses (``get_joint_limits`` with
    ``is_syn``/``pos_rot``), not just the zero/rest state. An articulated object
    spans far more when extended than when folded, so a rest-only box makes the
    render scale too large and posed parts fall outside the camera frustum. The
    pose-aware span fixes that. Each visual's local geometry is loaded once and
    only the per-configuration link transforms are recomputed, so the extra
    configurations are cheap.
    """

    base.setup_simulation_import(sim_src)
    import pybullet
    import trimesh
    from s2u.utils.saver import get_mesh_pose
    from s2u.utils.visual import as_mesh

    rng = np.random.default_rng(seed)
    client = pybullet.connect(pybullet.DIRECT)
    try:
        body = pybullet.loadURDF(
            str(urdf_path),
            useFixedBase=True,
            physicsClientId=client,
        )

        # 1) Cache each visual tuple and its scaled local-frame vertices once.
        cached: List[Tuple[Tuple[Any, ...], np.ndarray]] = []
        for visual in pybullet.getVisualShapeData(body, physicsClientId=client):
            geom_type = visual[2]
            dimensions = np.asarray(visual[3], dtype=np.float64)
            mesh_file = visual[4]
            if isinstance(mesh_file, bytes):
                mesh_file = mesh_file.decode("utf-8")

            if geom_type == pybullet.GEOM_MESH and mesh_file:
                mesh_path = Path(mesh_file)
                if not mesh_path.is_absolute():
                    mesh_path = (urdf_path.parent / mesh_path).resolve()
                mesh = as_mesh(trimesh.load(str(mesh_path), force="mesh"))
                local = np.asarray(mesh.vertices, dtype=np.float64) * dimensions
            else:
                half = _primitive_half_extent(pybullet, geom_type, dimensions)
                local = np.array(
                    [
                        [sx * half[0], sy * half[1], sz * half[2]]
                        for sx in (-1.0, 1.0)
                        for sy in (-1.0, 1.0)
                        for sz in (-1.0, 1.0)
                    ]
                )
            if len(local):
                cached.append((visual, local))

        if not cached:
            raise SampleGenerationError(
                f"URDF has no usable visual geometry: {urdf_path}"
            )

        # 2) Determine each movable joint's sampling range (same as the generator).
        movable: List[Tuple[int, float, float]] = []
        for joint_id in range(pybullet.getNumJoints(body, physicsClientId=client)):
            info = pybullet.getJointInfo(body, joint_id, physicsClientId=client)
            if int(info[2]) not in (base.REVOLUTE, base.PRISMATIC):
                continue
            lower, upper = base.get_joint_limits(info, is_syn, pos_rot)
            if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower):
                lower = upper = 0.0
            movable.append((joint_id, lower, upper))

        # 3) Build configurations to union over: rest, both extremes, randoms.
        configs: List[Dict[int, float]] = [{j: 0.0 for j, _, _ in movable}]
        if movable:
            configs.append({j: lo for j, lo, _ in movable})
            configs.append({j: up for j, _, up in movable})
            for _ in range(max(num_pose_samples, 0)):
                configs.append(
                    {j: float(rng.uniform(lo, up)) for j, lo, up in movable}
                )

        # 4) Union the world AABB across every configuration.
        mins: List[np.ndarray] = []
        maxs: List[np.ndarray] = []
        for config in configs:
            for joint_id, value in config.items():
                pybullet.resetJointState(body, joint_id, value, physicsClientId=client)
            for visual, local in cached:
                _, _, _, transform = get_mesh_pose(visual, client)
                world = (
                    transform.as_matrix() @ np.c_[local, np.ones(len(local))].T
                ).T[:, :3]
                mins.append(world.min(axis=0))
                maxs.append(world.max(axis=0))
    finally:
        pybullet.disconnect(client)

    return np.min(mins, axis=0), np.max(maxs, axis=0)

def _make_primitive_mesh(bullet: Any, geom_type: int, dims: np.ndarray) -> "trimesh.Trimesh":
    """Build a trimesh primitive in its local frame (dims match getVisualShapeData)."""

    from trimesh import creation

    if geom_type == bullet.GEOM_BOX:
        # dims are full side lengths (confirmed for getVisualShapeData).
        return creation.box(extents=np.abs(dims[:3]))
    if geom_type == bullet.GEOM_SPHERE:
        return creation.uv_sphere(radius=float(dims[0]))
    if geom_type == bullet.GEOM_CYLINDER:
        # dims[0] = length (height along local z), dims[1] = radius.
        return creation.cylinder(radius=float(dims[1]), height=float(dims[0]))
    if geom_type == bullet.GEOM_CAPSULE:
        return creation.capsule(radius=float(dims[1]), height=float(dims[0]))
    # Fallback: a box from whatever dimensions were reported.
    return creation.box(extents=np.abs(dims[:3]))


def sample_part_labeled_point_cloud_from_meshes(
    sim: Any,
    structure: Mapping[str, Any],
    num_points: int,
    rng: np.random.Generator,
    min_points_per_part: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directly sample labeled surface points from the posed meshes.

    Occlusion-free alternative to the multi-view scan: every part is guaranteed
    points regardless of camera coverage or object scale. Returns world-space
    points, contiguous part labels, and per-point surface normals -- the same
    contract as ``base.acquire_part_labeled_point_cloud``.
    """

    from trimesh import sample as tm_sample, util as tm_util
    from s2u.utils.saver import get_mesh_pose
    from s2u.utils.visual import as_mesh

    bullet = sim.world.p
    client = bullet._client
    uid = sim.object.uid
    link_to_part = structure["source_link_to_part"]  # dict: pybullet link id -> part id
    urdf_dir = Path(sim.object_urdfs[sim.object_idx]).parent

    num_parts = len(structure["ordered_owners"])
    per_part_meshes: Dict[int, List[Any]] = {p: [] for p in range(num_parts)}

    # 1) Collect each visual as a world-space trimesh, tagged with its part id.
    for visual in bullet.getVisualShapeData(uid):
        link_index = visual[1]
        geom_type = visual[2]
        dims = np.asarray(visual[3], dtype=np.float64)
        mesh_file = visual[4].decode() if isinstance(visual[4], bytes) else visual[4]
        if link_index not in link_to_part:
            continue

        _, _, scale, transform = get_mesh_pose(visual, client)
        world_from_visual = transform.as_matrix()

        if geom_type == bullet.GEOM_MESH and mesh_file:
            mesh_path = Path(mesh_file)
            if not mesh_path.is_absolute():
                mesh_path = (urdf_dir / mesh_path).resolve()
            mesh = as_mesh(trimesh.load(str(mesh_path), force="mesh"))
            mesh.apply_scale(np.asarray(scale, dtype=np.float64))
        else:
            mesh = _make_primitive_mesh(bullet, geom_type, dims)

        mesh.apply_transform(world_from_visual)
        per_part_meshes[link_to_part[link_index]].append(mesh)

    # 2) Sample each part's surface, proportional to area with a floor per part,
    #    so even tiny parts are represented (the whole point of mesh sampling).
    part_meshes: Dict[int, Any] = {}
    part_areas: Dict[int, float] = {}
    for part_id, meshes in per_part_meshes.items():
        if not meshes:
            raise SampleGenerationError(
                f"Part {part_id} has no visual geometry to sample"
            )
        combined = tm_util.concatenate(meshes)
        part_meshes[part_id] = combined
        part_areas[part_id] = float(combined.area)

    total_area = sum(part_areas.values())
    pool_target = max(num_points * 2, num_parts * min_points_per_part)

    points: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    for part_id, mesh in part_meshes.items():
        if total_area > 0 and part_areas[part_id] > 0:
            share = int(round(pool_target * part_areas[part_id] / total_area))
        else:
            share = min_points_per_part
        count = max(min_points_per_part, share)

        if mesh.area > 0:
            sampled, face_index = tm_sample.sample_surface(mesh, count)
            sampled_normals = mesh.face_normals[face_index]
        else:
            # Degenerate (zero-area) geometry: fall back to its vertices.
            choice = rng.integers(0, len(mesh.vertices), size=count)
            sampled = np.asarray(mesh.vertices)[choice]
            sampled_normals = np.tile([0.0, 0.0, 1.0], (count, 1))

        points.append(np.asarray(sampled, dtype=np.float64))
        labels.append(np.full(count, part_id, dtype=np.int16))
        normals.append(np.asarray(sampled_normals, dtype=np.float64))

    return (
        np.concatenate(points, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(normals, axis=0),
    )


def capture_mesh_visuals(
    sim: Any,
    structure: Mapping[str, Any],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Record, for the object's *current* configuration, every visual's pose.

    Returns ``(metadata, transforms)`` where ``metadata`` holds the pose-
    invariant per-visual description and ``transforms`` is ``(V, 4, 4)`` rigid
    world poses at the current joint state. To rebuild a visual's world geometry
    at a saved pose::

        local = load_obj(file).vertices if is_mesh else primitive(geom_type, dims)
        world = (transform @ np.c_[local * scale, ones].T).T[:, :3]
        normalized = (world - normalization_center) * normalization_scale

    ``scale`` already includes ``global_scaling`` (PyBullet folds it into the
    returned mesh scale), and the ``transform`` is in raw PyBullet world space,
    so the final normalization step is what aligns it with the saved clouds.
    """

    from s2u.utils.saver import get_mesh_pose

    bullet = sim.world.p
    client = bullet._client
    uid = sim.object.uid
    link_to_part = structure["source_link_to_part"]

    links: List[int] = []
    parts: List[int] = []
    geom_types: List[int] = []
    is_mesh: List[bool] = []
    scales: List[np.ndarray] = []
    dims_list: List[np.ndarray] = []
    files: List[str] = []
    transforms: List[np.ndarray] = []

    for visual in bullet.getVisualShapeData(uid):
        link_index = visual[1]
        if link_index not in link_to_part:
            continue
        geom_type = int(visual[2])
        dims = np.asarray(visual[3], dtype=np.float64)
        mesh_file = visual[4].decode() if isinstance(visual[4], bytes) else visual[4]
        _, _, scale, transform = get_mesh_pose(visual, client)

        links.append(int(link_index))
        parts.append(int(link_to_part[link_index]))
        geom_types.append(geom_type)
        is_mesh.append(geom_type == bullet.GEOM_MESH and bool(mesh_file))
        scales.append(np.asarray(scale, dtype=np.float64).reshape(3))
        dims_list.append(dims.reshape(-1)[:3])
        files.append(mesh_file or "")
        transforms.append(transform.as_matrix())

    metadata = {
        "mesh_visual_link": np.asarray(links, dtype=np.int16),
        "mesh_visual_part": np.asarray(parts, dtype=np.int16),
        "mesh_visual_geom_type": np.asarray(geom_types, dtype=np.int16),
        "mesh_visual_is_mesh": np.asarray(is_mesh, dtype=np.bool_),
        "mesh_visual_scale": np.asarray(scales, dtype=np.float32).reshape(-1, 3),
        "mesh_visual_dims": np.asarray(dims_list, dtype=np.float32).reshape(-1, 3),
        "mesh_visual_file": np.asarray(files),
    }
    return metadata, np.asarray(transforms, dtype=np.float32).reshape(-1, 4, 4)


def _primitive_half_extent(pybullet: Any, geom_type: int, dims: np.ndarray) -> np.ndarray:
    if geom_type == pybullet.GEOM_BOX:
        return dims / 2.0
    if geom_type == pybullet.GEOM_SPHERE:
        radius = float(dims[0])
        return np.array([radius, radius, radius])
    if geom_type == pybullet.GEOM_CYLINDER:
        length, radius = float(dims[0]), float(dims[1])
        return np.array([radius, radius, length / 2.0])
    if geom_type == pybullet.GEOM_CAPSULE:
        length, radius = float(dims[0]), float(dims[1])
        return np.array([radius, radius, length / 2.0 + radius])
    # Fallback: treat the reported dimensions as full extents.
    return np.abs(dims) / 2.0


def stage_objects(
    fit_dirs: Sequence[Path],
    staging_root: Path,
    object_set: str,
    sim_src: Path,
    overwrite: bool,
    bbox_options: Mapping[str, Any] | None = None,
) -> Tuple[Path, List[str]]:
    """Extract every ``fit`` archive and write a synthesized ``bounding_box.json``."""

    bbox_options = dict(bbox_options or {})
    object_set_dir = staging_root / object_set
    object_set_dir.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []

    for fit_dir in fit_dirs:
        subcategory = fit_dir.parent.name
        for archive in sorted(fit_dir.glob("*.tar.gz")):
            record_id = archive.name[: -len(".tar.gz")]
            dest = object_set_dir / f"{subcategory}__{record_id}"
            bbox_path = dest / "bounding_box.json"
            if dest.is_dir() and bbox_path.exists() and not overwrite:
                continue
            try:
                _stage_single_archive(
                    archive,
                    object_set_dir,
                    dest,
                    sim_src,
                    bbox_options,
                )
            except (tarfile.TarError, OSError, RuntimeError, ValueError) as error:
                message = f"staging failed for {archive.name}: {error}"
                errors.append(message)
                print(message, file=sys.stderr, flush=True)
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)

    staged = sorted(p for p in object_set_dir.iterdir() if p.is_dir())
    if not staged:
        raise SampleGenerationError(
            f"No objects were staged into {object_set_dir}; cannot generate data"
        )
    return object_set_dir, errors


def _stage_single_archive(
    archive: Path,
    object_set_dir: Path,
    dest: Path,
    sim_src: Path,
    bbox_options: Mapping[str, Any],
) -> None:
    if dest.is_dir():
        shutil.rmtree(dest)

    with tarfile.open(archive, "r:gz") as handle:
        top_levels = {Path(name).parts[0] for name in handle.getnames() if name.strip()}
        handle.extractall(object_set_dir)

    if len(top_levels) == 1:
        extracted = object_set_dir / next(iter(top_levels))
        if extracted.resolve() != dest.resolve():
            extracted.rename(dest)
    else:
        # Unexpected layout: gather everything into ``dest``.
        dest.mkdir(parents=True, exist_ok=True)
        for top in top_levels:
            (object_set_dir / top).rename(dest / top)

    urdf_candidates = sorted(dest.rglob("*.urdf"))
    if not urdf_candidates:
        raise SampleGenerationError(f"No .urdf found inside {archive.name}")
    urdf_path = urdf_candidates[0]

    bbox_min, bbox_max = compute_urdf_aabb(urdf_path, sim_src, **bbox_options)
    with (dest / "bounding_box.json").open("w", encoding="utf-8") as output:
        json.dump(
            {"min": bbox_min.tolist(), "max": bbox_max.tolist()},
            output,
            indent=2,
        )


# --------------------------------------------------------------------------- #
# Physical collision: force visual geometry to act as collision geometry and
# load objects with self-collision so overlapping poses can be detected.
# --------------------------------------------------------------------------- #
def force_visual_as_collision(urdf_path: str) -> str:
    """Write a sibling URDF whose every visual is duplicated as collision.

    Many Articraft objects ship visual-only links, so PyBullet creates no
    collision shapes and links can freely interpenetrate. Mirroring each visual
    into a ``<collision>`` element gives every link a real collision shape. The
    rewritten URDF is a temp file in the same directory (so relative mesh paths
    still resolve) and the caller is responsible for deleting it.
    """

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        visuals = link.findall("visual")
        if not visuals:
            continue
        for col in link.findall("collision"):
            link.remove(col)
        for visual in visuals:
            collision = copy.deepcopy(visual)
            collision.tag = "collision"
            link.append(collision)
    urdf_dir = os.path.dirname(urdf_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".urdf", dir=urdf_dir)
    os.close(fd)
    tree.write(tmp_path, xml_declaration=True)
    return tmp_path


def install_collision_loader(sim_src: Path) -> None:
    """Patch the simulator's URDF loader to add collisions + self-collision.

    The S2U simulator loads objects through ``btsim.Body.from_urdf`` with a
    hardcoded ``loadURDF`` call. We wrap it so that every object is rewritten
    via :func:`force_visual_as_collision` and loaded with
    ``URDF_USE_SELF_COLLISION``. The patch is idempotent and must be installed
    inside every worker process (it does not survive ``spawn``).
    """

    base.setup_simulation_import(sim_src)
    import pybullet
    from s2u.utils import btsim

    if getattr(btsim.Body.from_urdf, "_twopose_collision_patched", False):
        return

    def from_urdf(cls, physics_client, urdf_path, pose, scale, useFixedBase):
        collision_urdf = force_visual_as_collision(str(urdf_path))
        try:
            body_uid = physics_client.loadURDF(
                str(collision_urdf),
                pose.translation,
                pose.rotation.as_quat(),
                globalScaling=scale,
                useFixedBase=useFixedBase,
                flags=pybullet.URDF_USE_SELF_COLLISION,
            )
        finally:
            try:
                os.remove(collision_urdf)
            except OSError:
                pass
        return cls(physics_client, body_uid, scale)

    from_urdf._twopose_collision_patched = True
    btsim.Body.from_urdf = classmethod(from_urdf)


def _penetrating_link_pairs(
    sim: Any,
    penetration_tol: float,
) -> set:
    """Set of frozenset link-index pairs currently interpenetrating."""

    bullet = sim.world.p
    bullet.performCollisionDetection()
    contacts = bullet.getContactPoints(bodyA=sim.object.uid, bodyB=sim.object.uid)
    # Index 8 is contactDistance (negative = penetration); 3/4 are link indices.
    return {
        frozenset((contact[3], contact[4]))
        for contact in contacts
        if contact[8] < -penetration_tol
    }


def object_baseline_contacts(
    sim: Any,
    active_joint_ids: Sequence[int],
    penetration_tol: float,
) -> set:
    """Link pairs that overlap at the neutral (all-zero) configuration.

    AI-generated visual geometry frequently interpenetrates at the joints by
    design (collars wrapping beams, etc.). When those visuals are forced into
    collision shapes, such pairs overlap in *every* pose, so they must be
    treated as an allowed baseline rather than disqualifying the object.
    """

    for joint_id in active_joint_ids:
        sim.set_joint_state(joint_id, 0.0)
    return _penetrating_link_pairs(sim, penetration_tol)


def object_self_collides(
    sim: Any,
    penetration_tol: float,
    baseline_contacts: set = frozenset(),
) -> bool:
    """True when articulation makes *new* non-adjacent links interpenetrate.

    ``URDF_USE_SELF_COLLISION`` (without ``INCLUDE_PARENT``) excludes
    parent-child link pairs, and ``baseline_contacts`` excludes the overlaps the
    object already exhibits at rest, so a hit here is a genuine pose-induced
    overlap between parts that should not touch. ``resetJointState`` teleports
    joints and ignores collisions, so this explicit check is what actually lets
    us reject overlapping poses.
    """

    return bool(_penetrating_link_pairs(sim, penetration_tol) - baseline_contacts)


# --------------------------------------------------------------------------- #
# Two-pose sample assembly.
# --------------------------------------------------------------------------- #
def shared_normalization(
    point_clouds: Sequence[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """One center + scale from the union of every pose's world point cloud."""

    stacked = np.concatenate(
        [np.asarray(cloud, dtype=np.float64) for cloud in point_clouds],
        axis=0,
    )
    bbox_min = stacked.min(axis=0)
    bbox_max = stacked.max(axis=0)
    longest_side = float((bbox_max - bbox_min).max())
    if longest_side < 1e-10:
        raise SampleGenerationError("Combined point cloud bounding box is degenerate")
    center = (bbox_min + bbox_max) * 0.5
    return center, 1.0 / longest_side


def sample_start_joint_states(
    sim: Any,
    all_joint_info: Mapping[int, Tuple[Any, ...]],
    active_joint_ids: Sequence[int],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> Dict[int, float]:
    """Uniform random start pose (gives variety of absolute configurations)."""

    return base.sample_joint_states(
        sim,
        all_joint_info,
        active_joint_ids,
        rng,
        is_syn=args.is_syn,
        pos_rot=bool(args.pos_rot),
        random_state=True,
    )


def sample_end_joint_states(
    sim: Any,
    all_joint_info: Mapping[int, Tuple[Any, ...]],
    active_joint_ids: Sequence[int],
    start_states: Mapping[int, float],
    rng: np.random.Generator,
    args: argparse.Namespace,
    attenuation: float = 1.0,
) -> Dict[int, float]:
    """Sample an end pose pushed away from the start pose, biased to big moves.

    A per-sample separation level ``sep = u ** (1 / bias)`` (``u ~ U(0, 1)``) is
    drawn once: ``bias = 1`` is unbiased, larger ``bias`` skews ``sep`` toward 1
    so most samples move far. Per joint, the end is displaced from the start by
    ``sep`` (plus per-joint jitter) of the travel room, in the direction that
    has more room (chosen stochastically, proportional to that room). This
    maximizes the average start->end difference while keeping a random spread of
    magnitudes -- including the occasional small move.

    ``attenuation`` (<= 1) shrinks the move toward the start pose; it is driven
    below 1 only when self-collision rejection keeps failing, so an aggressive
    move can back off to a feasible (collision-free) one near the valid start.
    """

    separation = float(rng.uniform()) ** (1.0 / args.pose_separation_bias)
    active = set(active_joint_ids)
    states: Dict[int, float] = {}
    for joint_id, joint_data in all_joint_info.items():
        joint_type = int(joint_data[2])
        if joint_type not in base.SUPPORTED_MOVABLE_TYPES or joint_id not in active:
            continue

        lower, upper = base.get_joint_limits(joint_data, args.is_syn, bool(args.pos_rot))
        start = float(start_states[joint_id])
        if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower):
            end = start
        else:
            span = upper - lower
            room_up = upper - start
            room_down = start - lower
            if rng.uniform() < (room_up / span):
                direction, room = 1.0, room_up
            else:
                direction, room = -1.0, room_down
            travel = (separation + rng.normal(0.0, args.pose_separation_jitter)) * attenuation
            travel = float(np.clip(travel, 0.0, 1.0))
            end = float(np.clip(start + direction * travel * room, lower, upper))

        sim.set_joint_state(joint_id, end)
        states[joint_id] = end
    return states


def acquire_pose_world(
    sim: Any,
    structure: Mapping[str, Any],
    num_parts: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    sampler: Callable[[int], Dict[int, float]],
    baseline_contacts: set = frozenset(),
) -> Tuple[Dict[int, float], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Pose the joints via ``sampler`` and acquire one cloud per point source.

    ``sampler(attempt)`` applies a fresh joint configuration to the simulator
    and returns it; ``attempt`` is the number of prior rejected tries, letting a
    sampler anneal. When ``--reject-self-collision`` is enabled it is called
    again until the posed object introduces no *new* interpenetrating parts
    beyond ``baseline_contacts`` (or the attempt budget is spent).

    Returns ``(states, clouds)`` where ``clouds`` maps a source name ("scan"
    and/or "mesh") to a ``(points_world, part_ids, normals)`` tuple.
    """

    attempt = 0
    states = sampler(attempt)
    if args.reject_self_collision:
        while object_self_collides(
            sim, args.collision_penetration_tol, baseline_contacts
        ):
            attempt += 1
            if attempt >= args.max_pose_attempts:
                raise SampleGenerationError(
                    f"No non-overlapping pose found in {args.max_pose_attempts} "
                    "attempts"
                )
            states = sampler(attempt)

    if args.point_source == "scan":
        sources: Tuple[str, ...] = ("scan",)
    elif args.point_source == "mesh-sampling":
        sources = ("mesh",)
    else:  # "both"
        sources = ("scan", "mesh")

    clouds: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for source in sources:
        if source == "scan":
            points_world, part_ids, normals = base.acquire_part_labeled_point_cloud(
                sim,
                structure["part_source_links"],
                num_views=args.num_views,
                tsdf_resolution=args.tsdf_resolution,
                min_segmentation_score=args.min_segmentation_score,
            )
        else:
            points_world, part_ids, normals = sample_part_labeled_point_cloud_from_meshes(
                sim,
                structure,
                num_points=args.num_points,
                rng=rng,
                min_points_per_part=args.min_points_per_part,
            )
        clouds[source] = base.stratified_sample(
            points_world,
            part_ids,
            normals,
            num_points=args.num_points,
            min_points_per_part=args.min_points_per_part,
            num_parts=num_parts,
            rng=rng,
        )
    return states, clouds


def finalize_pose(
    sim: Any,
    all_joint_info: Mapping[int, Tuple[Any, ...]],
    structure: Mapping[str, Any],
    states: Mapping[int, float],
    points_world: np.ndarray,
    part_ids: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    scale: float,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    """Build every per-pose field in the shared normalized frame."""

    points = (np.asarray(points_world, dtype=np.float64) - center) * scale
    normals = base.sanitize_normals(points, normals, args.normal_neighbors)

    # ``build_motion_targets`` reads joint screws live from the simulator via
    # forward kinematics, so the object must be restored to *this* pose's
    # configuration first (both poses are acquired before either is finalized).
    for joint_id, state in states.items():
        sim.set_joint_state(joint_id, state)

    motion = base.build_motion_targets(
        sim,
        all_joint_info,
        structure,
        states,
        center,
        scale,
        is_syn=args.is_syn,
        pos_rot=bool(args.pos_rot),
    )
    axes_plucker = motion["link_axes_plucker"]
    ranges = motion["link_range"]
    motion_class = motion["part_motion_class"]
    closest_points = base.closest_points_on_revolute_axes(
        points,
        part_ids,
        axes_plucker,
        motion_class,
    )

    return {
        "points": points.astype(np.float32),
        "normals": normals.astype(np.float32),
        "point_to_bone": part_ids.astype(np.int16),
        "point_from_sharp": np.zeros(len(points), dtype=np.bool_),
        "link_axes_plucker": axes_plucker,
        "link_range": ranges,
        "part_ids": part_ids.astype(np.int16),
        "gt_part_motion_class": motion_class,
        "gt_revolute_plucker": axes_plucker[:, :6],
        "gt_revolute_range": ranges[:, :2],
        "gt_prismatic_axis": axes_plucker[:, 6:9],
        "gt_prismatic_range": ranges[:, 2:],
        "gt_closest_point_on_axis": closest_points,
        "is_part_revolute": np.isin(motion_class, (1, 3)),
        "is_part_prismatic": np.isin(motion_class, (2, 3)),
        "joint_origins": motion["joint_origins"],
        "moment_axis_points": motion["moment_axis_points"],
        "joint_origins_world": motion["joint_origins_world"],
        "joint_axes_world": motion["joint_axes_world"],
        "simulator_moments_world": motion["simulator_moments_world"],
        "joint_state": motion["joint_state"],
        "joint_limits": motion["joint_limits"],
        "source_joint_ids": motion["source_joint_ids"],
    }


POSE_SUFFIXES = ("start", "end")

# Fields that depend on the point cloud itself (everything else a pose produces
# is geometry/joint metadata that is identical across point sources). When both
# sources are saved, the secondary ("mesh") source stores only these.
POINT_LEVEL_KEYS = (
    "points",
    "normals",
    "point_to_bone",
    "point_from_sharp",
    "part_ids",
    "gt_closest_point_on_axis",
)


def create_two_pose_sample(
    sim: Any,
    args: argparse.Namespace,
    rng: np.random.Generator,
    sample_index: int,
) -> Dict[str, np.ndarray]:
    all_joint_info = base.get_all_joint_info(sim)
    structure = base.build_part_structure(
        all_joint_info,
        max_parts=(
            len(all_joint_info) + 1
            if args.max_moving_joints is not None
            else args.max_parts
        ),
        require_serial=not args.allow_tree,
    )
    if args.max_moving_joints is not None:
        structure = base.limit_moving_joints(
            all_joint_info,
            structure,
            max_moving_joints=args.max_moving_joints,
            max_parts=args.max_parts,
        )
    num_parts = len(structure["ordered_owners"])
    if num_parts < 2:
        raise SampleGenerationError("Particulate training requires at least two parts")

    # Pose the same object twice without re-loading it, so both poses share the
    # identical base placement (and therefore a meaningful shared frame). The
    # start pose is uniform; the end pose is biased far from it (unless
    # --independent-poses) so the articulation between the two is large.
    active_joint_ids = structure["active_joint_ids"]

    # Overlaps the object already has at rest are tolerated; only pose-induced
    # ones disqualify a configuration.
    baseline_contacts = (
        object_baseline_contacts(
            sim, active_joint_ids, args.collision_penetration_tol
        )
        if args.reject_self_collision
        else frozenset()
    )

    def start_sampler(attempt: int) -> Dict[int, float]:
        return sample_start_joint_states(
            sim, all_joint_info, active_joint_ids, rng, args
        )

    start_raw = acquire_pose_world(
        sim, structure, num_parts, args, rng, start_sampler, baseline_contacts
    )
    start_states = start_raw[0]

    if args.independent_poses:
        end_sampler = start_sampler
    else:
        def end_sampler(attempt: int) -> Dict[int, float]:
            # Back off toward the (valid) start pose as collision retries mount.
            attenuation = args.pose_separation_decay ** attempt
            return sample_end_joint_states(
                sim,
                all_joint_info,
                active_joint_ids,
                start_states,
                rng,
                args,
                attenuation=attenuation,
            )

    end_raw = acquire_pose_world(
        sim, structure, num_parts, args, rng, end_sampler, baseline_contacts
    )

    # poses[suffix] = (states, {source: (points_world, part_ids, normals)})
    poses = {"start": start_raw, "end": end_raw}
    center, scale = shared_normalization(
        [
            cloud[0]
            for _, clouds in poses.values()
            for cloud in clouds.values()
        ]
    )

    bone_structure = structure["bone_structure"]
    part_structure_matrix = np.zeros((num_parts, num_parts), dtype=np.bool_)
    if len(bone_structure):
        part_structure_matrix[bone_structure[:, 0], bone_structure[:, 1]] = True
    source_link_to_part = np.asarray(
        [
            structure["source_link_to_part"][link_id]
            for link_id in [-1, *sorted(all_joint_info)]
        ],
        dtype=np.int16,
    )
    object_path = str(sim.object_urdfs[sim.object_idx])

    # Fields that describe the object as a whole are stored once (unsuffixed).
    sample: Dict[str, np.ndarray] = {
        "bone_structure": bone_structure,
        "motion_hierarchy": bone_structure,
        "part_structure_matrix": part_structure_matrix,
        "num_valid_parts": np.asarray(num_parts, dtype=np.int16),
        "source_link_to_part": source_link_to_part,
        "active_source_joint_ids": np.asarray(
            structure["active_joint_ids"], dtype=np.int16
        ),
        "merged_source_joint_ids": np.asarray(
            structure["merged_joint_ids"], dtype=np.int16
        ),
        "normalization_center": center.astype(np.float32),
        "normalization_scale": np.asarray(scale, dtype=np.float32),
        "sample_index": np.asarray(sample_index, dtype=np.int64),
        "object_path": np.asarray(object_path),
        "pose_keys": np.asarray(list(POSE_SUFFIXES)),
    }

    for suffix in POSE_SUFFIXES:
        states, clouds = poses[suffix]

        # The "primary" source owns the base field names (and the shared joint /
        # motion metadata). Scan wins when present; otherwise mesh is primary.
        primary = "scan" if "scan" in clouds else "mesh"
        for source, (points_world, part_ids, normals) in clouds.items():
            pose = finalize_pose(
                sim,
                all_joint_info,
                structure,
                states,
                points_world,
                part_ids,
                normals,
                center,
                scale,
                args,
            )
            # Validate each cloud with the single-pose validator by lending it
            # the shared structure fields it expects.
            validation_view = dict(pose)
            validation_view["num_valid_parts"] = sample["num_valid_parts"]
            validation_view["bone_structure"] = bone_structure
            base.validate_sample(validation_view, args.max_parts)

            if source == primary:
                for key, value in pose.items():
                    sample[f"{key}_{suffix}"] = value
            else:
                # Secondary source: keep only the cloud-dependent fields, under a
                # "<field>_mesh_<suffix>" name (joint/motion fields are shared).
                for key in POINT_LEVEL_KEYS:
                    sample[f"{key}_mesh_{suffix}"] = pose[key]

    # Per-pose mesh visual poses, so the full meshes can be reconstructed at the
    # start/end states (see capture_mesh_visuals for the reconstruction recipe).
    if args.save_mesh_poses:
        mesh_metadata = None
        for suffix in POSE_SUFFIXES:
            states, _ = poses[suffix]
            for joint_id, state in states.items():
                sim.set_joint_state(joint_id, state)
            metadata, transforms = capture_mesh_visuals(sim, structure)
            if mesh_metadata is None:
                mesh_metadata = metadata  # pose-invariant; store once
            sample[f"mesh_visual_transform_{suffix}"] = transforms
        sample.update(mesh_metadata)

    return sample


def write_pose_visualization(
    path: Path,
    sample: Mapping[str, np.ndarray],
    suffix: str,
    axis_length: float,
    axis_samples: int,
) -> None:
    pose_view = {
        "points": sample[f"points_{suffix}"],
        "part_ids": sample[f"part_ids_{suffix}"],
        "gt_part_motion_class": sample[f"gt_part_motion_class_{suffix}"],
        "num_valid_parts": sample["num_valid_parts"],
        "joint_origins": sample[f"joint_origins_{suffix}"],
        "link_axes_plucker": sample[f"link_axes_plucker_{suffix}"],
    }
    base.write_visualization(path, pose_view, axis_length, axis_samples)


# --------------------------------------------------------------------------- #
# Worker / driver.
# --------------------------------------------------------------------------- #
def generate_worker(
    args: argparse.Namespace,
    worker_id: int,
    sample_indices: Sequence[int],
) -> List[str]:
    simulation_class = base.get_simulation_class(args.sim_src)
    install_collision_loader(args.sim_src)
    worker_seed = args.seed + worker_id * 100_003
    np.random.seed(worker_seed)
    rng = np.random.default_rng(worker_seed)
    sim = simulation_class(
        args.object_set,
        size=args.workspace_size,
        gui=args.sim_gui,
        global_scaling=args.global_scaling,
        dense_photo=args.dense_photo,
        seed=worker_seed + 1,
        urdf_root=args.urdf_root,
    )
    if not sim.object_urdfs:
        raise SampleGenerationError(
            f"No URDFs found for staged object set '{args.object_set}' under "
            f"{args.urdf_root}"
        )
    if len(sim.object_urdfs) != len(sim.object_bbox):
        raise SampleGenerationError(
            "Every staged URDF must have a matching bounding_box.json"
        )

    errors: List[str] = []
    for position, sample_index in enumerate(sample_indices):
        output_path = args.root / "samples" / f"{sample_index:08d}.npz"
        if output_path.exists() and not args.overwrite:
            continue

        last_error: Exception | None = None
        for _ in range(args.max_attempts_per_sample):
            try:
                sim.reset(canonical=args.canonical)
                sample = create_two_pose_sample(sim, args, rng, sample_index)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output_path, **sample)
                if args.save_ply:
                    for suffix in POSE_SUFFIXES:
                        write_pose_visualization(
                            args.root
                            / "visualizations"
                            / f"{sample_index:08d}_{suffix}.ply",
                            sample,
                            suffix,
                            axis_length=args.axis_visual_length,
                            axis_samples=args.axis_visual_samples,
                        )
                last_error = None
                break
            except (SampleGenerationError, ValueError, RuntimeError) as error:
                last_error = error

        if last_error is not None:
            message = (
                f"worker {worker_id}: failed sample {sample_index} after "
                f"{args.max_attempts_per_sample} attempts: {last_error}"
            )
            errors.append(message)
            print(message, file=sys.stderr, flush=True)

        if worker_id == 0 and (
            (position + 1) % max(args.log_every, 1) == 0
            or position + 1 == len(sample_indices)
        ):
            print(
                f"worker 0: processed {position + 1}/{len(sample_indices)} samples",
                flush=True,
            )
    return errors


def schema_description() -> Dict[str, str]:
    per_pose = base.schema_description()
    described = {
        "<field>_start / <field>_end": (
            "Every per-pose field below is stored twice, once for the random "
            "start state and once for the random end state"
        ),
        "pose_keys": "(2,) str pose suffixes, i.e. ['start', 'end']",
        "normalization_center": "(3,) float32 SHARED world-space bbox center",
        "normalization_scale": "scalar float32 SHARED reciprocal longest side",
        "num_valid_parts": "int16 part count (shared across both poses)",
        "bone_structure": "(P-1,2) int16 directed parent-child edges (shared)",
    }
    for key, value in per_pose.items():
        if key in ("normalization_center", "normalization_scale"):
            continue
        described.setdefault(f"{key}_start / {key}_end", value)
    return described


def write_dataset_metadata(
    args: argparse.Namespace,
    errors: Iterable[str],
) -> None:
    args.root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": schema_description(),
        "poses_per_sample": 2,
        "generator_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "errors": list(errors),
    }
    with (args.root / "dataset_info.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)

    sample_paths = sorted((args.root / "samples").glob("*.npz"))
    with (args.root / "manifest.txt").open("w", encoding="utf-8") as output:
        for sample_path in sample_paths:
            output.write(f"{sample_path.relative_to(args.root)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate two-pose (start/end) Particulate-compatible articulated "
            "point-cloud samples from Articraft-10K 'fit' objects."
        )
    )
    parser.add_argument("root", type=Path, help="Output dataset directory")

    # Object source.
    parser.add_argument(
        "--articraft-root",
        type=Path,
        default=Path("~/Articraft-10K"),
        help="Root of the Articraft-10K dataset",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Sub-category name whose 'fit' objects are used (repeatable)",
    )
    parser.add_argument(
        "--fit-dir",
        action="append",
        default=[],
        help="Explicit path to a 'fit' folder (repeatable)",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Use every 'fit' folder found under --articraft-root",
    )
    parser.add_argument(
        "--dataset-type",
        choices=("train", "val", "test"),
        help="Dataset type (used to select categories when --all-categories)",
        default="train"
    )
    parser.add_argument(
        "--object-set",
        default="articraft_fit",
        help="Name of the staged object set directory",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help="Where to extract objects (default: <root>/staging)",
    )
    parser.add_argument(
        "--sim-src",
        type=Path,
        default=Path("~/Articulated_object_simulation-main/src"),
    )

    # Generation controls (mirrors the single-pose generator).
    parser.add_argument("--point-source", choices=("scan", "mesh-sampling", "both"), default="scan")
    parser.add_argument("--num-scenes", type=int, default=1000)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-points", type=int, default=8000)
    parser.add_argument("--num-views", type=int, default=6)
    parser.add_argument("--tsdf-resolution", type=int, default=196)
    parser.add_argument("--normal-neighbors", type=int, default=24)
    parser.add_argument("--min-points-per-part", type=int, default=256)
    parser.add_argument("--min-segmentation-score", type=float, default=0.25)
    parser.add_argument("--max-parts", type=int, default=16)
    parser.add_argument(
        "--max-moving-joints",
        type=int,
        default=None,
        help=(
            "For serial objects, keep only the first N movable joints and "
            "rigidly merge every later link into the final retained part"
        ),
    )
    parser.add_argument("--workspace-size", type=float, default=0.3)
    parser.add_argument("--global-scaling", type=float, default=0.7)
    parser.add_argument("--pos-rot", type=int, choices=(0, 1), default=1)
    parser.add_argument("--canonical", action="store_true", default=True)
    parser.add_argument("--dense-photo", action="store_true", default=True)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument(
        "--is-syn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use synthetic-object URDF limits; defaults to object-set name detection",
    )
    parser.add_argument(
        "--allow-tree",
        action="store_true",
        help="Allow branching kinematic trees instead of requiring a serial chain",
    )
    parser.add_argument(
        "--reject-self-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resample joint states until the posed object has no overlapping "
            "parts (objects are loaded with collision geometry forced from "
            "visuals and URDF_USE_SELF_COLLISION)"
        ),
    )
    parser.add_argument(
        "--max-pose-attempts",
        type=int,
        default=20,
        help="Max joint-state resamples per pose when rejecting self-collisions",
    )
    parser.add_argument(
        "--collision-penetration-tol",
        type=float,
        default=1e-4,
        help="Penetration depth (meters) tolerated before a pose is rejected",
    )
    parser.add_argument(
        "--pose-separation-bias",
        type=float,
        default=3.0,
        help=(
            "Bias the start->end joint separation toward large moves. The "
            "per-sample separation level is u**(1/bias) for u~U(0,1): bias=1 is "
            "unbiased, larger values push the two poses further apart"
        ),
    )
    parser.add_argument(
        "--pose-separation-jitter",
        type=float,
        default=0.15,
        help="Per-joint stddev added to the separation level, for variety",
    )
    parser.add_argument(
        "--pose-separation-decay",
        type=float,
        default=0.8,
        help=(
            "Per-failed-attempt backoff factor for the end pose: when self-"
            "collision rejection keeps failing, the move shrinks toward the "
            "start pose by this factor until a feasible pose is found"
        ),
    )
    parser.add_argument(
        "--independent-poses",
        action="store_true",
        help="Sample both poses independently and uniformly (no separation bias)",
    )
    parser.add_argument(
        "--save-mesh-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Store per-visual world transforms (+ scale, geom type, dims, mesh "
            "file) for the start and end poses, so the full meshes can be "
            "reconstructed at each pose"
        ),
    )
    parser.add_argument("--save-ply", action="store_true")
    parser.add_argument("--axis-visual-length", type=float, default=0.75)
    parser.add_argument("--axis-visual-samples", type=int, default=160)
    parser.add_argument("--max-attempts-per-sample", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--restage",
        action="store_true",
        help="Re-extract archives and recompute bounding boxes even if staged",
    )
    parser.add_argument(
        "--bbox-pose-samples",
        type=int,
        default=16,
        help=(
            "Random joint configurations (plus rest + both extremes) unioned "
            "when synthesizing each object's bounding box, so the render scale "
            "keeps articulated objects in frame across poses. 0 = rest only"
        ),
    )
    args = parser.parse_args()

    args.sim_src = args.sim_src.expanduser().resolve()
    args.articraft_root = args.articraft_root.expanduser().resolve()
    args.root = args.root.expanduser().resolve()
    if args.staging_root is None:
        args.staging_root = args.root / "staging"
    args.staging_root = args.staging_root.expanduser().resolve()
    if args.is_syn is None:
        args.is_syn = "syn" in args.object_set.lower()

    if args.num_scenes < 1:
        parser.error("--num-scenes must be positive")
    if args.num_proc < 1:
        parser.error("--num-proc must be positive")
    if args.num_points < 1:
        parser.error("--num-points must be positive")
    if args.bbox_pose_samples < 0:
        parser.error("--bbox-pose-samples must be non-negative")
    if args.max_parts < 2:
        parser.error("--max-parts must be at least 2")
    if args.max_moving_joints is not None and args.max_moving_joints < 1:
        parser.error("--max-moving-joints must be positive")
    if args.max_moving_joints is not None and args.allow_tree:
        parser.error("--max-moving-joints requires serial kinematics")
    if args.max_pose_attempts < 1:
        parser.error("--max-pose-attempts must be positive")
    if args.pose_separation_bias <= 0:
        parser.error("--pose-separation-bias must be positive")
    if not 0.0 < args.pose_separation_decay <= 1.0:
        parser.error("--pose-separation-decay must be in (0, 1]")
    if args.sim_gui and args.num_proc != 1:
        parser.error("--sim-gui requires --num-proc 1")
    return args


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "samples").mkdir(parents=True, exist_ok=True)

    fit_dirs = collect_fit_dirs(
        args.articraft_root,
        args.category,
        args.fit_dir,
        args.all_categories,
        args.dataset_type
    )
    print(f"Staging objects from {len(fit_dirs)} 'fit' folder(s)...", flush=True)
    object_set_dir, staging_errors = stage_objects(
        fit_dirs,
        args.staging_root,
        args.object_set,
        args.sim_src,
        overwrite=args.restage,
        bbox_options={
            "is_syn": args.is_syn,
            "pos_rot": bool(args.pos_rot),
            "num_pose_samples": args.bbox_pose_samples,
        },
    )
    staged_count = len([p for p in object_set_dir.iterdir() if p.is_dir()])
    print(f"Staged {staged_count} objects into {object_set_dir}", flush=True)

    # Point the simulator at the staged object set.
    args.urdf_root = args.staging_root

    indices_per_worker = [
        list(range(worker_id, args.num_scenes, args.num_proc))
        for worker_id in range(args.num_proc)
    ]

    if args.num_proc == 1:
        errors = generate_worker(args, 0, indices_per_worker[0])
    else:
        context = mp.get_context("spawn")
        with context.Pool(args.num_proc) as pool:
            worker_errors = pool.starmap(
                generate_worker,
                [
                    (args, worker_id, indices)
                    for worker_id, indices in enumerate(indices_per_worker)
                ],
            )
        errors = [error for worker in worker_errors for error in worker]

    errors = list(staging_errors) + errors
    write_dataset_metadata(args, errors)
    generated = len(list((args.root / "samples").glob("*.npz")))
    print(f"Generated {generated} two-pose samples in {args.root}")
    if errors:
        raise SystemExit(f"{len(errors)} issues occurred during generation")


if __name__ == "__main__":
    main()
