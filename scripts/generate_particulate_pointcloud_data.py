#!/usr/bin/env python3
"""Generate point-cloud training samples for Particulate from the S2U simulator.

Each output ``.npz`` contains the fields produced by Particulate's
``cache_points.py`` plus targets consumed directly by ``PAT.forward``. Visual
meshes are not exported. Optional PLY files contain only colored object points
and sampled joint-axis points for inspection.

The S2U simulator uses the screw moment convention ``moment = origin x axis``.
Particulate uses Plucker coordinates ``[axis, axis x origin]``. This script
recovers the closest point on the axis as ``axis x moment``, applies the same
normalization as the point cloud, and rebuilds the Particulate Plucker moment.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REVOLUTE = 0
PRISMATIC = 1
FIXED = 4
SUPPORTED_MOVABLE_TYPES = (REVOLUTE, PRISMATIC)

PART_COLORS = np.asarray(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [174, 199, 232],
        [255, 187, 120],
        [152, 223, 138],
        [255, 152, 150],
        [197, 176, 213],
        [196, 156, 148],
    ],
    dtype=np.uint8,
)


class SampleGenerationError(RuntimeError):
    """Raised when a simulator scene cannot form a valid training sample."""


def setup_simulation_import(sim_src: Path) -> None:
    sim_src = sim_src.expanduser().resolve()
    if str(sim_src) not in sys.path:
        sys.path.insert(0, str(sim_src))


def get_simulation_class(sim_src: Path):
    setup_simulation_import(sim_src)
    from s2u.simulation import ArticulatedObjectManipulationSim

    return ArticulatedObjectManipulationSim


def normalized_axis(axis: Sequence[float]) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis_array)
    if norm < 1e-10:
        raise SampleGenerationError("Joint axis has near-zero length")
    return axis_array / norm


def simulator_moment_to_axis_origin(
    axis: Sequence[float],
    simulator_moment: Sequence[float],
) -> np.ndarray:
    """Recover the closest world-origin point on a screw axis.

    S2U stores ``moment = point x axis``. For a unit axis, ``axis x moment`` is
    the point on the line closest to the coordinate-system origin.
    """

    unit_axis = normalized_axis(axis)
    moment = np.asarray(simulator_moment, dtype=np.float64)
    return np.cross(unit_axis, moment)


def axis_point_to_plucker(
    axis: Sequence[float],
    point: Sequence[float],
) -> np.ndarray:
    unit_axis = normalized_axis(axis)
    point_array = np.asarray(point, dtype=np.float64)
    return np.concatenate([unit_axis, np.cross(unit_axis, point_array)])


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise SampleGenerationError("Expected a non-empty point cloud shaped (N, 3)")

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    longest_side = float((bbox_max - bbox_min).max())
    if longest_side < 1e-10:
        raise SampleGenerationError("Point cloud bounding box is degenerate")

    scale = 1.0 / longest_side
    return (points - center) * scale, center, scale


def get_joint_limits(
    joint_data: Tuple[Any, ...],
    is_syn: bool,
    pos_rot: bool,
) -> Tuple[float, float]:
    joint_type = int(joint_data[2])
    if joint_type == REVOLUTE and not is_syn:
        if pos_rot:
            return 0.0, float(np.pi / 2.0)
        return float(-np.pi / 4.0), float(np.pi / 4.0)
    return float(joint_data[8]), float(joint_data[9])


def get_all_joint_info(sim: Any) -> Dict[int, Tuple[Any, ...]]:
    bullet = sim.world.p
    count = bullet.getNumJoints(sim.object.uid)
    return {
        joint_id: bullet.getJointInfo(sim.object.uid, joint_id)
        for joint_id in range(count)
    }


def object_frame_source_link_ids(
    all_joint_info: Mapping[int, Tuple[Any, ...]],
    ordered_owners: Sequence[int],
) -> List[int]:
    """Choose the physical link frame representing each merged rigid part.

    Verified URDFs can contain a geometry-free ``articraft_global`` root whose
    fixed child is the original object base.  For that explicit wrapper, use
    the child frame as the base-part/NOCS frame while retaining ``-1`` as the
    rigid part owner.  Ordinary URDFs keep their previous owner-frame mapping.
    """
    source_ids = list(ordered_owners)
    if not source_ids or source_ids[0] != -1:
        return source_ids
    for joint_id, joint_data in all_joint_info.items():
        joint_name = joint_data[1]
        if isinstance(joint_name, bytes):
            joint_name = joint_name.decode("utf-8", errors="replace")
        if (
            int(joint_data[2]) == FIXED
            and int(joint_data[-1]) == -1
            and str(joint_name).startswith("articraft_global_to_base")
        ):
            source_ids[0] = int(joint_id)
            break
    return source_ids


def build_part_structure(
    all_joint_info: Mapping[int, Tuple[Any, ...]],
    max_parts: int,
    require_serial: bool,
) -> Dict[str, Any]:
    """Merge fixed links and map PyBullet link IDs to contiguous part IDs."""

    owner_cache: Dict[int, int] = {-1: -1}

    def rigid_owner(link_id: int) -> int:
        if link_id in owner_cache:
            return owner_cache[link_id]
        if link_id not in all_joint_info:
            raise SampleGenerationError(f"Unknown PyBullet link ID {link_id}")

        joint_data = all_joint_info[link_id]
        joint_type = int(joint_data[2])
        parent_link = int(joint_data[-1])
        if joint_type in SUPPORTED_MOVABLE_TYPES:
            owner = link_id
        elif joint_type == FIXED:
            owner = rigid_owner(parent_link)
        else:
            raise SampleGenerationError(
                f"Unsupported PyBullet joint type {joint_type} at joint {link_id}"
            )
        owner_cache[link_id] = owner
        return owner

    source_link_to_owner = {-1: -1}
    for link_id in sorted(all_joint_info):
        source_link_to_owner[link_id] = rigid_owner(link_id)

    owner_children: Dict[int, List[int]] = {}
    owner_parent: Dict[int, int] = {}
    for joint_id, joint_data in all_joint_info.items():
        if int(joint_data[2]) not in SUPPORTED_MOVABLE_TYPES:
            continue
        child_owner = rigid_owner(joint_id)
        parent_owner = rigid_owner(int(joint_data[-1]))
        if child_owner == parent_owner:
            raise SampleGenerationError(
                f"Movable joint {joint_id} collapsed into its parent rigid part"
            )
        if child_owner in owner_parent and owner_parent[child_owner] != parent_owner:
            raise SampleGenerationError(
                f"Part owned by link {child_owner} has multiple parents"
            )
        owner_parent[child_owner] = parent_owner
        owner_children.setdefault(parent_owner, []).append(child_owner)

    ordered_owners: List[int] = []

    def visit(owner: int) -> None:
        if owner in ordered_owners:
            raise SampleGenerationError("Kinematic graph contains a cycle")
        ordered_owners.append(owner)
        for child in sorted(set(owner_children.get(owner, []))):
            visit(child)

    visit(-1)
    expected_owners = {-1}
    expected_owners.update(
        rigid_owner(joint_id)
        for joint_id, data in all_joint_info.items()
        if int(data[2]) in SUPPORTED_MOVABLE_TYPES
    )
    if set(ordered_owners) != expected_owners:
        missing = sorted(expected_owners - set(ordered_owners))
        raise SampleGenerationError(
            f"Kinematic graph is disconnected from the base; missing owners {missing}"
        )

    if len(ordered_owners) > max_parts:
        raise SampleGenerationError(
            f"Object has {len(ordered_owners)} parts after fixed-link merging; "
            f"maximum is {max_parts}"
        )

    if require_serial:
        branching = {
            owner: children
            for owner, children in owner_children.items()
            if len(set(children)) > 1
        }
        if branching:
            raise SampleGenerationError(
                f"Expected serial kinematics, but found branching owners {branching}"
            )

    owner_to_part = {owner: part_id for part_id, owner in enumerate(ordered_owners)}
    source_link_to_part = {
        link_id: owner_to_part[owner]
        for link_id, owner in source_link_to_owner.items()
    }
    bone_structure = np.asarray(
        [
            (owner_to_part[parent], owner_to_part[child])
            for child, parent in owner_parent.items()
        ],
        dtype=np.int16,
    )
    if len(bone_structure):
        bone_structure = bone_structure[np.argsort(bone_structure[:, 1])]
    else:
        bone_structure = np.empty((0, 2), dtype=np.int16)

    if len(ordered_owners) > 1 and len(bone_structure) != len(ordered_owners) - 1:
        raise SampleGenerationError("Merged kinematic structure is not a tree")

    part_source_links = {
        part_id: sorted(
            link_id
            for link_id, mapped_part_id in source_link_to_part.items()
            if mapped_part_id == part_id
        )
        for part_id in range(len(ordered_owners))
    }
    return {
        "ordered_owners": ordered_owners,
        "frame_source_link_ids": object_frame_source_link_ids(
            all_joint_info,
            ordered_owners,
        ),
        "owner_to_part": owner_to_part,
        "source_link_to_part": source_link_to_part,
        "part_source_links": part_source_links,
        "bone_structure": bone_structure,
        "active_joint_ids": ordered_owners[1:],
        "merged_joint_ids": [],
    }


def limit_moving_joints(
    all_joint_info: Mapping[int, Tuple[Any, ...]],
    structure: Mapping[str, Any],
    max_moving_joints: int,
    max_parts: int,
) -> Dict[str, Any]:
    """Keep the first movable joints and rigidly merge the remaining tail."""

    movable_order = list(structure["ordered_owners"][1:])
    active_joint_ids = movable_order[:max_moving_joints]
    merged_joint_ids = movable_order[max_moving_joints:]
    if not merged_joint_ids:
        if len(structure["ordered_owners"]) > max_parts:
            raise SampleGenerationError(
                f"Keeping {len(active_joint_ids)} movable joints produces "
                f"{len(structure['ordered_owners'])} parts, exceeding "
                f"maximum {max_parts}"
            )
        limited = dict(structure)
        limited["active_joint_ids"] = active_joint_ids
        limited["merged_joint_ids"] = []
        return limited

    active_joint_set = set(active_joint_ids)
    owner_cache: Dict[int, int] = {-1: -1}

    def retained_owner(link_id: int) -> int:
        if link_id in owner_cache:
            return owner_cache[link_id]
        if link_id not in all_joint_info:
            raise SampleGenerationError(f"Unknown PyBullet link ID {link_id}")

        joint_data = all_joint_info[link_id]
        parent_link = int(joint_data[-1])
        if (
            int(joint_data[2]) in SUPPORTED_MOVABLE_TYPES
            and link_id in active_joint_set
        ):
            owner = link_id
        else:
            owner = retained_owner(parent_link)
        owner_cache[link_id] = owner
        return owner

    source_link_to_owner = {-1: -1}
    for link_id in sorted(all_joint_info):
        source_link_to_owner[link_id] = retained_owner(link_id)

    ordered_owners = [-1, *active_joint_ids]
    if len(ordered_owners) > max_parts:
        raise SampleGenerationError(
            f"Keeping {len(active_joint_ids)} movable joints produces "
            f"{len(ordered_owners)} parts, exceeding maximum {max_parts}"
        )

    owner_to_part = {
        owner: part_id
        for part_id, owner in enumerate(ordered_owners)
    }
    source_link_to_part = {
        link_id: owner_to_part[owner]
        for link_id, owner in source_link_to_owner.items()
    }
    bone_structure = np.asarray(
        [
            (
                owner_to_part[retained_owner(int(all_joint_info[joint_id][-1]))],
                owner_to_part[joint_id],
            )
            for joint_id in active_joint_ids
        ],
        dtype=np.int16,
    ).reshape(-1, 2)
    if len(bone_structure):
        bone_structure = bone_structure[np.argsort(bone_structure[:, 1])]

    part_source_links = {
        part_id: sorted(
            link_id
            for link_id, mapped_part_id in source_link_to_part.items()
            if mapped_part_id == part_id
        )
        for part_id in range(len(ordered_owners))
    }
    return {
        "ordered_owners": ordered_owners,
        "frame_source_link_ids": object_frame_source_link_ids(
            all_joint_info,
            ordered_owners,
        ),
        "owner_to_part": owner_to_part,
        "source_link_to_part": source_link_to_part,
        "part_source_links": part_source_links,
        "bone_structure": bone_structure,
        "active_joint_ids": active_joint_ids,
        "merged_joint_ids": merged_joint_ids,
    }


def sample_joint_states(
    sim: Any,
    all_joint_info: Mapping[int, Tuple[Any, ...]],
    active_joint_ids: Sequence[int],
    rng: np.random.Generator,
    is_syn: bool,
    pos_rot: bool,
    random_state: bool,
) -> Dict[int, float]:
    states: Dict[int, float] = {}
    active_joint_set = set(active_joint_ids)
    for joint_id, joint_data in all_joint_info.items():
        joint_type = int(joint_data[2])
        if (
            joint_type not in SUPPORTED_MOVABLE_TYPES
            or joint_id not in active_joint_set
        ):
            continue

        lower, upper = get_joint_limits(joint_data, is_syn, pos_rot)
        initial_state = float(joint_data[10])
        if random_state and np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            state = float(rng.uniform(lower, upper))
        else:
            state = initial_state
            state = 0.0
        sim.set_joint_state(joint_id, state)
        states[joint_id] = state
    return states


def render_viewpoint_extrinsics(sim: Any, num_views: int) -> List[Any]:
    from s2u.perception import camera_on_sphere
    from s2u.utils.transform import Rotation, Transform

    origin = Transform(
        Rotation.identity(),
        np.r_[sim.size / 2.0, sim.size / 2.0, sim.size / 2.0],
    )
    radius = 1.2 * sim.size
    if sim.dense_photo:
        extrinsics = []
        for offset, theta in zip((0, 1, 2), (np.pi / 8, np.pi / 4, np.pi / 2)):
            phi_values = (
                2.0 * np.pi * np.arange(num_views) / num_views
                + offset * 2.0 * np.pi / (num_views * 3)
            )
            extrinsics.extend(
                camera_on_sphere(origin, radius, theta, phi)
                for phi in phi_values
            )
        return extrinsics

    theta = np.pi / 4.0
    phi_values = 2.0 * np.pi * np.arange(num_views) / num_views
    return [
        camera_on_sphere(origin, radius, theta, phi)
        for phi in phi_values
    ]


def acquire_part_labeled_point_cloud(
    sim: Any,
    part_source_links: Mapping[int, Sequence[int]],
    num_views: int,
    tsdf_resolution: int,
    min_segmentation_score: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate one common TSDF geometry with a binary color channel per part."""

    import pybullet
    import open3d as o3d
    from scipy.spatial import cKDTree
    from s2u.perception import TSDFVolume

    part_ids = sorted(part_source_links)
    volumes = {
        part_id: TSDFVolume(
            sim.size,
            tsdf_resolution,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        for part_id in part_ids
    }

    for extrinsic in render_viewpoint_extrinsics(sim, num_views):
        _, depth, (seg_uid, seg_link) = sim.camera.render(
            extrinsic,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )
        object_mask = (seg_uid + 1).astype(bool)
        for part_id in part_ids:
            source_links = np.asarray(part_source_links[part_id], dtype=np.int32)
            part_mask = object_mask & np.isin(seg_link, source_links)
            color = np.repeat(
                (part_mask.astype(np.uint8) * 255)[..., None],
                3,
                axis=-1,
            )
            volumes[part_id].integrate_rgb(
                depth,
                color,
                sim.camera.intrinsic,
                extrinsic,
            )

    clouds = {
        part_id: volumes[part_id].get_cloud()
        for part_id in part_ids
    }
    reference_cloud = clouds[part_ids[0]]
    points = np.asarray(reference_cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise SampleGenerationError("TSDF reconstruction produced no points")

    if reference_cloud.has_normals():
        normals = np.asarray(reference_cloud.normals, dtype=np.float64)
    else:
        normals = np.empty((0, 3), dtype=np.float64)

    scores = np.zeros((len(part_ids), len(points)), dtype=np.float32)
    for score_row, part_id in enumerate(part_ids):
        cloud = clouds[part_id]
        part_points = np.asarray(cloud.points, dtype=np.float64)
        colors = np.asarray(cloud.colors, dtype=np.float32)
        if len(part_points) == 0 or len(colors) == 0:
            continue
        intensity = colors.mean(axis=1)

        if part_points.shape == points.shape and np.allclose(
            part_points,
            points,
            atol=1e-7,
            rtol=0.0,
        ):
            scores[score_row] = intensity
        else:
            distances, nearest = cKDTree(part_points).query(points, k=1)
            aligned = intensity[nearest]
            aligned[distances > (2.5 * sim.size / tsdf_resolution)] = 0.0
            scores[score_row] = aligned

    labels = np.argmax(scores, axis=0).astype(np.int16)
    confidence = scores.max(axis=0)
    keep = confidence >= min_segmentation_score
    if not np.any(keep):
        raise SampleGenerationError(
            "No reconstructed points passed the segmentation confidence threshold"
        )

    points = points[keep]
    labels = labels[keep]
    if len(normals) == len(keep):
        normals = normals[keep]
    else:
        normals = np.empty((0, 3), dtype=np.float64)
    return points, labels, normals


def stratified_sample(
    points: np.ndarray,
    labels: np.ndarray,
    normals: np.ndarray,
    num_points: int,
    min_points_per_part: int,
    num_parts: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected: List[np.ndarray] = []
    per_part_minimum = min(min_points_per_part, num_points // num_parts)

    for part_id in range(num_parts):
        candidates = np.flatnonzero(labels == part_id)
        if len(candidates) == 0:
            raise SampleGenerationError(
                f"Part {part_id} has no reconstructed surface points"
            )
        selected.append(
            rng.choice(
                candidates,
                size=per_part_minimum,
                replace=len(candidates) < per_part_minimum,
            )
        )

    selected_indices = np.concatenate(selected)
    remaining_count = num_points - len(selected_indices)
    if remaining_count > 0:
        available_mask = np.ones(len(points), dtype=bool)
        available_mask[selected_indices] = False
        available = np.flatnonzero(available_mask)
        if len(available) >= remaining_count:
            remaining = rng.choice(available, size=remaining_count, replace=False)
        else:
            extra_source = available if len(available) else np.arange(len(points))
            first = available
            extra = rng.choice(
                extra_source,
                size=remaining_count - len(first),
                replace=True,
            )
            remaining = np.concatenate([first, extra])
        selected_indices = np.concatenate([selected_indices, remaining])

    rng.shuffle(selected_indices)
    sampled_normals = (
        normals[selected_indices]
        if len(normals) == len(points)
        else np.empty((0, 3), dtype=np.float64)
    )
    return (
        points[selected_indices],
        labels[selected_indices],
        sampled_normals,
    )


def estimate_normals(
    points: np.ndarray,
    num_neighbors: int,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    if len(points) < 4:
        raise SampleGenerationError("At least four points are needed for normals")
    k = min(max(num_neighbors, 3), len(points))
    _, neighbor_indices = cKDTree(points).query(points, k=k)
    neighborhoods = points[neighbor_indices]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    _, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]

    outward = points - points.mean(axis=0, keepdims=True)
    flip = np.einsum("ni,ni->n", normals, outward) < 0
    normals[flip] *= -1.0
    return normals


def sanitize_normals(
    points: np.ndarray,
    normals: np.ndarray,
    num_neighbors: int,
) -> np.ndarray:
    if normals.shape != points.shape:
        return estimate_normals(points, num_neighbors)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-8
    if not np.all(valid):
        return estimate_normals(points, num_neighbors)
    normals = normals / lengths

    outward = points - points.mean(axis=0, keepdims=True)
    flip = np.einsum("ni,ni->n", normals, outward) < 0
    normals[flip] *= -1.0
    return normals


def build_motion_targets(
    sim: Any,
    all_joint_info: Mapping[int, Tuple[Any, ...]],
    structure: Mapping[str, Any],
    states: Mapping[int, float],
    center: np.ndarray,
    scale: float,
    is_syn: bool,
    pos_rot: bool,
) -> Dict[str, np.ndarray]:
    num_parts = len(structure["ordered_owners"])
    axes_plucker = np.zeros((num_parts, 12), dtype=np.float32)
    ranges = np.zeros((num_parts, 4), dtype=np.float32)
    motion_class = np.zeros(num_parts, dtype=np.int8)
    joint_origins = np.zeros((num_parts, 3), dtype=np.float32)
    moment_axis_points = np.zeros((num_parts, 3), dtype=np.float32)
    joint_origins_world = np.zeros((num_parts, 3), dtype=np.float32)
    joint_axes_world = np.zeros((num_parts, 3), dtype=np.float32)
    simulator_moments_world = np.zeros((num_parts, 3), dtype=np.float32)
    joint_states = np.zeros(num_parts, dtype=np.float32)
    joint_limits = np.zeros((num_parts, 2), dtype=np.float32)
    source_joint_ids = np.full(num_parts, -1, dtype=np.int16)

    owner_to_part = structure["owner_to_part"]
    for joint_id, joint_data in all_joint_info.items():
        joint_type = int(joint_data[2])
        if (
            joint_type not in SUPPORTED_MOVABLE_TYPES
            or joint_id not in owner_to_part
        ):
            continue

        part_id = owner_to_part[joint_id]
        axis_world, simulator_moment = sim.get_joint_screw(joint_id)
        axis_world = normalized_axis(axis_world)
        origin_world = simulator_moment_to_axis_origin(
            axis_world,
            simulator_moment,
        )
        moment_axis_point = (origin_world - center) * scale
        plucker = axis_point_to_plucker(axis_world, moment_axis_point)
        origin_normalized = np.cross(plucker[3:], plucker[:3])

        lower, upper = get_joint_limits(joint_data, is_syn, pos_rot)
        state = float(states[joint_id])
        source_joint_ids[part_id] = joint_id
        joint_origins[part_id] = origin_normalized
        moment_axis_points[part_id] = moment_axis_point
        joint_origins_world[part_id] = origin_world
        joint_axes_world[part_id] = axis_world
        simulator_moments_world[part_id] = simulator_moment
        joint_states[part_id] = state
        joint_limits[part_id] = (lower, upper)

        if joint_type == REVOLUTE:
            axes_plucker[part_id, :6] = plucker
            ranges[part_id, :2] = (lower - state, upper - state)
            motion_class[part_id] = 1
        else:
            axes_plucker[part_id, 6:] = plucker
            ranges[part_id, 2:] = (
                (lower - state) * scale,
                (upper - state) * scale,
            )
            motion_class[part_id] = 2

    return {
        "link_axes_plucker": axes_plucker,
        "link_range": ranges,
        "part_motion_class": motion_class,
        "joint_origins": joint_origins,
        "moment_axis_points": moment_axis_points,
        "joint_origins_world": joint_origins_world,
        "joint_axes_world": joint_axes_world,
        "simulator_moments_world": simulator_moments_world,
        "joint_state": joint_states,
        "joint_limits": joint_limits,
        "source_joint_ids": source_joint_ids,
    }


def closest_points_on_revolute_axes(
    points: np.ndarray,
    part_ids: np.ndarray,
    axes_plucker: np.ndarray,
    motion_class: np.ndarray,
) -> np.ndarray:
    targets = np.zeros_like(points, dtype=np.float32)
    for part_id in range(len(motion_class)):
        if motion_class[part_id] not in (1, 3):
            continue
        point_mask = part_ids == part_id
        if not np.any(point_mask):
            continue

        axis = normalized_axis(axes_plucker[part_id, :3])
        moment = axes_plucker[part_id, 3:6]
        origin = np.cross(moment, axis)
        offsets = points[point_mask] - origin
        targets[point_mask] = (
            origin + np.outer(offsets @ axis, axis)
        ).astype(np.float32)
    return targets


def create_training_sample(
    sim: Any,
    args: argparse.Namespace,
    rng: np.random.Generator,
    sample_index: int,
) -> Dict[str, np.ndarray]:
    all_joint_info = get_all_joint_info(sim)
    structure = build_part_structure(
        all_joint_info,
        max_parts=(
            len(all_joint_info) + 1
            if args.max_moving_joints is not None
            else args.max_parts
        ),
        require_serial=not args.allow_tree,
    )
    if args.max_moving_joints is not None:
        structure = limit_moving_joints(
            all_joint_info,
            structure,
            max_moving_joints=args.max_moving_joints,
            max_parts=args.max_parts,
        )
    num_parts = len(structure["ordered_owners"])
    if num_parts < 2:
        raise SampleGenerationError("Particulate training requires at least two parts")

    states = sample_joint_states(
        sim,
        all_joint_info,
        structure["active_joint_ids"],
        rng,
        is_syn=args.is_syn,
        pos_rot=bool(args.pos_rot),
        random_state=args.rand_state,
    )
    points_world, part_ids, normals = acquire_part_labeled_point_cloud(
        sim,
        structure["part_source_links"],
        num_views=args.num_views,
        tsdf_resolution=args.tsdf_resolution,
        min_segmentation_score=args.min_segmentation_score,
    )
    points_world, part_ids, normals = stratified_sample(
        points_world,
        part_ids,
        normals,
        num_points=args.num_points,
        min_points_per_part=args.min_points_per_part,
        num_parts=num_parts,
        rng=rng,
    )
    points, center, scale = normalize_points(points_world)
    normals = sanitize_normals(points, normals, args.normal_neighbors)

    motion = build_motion_targets(
        sim,
        all_joint_info,
        structure,
        states,
        center,
        scale,
        is_syn=args.is_syn,
        pos_rot=bool(args.pos_rot),
    )
    bone_structure = structure["bone_structure"]
    part_structure_matrix = np.zeros((num_parts, num_parts), dtype=np.bool_)
    if len(bone_structure):
        part_structure_matrix[
            bone_structure[:, 0],
            bone_structure[:, 1],
        ] = True

    axes_plucker = motion["link_axes_plucker"]
    ranges = motion["link_range"]
    motion_class = motion["part_motion_class"]
    closest_points = closest_points_on_revolute_axes(
        points,
        part_ids,
        axes_plucker,
        motion_class,
    )

    source_link_to_part = np.asarray(
        [
            structure["source_link_to_part"][link_id]
            for link_id in [-1, *sorted(all_joint_info)]
        ],
        dtype=np.int16,
    )
    object_path = str(sim.object_urdfs[sim.object_idx])
    sample = {
        # Native Particulate cached training fields.
        "points": points.astype(np.float32),
        "normals": normals.astype(np.float32),
        "point_to_bone": part_ids.astype(np.int16),
        "point_from_sharp": np.zeros(len(points), dtype=np.bool_),
        "bone_structure": bone_structure,
        "link_axes_plucker": axes_plucker,
        "link_range": ranges,
        # Direct PAT.forward targets.
        "part_ids": part_ids.astype(np.int16),
        "num_valid_parts": np.asarray(num_parts, dtype=np.int16),
        "part_structure_matrix": part_structure_matrix,
        "gt_part_motion_class": motion_class,
        "gt_revolute_plucker": axes_plucker[:, :6],
        "gt_revolute_range": ranges[:, :2],
        "gt_prismatic_axis": axes_plucker[:, 6:9],
        "gt_prismatic_range": ranges[:, 2:],
        "gt_closest_point_on_axis": closest_points,
        "motion_hierarchy": bone_structure,
        "is_part_revolute": np.isin(motion_class, (1, 3)),
        "is_part_prismatic": np.isin(motion_class, (2, 3)),
        # Traceability and visualization metadata.
        "joint_origins": motion["joint_origins"],
        "moment_axis_points": motion["moment_axis_points"],
        "joint_origins_world": motion["joint_origins_world"],
        "joint_axes_world": motion["joint_axes_world"],
        "simulator_moments_world": motion["simulator_moments_world"],
        "joint_state": motion["joint_state"],
        "joint_limits": motion["joint_limits"],
        "source_joint_ids": motion["source_joint_ids"],
        "source_link_to_part": source_link_to_part,
        "active_source_joint_ids": np.asarray(
            structure["active_joint_ids"],
            dtype=np.int16,
        ),
        "merged_source_joint_ids": np.asarray(
            structure["merged_joint_ids"],
            dtype=np.int16,
        ),
        "normalization_center": center.astype(np.float32),
        "normalization_scale": np.asarray(scale, dtype=np.float32),
        "sample_index": np.asarray(sample_index, dtype=np.int64),
        "object_path": np.asarray(object_path),
    }
    validate_sample(sample, args.max_parts)
    return sample


def validate_sample(sample: Mapping[str, np.ndarray], max_parts: int) -> None:
    points = sample["points"]
    normals = sample["normals"]
    part_ids = sample["part_ids"]
    num_parts = int(sample["num_valid_parts"])
    hierarchy = sample["bone_structure"]

    if points.ndim != 2 or points.shape[1] != 3:
        raise SampleGenerationError("points must have shape (N, 3)")
    if normals.shape != points.shape:
        raise SampleGenerationError("normals must have the same shape as points")
    if part_ids.shape != (len(points),):
        raise SampleGenerationError("part_ids must have shape (N,)")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise SampleGenerationError("points or normals contain non-finite values")
    if not (2 <= num_parts <= max_parts):
        raise SampleGenerationError(f"Invalid number of parts: {num_parts}")
    if not np.array_equal(np.unique(part_ids), np.arange(num_parts)):
        raise SampleGenerationError("Every contiguous part ID must own at least one point")
    if hierarchy.shape != (num_parts - 1, 2):
        raise SampleGenerationError(
            f"Expected hierarchy shape {(num_parts - 1, 2)}, got {hierarchy.shape}"
        )
    if sample["link_axes_plucker"].shape != (num_parts, 12):
        raise SampleGenerationError("link_axes_plucker has the wrong shape")
    if sample["link_range"].shape != (num_parts, 4):
        raise SampleGenerationError("link_range has the wrong shape")
    if sample["gt_closest_point_on_axis"].shape != points.shape:
        raise SampleGenerationError("gt_closest_point_on_axis has the wrong shape")
    if not np.isfinite(sample["link_axes_plucker"]).all():
        raise SampleGenerationError("link_axes_plucker contains non-finite values")
    if not np.isfinite(sample["link_range"]).all():
        raise SampleGenerationError("link_range contains non-finite values")
    if np.any(sample["gt_part_motion_class"][1:] == 0):
        raise SampleGenerationError("Every non-root serial part must have a moving joint")


def write_binary_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(points), dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as output:
        output.write(header)
        vertices.tofile(output)


def write_visualization(
    path: Path,
    sample: Mapping[str, np.ndarray],
    axis_length: float,
    axis_samples: int,
) -> None:
    points = sample["points"]
    part_ids = sample["part_ids"]
    colors = PART_COLORS[part_ids % len(PART_COLORS)]
    vis_points = [points]
    vis_colors = [colors]

    motion_class = sample["gt_part_motion_class"]
    origins = sample["joint_origins"]
    axes_plucker = sample["link_axes_plucker"]
    t_values = np.linspace(
        -axis_length * 0.5,
        axis_length * 0.5,
        axis_samples,
    )
    for part_id in range(int(sample["num_valid_parts"])):
        if motion_class[part_id] == 0:
            continue
        if motion_class[part_id] in (1, 3):
            axis = normalized_axis(axes_plucker[part_id, :3])
            axis_color = np.array([255, 40, 40], dtype=np.uint8)
        else:
            axis = normalized_axis(axes_plucker[part_id, 6:9])
            axis_color = np.array([40, 100, 255], dtype=np.uint8)
        axis_points = origins[part_id] + np.outer(t_values, axis)
        vis_points.append(axis_points)
        vis_colors.append(np.repeat(axis_color[None, :], axis_samples, axis=0))

        origin_offsets = np.eye(3, dtype=np.float32) * 0.008
        origin_points = np.concatenate(
            [origins[part_id] + origin_offsets, origins[part_id] - origin_offsets],
            axis=0,
        )
        vis_points.append(origin_points)
        vis_colors.append(
            np.repeat(
                np.array([[255, 255, 255]], dtype=np.uint8),
                len(origin_points),
                axis=0,
            )
        )

    write_binary_ply(
        path,
        np.concatenate(vis_points, axis=0).astype(np.float32),
        np.concatenate(vis_colors, axis=0).astype(np.uint8),
    )


def schema_description() -> Dict[str, str]:
    return {
        "points": "(N,3) float32 normalized point coordinates",
        "normals": "(N,3) float32 normalized surface normals",
        "point_to_bone": "(N,) int16 contiguous part labels",
        "point_from_sharp": "(N,) bool; always false for scan point clouds",
        "bone_structure": "(P-1,2) int16 directed parent-child edges",
        "link_axes_plucker": "(P,12) float32; revolute [0:6], prismatic [6:12]",
        "link_range": "(P,4) float32; revolute [0:2], prismatic [2:4]",
        "part_structure_matrix": "(P,P) bool directed adjacency matrix",
        "gt_part_motion_class": "(P,) int8; 0 fixed, 1 revolute, 2 prismatic",
        "gt_closest_point_on_axis": "(N,3) float32 per-point revolute target",
        "joint_origins": "(P,3) float32 closest normalized points on joint axes",
        "moment_axis_points": (
            "(P,3) float32 normalized points recovered directly from S2U moments"
        ),
        "simulator_moments_world": "(P,3) float32 S2U point-x-axis moments",
        "active_source_joint_ids": (
            "(P-1,) int16 movable source joints retained as Particulate joints"
        ),
        "merged_source_joint_ids": (
            "(K,) int16 downstream source joints rigidly merged into the last part"
        ),
        "normalization_center": "(3,) float32 world-space bounding-box center",
        "normalization_scale": "scalar float32 reciprocal longest bounding-box side",
    }


def generate_worker(
    args: argparse.Namespace,
    worker_id: int,
    sample_indices: Sequence[int],
) -> List[str]:
    simulation_class = get_simulation_class(args.sim_src)
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
            f"No URDFs found for object set '{args.object_set}' under "
            f"{args.urdf_root}. Include the split suffix when needed, for "
            "example 'syn/drawer/train'."
        )
    if len(sim.object_urdfs) != len(sim.object_bbox):
        raise SampleGenerationError(
            "Every discovered URDF must have a matching bounding_box.json"
        )

    one_per_object = getattr(args, "one_per_object", False)
    if one_per_object:
        # Simulation discovery follows filesystem iteration order. Sort the URDF
        # and bounding-box pairs together so sample index i always selects the
        # same object, including when indices are distributed across workers.
        object_pairs = sorted(
            zip(sim.object_urdfs, sim.object_bbox),
            key=lambda pair: str(pair[0]),
        )
        sim.object_urdfs = [pair[0] for pair in object_pairs]
        sim.object_bbox = [pair[1] for pair in object_pairs]

    errors: List[str] = []
    for position, sample_index in enumerate(sample_indices):
        output_path = args.root / "samples" / f"{sample_index:08d}.npz"
        if output_path.exists() and not args.overwrite:
            continue

        last_error: Exception | None = None
        for _ in range(args.max_attempts_per_sample):
            try:
                object_index = sample_index if one_per_object else None
                sim.reset(index=object_index, canonical=args.canonical)
                sample = create_training_sample(sim, args, rng, sample_index)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output_path, **sample)
                if args.save_ply:
                    write_visualization(
                        args.root / "visualizations" / f"{sample_index:08d}.ply",
                        sample,
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


def write_dataset_metadata(args: argparse.Namespace, errors: Iterable[str]) -> None:
    args.root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": schema_description(),
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
            "Generate Particulate-compatible articulated point-cloud samples "
            "using the S2U PyBullet simulator."
        )
    )
    parser.add_argument("root", type=Path, help="Output dataset directory")
    parser.add_argument("--object-set", required=True)
    parser.add_argument(
        "--sim-src",
        type=Path,
        default=Path("~/Articulated_object_simulation-main/src"),
    )
    parser.add_argument(
        "--urdf-root",
        type=Path,
        default=Path("~/Articulated_object_simulation-main/data/urdfs"),
    )
    parser.add_argument("--num-scenes", type=int, default=1000)
    parser.add_argument(
        "--one-per-object",
        action="store_true",
        help=(
            "Generate exactly one sample for every discovered URDF, in sorted "
            "path order; overrides --num-scenes and disables random object "
            "selection"
        ),
    )
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
            "rigidly merge every later link into the final retained part; "
            "use 3 for at most 3 joints and 4 parts"
        ),
    )
    parser.add_argument("--workspace-size", type=float, default=0.3)
    parser.add_argument("--global-scaling", type=float, default=0.7)
    parser.add_argument("--pos-rot", type=int, choices=(0, 1), default=1)
    parser.add_argument("--rand-state", action="store_true")
    parser.add_argument("--canonical", action="store_true", default= True)
    parser.add_argument("--dense-photo", action="store_true", default= True)
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
    parser.add_argument("--save-ply", action="store_true")
    parser.add_argument(
        "--save-mesh",
        action="store_true",
        help="Store per-part mesh vertices and faces in the output NPZ",
    )
    parser.add_argument("--axis-visual-length", type=float, default=0.75)
    parser.add_argument("--axis-visual-samples", type=int, default=160)
    parser.add_argument("--max-attempts-per-sample", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.sim_src = args.sim_src.expanduser().resolve()
    args.urdf_root = args.urdf_root.expanduser().resolve()
    args.root = args.root.expanduser().resolve()
    if args.is_syn is None:
        args.is_syn = "syn" in args.object_set.lower()
    if args.num_scenes < 1:
        parser.error("--num-scenes must be positive")
    if args.num_proc < 1:
        parser.error("--num-proc must be positive")
    if args.num_points < 1:
        parser.error("--num-points must be positive")
    if args.max_parts < 2:
        parser.error("--max-parts must be at least 2")
    if args.max_moving_joints is not None and args.max_moving_joints < 1:
        parser.error("--max-moving-joints must be positive")
    if args.max_moving_joints is not None and args.allow_tree:
        parser.error("--max-moving-joints requires serial kinematics")
    if args.sim_gui and args.num_proc != 1:
        parser.error("--sim-gui requires --num-proc 1")
    return args


def main() -> None:
    args = parse_args()
    if args.one_per_object:
        object_root = args.urdf_root / args.object_set
        if not object_root.is_dir():
            raise SampleGenerationError(
                f"Object set directory does not exist: {object_root}"
            )
        args.num_scenes = sum(
            1
            for object_dir in object_root.iterdir()
            if object_dir.is_dir()
            and any(path.suffix == ".urdf" for path in object_dir.iterdir())
        )
        if args.num_scenes == 0:
            raise SampleGenerationError(
                f"No URDF object directories found under {object_root}"
            )
        print(
            f"One-per-object mode: generating {args.num_scenes} sample(s) "
            "in sorted URDF order.",
            flush=True,
        )
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "samples").mkdir(parents=True, exist_ok=True)
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

    write_dataset_metadata(args, errors)
    generated = len(list((args.root / "samples").glob("*.npz")))
    print(f"Generated {generated} samples in {args.root}")
    if errors:
        raise SystemExit(f"{len(errors)} samples could not be generated")


if __name__ == "__main__":
    main()
