#!/usr/bin/env python3
"""Generate Particulate samples with world-aligned part/global NOCS targets.

This variant keeps the CLI and output schema of
``generate_particulate_pointcloud_nocs_data.py`` but changes the coordinate
representation used before NOCS normalization:

* ``nocs_p`` is computed from ``P_world - t_part``.
* ``nocs_g`` is computed from ``P_world - t_object``.

Compared with the regular NOCS generator, this removes inverse part/object
rotations from the canonical labels while still subtracting the relevant
translation. The resulting part and global canonical axes stay aligned to the
world axes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import generate_particulate_pointcloud_nocs_data as nocs_base


_BASE_SCHEMA_DESCRIPTION = nocs_base.schema_description


def translation_centered_part_points(
    points_world: np.ndarray,
    part_ids: np.ndarray,
    ordered_owners: Sequence[int],
    owner_transforms: Mapping[int, Any],
) -> np.ndarray:
    centered = np.zeros_like(points_world, dtype=np.float64)
    for part_id, owner_id in enumerate(ordered_owners):
        mask = part_ids == part_id
        if not np.any(mask):
            continue
        centered[mask] = (
            points_world[mask]
            - owner_transforms[owner_id].translation.reshape(1, 3)
        )
    return centered


def translation_centered_global_points(
    points_world: np.ndarray,
    object_transform: Any,
) -> np.ndarray:
    return points_world - object_transform.translation.reshape(1, 3)


def collect_world_aligned_reference_points(
    sim: Any,
    structure: Mapping[str, Any],
    owner_transforms: Mapping[int, Any],
    object_path: Path,
    mode: str,
) -> List[List[np.ndarray]]:
    from s2u.utils.saver import get_mesh_pose

    bullet = sim.world.p
    client_id = getattr(bullet, "_client", 0)
    num_parts = len(structure["ordered_owners"])
    points_by_part: List[List[np.ndarray]] = [[] for _ in range(num_parts)]
    object_translation = owner_transforms[-1].translation.reshape(1, 3)

    for visual in bullet.getVisualShapeData(sim.object.uid):
        source_link = int(visual[1])
        if source_link not in structure["source_link_to_part"]:
            continue

        part_id = int(structure["source_link_to_part"][source_link])
        geometry_points = nocs_base.visual_geometry_points(visual, object_path)
        if len(geometry_points) == 0:
            continue

        _, _, _, visual_world_transform = get_mesh_pose(visual, client_id)
        points_world = nocs_base.transform_points(
            visual_world_transform,
            geometry_points,
        )

        if mode == "part":
            owner_id = int(structure["ordered_owners"][part_id])
            points = (
                points_world
                - owner_transforms[owner_id].translation.reshape(1, 3)
            )
        elif mode == "global":
            points = points_world - object_translation
        else:
            raise ValueError(f"Unknown reference point mode: {mode}")
        points_by_part[part_id].append(points)

    return points_by_part


def build_world_aligned_nocs_targets(
    sim: Any,
    all_joint_info: Mapping[int, Tuple[Any, ...]],
    structure: Mapping[str, Any],
    points_world: np.ndarray,
    part_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    ordered_owners = list(structure["ordered_owners"])
    object_path = Path(sim.object_urdfs[sim.object_idx])
    current_transforms = nocs_base.get_owner_world_transforms(sim, ordered_owners)

    part_coords = translation_centered_part_points(
        points_world,
        part_ids,
        ordered_owners,
        current_transforms,
    )
    global_coords = translation_centered_global_points(
        points_world,
        current_transforms[-1],
    )

    part_reference = collect_world_aligned_reference_points(
        sim,
        structure,
        current_transforms,
        object_path,
        mode="part",
    )
    global_reference = collect_world_aligned_reference_points(
        sim,
        structure,
        current_transforms,
        object_path,
        mode="global",
    )

    bullet = sim.world.p
    state_id = bullet.saveState()
    try:
        nocs_base.reset_movable_joints_to_rest(sim, all_joint_info)
        rest_transforms = nocs_base.get_owner_world_transforms(sim, ordered_owners)
    finally:
        bullet.restoreState(state_id)
        bullet.removeState(state_id)

    nocs_p = np.zeros_like(points_world, dtype=np.float64)
    part_factors = np.zeros(len(ordered_owners), dtype=np.float32)
    part_bbox_min = np.zeros((len(ordered_owners), 3), dtype=np.float32)
    part_bbox_max = np.zeros((len(ordered_owners), 3), dtype=np.float32)
    for part_id in range(len(ordered_owners)):
        mask = part_ids == part_id
        reference_points = nocs_base.concatenate_or_none(part_reference[part_id])
        if reference_points is None:
            reference_points = part_coords[mask]
        factor, bbox_min, bbox_max = nocs_base.nocs_factor_and_corners(
            reference_points
        )
        nocs_p[mask] = nocs_base.normalize_nocs(
            part_coords[mask],
            factor,
            bbox_min,
            bbox_max,
        )
        part_factors[part_id] = factor
        part_bbox_min[part_id] = bbox_min.reshape(3)
        part_bbox_max[part_id] = bbox_max.reshape(3)

    global_reference_points = nocs_base.concatenate_or_none(
        [
            nocs_base.concatenate_or_none(part_chunks)
            for part_chunks in global_reference
        ]
    )
    if global_reference_points is None:
        global_reference_points = global_coords
    global_factor, global_bbox_min, global_bbox_max = (
        nocs_base.nocs_factor_and_corners(global_reference_points)
    )
    nocs_g = nocs_base.normalize_nocs(
        global_coords,
        global_factor,
        global_bbox_min,
        global_bbox_max,
    )

    return {
        "nocs_p": nocs_p.astype(np.float32),
        "nocs_g": nocs_g.astype(np.float32),
        "nocs_part_coords": part_coords.astype(np.float32),
        "nocs_global_coords": global_coords.astype(np.float32),
        "nocs_part_factor": part_factors,
        "nocs_part_bbox_min": part_bbox_min,
        "nocs_part_bbox_max": part_bbox_max,
        "nocs_global_factor": np.asarray(global_factor, dtype=np.float32),
        "nocs_global_bbox_min": global_bbox_min.reshape(3).astype(np.float32),
        "nocs_global_bbox_max": global_bbox_max.reshape(3).astype(np.float32),
        "urdf_frame_source_link_ids": np.asarray(ordered_owners, dtype=np.int16),
        "urdf_object_current_frame": nocs_base.transform_matrix(
            current_transforms[-1]
        ),
        "urdf_link_current_frames": nocs_base.transform_matrix_array(
            current_transforms,
            ordered_owners,
        ),
        "urdf_link_current_frames_object": nocs_base.relative_transform_matrix_array(
            current_transforms[-1],
            current_transforms,
            ordered_owners,
        ),
        "urdf_object_rest_frame": nocs_base.transform_matrix(rest_transforms[-1]),
        "urdf_link_rest_frames": nocs_base.transform_matrix_array(
            rest_transforms,
            ordered_owners,
        ),
        "urdf_link_rest_frames_object": nocs_base.relative_transform_matrix_array(
            rest_transforms[-1],
            rest_transforms,
            ordered_owners,
        ),
    }


def schema_description() -> Dict[str, str]:
    schema = _BASE_SCHEMA_DESCRIPTION()
    schema.update(
        {
            "nocs_p": (
                "(N,3) float32 world-aligned part NOCS coordinates computed "
                "from P_world - t_part"
            ),
            "nocs_g": (
                "(N,3) float32 world-aligned global NOCS coordinates computed "
                "from P_world - t_object"
            ),
            "nocs_part_coords": (
                "(N,3) float32 world-aligned coordinates before part NOCS "
                "normalization; equal to P_world - t_part"
            ),
            "nocs_global_coords": (
                "(N,3) float32 world-aligned coordinates before global NOCS "
                "normalization; equal to P_world - t_object"
            ),
        }
    )
    return schema


def main() -> None:
    nocs_base.build_nocs_targets = build_world_aligned_nocs_targets
    nocs_base.schema_description = schema_description
    nocs_base.main()


if __name__ == "__main__":
    main()
