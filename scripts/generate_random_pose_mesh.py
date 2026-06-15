"""Export one randomly articulated object pose as a single mesh file."""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from s2u.simulation import ArticulatedObjectManipulationSim
from s2u.utils.saver import get_mesh_pose_dict_from_world


def get_joint_limits(joint_data, is_syn, pos_rot):
    joint_type = joint_data[2]
    if joint_type == 0 and not is_syn:
        if pos_rot:
            return 0.0, float(np.pi / 2)
        return float(-np.pi / 4), float(np.pi / 4)
    return float(joint_data[8]), float(joint_data[9])


def randomize_all_joints(sim, is_syn, pos_rot, rng):
    if is_syn:
        joint_info = sim.get_joint_info_w_sub()
    else:
        joint_info = sim.get_joint_info()

    for joint_id, raw_joint_data in joint_info.items():
        joint_data = raw_joint_data[0] if is_syn else raw_joint_data
        lower_limit, upper_limit = get_joint_limits(
            joint_data, is_syn, pos_rot
        )
        initial_state = float(joint_data[10])

        if (
            np.isfinite(lower_limit)
            and np.isfinite(upper_limit)
            and upper_limit > lower_limit
        ):
            state = float(rng.uniform(lower_limit, upper_limit))
        else:
            state = initial_state

        sim.set_joint_state(joint_id, state)

    return len(joint_info)


def load_transformed_mesh(mesh_path, scale, pose):
    if mesh_path.startswith("#"):
        return trimesh.creation.box(extents=scale, transform=pose)

    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            return None
        if hasattr(loaded, "to_geometry"):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise TypeError(
            "Unsupported geometry type for {}: {}".format(
                mesh_path, type(loaded).__name__
            )
        )

    mesh.apply_scale(scale)
    mesh.apply_transform(pose)
    return mesh


def export_object_mesh(sim, output_path, output_format):
    mesh_pose_dict = get_mesh_pose_dict_from_world(
        sim.world, exclude_plane=True
    )
    object_prefix = "{}_".format(sim.object.uid)
    meshes = []

    for object_name, mesh_entries in mesh_pose_dict.items():
        if not object_name.startswith(object_prefix):
            continue
        for mesh_path, scale, pose in mesh_entries:
            mesh = load_transformed_mesh(mesh_path, scale, pose)
            if mesh is not None:
                meshes.append(mesh)

    if not meshes:
        raise RuntimeError("The loaded object has no exportable visual meshes.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_mesh = trimesh.util.concatenate(meshes)
    combined_mesh.export(str(output_path), file_type=output_format)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load one articulated object, randomly move every movable joint, "
            "and export the resulting pose as one mesh file."
        )
    )
    parser.add_argument("output", type=Path, help="Output mesh file path.")
    parser.add_argument("--object-set", required=True, help="URDF object set.")
    parser.add_argument(
        "--format",
        choices=("glb", "obj"),
        default="glb",
        help="Output mesh format (default: glb).",
    )
    parser.add_argument(
        "--object-index",
        type=int,
        default=None,
        help="Object index within the set; a random object is used by default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for object selection and joint states.",
    )
    parser.add_argument(
        "--pos-rot",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "For non-synthetic revolute joints, use [0, pi/2] when 1 or "
            "[-pi/4, pi/4] when 0."
        ),
    )
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--global-scaling", type=float, default=0.5)
    parser.add_argument("--sim-gui", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output.with_suffix(".{}".format(args.format))

    is_syn = "syn" in args.object_set
    # rng = np.random.default_rng(args.seed)
    seed = np.random.randint(0, 10000 - 1)
    rng = np.random.default_rng(seed)
    sim = ArticulatedObjectManipulationSim(
        args.object_set,
        size=0.3,
        gui=args.sim_gui,
        global_scaling=args.global_scaling,
        seed=seed,
    )
    print("Loaded {} objects from set '{}'.".format(
        len(sim.object_urdfs), args.object_set
    ))
    try:
        if not sim.object_urdfs:
            raise RuntimeError(
                "No URDF objects found in object set '{}'.".format(
                    args.object_set
                )
            )
        if args.object_index is not None and not (
            0 <= args.object_index < len(sim.object_urdfs)
        ):
            raise ValueError(
                "Object index {} is outside [0, {}).".format(
                    args.object_index, len(sim.object_urdfs)
                )
            )

        sim.reset(index=args.object_index, canonical=args.canonical)
        joint_count = randomize_all_joints(
            sim, is_syn, args.pos_rot, rng
        )
        export_object_mesh(sim, output_path, args.format)

        print("Object: {}".format(sim.object_urdfs[sim.object_idx]))
        print("Randomized joints: {}".format(joint_count))
        print("Saved mesh: {}".format(output_path))
    finally:
        sim.world.close()


if __name__ == "__main__":
    main()
