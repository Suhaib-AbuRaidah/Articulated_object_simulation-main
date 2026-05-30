#!/usr/bin/env python3
"""
Generate a VLM fine-tuning dataset from articulated-object simulation.

Unlike generate_data_seg.py, this exports one articulation pose per scene instead
of a 4-pose trajectory. For each sampled scene, it saves rendered RGB images and
one JSONL annotation per image:

dataset_root/
  images/
    scene_000000_view_00.png
    scene_000000_view_01.png
  train.jsonl
  meta.json

Each JSONL row is compatible with VLM_finetuning.py:

{"image": "scene_000000_view_00.png", "joint_parameters": {...}}
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


COLORS_PALETTE = [
    "#d62728",  # red
    "#2ca02c",  # green
    "#1f77b4",  # blue
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#ff7f0e",  # orange
    "#7f7f7f",  # gray
]


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


SEGMENTATION_COLORS = np.asarray(
    [hex_to_rgb(color) for color in COLORS_PALETTE], dtype=np.uint8
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3:
        raise ValueError(f"Expected HxWxC RGB image, got shape {rgb.shape}")
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def save_rgb_image(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_to_uint8(rgb)).save(path)


def joint_type_name(joint_type: int) -> str:
    if joint_type == 0:
        return "revolute"
    if joint_type == 1:
        return "prismatic"
    return f"pybullet_type_{joint_type}"


def get_limit(v: Tuple[Any, ...], args: argparse.Namespace) -> Tuple[float, float, float]:
    joint_type = int(v[2])
    if joint_type == 0 and not args.is_syn:
        if args.pos_rot:
            lower_limit = 0.0
            higher_limit = float(np.pi / 2)
        else:
            lower_limit = float(-np.pi / 4)
            higher_limit = float(np.pi / 4)
        range_lim = higher_limit - lower_limit
    else:
        lower_limit = float(v[8])
        higher_limit = float(v[9])
        range_lim = higher_limit - lower_limit
    return lower_limit, higher_limit, range_lim


def setup_simulation_import(sim_src: Path) -> None:
    sim_src = sim_src.expanduser().resolve()
    if str(sim_src) not in sys.path:
        sys.path.insert(0, str(sim_src))


def get_simulation_class(sim_src: Path):
    setup_simulation_import(sim_src)
    from s2u.simulation import ArticulatedObjectManipulationSim

    return ArticulatedObjectManipulationSim


def get_mobile_joint_info(sim: Any, args: argparse.Namespace):
    if args.is_syn:
        return sim.get_joint_info_w_sub()
    return sim.get_joint_info()


def unwrap_joint_info(raw_info: Any, args: argparse.Namespace) -> Tuple[Any, ...]:
    if args.is_syn:
        return raw_info[0]
    return raw_info


def sample_one_pose(
    sim: Any,
    joint_info: Dict[int, Any],
    all_joints: List[int],
    args: argparse.Namespace,
) -> List[float]:
    if not all_joints:
        return []

    max_joint = max(all_joints) + 1
    q = [0.0] * max_joint

    for joint_id in all_joints:
        v = unwrap_joint_info(joint_info[joint_id], args)
        lower_limit, higher_limit, _ = get_limit(v, args)
        init_state = float(v[10])

        if args.rand_state and higher_limit > lower_limit:
            q[joint_id] = float(np.random.uniform(lower_limit, higher_limit))
        else:
            q[joint_id] = init_state

        sim.set_joint_state(joint_id, q[joint_id])

    return q


def links_for_segmentation(all_joints: List[int]) -> List[int]:
    if not all_joints:
        return [0]
    links_real = [joint_index for joint_index in all_joints]
    if all_joints[0] != 0:
        links_real.insert(0, 0)
    else:
        links_real.insert(0, -1)
    return links_real


def render_viewpoint_extrinsics(sim: Any, n: int, N: int | None = None) -> List[Any]:
    from s2u.perception import camera_on_sphere
    from s2u.utils.transform import Rotation, Transform

    origin = Transform(
        Rotation.identity(),
        np.r_[sim.size / 2, sim.size / 2, sim.size / 2],
    )
    radius = 1.2 * sim.size
    extrinsics = []

    if sim.dense_photo:
        for offset, theta in zip([0, 1, 2], [np.pi / 8.0, np.pi / 4.0, np.pi / 2.0]):
            view_count = N if N else n
            phi_list = (
                2.0 * np.pi * np.arange(n) / view_count
                + offset * 2.0 * np.pi / (view_count * 3)
            )
            extrinsics += [
                camera_on_sphere(origin, radius, theta, phi) for phi in phi_list
            ]
    else:
        theta = np.pi / 4.0
        view_count = N if N else n
        phi_list = 2.0 * np.pi * np.arange(n) / view_count
        extrinsics += [camera_on_sphere(origin, radius, theta, phi) for phi in phi_list]

    return extrinsics


def colorize_segmentation(seg_uid: np.ndarray, seg_link: np.ndarray, links: List[int]) -> np.ndarray:
    seg_color = np.ones((*seg_link.shape, 3), dtype=np.uint8)*255
    foreground = (seg_uid + 1).astype(bool)

    for color_idx, link_id in enumerate(links):
        color = SEGMENTATION_COLORS[color_idx % len(SEGMENTATION_COLORS)]
        mask = foreground & (seg_link == link_id)
        seg_color[mask] = color

    return seg_color


def render_segmented_images(sim: Any, n: int, links: List[int]) -> List[np.ndarray]:
    import pybullet

    segmented_images = []
    for extrinsic in render_viewpoint_extrinsics(sim, n):
        _, _, (seg_uid, seg_link) = sim.camera.render(
            extrinsic,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )
        segmented_images.append(colorize_segmentation(seg_uid, seg_link, links))

    return segmented_images


def collect_joint_parameters(
    sim: Any,
    joint_info: Dict[int, Any],
    all_joints: List[int],
    q: List[float],
    object_path: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    joints = []

    for joint_id in all_joints:
        v = unwrap_joint_info(joint_info[joint_id], args)
        joint_type = int(v[2])
        lower_limit, higher_limit, _ = get_limit(v, args)
        axis, moment = sim.get_joint_screw(joint_id)

        joints.append(
            {
                "id": int(joint_id),
                "name": to_jsonable(v[1]),
                "type": joint_type_name(joint_type),
                "type_id": joint_type,
                "state": q[joint_id] if joint_id < len(q) else 0.0,
                "axis": to_jsonable(axis),
                "screw_moment": to_jsonable(moment),
                "lower_limit": lower_limit,
                "upper_limit": higher_limit,
                "initial_state": float(v[10]),
            }
        )

    return {
        "object_path": object_path,
        "joint_states": q,
        "joints": joints,
    }


def collect_one_pose_scene(
    sim: Any,
    args: argparse.Namespace,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    joint_info = get_mobile_joint_info(sim, args)
    all_joints = list(joint_info.keys())
    q = sample_one_pose(sim, joint_info, all_joints, args)
    links_real = links_for_segmentation(all_joints)

    if args.segmented:
        rgb_imgs = render_segmented_images(sim, args.num_views, links_real)
    else:
        _, rgb_imgs, _, _, _, _ = sim.acquire_segmented_pcs(args.num_views, links_real)

    object_path = str(sim.object_urdfs[sim.object_idx])
    joint_parameters = collect_joint_parameters(
        sim, joint_info, all_joints, q, object_path, args
    )
    return rgb_imgs, joint_parameters


def make_annotation(
    image_name: str,
    scene_id: str,
    view_idx: int,
    joint_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "image": image_name,
        "question": (
            "<image>\n"
            "Estimate this articulated object's joint parameters. Return JSON "
            "with the joint type, state, axis, screw moment, and limits."
        ),
        "joint_parameters": {
            **joint_parameters,
            "scene_id": scene_id,
            "view": view_idx,
        },
    }


def write_meta_file(args: argparse.Namespace) -> None:
    meta = {
        "articulated_objects": {
            "root": str((args.root / "images").resolve()),
            "annotation": str((args.root / "train.jsonl").resolve()),
            "data_augment": False,
            "segmented": args.segmented,
            "max_dynamic_patch": 12,
            "repeat_time": 1,
            "length": args.num_scenes * args.effective_num_views,
        }
    }
    with (args.root / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def worker_main(args: argparse.Namespace, rank: int) -> None:
    np.random.seed()
    seed = np.random.randint(0, 1000) + rank
    np.random.seed(seed)

    simulation_class = get_simulation_class(args.sim_src)
    sim = simulation_class(
        args.object_set,
        size=0.3,
        gui=args.sim_gui,
        global_scaling=args.global_scaling,
        dense_photo=args.dense_photo,
        urdf_root=args.urdf_root,
    )

    scenes_per_worker = args.num_scenes // args.num_proc
    if rank < args.num_scenes % args.num_proc:
        scenes_per_worker += 1

    annotation_path = args.root / f"train_rank_{rank:03d}.jsonl"
    pbar = tqdm(total=scenes_per_worker, disable=rank != 0)

    if rank == 0:
        print(f"Number of objects: {len(sim.object_urdfs)}")

    with annotation_path.open("w", encoding="utf-8") as annotation_file:
        for local_idx in range(scenes_per_worker):
            sim.reset(canonical=args.canonical)
            scene_id = f"rank{rank:03d}_scene{local_idx:06d}_{uuid.uuid4().hex[:8]}"
            rgb_imgs, joint_parameters = collect_one_pose_scene(sim, args)

            for view_idx, rgb in enumerate(rgb_imgs):
                image_name = f"{scene_id}_view{view_idx:02d}.{args.image_format}"
                image_path = args.root / "images" / image_name
                save_rgb_image(rgb, image_path)

                annotation = make_annotation(
                    image_name=image_name,
                    scene_id=scene_id,
                    view_idx=view_idx,
                    joint_parameters=joint_parameters,
                )
                annotation_file.write(json.dumps(to_jsonable(annotation)) + "\n")

            pbar.update()

    pbar.close()
    print(f"Process {rank} finished!")


def merge_annotations(args: argparse.Namespace) -> None:
    final_path = args.root / "train.jsonl"
    rank_paths = sorted(args.root.glob("train_rank_*.jsonl"))

    with final_path.open("w", encoding="utf-8") as final_file:
        for rank_path in rank_paths:
            with rank_path.open("r", encoding="utf-8") as rank_file:
                for line in rank_file:
                    final_file.write(line)
            rank_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--object-set", type=str, required=True)
    parser.add_argument("--num-scenes", type=int, default=1000)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=6)
    parser.add_argument("--image-format", choices=["png", "jpg", "jpeg"], default="png")
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--pos-rot", type=int, required=True)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--rand-state", action="store_true")
    parser.add_argument(
        "--segmented",
        action="store_true",
        help="Save part-colored segmentation images instead of normal RGB renders.",
    )
    parser.add_argument("--global-scaling", type=float, default=0.5)
    parser.add_argument("--dense-photo", action="store_true")
    parser.add_argument(
        "--urdf-root",
        type=Path,
        default=Path("/home/suhaib/Ditto/Articulated_object_simulation-main/data/urdfs"),
        help="Path to the directory that contains articulated-object URDF sets.",
    )
    parser.add_argument(
        "--sim-src",
        type=Path,
        default=Path("/home/suhaib/Ditto/Articulated_object_simulation-main/src"),
        help="Path to the simulation repo src directory that contains the s2u package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.is_syn = "syn" in args.object_set
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "images").mkdir(parents=True, exist_ok=True)

    args.effective_num_views = args.num_views * 3 if args.dense_photo else args.num_views

    print(f"Is synthetic: {args.is_syn}")
    print(f"Segmented images: {args.segmented}")
    print(f"Writing images to: {args.root / 'images'}")
    print(f"Writing annotations to: {args.root / 'train.jsonl'}")

    if args.num_proc > 1:
        pool = mp.get_context("spawn").Pool(processes=args.num_proc)
        results = []
        for rank in range(args.num_proc):
            results.append(pool.apply_async(func=worker_main, args=(args, rank)))
        pool.close()
        pool.join()
        for result in results:
            result.get()
    else:
        worker_main(args, 0)

    merge_annotations(args)
    write_meta_file(args)


if __name__ == "__main__":
    main()
