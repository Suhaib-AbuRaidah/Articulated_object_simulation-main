import os
import trimesh
import argparse
from pathlib import Path
import sys
sys.path.append(os.path.expanduser("~/Articulated_object_simulation-main/src/"))
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from collections import OrderedDict

from s2u.simulation import ArticulatedObjectManipulationSim
from s2u.utils.axis2transform import axis2transformation
from s2u.utils.saver import get_mesh_pose_dict_from_world
from s2u.utils.visual import as_mesh
from s2u.utils.implicit import sample_iou_points_occ
from s2u.utils.io import write_data
from s2u.canonical_cf import estimate_normals_knn, optimize_canonical_frame

def normalize(points):
    bound_max = points.max(0)
    bound_min = points.min(0)
    center = (bound_max+bound_min)/2
    scale = bound_max-bound_min
    return (points-center)/scale

def downsample_point_cloud(points, labels=None, num_points=1024):
    """
    Randomly downsample the point cloud to a fixed size.
    """
    N = points.shape[0]
    if N >= num_points:
        np.random.seed(97)
        indices = np.random.choice(N, num_points, replace=False)
    else:
        np.random.seed(97)
        indices = np.random.choice(N, num_points, replace=True)  # pad if too small
    if labels is not None:
        labels = labels[indices]
        return points[indices], labels
    else:
        return points[indices]

def calculate_canonical_frames(points, part_masks):
    """Calculate a world-space canonical frame for each segmented part."""
    canonical_frames = {}
    for part_id, mask in part_masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != points.shape[0]:
            raise ValueError(
                f"Part {part_id} mask has {mask.shape[0]} entries for "
                f"{points.shape[0]} points")

        part_points = np.asarray(points[mask], dtype=np.float32)
        if part_points.shape[0] < 3:
            canonical_frames[part_id] = {
                'status': 'insufficient_points',
                'num_points': part_points.shape[0],
            }
            continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        part_points_tensor = torch.from_numpy(part_points).to(device)
        normal_k = min(24, part_points.shape[0] - 1)
        part_normals = estimate_normals_knn(part_points_tensor, k=normal_k)
        frame = optimize_canonical_frame(
            part_points_tensor,
            normals=part_normals,
            alpha=0.5,
            beta=2.0,
            restarts=1,
            steps=250,
            random_rotation_degrees=20.0,
        )
        canonical_frames[part_id] = {
            'status': 'ok',
            'num_points': part_points.shape[0],
            'center': frame.center.detach().cpu().numpy(),
            'axes': frame.axes.detach().cpu().numpy(),
            'scale': float(frame.scale.detach().cpu()),
            'energy': frame.energy,
            'info': frame.info,
        }
    return canonical_frames


def smoothstep(t):
    # cubic smooth interpolation, t in [0, 1]
    return 3.0 * t**2 - 2.0 * t**3

def sample_joint_trajectory_end(q_start, lower_limit, higher_limit):
    joint_range = higher_limit - lower_limit
    min_delta_ratio = np.random.choice([0.15, 0.4], p=[0.2, 0.8])
    min_delta = min_delta_ratio * joint_range
    max_delta = 0.9 * joint_range

    valid_delta_ranges = []
    if q_start + min_delta <= higher_limit:
        valid_delta_ranges.append(
            (min_delta, min(max_delta, higher_limit - q_start)))
    if q_start - min_delta >= lower_limit:
        valid_delta_ranges.append(
            (-min(max_delta, q_start - lower_limit), -min_delta))

    delta_min, delta_max = valid_delta_ranges[
        np.random.randint(len(valid_delta_ranges))]
    return q_start + np.random.uniform(delta_min, delta_max)

def main(args, rank):
    
    np.random.seed()
    seed = np.random.randint(0, 1000) + rank
    np.random.seed(seed)
    sim = ArticulatedObjectManipulationSim(args.object_set,
                                           size=0.3,
                                           gui=args.sim_gui,
                                           global_scaling=args.global_scaling,
                                           dense_photo=args.dense_photo)
    scenes_per_worker = args.num_scenes // args.num_proc
    pbar = tqdm(total=scenes_per_worker, disable=rank != 0)
    
    if rank == 0:
        print(f'Number of objects: {len(sim.object_urdfs)}')
    
    for i in range(scenes_per_worker):
        
        sim.reset(canonical=args.canonical)
        object_path = str(sim.object_urdfs[sim.object_idx])
        result = collect_observations(
            sim, args)
        result['object_path'] = object_path
        # print(result.keys())
        write_data(args.root, result)
        
        pbar.update()
    
    pbar.close()
    print('Process %d finished!' % rank)

def get_limit(v, args):
    joint_type = v[2]
    # specify revolute angle range for shape2motion
    if joint_type == 0 and not args.is_syn:
        if args.pos_rot:
            lower_limit = 0
            range_lim = np.pi / 2
            higher_limit = np.pi / 2
        else:
            lower_limit = - np.pi / 4
            range_lim = np.pi / 2
            higher_limit = np.pi / 4
    else:
        lower_limit = v[8]
        higher_limit = v[9]
        range_lim = higher_limit - lower_limit
    return lower_limit, higher_limit, range_lim

def sample_joint_configuration(all_joints, joint_info, args):
    max_joint = max(all_joints) + 1
    q = [0.0] * max_joint

    for j in all_joints:
        v = joint_info[j]
        if args.is_syn:
            v = v[0]

        lower_limit, higher_limit, _ = get_limit(v, args)

        # reset to nominal/initial state first
        sim_init = v[10]

        # if limits are invalid or too tight, just keep nominal state
        if higher_limit <= lower_limit:
            q[j] = sim_init
        else:
            q[j] = np.random.uniform(lower_limit, higher_limit)

    return q


def collect_observations(sim, args):
    # print(str(sim.object_urdfs[sim.object_idx]))
    if args.is_syn:
        joint_info = sim.get_joint_info_w_sub()
    else:
        joint_info = sim.get_joint_info()

    all_joints = list(joint_info.keys())
    max_joint = max(all_joints) + 1

    # number of frames in the smooth trajectory
    num_frames = 2

    # base link handling exactly as in your original code
    links_real = [joint_index for joint_index in all_joints]
    if all_joints[0] != 0:
        links_real.insert(0, 0)   # add the base link
    else:
        links_real.insert(0, -1)  # add the base link

    # ---------- sample smooth start/end configurations ----------
    q_start = [0.0] * max_joint
    q_end = [0.0] * max_joint

    for j in all_joints:
        v = joint_info[j]
        # print("joint id:", j)
        # print("parent index:", v[-1])
        # print("joint axis:", v[-4])
        # print("parent frame pos:", v[-3])
        # print("parent frame orn:", v[-2])
        if args.is_syn:
            v = v[0]

        lower_limit, higher_limit, _ = get_limit(v, args)
        init_state = v[10]

        # reset first
        sim.set_joint_state(j, init_state)

        if args.rand_state and higher_limit > lower_limit:
            q_start[j] = np.random.uniform(lower_limit, higher_limit)
            q_end[j] = sample_joint_trajectory_end(
                q_start[j], lower_limit, higher_limit)
        else:
            q_start[j] = init_state
            q_end[j] = init_state

    # ---------- collect sequence ----------
    poses_pcs = []
    poses_seg = []
    joint_states = []
    axis_list = []
    moment_list = []
    joint_type_list = []
    canonical_frames_lst = []

    num_points = 0

    for frame_idx in range(num_frames):
        if num_frames == 1:
            tau = 0.0
        else:
            tau = frame_idx / (num_frames - 1)

        s = smoothstep(tau)

        # interpolate all joints smoothly
        q_t = [0.0] * max_joint
        for j in all_joints:
            q_t[j] = q_start[j] + s * (q_end[j] - q_start[j])
            sim.set_joint_state(j, q_t[j])

        # capture observation
        _, _, pc, seg_label, _, _ = sim.acquire_segmented_pcs(6, links_real)

        if pc.shape[0] > num_points:
            num_points = pc.shape[0]

        poses_pcs.append(pc)
        poses_seg.append(seg_label)
        joint_states.append(q_t.copy())

        # store screw info for all joints at this frame
        frame_axes = {}
        frame_moments = {}
        frame_joint_types = {}
        for j in all_joints:
            if j >1 and not args.is_syn:
                axis, moment, previous_axis = sim.get_joint_screw1(j,joint_info[j-1][-4])
                if np.dot(frame_axes[j-1], previous_axis) < -0.5:
                    frame_axes[j-1] = -frame_axes[j-1]
                    frame_moments[j-1] = -frame_moments[j-1]
            else:
                axis, moment = sim.get_joint_screw(j)
            v = joint_info[j]
            if args.is_syn:
                v = v[0]
            joint_type = v[2]

            frame_axes[j] = axis
            frame_moments[j] = moment
            frame_joint_types[j] = joint_type

        axis_list.append(frame_axes)
        moment_list.append(frame_moments)
        joint_type_list.append(frame_joint_types)

        # Canonical frames describe each part in the world coordinates of frame 0.
        canonical_frames = calculate_canonical_frames(poses_pcs[frame_idx], poses_seg[frame_idx])
        canonical_frames_lst.append(canonical_frames)

    # ---------- make all point clouds/segmentations same size ----------
    for i, (pc, labels) in enumerate(zip(poses_pcs, poses_seg)):
        poses_pcs[i] = downsample_point_cloud(pc, num_points=num_points)
        for key in labels.keys():
            poses_seg[i][key] = downsample_point_cloud(labels[key], num_points=num_points)

    state0 = joint_states[0]
    # print(f"Joint states abs:\n{joint_states}")
    joint_states_rel = np.asarray(joint_states) - np.array(state0)
    # print(f"Joint states relative:\n{joint_states_rel}")
    result = {
        'poses_pcs': poses_pcs,                 # list of point clouds over time
        'poses_seg': poses_seg,                 # list of segmentation dicts over time
        'joint_states': joint_states,           # full joint vector at each frame
        'joint_states_rel': joint_states_rel,   # relative joint states
        'axes': axis_list,                      # list of dicts: frame -> joint -> axis
        'screw_moments': moment_list,           # list of dicts: frame -> joint -> moment
        'joints_type': joint_type_list,         # list of dicts: frame -> joint -> type
        'canonical_frames': canonical_frames_lst,   # list of dicts: frame -> part -> canonical frame at frame 0
        # 'q_start': q_start,
        # 'q_end': q_end,
        # 'joint_ids': all_joints,
    }

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--object-set", type=str)
    parser.add_argument("--num-scenes", type=int, default=10000)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--range-scale", type=float, default=0.3)
    parser.add_argument("--num-point-occ", type=int, default=100000)
    parser.add_argument("--occ-var", type=float, default=0.005)
    parser.add_argument("--pos-rot", type=int, required=True)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--sample-method", type=str, default='mix')
    parser.add_argument("--rand-state", action="store_true", help='set static joints at random state')
    parser.add_argument("--global-scaling", type=float, default=0.5)
    parser.add_argument("--dense-photo", action="store_true")


    args = parser.parse_args()
    if 'syn' in args.object_set:
        args.is_syn = True
    else:
        args.is_syn = False
    print(f"Is synthetic: {args.is_syn}")
    (args.root / "scenes").mkdir(parents=True, exist_ok=True)
    if args.num_proc > 1:
        #print(args.num_proc)
        pool = mp.get_context("spawn").Pool(processes=args.num_proc)
        for i in range(args.num_proc):
            pool.apply_async(func=main, args=(args, i))
        pool.close()
        pool.join()
    else:
        main(args, 0)
