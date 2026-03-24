import trimesh
import argparse
from pathlib import Path

import numpy as np
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

def collect_observations(sim, args):

    if args.is_syn:
        joint_info = sim.get_joint_info_w_sub()
    else:
        joint_info = sim.get_joint_info()

    all_joints = list(joint_info.keys())
    max_joint = max(all_joints) + 1
    start_state_list = [0.0] * max_joint
    if args.rand_state:          
        for x in all_joints:
            v = joint_info[x]
            if args.is_syn:
                v = v[0]
            lower_limit, higher_limit, range_lim = get_limit(v, args)
            start_state = np.random.uniform(lower_limit, higher_limit)
            start_state_list[x]=(start_state)
            sim.set_joint_state(x, v[10]) # set to initial state before randomization
            sim.set_joint_state(x, start_state) # set to random state


    index_chosen = np.random.randint(len(all_joints[:-1]))

    v = joint_info[all_joints[index_chosen]]
    lower_limit, higher_limit, range_lim = get_limit(v, args)
    pose_range_upper = higher_limit - start_state_list[all_joints[index_chosen]]
    pose_range_lower = start_state_list[all_joints[index_chosen]] - lower_limit
    if pose_range_upper > pose_range_lower:
        end_states = [start_state_list[all_joints[index_chosen]] + pose_range_upper*i/4 for i in range(5)]
    else:
        end_states = [start_state_list[all_joints[index_chosen]] - pose_range_lower*i/4 for i in range(5)]
    

    links_real = [joint_index for joint_index in all_joints]
    if all_joints[0] != 0:
        links_real.insert(0, 0)  # add the base link
    else:
        links_real.insert(0, -1)  # add the base link

    poses_pcs = []
    poses_seg = []
    axis_list = []
    moment_list = []
    joint_type_list = []
    num_points = 0
    for i in range(len(end_states)):
        sim.set_joint_state(all_joints[index_chosen], end_states[i])
        _, _, pc, seg_label, _, _ = sim.acquire_segmented_pcs(6, links_real)
        if pc.shape[0] > num_points:    
            num_points = pc.shape[0]
        poses_pcs.append(pc)
        poses_seg.append(seg_label)
        
        axis, moment = sim.get_joint_screw(all_joints[index_chosen])
        joint_type = v[2]
        axis_list.append(axis)
        moment_list.append(moment)
        joint_type_list.append(joint_type)

    for i, (pc, labels) in enumerate(zip(poses_pcs, poses_seg)):

        poses_pcs[i]= downsample_point_cloud(pc, num_points=num_points)
        for key in labels.keys():
            poses_seg[i][key] = downsample_point_cloud(labels[key], num_points=num_points)

    result = {
            f'poses_pcs': poses_pcs,
            f'poses_seg': poses_seg,
            f'axes': axis_list,
            f'screw_moments': moment_list,
            f'joints_type': joint_type_list,
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
