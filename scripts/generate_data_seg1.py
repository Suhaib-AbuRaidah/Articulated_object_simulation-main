import trimesh
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist

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

def downsample_point_cloud(points, labels, num_points=1024):
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
    labels = labels[indices]
    return points[indices], labels

def build_graph_from_parts(pointclouds, threshold):
    """
    pointclouds: list of np.ndarray, each of shape (N_i, 3)
    threshold: float, maximum allowed distance for connection
    returns: adjacency matrix (n_parts x n_parts)
    """
    n = len(pointclouds)
    adj = np.zeros((n, n), dtype=int)

    # Precompute centroids or use full point distance
    centroids = [pc.mean(axis=0) for pc in pointclouds]


    for i in range(n):
        for j in range(i + 1, n):
            # compute minimum distance between two parts
            d = np.min(cdist(pointclouds[i], pointclouds[j]))
            # print(f"distance between joints {i} and {j} is {d}")
            if d < threshold:
                adj[i, j] = adj[j, i] = 1

    return adj

def binary_occ(occ_list, idx):
    occ_fore = occ_list.pop(idx)
    occ_back = np.zeros_like(occ_fore)
    for o in occ_list:
        occ_back += o
    return occ_fore, occ_back


def sample_occ(sim, num_point, method, var=0.005):
    result_dict = get_mesh_pose_dict_from_world(sim.world, False)
    obj_name = str(sim.object_urdfs[sim.object_idx])
    obj_name = '/'.join(obj_name.split('/')[-4:-1])
    mesh_dict = {}
    whole_scene = trimesh.Scene()
    for k, v in result_dict.items():
        scene = trimesh.Scene()
        for mesh_path, scale, pose in v:
            mesh = trimesh.load(mesh_path)
            mesh.apply_scale(scale)
            mesh.apply_transform(pose)
            scene.add_geometry(mesh)
            whole_scene.add_geometry(mesh)
        mesh_dict[k] = as_mesh(scene)
    points_occ, occ_list = sample_iou_points_occ(mesh_dict.values(),
                                                      whole_scene.bounds,
                                                      num_point,
                                                      method,
                                                      var=var)
    return points_occ, occ_list

def sample_occ_binary(sim, mobile_links, num_point, method, var=0.005):
    result_dict = get_mesh_pose_dict_from_world(sim.world, False)
    new_dict = {'0_0': [], '0_1': []}
    obj_name = str(sim.object_urdfs[sim.object_idx])
    obj_name = '/'.join(obj_name.split('/')[-4:-1])
    whole_scene = trimesh.Scene()
    static_scene = trimesh.Scene()
    mobile_scene = trimesh.Scene()
    for k, v in result_dict.items():
        body_uid, link_index = k.split('_')
        link_index = int(link_index)
        if link_index in mobile_links:
            new_dict['0_1'] += v
        else:
            new_dict['0_0'] += v
        for mesh_path, scale, pose in v:
            if mesh_path.startswith('#'): # primitive
                mesh = trimesh.creation.box(extents=scale, transform=pose)
            else:
                mesh = trimesh.load(mesh_path)
                mesh.apply_scale(scale)
                mesh.apply_transform(pose)
            if link_index in mobile_links:
                mobile_scene.add_geometry(mesh)
            else:
                static_scene.add_geometry(mesh)
            whole_scene.add_geometry(mesh)
    static_mesh = as_mesh(static_scene)
    mobile_mesh = as_mesh(mobile_scene)
    points_occ, occ_list = sample_iou_points_occ((static_mesh, mobile_mesh),
                                                      whole_scene.bounds,
                                                      num_point,
                                                      method,
                                                      var=var)
    return points_occ, occ_list, new_dict


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
    
    for _ in range(scenes_per_worker):
        
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
    total_result = []
    parts_connections = []
    if args.is_syn:
        joint_info = sim.get_joint_info_w_sub()
    else:
        joint_info = sim.get_joint_info()
    all_joints = list(joint_info.keys())
    # print(all_joints)
    start_state_list = []
    initial_state_list = []
    if args.rand_state:
        for x in all_joints:
            v = joint_info[x]
            if args.is_syn:
                v = v[0]
            lower_limit, higher_limit, range_lim = get_limit(v, args)
            start_state = np.random.uniform(lower_limit, higher_limit)
            start_state_list.append(start_state)
            # print(v[10])
            initial_state_list.append(v[10])
            sim.set_joint_state(x, start_state)

    for joint in all_joints:
        # joint_index = all_joints.pop(np.random.randint(len(all_joints)))
        for index in all_joints:
            sim.set_joint_state(x, initial_state_list[index])
        if args.rand_state:
            for x in all_joints:
                sim.set_joint_state(x, start_state_list[x])
        joint_index = joint
        parts_connections.append((joint_index+1, joint_info[joint][-1]+1))
        # print(f'Process joint {joint_index}')

        v = joint_info[joint_index]
        if args.is_syn:
            v = v[0]
        axis, moment = sim.get_joint_screw(joint_index)
        joint_type = v[2]
        # not set
        lower_limit, higher_limit, range_lim = get_limit(v, args)

        # move_range = np.random.uniform(range_lim * args.range_scale, range_lim)
        # start_state = np.random.uniform(lower_limit, higher_limit - move_range)
        start_state = sim.get_joint_state(joint_index)[0]
        # print(start_state)
        # print(start_state)
        # end_state = start_state + move_range
        if (start_state-0.1*range_lim-lower_limit) > (higher_limit-start_state-0.1*range_lim):

            end_state = np.random.uniform(lower_limit, start_state-0.1*range_lim)
        else:
            end_state = np.random.uniform(start_state+0.1*range_lim, higher_limit)
        # if np.random.uniform(0, 1) > 0.5:
        #     start_state, end_state = end_state, start_state
        
        # sim.set_joint_state(joint_index, start_state)
        if joint == 0:
            
            if args.is_syn:
                _, _, start_pc, start_seg_label, start_mesh_pose_dict = sim.acquire_segmented_pc(6, joint_info[joint_index][1])
                start_p_occ, start_occ_list, start_mesh_pose_dict = sample_occ_binary(sim, joint_info[joint_index][1], args.num_point_occ, args.sample_method, args.occ_var)
            else:
                _, _, start_pc, start_seg_label, start_mesh_pose_dict = sim.acquire_segmented_pc(6, [joint_index])
                start_p_occ, start_occ_list = sample_occ(sim, args.num_point_occ, args.sample_method, args.occ_var)
            num_points = start_pc.shape[0]

        else:
            if args.is_syn:
                _, _, start_pc, start_seg_label, start_mesh_pose_dict = sim.acquire_segmented_pc(6, joint_info[joint_index][1])
                # start_pc, start_seg_label = downsample_point_cloud(start_pc, start_seg_label, num_points)
                start_p_occ, start_occ_list, start_mesh_pose_dict = sample_occ_binary(sim, joint_info[joint_index][1], args.num_point_occ, args.sample_method, args.occ_var)
            else:
                _, _, start_pc, start_seg_label, start_mesh_pose_dict = sim.acquire_segmented_pc(6, [joint_index])
                # start_pc, start_seg_label = downsample_point_cloud(start_pc, start_seg_label, num_points)
                start_p_occ, start_occ_list = sample_occ(sim, args.num_point_occ, args.sample_method, args.occ_var)
    
        # canonicalize start pc
        axis, moment = sim.get_joint_screw(joint_index)
        state_change = end_state - start_state
        if joint_type == 0:
            transformation = axis2transformation(axis, np.cross(axis, moment), state_change)
        else:
            transformation = np.eye(4)
            transformation[:3, 3] = axis * state_change
        
        mobile_start_pc = start_pc[start_seg_label].copy()
        rotated = transformation[:3, :3].dot(mobile_start_pc.T) + transformation[:3, [3]]
        rotated = rotated.T
        canonical_start_pc = start_pc.copy()
        canonical_start_pc[start_seg_label] = rotated

        sim.set_joint_state(joint_index, end_state)
        
        if args.is_syn:
            _, _, end_pc, end_seg_label, end_mesh_pose_dict = sim.acquire_segmented_pc(6, joint_info[joint_index][1])
            end_p_occ, end_occ_list, end_mesh_pose_dict = sample_occ_binary(sim, joint_info[joint_index][1], args.num_point_occ, args.sample_method, args.occ_var)
        else:
            _, _, end_pc, end_seg_label, end_mesh_pose_dict = sim.acquire_segmented_pc(6, [joint_index])
            end_p_occ, end_occ_list = sample_occ(sim, args.num_point_occ, args.sample_method, args.occ_var)
        # canonicalize end pc
        axis, moment = sim.get_joint_screw(joint_index)
        state_change = start_state - end_state
        if joint_type == 0:
            transformation = axis2transformation(axis, np.cross(axis, moment), state_change)
        else:
            transformation = np.eye(4)
            transformation[:3, 3] = axis * state_change
        mobile_end_pc = end_pc[end_seg_label].copy()
        rotated = transformation[:3, :3].dot(mobile_end_pc.T) + transformation[:3, [3]]
        rotated = rotated.T
        canonical_end_pc = end_pc.copy()
        canonical_end_pc[end_seg_label] = rotated

        result = {
                f'pc_start_{joint}': start_pc,
                f'pc_start_end_{joint}': canonical_start_pc,
                f'pc_seg_start_{joint}': start_seg_label,
                f'pc_end_{joint}': end_pc,
                f'pc_end_start_{joint}': canonical_end_pc,
                f'pc_seg_end_{joint}': end_seg_label,
                f'state_start_{joint}': start_state,
                f'state_end_{joint}': end_state,
                f'screw_axis_{joint}': axis,
                f'screw_moment_{joint}': moment,
                f'joint_type_{joint}': joint_type,
                f'joint_index_{joint}': 1 if args.is_syn else joint_index,
                f'start_p_occ_{joint}': start_p_occ, 
                f'start_occ_list_{joint}': start_occ_list, 
                f'end_p_occ_{joint}': end_p_occ, 
                f'end_occ_list_{joint}': end_occ_list,
                f'start_mesh_pose_dict_{joint}': start_mesh_pose_dict,
                f'end_mesh_pose_dict_{joint}': end_mesh_pose_dict
            }
        total_result.append(result)
    result_dict = {}
    for i in range(len(all_joints)):
        result_dict = {**result_dict, **total_result[i]}

    mask_start_list = []
    for joint in range(len(all_joints)):
        mask_start = result_dict[f'pc_seg_start_{joint}']
        mask_start_list.append(mask_start)

    total = np.zeros_like(mask_start_list[0])
    for mask in mask_start_list:
        total+=mask
    base_mask = np.zeros_like(total)
    base_mask[np.where(total==0)[0]]=1

    result_dict['pc_seg_start_base'] = base_mask


    parts_conne_gt = np.zeros((len(all_joints)+1, len(all_joints)+1))
    for i in range(len(parts_connections)):
        parts_conne_gt[parts_connections[i][0], parts_connections[i][1]] = 1
        parts_conne_gt[parts_connections[i][1], parts_connections[i][0]] = 1

    result_dict['parts_conne_gt'] = parts_conne_gt

    mask_start_list.insert(0, base_mask)

    start_pc = normalize(start_pc)
    parts_pcs = []
    for i in range(len(all_joints)+1):
        part = start_pc[mask_start_list[i]]
        parts_pcs.append(part)


    adj = build_graph_from_parts(parts_pcs, 0.05)
    # print(f"adj:\n{adj}")
    result_dict['adj'] = adj

    return result_dict


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
