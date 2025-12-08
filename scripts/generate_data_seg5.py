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
            print(mesh_path)
            if mesh_path.startswith('#'): # primitive
                continue
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
    
    max_joint = max(all_joints) + 1
    start_state_list = [0.0] * max_joint
    end_state_list = [0.0] * max_joint
    initial_state_list = [0.0] * max_joint
    if args.rand_state:
        for x in all_joints:
            v = joint_info[x]
            if args.is_syn:
                v = v[0]
            lower_limit, higher_limit, range_lim = get_limit(v, args)
            start_state = np.random.uniform(lower_limit, higher_limit)
            start_state_list[x]=(start_state)

            if (start_state-0.1*range_lim-lower_limit) > (higher_limit-start_state-0.1*range_lim):
                end_state = np.random.uniform(lower_limit, start_state-0.1*range_lim)
            else:
                end_state = np.random.uniform(start_state+0.1*range_lim, higher_limit)
            end_state_list[x]=(end_state)

            initial_state_list[x]=(v[10])
            sim.set_joint_state(x, start_state)

    state_diff = np.array(end_state_list) - np.array(start_state_list)

    # joint_index = all_joints.pop(np.random.randint(len(all_joints)))
    for index in all_joints:
        sim.set_joint_state(x, initial_state_list[index])
    if args.rand_state:
        for x in all_joints:
            sim.set_joint_state(x, start_state_list[x])
            
    all_joints_array = np.array(all_joints)
    axis_list = []
    moment_list = []
    joint_type_list = []
    for joint_index in all_joints:
        a = np.where(all_joints_array == joint_index)[0] + 1
        b = np.where(all_joints_array == joint_info[joint_index][-1])[0] + 1

        if a.size == 0:
            a = np.array([0])
        if b.size == 0:
            b = np.array([0])

        parts_connections.append((a, b))

        v = joint_info[joint_index]
        if args.is_syn:
            v = v[0]
        axis, moment = sim.get_joint_screw(joint_index)
        joint_type = v[2]
        axis_list.append(axis)
        moment_list.append(moment)
        joint_type_list.append(joint_type)


    joints_syn = [joint_info[joint_index][1] for joint_index in all_joints]
    joints_real = [joint_index for joint_index in all_joints]
    joints_real.insert(0, 0)  # add the base link
    # print(f"joints_syn:{joints_syn}")
    # print(f"joints_real:{joints_real}")
    if args.is_syn:
        _, _, start_pc, start_seg_label, start_mesh_pose_dict = sim.acquire_segmented_pcs(6, joints_syn)
        # start_pc, start_seg_label = downsample_point_cloud(start_pc, start_seg_label, num_points)
        start_p_occ, start_occ_list, start_mesh_pose_dict = sample_occ_binary(sim, joints_syn, args.num_point_occ, args.sample_method, args.occ_var)

    else:
        _, _, start_pc, start_seg_label, start_mesh_pose_dict = sim.acquire_segmented_pcs(6, joints_real)
        # start_pc, start_seg_label = downsample_point_cloud(start_pc, start_seg_label, num_points)
        # start_p_occ, start_occ_list = sample_occ(sim, args.num_point_occ, args.sample_method, args.occ_var)
        start_p_occ, start_occ_list = None, None


    for x in all_joints:
        sim.set_joint_state(x, end_state_list[x])

    
    if args.is_syn:
        _, _, end_pc, end_seg_label, end_mesh_pose_dict = sim.acquire_segmented_pcs(6, joints_syn)
        end_p_occ, end_occ_list, end_mesh_pose_dict = sample_occ_binary(sim, joints_syn, args.num_point_occ, args.sample_method, args.occ_var)
    else:
        _, _, end_pc, end_seg_label, end_mesh_pose_dict = sim.acquire_segmented_pcs(6, joints_real)
        # end_p_occ, end_occ_list = sample_occ(sim, args.num_point_occ, args.sample_method, args.occ_var)
        end_p_occ, end_occ_list = None, None


    result = {
            f'pc_start': start_pc,
            f'pc_seg_start': start_seg_label,
            f'pc_end': end_pc,
            f'pc_seg_end': end_seg_label,
            f'state_start': start_state_list,
            f'state_end': end_state_list,
            f'state_diff': state_diff,
            f'screw_axis': axis_list,
            f'screw_moment': moment_list,
            f'joint_type': joint_type_list,
            f'joint_index': joints_syn if args.is_syn else joints_real,
        }



    parts_conne_gt = np.zeros((len(all_joints)+1, len(all_joints)+1))
    # print(f"parts_connections:{parts_connections}")
    for i in range(len(parts_connections)):
        parts_conne_gt[parts_connections[i][0], parts_connections[i][1]] = 1
        parts_conne_gt[parts_connections[i][1], parts_connections[i][0]] = 1
    # print(parts_conne_gt)
    result['parts_conne_gt'] = parts_conne_gt


    start_pc = normalize(start_pc)
    parts_pcs = []
    for i in range(len(all_joints)+1):
        part = start_pc[start_seg_label[i]]
        parts_pcs.append(part)


    adj = build_graph_from_parts(parts_pcs, 0.05)
    # print(f"adj:\n{adj}")
    result['adj'] = adj

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
