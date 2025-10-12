import argparse
from pathlib import Path
import uuid
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from s2u.simulation import ArticulatedObjectManipulationSim
from s2u.utils.rgb_feature_matching import orb_matching, loftr_matching
import trimesh
from s2u.utils.visual import as_mesh
from s2u.utils.implicit import sample_iou_points_occ
from s2u.utils.saver import get_mesh_pose_dict_from_world



def write_data(root, data_dict,i):
    scene_id = uuid.uuid4().hex
    path = root/ f"{i:06d}.npz"
    #assert not path.exists()
    np.savez_compressed(path, **data_dict)
    return scene_id

def downsample_point_cloud(points, labels, num_points=1024):
    """
    Randomly downsample the point cloud to a fixed size.
    """
    N = points.shape[0]
    if N >= num_points:
        indices = np.random.choice(N, num_points, replace=False)
    else:
        indices = np.random.choice(N, num_points, replace=True)  # pad if too small
    return points[indices], labels[indices]


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

def points_to_occ_grid(points_occ, occ_list, grid_res=64):
    """
    Turn sampled points + occupancy labels into a voxel grid.
    """
    # # Normalize coordinates to [0, 1] cube
    mins = points_occ.min(0)
    maxs = points_occ.max(0)
    # normed = (points_occ - mins) / (maxs - mins + 1e-8)
    
    # Scale to grid
    idxs = (points_occ * (grid_res - 1)).astype(int)

    # Build occupancy grid
    occ_grid = np.zeros((grid_res, grid_res, grid_res), dtype=np.uint8)
    for i, occ in zip(idxs, occ_list[0]):
        x, y, z = i
        occ_grid[x, y, z] = max(occ_grid[x, y, z], occ)  # mark occupied if any point inside
    
    return occ_grid, mins, maxs


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
        #print(f'Number of objects: {len(sim.object_urdfs)}')
        pass
    for i in range(scenes_per_worker):
        
        sim.reset(canonical=args.canonical)        
        
        result = collect_observations(sim, args,i)
        result['pc_start'] = result['pc_start'].astype(np.float32)
        result['pc_seg_start'] = result['pc_seg_start'].astype(np.int64)
       
        write_data(args.root, result, i)
        
        pbar.update()
    
    pbar.close()
    print('Process %d finished!' % rank)




def collect_observations(sim, args,i):
    for pose in ["start","target"]: 
        seg_label_list = []
        if args.is_syn:
            joint_info = sim.get_joint_info_w_sub()
        else:
            joint_info = sim.get_joint_info()
        all_joints = list(joint_info.keys())
        if args.rand_state:
            if pose == "start":               
                for x in all_joints:
                    v = joint_info[x]
                    if args.is_syn:
                        v = v[0]
                    lower_limit, higher_limit, range_lim = get_limit(v, args)
                    # print(f"Lower Limit: {lower_limit}, Higher Limit: {higher_limit}, Range Limit: {range_lim}")
                    start_state = np.random.choice([i for i in range(0,21,1)])
                    sim.set_joint_state(x, lower_limit+(start_state/20)*range_lim)
                sim.world.p.stepSimulation()

            elif pose == "target":
                for x in all_joints:
                    v = joint_info[x]
                    if args.is_syn:
                        v = v[0]
                    end_state = np.random.choice([i for i in range(0,start_state-3,1)]+[i for i in range(start_state+4,21,1)])
                    sim.set_joint_state(x, lower_limit+(end_state/20)*range_lim)
                sim.world.p.stepSimulation()
        if pose == "start":
            seg_label_list = []
            for joint_index in all_joints:
                v = joint_info[joint_index]
                if args.is_syn:
                    v = v[0]

                if args.is_syn:
                    depth_imgs_start, rgb_imgs_start, start_pc, start_seg_label, _ = sim.acquire_segmented_pc(6, joint_info[joint_index][1])
                    start_p_occ, start_occ_list, new_dict = sample_occ_binary(sim,joint_info[joint_index][1], args.num_point_occ, args.sample_method, args.occ_var)

                else:
                    depth_imgs_start, rgb_imgs_start, start_pc, start_seg_label, _ = sim.acquire_segmented_pc(6, [joint_index])
                    start_p_occ, start_occ_list = sample_occ(sim, args.num_point_occ, args.sample_method, args.occ_var)

                start_seg_label = np.where(start_seg_label == 1, joint_index + 1, 0)

                seg_label_list.append(start_seg_label)
            

            start_seg_label = np.max(np.stack(seg_label_list, axis=0), axis=0)
            # Downsample to fixed size
            start_pc, start_seg_label = downsample_point_cloud(start_pc, start_seg_label, num_points=int(1024*1000))
            # occ_grid_start, mins, maxs = points_to_occ_grid(start_p_occ, start_occ_list)

        else:
            seg_label_list = []
            for joint_index in all_joints:

                v = joint_info[joint_index]
                if args.is_syn:
                    v = v[0]

                if args.is_syn:
                    depth_imgs_target, rgb_imgs_target, target_pc, target_seg_label, _ = sim.acquire_segmented_pc(6, joint_info[joint_index][1])
                    target_p_occ, target_occ_list, mesh_dict = sample_occ_binary(sim, joint_info[joint_index][1], args.num_point_occ, args.sample_method, args.occ_var)

                else:
                    depth_imgs_target, rgb_imgs_target, target_pc, target_seg_label, _ = sim.acquire_segmented_pc(6, [joint_index])
                    target_p_occ, target_occ_list = sample_occ(sim, args.num_point_occ, args.sample_method, args.occ_var)

                target_seg_label = np.where(target_seg_label == 1, joint_index + 1, 0)

                seg_label_list.append(target_seg_label)

            target_seg_label = np.max(np.stack(seg_label_list, axis=0), axis=0)
            # Downsample to fixed size
            target_pc, target_seg_label = downsample_point_cloud(target_pc, target_seg_label, num_points=int(1024*1000))
            # occ_grid_target, _, _ = points_to_occ_grid(target_p_occ, target_occ_list)       
    correspondences = orb_matching(rgb_imgs_start,depth_imgs_start, rgb_imgs_target, depth_imgs_target, sim.camera.intrinsic)

    result = {
            'pc_start': start_pc,
            'pc_seg_start': start_seg_label,
            'pc_target': target_pc,
            'pc_seg_target': target_seg_label,
            'p_occ_start': start_p_occ,
            'occ_list_start': start_occ_list,
            'p_occ_target': target_p_occ,
            'occ_list_target': target_occ_list,
            'correspondences': correspondences
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
    parser.add_argument("--pos-rot", type=int, required=True)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--num-point-occ", type=int, default=100000)
    parser.add_argument("--occ-var", type=float, default=0.005)
    parser.add_argument("--sample-method", type=str, default='mix')
    parser.add_argument("--rand-state", action="store_true", help='set static joints at random state')
    parser.add_argument("--global-scaling", type=float, default=0.5)
    parser.add_argument("--dense-photo", action="store_true")


    args = parser.parse_args()
    if 'syn' in args.object_set:
        args.is_syn = True
    else:
        args.is_syn = False
    (args.root).mkdir(parents=True, exist_ok=True)
    if args.num_proc > 1:
        #print(args.num_proc)
        pool = mp.get_context("spawn").Pool(processes=args.num_proc)
        for i in range(args.num_proc):
            pool.apply_async(func=main, args=(args, i))
        pool.close()
        pool.join()
    else:
        main(args, 0)
