import argparse
from pathlib import Path
import uuid
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from s2u.simulation import ArticulatedObjectManipulationSim

def write_data(root, data_dict,i):
    scene_id = uuid.uuid4().hex
    path = root/ f"{i:06d}.npz"
    assert not path.exists()
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
    seg_label_list = []
    if args.is_syn:
        joint_info = sim.get_joint_info_w_sub()
    else:
        joint_info = sim.get_joint_info()
    all_joints = list(joint_info.keys())
    if args.rand_state:
        for x in all_joints:
            v = joint_info[x]
            if args.is_syn:
                v = v[0]
            lower_limit, higher_limit, range_lim = get_limit(v, args)
            start_state = np.random.uniform(lower_limit, higher_limit)
            sim.set_joint_state(x, i*np.pi/20)
        sim.world.p.stepSimulation()


    for joint_index in all_joints:

        v = joint_info[joint_index]
        if args.is_syn:
            v = v[0]

        if args.is_syn:
            _, start_pc, start_seg_label, _ = sim.acquire_segmented_pc(6, joint_info[joint_index][1])
        else:
            _, start_pc, start_seg_label, _ = sim.acquire_segmented_pc(6, [joint_index])
        
        start_seg_label = np.where(start_seg_label == 1, joint_index + 1, 0)

        seg_label_list.append(start_seg_label)

    start_seg_label = np.max(np.stack(seg_label_list, axis=0), axis=0)
    # Downsample to fixed size
    start_pc, start_seg_label = downsample_point_cloud(start_pc, start_seg_label, num_points=1024)

    result = {
            'pc_start': start_pc,
            'pc_seg_start': start_seg_label,
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
    parser.add_argument("--sample-method", type=str, default='mix')
    parser.add_argument("--rand-state", action="store_true", help='set static joints at random state')
    parser.add_argument("--global-scaling", type=float, default=0.5)
    parser.add_argument("--dense-photo", action="store_true")


    args = parser.parse_args()
    if 'syn' in args.object_set:
        args.is_syn = True
    else:
        args.is_syn = False
    (args.root).mkdir(parents=True)
    if args.num_proc > 1:
        #print(args.num_proc)
        pool = mp.get_context("spawn").Pool(processes=args.num_proc)
        for i in range(args.num_proc):
            pool.apply_async(func=main, args=(args, i))
        pool.close()
        pool.join()
    else:
        main(args, 0)
