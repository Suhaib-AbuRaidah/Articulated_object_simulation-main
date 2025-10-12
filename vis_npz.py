import numpy as np
import open3d as o3d
import pathlib
import glob
import os   
def create_color_array_grouped(N, colors=None, group_size=1500):
    """
    Create a numpy array where colors are applied in groups
    """
    if colors is None:
        colors = [
            [255, 0, 0],    # Red
            [0, 0, 255],    # Blue
            [0, 255, 0],    # Green
            [0, 0, 0]       # Black
        ]
    
    # Calculate how many complete groups we need
    num_groups = int(np.ceil(N / group_size))
    
    # Create result array
    result = np.zeros((N, 3), dtype=np.uint8)
    
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, N)
        
        # Randomly select a color for this group
        color_idx = np.random.randint(0, len(colors))
        result[start_idx:end_idx] = colors[color_idx]
    
    return result

file_paths = "./data/Shape2Motion_local/cabinet/scenes/*.npz"

data_list = []
for f in glob.glob(file_paths):
    data = np.load(f, allow_pickle=True)
    data_list.append(data)
print(len(data_list))
for i in range(len(data_list)):
    pc_start_list = []
    pc_end_list = []
    mask_start_list = []
    mask_end_list = []
    index = i
    file = data_list[index]
    print(len(file))
    num_joints = int((len(file)-4)/18)
    print(num_joints)
    for joint in range(num_joints):
        pc_start = file[f'pc_start_{joint}']
        pc_end = file[f'pc_end_{joint}']
        mask_start = file[f'pc_seg_start_{joint}']
        mask_end = file[f'pc_seg_end_{joint}']
        pc_start_list.append(pc_start)
        pc_end_list.append(pc_end)
        mask_start_list.append(mask_start)
        mask_end_list.append(mask_end)

    Adjacency_matrix = data_list[index]['adj']
    parts_conne_gt = data_list[index]['parts_conne_gt']
    print(Adjacency_matrix)
    print("\n")
    print(parts_conne_gt)
    # total_pc_list = []
    # for i in range(num_joints):
    #     pc_start = pc_start_list[i]
    #     mask = mask_start_list[i]
    #     pc_start = pc_start[mask]
    #     total_pc_list.append(pc_start)
    # pc_start = np.concatenate(total_pc_list, axis=0)

    print(pc_start_list[0].shape[0])
    pcd_start = o3d.geometry.PointCloud()
    pcd_start.points = o3d.utility.Vector3dVector(pc_start_list[0])
    color = np.zeros_like(pc_start)
    # color = create_color_array_grouped(pc_start_list[0].shape[0])
    for i in range(num_joints):
        if i==0:
            color[mask_start_list[i]] = np.array([1, 0, 0])
        elif i==1:
            color[mask_start_list[i]] = np.array([0, 1, 0])
        elif i==2:
            color[mask_start_list[i]] = np.array([0, 0, 1])
        elif i==3:
            color[mask_start_list[i]] = np.array([1, 1, 0])
        elif i==4:
            color[mask_start_list[i]] = np.array([1, 0, 1])
        elif i==5:
            color[mask_start_list[i]] = np.array([0, 1, 1])
        elif i==6:
            color[mask_start_list[i]] = np.array([0.5, 0.5, 0.5])
    pcd_start.colors = o3d.utility.Vector3dVector(color)
    pcd_start.colors = o3d.utility.Vector3dVector(color)
    pcd_end = o3d.geometry.PointCloud()
    pcd_end.points = o3d.utility.Vector3dVector(pc_end)
    o3d.visualization.draw_geometries([pcd_start])

