import numpy as np
import open3d as o3d
import sys
import os 
import glob
sys.path.append('/home/suhaib/Ditto/Articulated_object_simulation-main')

data_path = os.path.expanduser("~/Articulated_object_simulation-main/data/dataset/parabolic_dish_on_azimuth_elevation_mount_Canonical_zero/samples")
# # data_path = os.path.expanduser("~/particulate/dataset/train/samples")
# print (f"Data path: {data_path}")
# npz_files = glob.glob(os.path.join(data_path, "*.npz"))
# index = np.random.randint(0, len(npz_files))
# index = 17
# # for index in range(len(npz_files)):
# print(f"Index: {index}")
# npz_file = npz_files[index]
# data = np.load(npz_file)
# print(f"keys: {data.files}")
# object_path = data["object_path"]
# print(f"Object path: {object_path}")
# print(f"joints ranges:\n{data['joint_limits_start']}")
# print(f"joint ranges:\n{data['joint_limits_end']}")
# structure = data['part_structure_matrix'].astype(int)
# points_start = data['points_start']
# masks_start = data['point_to_bone_start']

# points_end = data['points_end']
# masks_end = data['point_to_bone_end']
# pcd_start = o3d.geometry.PointCloud()
# pcd_start.points = o3d.utility.Vector3dVector(points_start)
# pcd_end = o3d.geometry.PointCloud()
# pcd_end.points = o3d.utility.Vector3dVector(points_end)
# print(f"Points shape: {points_start.shape}")
# print(f"Masks shape: {masks_start.shape}")
# print(f"Unique points values: {np.unique(points_start, axis=0).shape}")
# colors_start = np.zeros_like(points_start)
# colors_end = np.zeros_like(points_end)
# base_color = np.array([0.5, 0.5, 0.5])
# light_dir = np.array([1.0, 1.0, 1.0])
# light_dir /= np.linalg.norm(light_dir)
# normals_start = data['normals_start']
# normals_end = data['normals_end']

# intensity_start = np.clip(normals_start @ light_dir, 0.0, 1.0)
# intensity_end = np.clip(normals_end @ light_dir, 0.0, 1.0)
# ambient = 0.3
# colors_start = (ambient + (1.0 - ambient) * intensity_start[:, None]) * base_color
# colors_end = (ambient + (1.0 - ambient) * intensity_end[:, None]) * base_color
# np.random.seed(42)  # For reproducibility
coloring = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,1], [0,1,1], [0.5,0.5,0.5], [1,0.5,0], [0.5,0,1], [0,0.5,1]])
# for i in range(structure.shape[0]):
#     part_indices_start = np.where(masks_start == i)[0]
#     # colors_start[part_indices_start] = coloring[i % len(coloring)]
#     part_indices_end = np.where(masks_end == i)[0]
#     colors_end[part_indices_end] = coloring[i % len(coloring)]
    

# pcd_start.colors = o3d.utility.Vector3dVector(colors_start)
# pcd_end.colors = o3d.utility.Vector3dVector(colors_end)

# # pcd_end.translate((1.0, 0, 0))  # Translate the second point cloud to the right for better visualization
# o3d.visualization.draw_geometries([pcd_start], window_name="Point Cloud and Mesh", width=800, height=600)
# o3d.visualization.draw_geometries([pcd_start, pcd_end], window_name="Point Cloud and Mesh", width=800, height=600)

print (f"Data path: {data_path}")
npz_files = glob.glob(os.path.join(data_path, "*.npz"))
index = np.random.randint(0, len(npz_files))

geometries = []
for index in range(30):
    print(f"Index: {index}")
    npz_file = npz_files[index]
    data = np.load(npz_file)
    # print(f"keys: {data.files}")
    points = data['points']
    masks = data['point_to_bone']
    nocs_g = data['nocs_g']
    nocs_p = data['nocs_p']
    obj_path = data["object_path"]
    print(f"Object path: {obj_path}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.asarray(nocs_g)
    structure = data['part_structure_matrix'].astype(int)
    masks = data['point_to_bone']
    # for i in range(structure.shape[0]):
    #     part_indices_start = np.where(masks == i)[0]
    #     colors[part_indices_start] = coloring[i % len(coloring)]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.translate((index * 1.5, 0, 0))  # Translate each point cloud for better visualization
    geometries.append(pcd)
o3d.visualization.draw_geometries(geometries, window_name="Point Cloud and Mesh", width=800, height=600)