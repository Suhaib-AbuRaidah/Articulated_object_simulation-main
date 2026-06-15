import numpy as np
import open3d as o3d
import sys
import os 
import glob
sys.path.append('/home/suhaib/Ditto/Articulated_object_simulation-main')

data_path = os.path.expanduser("~/Articulated_object_simulation-main/data/Serial3/train/samples")
npz_files = glob.glob(os.path.join(data_path, "*.npz"))
index = np.random.randint(0, len(npz_files))
print(f"Index: {index}")
npz_file = npz_files[index]
data = np.load(npz_file)
object_path = data["object_path"]
print(f"Object path: {object_path}")
structure = data['part_structure_matrix'].astype(int)

points = data['points']
masks = data['point_to_bone']
print(f"Points shape: {points.shape}")
print(f"Masks shape: {masks.shape}")
print(f"Unique points values: {np.unique(points).shape}")
colors = np.zeros_like(points)
np.random.seed(42)  # For reproducibility
for i in range(structure.shape[0]):
    part_indices = np.where(masks == i)[0]
    colors[part_indices] = np.random.rand(3)
     
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])