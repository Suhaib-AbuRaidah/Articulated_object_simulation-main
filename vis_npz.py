import numpy as np
import open3d as o3d
import pathlib
import glob
import os
import sys
sys.path.append('/home/suhaib/superv_Articulation')
sys.path.append('/home/suhaib/Ditto/Articulated_object_simulation-main')

# def downsample_point_cloud(points, labels=None, num_points=1024):
#     """
#     Randomly downsample the point cloud to a fixed size.
#     """
#     N = points.shape[0]
#     if N >= num_points:
#         np.random.seed(97)
#         indices = np.random.choice(N, num_points, replace=False)
#     else:
#         np.random.seed(97)
#         indices = np.random.choice(N, num_points, replace=True)  # pad if too small

#     if labels is not None:
#         labels = labels[indices]
#         return points[indices], labels
    
#     return points[indices]

def create_axis_arrow(origin, direction, length=0.08, radius=0.0001, color=[0, 0, 1],tr= np.array([0,0,0])):
    """
    Create an Open3D arrow pointing in `direction` starting at `origin`.
    """
    from scipy.spatial.transform import Rotation as R
# Translate to origin
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    end = origin + direction * (length/2 + length/10)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius*10,
        cone_radius=radius*10,
        cylinder_height=length,
        cone_height=length/10
    )
    arrow.paint_uniform_color(color)

    # Rotate arrow from +Z (default) to desired direction
    default_dir = np.array([0, 0, 1])
    rot_axis = np.cross(default_dir, direction)
    if np.linalg.norm(rot_axis) < 1e-6:
        rot_matrix = np.eye(3) if np.dot(default_dir, direction) > 0 else -np.eye(3)
    else:
        rot_angle = np.arccos(np.clip(np.dot(default_dir, direction), -1, 1))
        rot_matrix = R.from_rotvec(rot_angle * rot_axis / np.linalg.norm(rot_axis)).as_matrix()
    arrow.rotate(rot_matrix, center=(0,0,0))

    # Translate to origin
    arrow.translate(end+tr, relative=False)
    return arrow   

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

def downsample_point_cloud(points, labels_list, num_points=1024):
    """
    Randomly downsample the point cloud to a fixed size.
    """
    N = points.shape[0]
    np.random.seed(97)
    if N >= num_points:
        indices = np.random.choice(N, num_points, replace=False)
    else:
        indices = np.random.choice(N, num_points, replace=True)

    # Downsample points and labels correctly
    points = points[indices]
    new_labels_list = [labels[indices] for labels in labels_list]

    return points, new_labels_list

file_paths = "/home/suhaib/Ditto/Articulated_object_simulation-main/data/Shape2Motion_gcn/robotic_arm_testing3/scenes/*.npz"

data_list = []
for f in glob.glob(file_paths):
    data = np.load(f, allow_pickle=True)
    data_list.append(data)
print(len(data_list))

for i in range(len(data_list)):
    data = data_list[i]
    print(data.files)

    print(f"Joint type list: \n{data['joint_type']}\n")
    print(f"Adjacency matrix: \n{data['adj']}\n")
    print(f"Parts connectivity ground truth: \n{data['parts_conne_gt']}\n")
    print(f"Axis of rotation: {data['screw_axis']}\n")
    print(f"Angles between parts: {data['state_diff']}\n")

    pc_start = data['pc_start']
    print(f"pc_start.shape: {pc_start.shape}")
    pc_end = data['pc_end']
    print(f"pc_end.shape: {pc_end.shape}")

    masks_start = data['pc_seg_start'].item()
    masks_end = data['pc_seg_end'].item()
    masks_start = [masks_start[i] for i in range(len(masks_start))]
    # pc_start, masks_start = downsample_point_cloud(pc_start, masks_start, 1000)
    colors_start = np.zeros_like(pc_start)
    np.random.seed(0)
    for i in range(len(masks_start)):
        colors_start[masks_start[i]] = np.random.rand(3)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc_start)
    pcd1.colors = o3d.utility.Vector3dVector(colors_start)
    pcd1.estimate_normals()
    pcd1.orient_normals_consistent_tangent_plane(30)

    mesh1, densities1 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd1, depth=9
    )
    np.random.seed(0)

    mesh_start = data['mesh_start'].item()
    start_mesh = o3d.geometry.TriangleMesh()
    for key, value in mesh_start.items():
        vertices, triangles = value
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.random.rand(3))
        start_mesh += mesh
    np.random.seed(0)
    mesh_end = data['mesh_end'].item()
    end_mesh = o3d.geometry.TriangleMesh()
    for key, value in mesh_end.items():
        vertices, triangles = value
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.random.rand(3))
        end_mesh += mesh

    colors_end = np.zeros_like(pc_end)
    np.random.seed(0)
    for i in range(len(masks_end)):
        colors_end[masks_end[i]] = np.random.rand(3)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc_end)
    pcd2.colors = o3d.utility.Vector3dVector(colors_end)

    pcd2.estimate_normals()
    pcd2.orient_normals_consistent_tangent_plane(30)
    mesh2, densities2 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd2, depth=9
    )

    o3d.visualization.draw_geometries([start_mesh, end_mesh], mesh_show_back_face=True)
    o3d.visualization.draw_geometries([pcd1, pcd2])


























# for i in [22]:
#     print(f"index:{i}")
#     pc_start_list = []
#     pc_end_list = []
#     mask_start_list = []
#     mask_end_list = []
#     index = i
#     file = data_list[index]
#     print(len(file))
#     num_joints = int((len(file)-4)/18)
#     print(num_joints)
#     for joint in range(num_joints):
#         pc_start = file[f'pc_start_{joint}']
#         pc_end = file[f'pc_end_{joint}']
#         mask_start = file[f'pc_seg_start_{joint}']
#         mask_end = file[f'pc_seg_end_{joint}']
#         pc_start_list.append(pc_start)
#         pc_end_list.append(pc_end)
#         mask_start_list.append(mask_start)
#         mask_end_list.append(mask_end)

#     Adjacency_matrix = data_list[index]['adj']
#     parts_conne_gt = data_list[index]['parts_conne_gt']
#     print(Adjacency_matrix)
#     print(parts_conne_gt)
#     print("\n")
#     # print(parts_conne_gt)
#     # total_pc_list = []
#     # for i in range(num_joints):
#     #     pc_start = pc_start_list[i]
#     #     mask = mask_start_list[i]
#     #     pc_start = pc_start[mask]
#     #     total_pc_list.append(pc_start)
#     # pc_start = np.concatenate(total_pc_list, axis=0)
#     pc_start = pc_start_list[0].copy()
#     pc_end = pc_end_list[0].copy()
#     # pc_start, mask_start_list = downsample_point_cloud(pc_start, mask_start_list, 90000)
#     print(pc_start_list[0].shape[0])
#     pcd_start = o3d.geometry.PointCloud()
#     pcd_start.points = o3d.utility.Vector3dVector(pc_start)
#     color = np.zeros_like(pc_start)
#     # color = create_color_array_grouped(pc_start_list[0].shape[0])
#     np.random.seed(0)
#     for i in range(num_joints):
#         if i==0:
#             color[mask_start_list[i]] = np.array([1, 0, 0])
#         elif i==1:
#             color[mask_start_list[i]] = np.array([0, 1, 0])
#         elif i==2:
#             color[mask_start_list[i]] = np.array([0, 0, 1])
#         elif i==3:
#             color[mask_start_list[i]] = np.array([1, 1, 0])
#         elif i==4:
#             color[mask_start_list[i]] = np.array([1, 0, 1])
#         elif i==5:
#             color[mask_start_list[i]] = np.array([0, 1, 1])
#         elif i==6:
#             color[mask_start_list[i]] = np.array([0.5, 0.7, 1])
#         elif i==7:
#             color[mask_start_list[i]] = np.array([0.7, 1, 0.5])

#     pcd_start.colors = o3d.utility.Vector3dVector(color)
#     pcd_start.colors = o3d.utility.Vector3dVector(color)
#     pcd_end = o3d.geometry.PointCloud()
#     pcd_end.points = o3d.utility.Vector3dVector(pc_end)
#     pivot_point_pred = np.array([0.2, 0.15, 0.2])
#     joint_direction = np.array([1, 0, 0.1])
#     pivot_point_pred2 = np.array([0.2, 0.16, 0.17])
#     joint_direction2 = np.array([0, 0.05, 1])
#     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
#     sphere.paint_uniform_color([1, 0, 1]) 
#     sphere.translate(np.asarray(pivot_point_pred),relative=False)

#     sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
#     sphere2.paint_uniform_color([1, 0, 1]) 
#     sphere2.translate(np.asarray(pivot_point_pred2),relative=False)

#     arrow = create_axis_arrow(pivot_point_pred, joint_direction, length=0.07,tr=np.array([0,0,0]))
#     arrow2 = create_axis_arrow(pivot_point_pred2, joint_direction2, length=0.07,tr=np.array([0,0,0]))

#     # mesh = mesh_from_pcd(pcd_target1)
#     o3d.visualization.draw_geometries([pcd_start, pcd_end])
#     # o3d.visualization.draw_geometries([pcd_start, arrow, arrow2, sphere, sphere2])

