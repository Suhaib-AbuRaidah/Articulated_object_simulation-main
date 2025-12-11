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

file_paths = "/home/suhaib/Ditto/Articulated_object_simulation-main/data/Shape2Motion_gcn/test/cabinet/scenes/*.npz"

data_list = []
for f in sorted(glob.glob(file_paths)):
    try:
        data = np.load(f, allow_pickle=True)
        data_list.append(data)
    except Exception as e:
        print(f"Error loading {f}: {e}")
print(len(data_list))

for j in range(len(data_list)):
    data = data_list[j]
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
    links_index = data['links_index']
    print(f"\n\nlinks_index: {links_index}\n\n")
    masks_start = [masks_start[i] for i in links_index]
    masks_end = [masks_end[i] for i in links_index]
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

    print(f"example #: {j+1}")
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
    o3d.visualization.draw_geometries([pcd1,pcd2])







