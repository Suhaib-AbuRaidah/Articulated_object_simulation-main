import numpy as np
import open3d as o3d
import sys
import os 
import glob
sys.path.append('/home/suhaib/Ditto/Articulated_object_simulation-main')

def create_aabb_lineset(
    min_point: np.ndarray,
    max_point: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> o3d.geometry.LineSet:
    """Create an AABB LineSet from its minimum and maximum xyz points."""
    min_point = np.asarray(min_point, dtype=np.float64).reshape(-1)
    max_point = np.asarray(max_point, dtype=np.float64).reshape(-1)

    if min_point.shape != (3,) or max_point.shape != (3,):
        raise ValueError("min_point and max_point must each contain 3 values: [x, y, z].")
    if not np.all(np.isfinite(min_point)) or not np.all(np.isfinite(max_point)):
        raise ValueError("AABB coordinates must be finite numbers.")
    if np.any(min_point > max_point):
        raise ValueError("Each min_point coordinate must be <= max_point.")

    x_min, y_min, z_min = min_point
    x_max, y_max, z_max = max_point

    # Eight AABB corners: four on the z-min face and four on the z-max face.
    points = np.array(
        [
            [x_min, y_min, z_min],  # 0
            [x_max, y_min, z_min],  # 1
            [x_max, y_max, z_min],  # 2
            [x_min, y_max, z_min],  # 3
            [x_min, y_min, z_max],  # 4
            [x_max, y_min, z_max],  # 5
            [x_max, y_max, z_max],  # 6
            [x_min, y_max, z_max],  # 7
        ],
        dtype=np.float64,
    )

    # Four bottom edges, four top edges, and four vertical edges.
    connections = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(connections),

    )
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(connections), 1))
    )
    return line_set
if __name__ == "__main__":
    data_path = os.path.expanduser("~/Articulated_object_simulation-main/data/dataset/robotic_arm_zero/samples")

    print (f"Data path: {data_path}")
    npz_files = glob.glob(os.path.join(data_path, "*.npz"))
    index = np.random.randint(0, len(npz_files))

    geometries = []
    index = 2
    print(f"Index: {index}")
    npz_file = npz_files[index]
    data = np.load(npz_file)
    # print(f"keys: {data.files}")
    points = data['points']
    points = points *np.random.uniform(1.0, 4.0)+ np.random.uniform(0.0, 3.0)
    max = np.max(points, axis=0)
    min = np.min(points, axis=0)
    mean = np.mean(points, axis=0)
    print(f"mean: {mean}, mean_aabb: {(min+max)/2}, min: {min}, max: {max}")
    bbox_center = (min + max) / 2
    diag = max-min
    len_diag = np.linalg.norm(diag)
    nocs_original = 0.5+(points-bbox_center) / len_diag if len_diag != 0 else points - mean
    nocs_mine  = (points-min) / len_diag if len_diag != 0 else points - min
    nocs_claude = (points-min)/ np.max(max-min) if np.max(max-min) != 0 else points - min
    # points = 0.5+(points - mean)/ len_diag if len_diag != 0 else points - mean




    line_set = create_aabb_lineset(min, max, color=(1.0, 0.0, 0.0))
    line_set_fixed = create_aabb_lineset(np.array([0.0,0.0,0.0]), np.array([1.0,1.0,1.0]), color=(0.0, 1.0, 0.0))
    nocs_g = data['nocs_g']
    nocs_p = data['nocs_p']
    obj_path = data["object_path"]
    print(f"Object path: {obj_path}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.ones_like(points) * np.array([[0.5, 0.5, 0.5]])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd)

    geometries.append(line_set)
    geometries.append(line_set_fixed)

    pcd2 = o3d.geometry.PointCloud()
    colors2 = np.asarray(nocs_mine)
    pcd2.points = o3d.utility.Vector3dVector(colors2)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)
    # geometries.append(pcd2)

    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(nocs_original)
    pcd3.colors = o3d.utility.Vector3dVector(nocs_original)
    geometries.append(pcd3)

    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(nocs_claude)
    pcd4.colors = o3d.utility.Vector3dVector(nocs_claude)
    geometries.append(pcd4)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(coord_frame)
    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud and Mesh", width=800, height=600,)