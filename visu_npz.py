import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R


# def visualize_segmented_point_cloud(points, labels):
#     """
#     Visualize a point cloud with segmentation labels.
    
#     Args:
#         points (numpy.ndarray): (N, 3) array of xyz coordinates
#         labels (numpy.ndarray): (N,) array of integer labels
#     """
#     # Normalize labels to start from 0
#     unique_labels = np.unique(labels)
#     label_to_color = {}

#     # Use a color map (tab20 can handle up to 20 unique labels nicely)
#     cmap = plt.get_cmap("tab20", len(unique_labels))

#     for i, label in enumerate(unique_labels):
#         if label == 0:  # Label 0 is often background
#             label_to_color[label] = [0.5, 0.5, 0.5]  # Gray for background
#         else:
#             label_to_color[label] = cmap(i % 20)[:3]  # RGB, ignore alpha

#     # Map each point to its color
#     colors = np.array([label_to_color[label] for label in labels])

#     # Create Open3D point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     # Visualize
#     o3d.visualization.draw_geometries([pcd],
#                                       window_name="Segmented Point Cloud",
#                                       width=800, height=600,
#                                       point_show_normal=False)

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

def points_to_occ_grid(points_occ_start, points_occ_target, occ_list_start, occ_list_target, grid_res=64):
    """
    Turn sampled points + occupancy labels into a voxel grid.
    """
    # # Normalize coordinates to [0, 1] cube

    occ_start = occ_list_start[0] | occ_list_start[1]
    occ_target = occ_list_target[0] | occ_list_target[1]
    mins = np.minimum(points_occ_start.min(0), points_occ_target.min(0))
    maxs = np.maximum(points_occ_start.max(0), points_occ_target.max(0))
    # normed_start = (points_occ_start - mins) / (maxs - mins + 1e-8)
    # normed_target = (points_occ_target-mins) / (maxs - mins + 1e-8)
    
    # Scale to grid
    idxs_start = (points_occ_start * (grid_res - 1)).astype(int)
    idxs_target = (points_occ_target * (grid_res - 1)).astype(int)
    # Build occupancy grid
    occ_grid_start = np.zeros((grid_res, grid_res, grid_res), dtype=np.uint8)
    for i, occ in zip(idxs_start, occ_start):
        x, y, z = i
        occ_grid_start[x, y, z] = max(occ_grid_start[x, y, z], occ)  # mark occupied if any point inside

    occ_grid_target = np.zeros((grid_res, grid_res, grid_res), dtype=np.uint8)
    for i, occ in zip(idxs_target, occ_target):
        x, y, z = i
        occ_grid_target[x, y, z] = max(occ_grid_target[x, y, z], occ)  # mark occupied if any point inside

    return occ_grid_start,occ_grid_target, mins, maxs

def occ_to_tsdf(occ_grid, voxel_size=0.1, trunc_dist=500):
    """
    Convert occupancy grid to TSDF.
    """
    # Distance to nearest occupied voxel (outside surface)
    outside_dist = distance_transform_edt(occ_grid == 0) * voxel_size
    # Distance to nearest free voxel (inside surface)
    inside_dist = distance_transform_edt(occ_grid == 1) * voxel_size

    tsdf = outside_dist - inside_dist  # positive outside, negative inside
    
    # Truncate
    tsdf = np.clip(tsdf, -trunc_dist * voxel_size, trunc_dist * voxel_size)

    return tsdf

def voxel_index_to_world(i, j, k, mins, maxs, grid_res=64):
    # step size along each axis
    step = (maxs - mins) / (grid_res - 1)
    return mins + step * np.array([i, j, k])

def tsdf_of_point(point, tsdf, mins, maxs, grid_res=64):
    i, j, k = point.astype(int)
    # print(f"i, j, k: {i, j, k}")
    return tsdf[i, j, k]


def tsdf_of_point1(point, tsdf, mins, maxs, grid_res=64):
    # normalize to [0, grid_res-1] float index
    norm = (point-mins) / (maxs - mins + 1e-8)
    idx_f = norm * (grid_res - 1)
    
    # integer indices
    i0 = np.floor(idx_f).astype(int)
    i1 = np.clip(i0 + 1, 0, grid_res - 1)
    d = idx_f - i0  # fractional part
    
    # get values from 8 neighbors
    c000 = tsdf[i0[0], i0[1], i0[2]]
    c001 = tsdf[i0[0], i0[1], i1[2]]
    c010 = tsdf[i0[0], i1[1], i0[2]]
    c011 = tsdf[i0[0], i1[1], i1[2]]
    c100 = tsdf[i1[0], i0[1], i0[2]]
    c101 = tsdf[i1[0], i0[1], i1[2]]
    c110 = tsdf[i1[0], i1[1], i0[2]]
    c111 = tsdf[i1[0], i1[1], i1[2]]

    # trilinear interpolation
    c00 = c000 * (1 - d[0]) + c100 * d[0]
    c01 = c001 * (1 - d[0]) + c101 * d[0]
    c10 = c010 * (1 - d[0]) + c110 * d[0]
    c11 = c011 * (1 - d[0]) + c111 * d[0]
    
    c0 = c00 * (1 - d[1]) + c10 * d[1]
    c1 = c01 * (1 - d[1]) + c11 * d[1]
    
    tsdf_val = c0 * (1 - d[2]) + c1 * d[2]
    
    return tsdf_val

def rotate_about_pivot(points,axis, angle, pivot):
    rot = R.from_rotvec(angle * axis).as_matrix()
    points_rot = ((points - pivot) @ rot.T) + pivot
    return points_rot

# Example usage
if __name__ == "__main__":
    data = np.load("./data/syn/real_cabinet_test/000000.npz")
    pc_start = data['pc_start']
    pc_seg_start = data['pc_seg_start']
    # print(pc_start.shape)
    pc_target = data['pc_target']
    pc_seg_target = data['pc_seg_target']

    start_p_occ = data['p_occ_start']
    occ_list_start = data['occ_list_start']

    target_p_occ = data['p_occ_target']
    occ_list_target = data['occ_list_target']
    print("Bounds:\n")
    print(start_p_occ.min(), start_p_occ.max())
    print(target_p_occ.min(), target_p_occ.max())
    print(pc_start.min(), pc_start.max())
    print(pc_target.min(), pc_target.max())

    bound_max = np.maximum(start_p_occ.max(0), target_p_occ.max(0))
    bound_min = np.minimum(start_p_occ.min(0), target_p_occ.min(0))
    center = (bound_min + bound_max) / 2
    scale = (bound_max - bound_min).max()
    start_p_occ = (start_p_occ - bound_min) / scale
    target_p_occ = (target_p_occ - bound_min) / scale



    occ_start_grid,occ_target_grid,mins,maxs = points_to_occ_grid(start_p_occ,target_p_occ, occ_list_start, occ_list_target,64)
    print(f"occ_start_grid shape: {occ_start_grid.shape}")
    tsdf_start = occ_to_tsdf(occ_start_grid)
    tsdf_target = occ_to_tsdf(occ_target_grid)

    print(tsdf_start.min(), tsdf_start.max())

    pc_start = (pc_start - bound_min)*63 / scale
    pc_target = (pc_target - bound_min)*63 / scale
    # pc_start, pc_seg_start = downsample_point_cloud(pc_start, pc_seg_start, num_points=int(1024*200))
    # pc_target, pc_seg_target = downsample_point_cloud(pc_target, pc_seg_target, num_points=int(1024*200))
    # print(pc_start.shape)
    correspondences = data['correspondences']

    pcd_start = o3d.geometry.PointCloud()
    pcd_start.points = o3d.utility.Vector3dVector(pc_start)
    colors_start = np.zeros_like(pc_start)
    colors_start[pc_seg_start==0] = [0.5,0.5,0.5]
    colors_start[pc_seg_start==1] = [1,0,0]
    pcd_start.colors = o3d.utility.Vector3dVector(colors_start)

    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(pc_target)
    colors_target = np.zeros_like(pc_target)
    colors_target[pc_seg_target==0] = [0.5,0.5,0.5]
    colors_target[pc_seg_target==1] = [0,1,0]
    pcd_target.colors = o3d.utility.Vector3dVector(colors_target)

    pcd_start_mobile = o3d.geometry.PointCloud()
    pcd_start_mobile.points = o3d.utility.Vector3dVector(pc_start[pc_seg_start==1])
    pcd_target_mobile = o3d.geometry.PointCloud()
    pcd_target_mobile.points = o3d.utility.Vector3dVector(pc_target[pc_seg_target==1])


    # Get AABB for both clouds
    bbox1 = pcd_start_mobile.get_axis_aligned_bounding_box()
    bbox2 = pcd_target_mobile.get_axis_aligned_bounding_box()

    # Combine bounds
    min_bound = np.minimum(bbox1.get_min_bound(), bbox2.get_min_bound())
    max_bound = np.maximum(bbox1.get_max_bound(), bbox2.get_max_bound())

    # Create new combined bounding box
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # Extract start and target points from correspondences
    correspondences[:, :3] = (correspondences[:,:3]-bound_min)*63/scale
    correspondences[:, 3:] = (correspondences[:,3:]-bound_min)*63/scale
    start_pts = correspondences[:, :3]
    target_pts = correspondences[:, 3:]

    # Check which correspondences have BOTH points inside bbox
    mask_start_inside = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(start_pts))
    mask_target_inside = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(target_pts))

    # Convert to boolean mask (True if both start and target are inside)
    mask_start_bool = np.isin(np.arange(len(start_pts)), mask_start_inside)
    mask_target_bool = np.isin(np.arange(len(target_pts)), mask_target_inside)
    mask = mask_start_bool & mask_target_bool

    # Filter correspondences
    correspondences = correspondences[mask]

    pc_corr_start = o3d.geometry.PointCloud()
    pc_corr_start.points = o3d.utility.Vector3dVector(correspondences[:,:3])
    pc_corr_start.colors = o3d.utility.Vector3dVector(np.zeros_like(correspondences[:,:3]))

    pc_corr_target = o3d.geometry.PointCloud()
    pc_corr_target.points = o3d.utility.Vector3dVector(correspondences[:,3:])
    pc_corr_target.colors = o3d.utility.Vector3dVector(np.zeros_like(correspondences[:,3:]))
    for i in range(len(correspondences)):
        if correspondences[i,:3].all==correspondences[i,3:].all:
            pc_corr_start.colors[i] = o3d.utility.Vector3dVector([0.3,1,0.7])
            pc_corr_target.colors[i] = o3d.utility.Vector3dVector([0.7,1,0.3])

    print(f"Kept {len(correspondences)} correspondences after filtering.")

    # Make LineSet from filtered correspondences
    all_points = np.vstack((correspondences[:, :3], correspondences[:, 3:]))
    lines = [[i, i + len(correspondences)] for i in range(len(correspondences))]
    line_colors = [np.random.rand(3) for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    occ_start_pcd = o3d.geometry.PointCloud()
    occ_start_pcd.points = o3d.utility.Vector3dVector(start_p_occ)
    occ_start_pcd.paint_uniform_color([0, 0, 0])

    occ_target_pcd = o3d.geometry.PointCloud()
    occ_target_pcd.points = o3d.utility.Vector3dVector(target_p_occ)
    occ_target_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    rand_idx = np.random.choice(pc_start.shape[0])
    print(f"Random Index: {rand_idx}")
    rand_pt = pc_start[rand_idx]
    print(f"Random Point: {rand_pt}")
    real_coor = rand_pt*scale +bound_min
    print(f"Point Real Value: {real_coor}")
    rotated = rotate_about_pivot(real_coor, np.array([0,0,1]),np.pi/4,np.array([0.5,0.5,0.5]))
    # rotated = (rotated-min_bound)/scale
    print(f"Rotated Point: {rotated}")
    tsdf_v = tsdf_of_point(rand_pt, tsdf_start, mins, maxs, 1)
    tsdf_v1 = tsdf_of_point(rotated, tsdf_target, mins, maxs) 
    print(f"TSDF Value: {tsdf_v}, tsdf max: {tsdf_start.max()}, tsdf min: {tsdf_start.min()}")
    print(f"TSDF Value1: {tsdf_v1}")
    n_occ_p = np.where(tsdf_start<=0.5)[0].shape[0]
    print(f"TSDF value below zero: {n_occ_p}")
    print(f"Percentage of Occupied Points: {n_occ_p/(64*64*64)*100}%")


    x, y, z = np.meshgrid(
    np.arange(tsdf_start.shape[0]),
    np.arange(tsdf_start.shape[1]),
    np.arange(tsdf_start.shape[2]),
    indexing="ij")

    coords = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    values = tsdf_start.flatten()

    # normalize values to [0,1] for coloring
    norm_values = (values - values.min()) / (values.max() - values.min())
    colors_s = plt.cm.seismic(norm_values)[:, :3]  # RGB colormap

    # create Open3D point cloud
    pcd_tsdf_start = o3d.geometry.PointCloud()
    pcd_tsdf_start.points = o3d.utility.Vector3dVector(coords)
    pcd_tsdf_start.colors = o3d.utility.Vector3dVector(colors_s)


    x, y, z = np.meshgrid(
    np.arange(tsdf_target.shape[0]),
    np.arange(tsdf_target.shape[1]),
    np.arange(tsdf_target.shape[2]),
    indexing="ij"
)
    # z = np.zeros_like(z)
    coords = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    values = tsdf_target.flatten()

    # normalize values to [0,1] for coloring
    norm_values = (values - values.min()) / (values.max() - values.min())
    colors = plt.cm.seismic(norm_values)[:, :3]  # RGB colormap

    # create Open3D point cloud
    pcd_tsdf_target = o3d.geometry.PointCloud()
    pcd_tsdf_target.points = o3d.utility.Vector3dVector(coords)
    pcd_tsdf_target.colors = o3d.utility.Vector3dVector(colors)

    # occ_start_pcd.points= o3d.utility.Vector3dVector((np.asarray(occ_start_pcd.points)-bound_min)*15/scale)
    # occ_target_pcd.points= o3d.utility.Vector3dVector((np.asarray(occ_target_pcd.points)-bound_min)*15/scale)



    # # Visualize
    o3d.visualization.draw_geometries([
        pcd_start,
        pcd_target,
        # occ_start_pcd,
        # occ_target_pcd,
        line_set,
        pc_corr_start,
        pc_corr_target,
        # pcd_tsdf_start,
        # pcd_tsdf_target,
    ])

    # o3d.visualization.draw_geometries([
    #     pcd_start,
    #     occ_start_pcd,])
# for i in range(27,28):
#     slice_idx = i  # middle slice
#     # print(f"Slice Number: {slice_idx}")
#     # print(tsdf_target[:, :, slice_idx])
#     # print(np.where(tsdf_target<=0)[0].shape)
#     # pic = np.hstack((tsdf_start[:, :, slice_idx], tsdf_target[:, :, slice_idx]))
#     # plt.imshow(pic, cmap='seismic')
#     plt.imshow(tsdf_start[:, :, slice_idx], cmap='seismic')  # red/blue for signed values
#     plt.colorbar()
#     # plt.show()
#     # p1 = (45, 26)   # row=20, col=15 (in start image)
#     # p2 = (38, 47)   # row=40, col=25 (in target image)
#     # p3 = (38,30)
#     # p4 = (36,37)
#     # # Shift the x-coordinate of p2 by width of the left image
#     # width = tsdf_start.shape[1]
#     # p2_shifted = (p2[0], p2[1] + width)
#     # p4_shifted = (p4[0], p4[1] + width)
#     # # Draw the line
#     # plt.plot([p1[1], p2_shifted[1]], [p1[0], p2_shifted[0]], 'y-', linewidth=0.8)

#     # # Mark the points
#     # plt.scatter([p1[1], p2_shifted[1]], [p1[0], p2_shifted[0]], c='green', s=30, marker='o')

#     # plt.plot([p3[1], p4_shifted[1]], [p3[0], p4_shifted[0]], 'y-', linewidth=0.8)

#     # # Mark the points
#     # plt.scatter([p3[1], p4_shifted[1]], [p3[0], p4_shifted[0]], c='green', s=30, marker='o')

#     plt.show()