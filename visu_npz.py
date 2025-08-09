import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def visualize_segmented_point_cloud(points, labels):
    """
    Visualize a point cloud with segmentation labels.
    
    Args:
        points (numpy.ndarray): (N, 3) array of xyz coordinates
        labels (numpy.ndarray): (N,) array of integer labels
    """
    # Normalize labels to start from 0
    unique_labels = np.unique(labels)
    label_to_color = {}

    # Use a color map (tab20 can handle up to 20 unique labels nicely)
    cmap = plt.get_cmap("tab20", len(unique_labels))

    for i, label in enumerate(unique_labels):
        if label == 0:  # Label 0 is often background
            label_to_color[label] = [0.5, 0.5, 0.5]  # Gray for background
        else:
            label_to_color[label] = cmap(i % 20)[:3]  # RGB, ignore alpha

    # Map each point to its color
    colors = np.array([label_to_color[label] for label in labels])

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Segmented Point Cloud",
                                      width=800, height=600,
                                      point_show_normal=False)


# Example usage
if __name__ == "__main__":
    # Load from .npy or whatever your format is
    pc_start = np.load("pc_start.npy")        # Shape (N, 3)
    pc_seg_start = np.load("pc_seg_start.npy")  # Shape (N,)

    visualize_segmented_point_cloud(pc_start, pc_seg_start)


