import numpy as np
import open3d as o3d
import sys
sys.path.append('/home/suhaib/Ditto/Articulated_object_simulation-main')

mesh = o3d.io.read_triangle_mesh("./a_HighFid.ply")
mesh.compute_vertex_normals()
mesh.paint_uniform_color([1, 0.706, 0.3])

# # Vertex list (Nx3)
# vertices = np.array([
#     [0.0, 0.0, 0.0],  # v0
#     [1.0, 0.0, 0.0],  # v1
#     [1.0, 1.0, 0.0],  # v2
#     [0.0, 1.0, 0.0],  # v3
#     [0.5, 0.5, 1.0],  # v4 (apex)
# ], dtype=np.float64)
# # Triangle list (Mx3), each row = one triangle
# triangles = np.array([
#     [0, 2, 1],  # base triangle 1
#     [0, 3, 2],  # base triangle 2
#     [0, 1, 4],  # side triangle
#     [2, 3, 4],  # side triangle
#     [1, 2, 4],  # side triangle
#     [3, 0, 4],  # side triangle
# ], dtype=np.int32)

# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# mesh.triangles = o3d.utility.Vector3iVector(triangles)
# mesh.paint_uniform_color([0.1, 0.9, 0.1])
# mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])