import open3d as o3d
import numpy as np

import sys
sys.path.append('/home/suhaib/Ditto/Articulated_object_simulation-main')

file_path = './c_HighFid.ply'

mesh = o3d.io.read_triangle_mesh(file_path)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.7, 0.7, 0.7])
o3d.visualization.draw_geometries([mesh])