import cv2
import numpy as np
import open3d as o3d

from typing import List, Tuple

from copy import deepcopy

class RayCaster:
    def __init__(self) -> None:
        self.mesh = None
        self.intrinsics = None
        self.extrinsics = None
        self.height = None
        self.width = None
        
        self.scene = o3d.t.geometry.RaycastingScene()
    
    def set_mesh(self, mesh:o3d.geometry.TriangleMesh):
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    
    def set_meshes(self, meshes:List[o3d.geometry.TriangleMesh]):
        for mesh in meshes:
            self.set_mesh(mesh)
    
    def set_intrinsics(self, intrinsics:np.ndarray, width:int, height:int):
        self.intrinsics = intrinsics
        self.width = width
        self.height = height
    
    def set_extrinsics(self, extrinsics:np.ndarray):
        self.extrinsics = extrinsics
    
    def cast_rays(self, pts:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
        # Init rays
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            self.intrinsics, self.extrinsics, self.width, self.height)
        
        # If pts are provided, cast rays only for those points
        if pts is not None:
            rays = rays.numpy()
            rays = rays[pts[:, 0], pts[:, 1]]
            rays = o3d.cpu.pybind.core.Tensor(rays)
        
        # Cast rays
        ans = self.scene.cast_rays(rays)
        
        # Get distances
        distance = ans["t_hit"]
        hit_mask = distance.isfinite()
        
        # Mask out infinite distances
        finite_distance = distance[hit_mask]
        finite_rays = rays[hit_mask]
        
        # Calculate 3d coordinates
        vertices = finite_rays[:, :3] + finite_rays[:, 3:] * finite_distance.reshape((-1, 1))
        mask = hit_mask.numpy()
        
        return vertices, mask