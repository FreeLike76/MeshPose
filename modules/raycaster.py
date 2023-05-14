import cv2
import numpy as np
import open3d as o3d

from copy import deepcopy
from typing import List, Tuple

from .data import View, ViewDescription

class RayCaster:
    def __init__(self, mesh:o3d.geometry.TriangleMesh = None) -> None:
        self.mesh = mesh
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
    
    def set_view(self, view:View):
        """
        Sets intrinsics and extrinsics from ARView.
        """
        h, w = view.image.shape[:2]
        self.set_intrinsics(view.camera.intrinsics, w, h)
        self.set_extrinsics(view.camera.extrinsics)
    
    def run(self, pts:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def run_view(self, view:View) -> Tuple[np.ndarray, np.ndarray]:
        """
        Casts all rays for a single view.
        """
        self.set_view(view)
        return self.run()
        
    def run_view_desc(self, view_desc:ViewDescription):
        """
        Casts 2D keypoints of a single ViewDescriptions into 3D.
        
        Parameters:
        --------
        views_desc: ViewDescription
            ViewDescriptions with 2D keypoints to cast into 3D.
        """
        self.set_view(view_desc.view)
        vertices, mask = self.run(pts=view_desc.keypoints_2d)
        view_desc.set_keypoints_3d(vertices, mask=mask)
    
    def run_views_desc(self, views_desc:List[ViewDescription]):
        """
        Casts 2D keypoints of each ViewDescriptions into 3D.
        
        Parameters:
        --------
        views_desc: List[ViewDescription]
            A list of ViewDescriptions with 2D keypoints to cast into 3D.
        """
        for view_desc in views_desc:
            self.run_view_desc(view_desc)