import cv2
import numpy as np
import open3d as o3d

from tqdm import tqdm
from typing import List, Tuple

from mesh_pose.utils import tqdm_description
from mesh_pose.data import View, ViewDescription

class RayCaster:
    def __init__(self, mesh:o3d.geometry.TriangleMesh = None,
                 intrinsics:np.ndarray = None,
                 height:int = None, width:int = None,
                 extrinsics:np.ndarray = None, verbose:bool = False) -> None:
        # Init empty
        self.mesh = None
        self.scene = None
        
        # No params
        self.intrinsics = intrinsics
        self.height = height
        self.width = width
        self.extrinsics = extrinsics
        
        self.verbose = verbose
        
        if mesh is not None:
            self.set_mesh(mesh)
    
    def set_mesh(self, mesh:o3d.geometry.TriangleMesh):
        self.mesh = mesh
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    
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
    
    def get_depth_buffer(self, center_intrinsics:bool=False) -> np.ndarray:
        """
        Returns depth buffer of the scene from the current view.
        """
        if center_intrinsics:
            intrinsics = self.intrinsics.copy()
            intrinsics[0, 2] = self.width / 2
            intrinsics[1, 2] = self.height / 2
        else:
            intrinsics = self.intrinsics
            
        # Init rays
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsics, self.extrinsics, self.width, self.height)
        
        # Cast rays
        ans = self.scene.cast_rays(rays)
        
        # Get distances, infinity -> -1, reshape back to image
        distance = ans["t_hit"]
        distance = distance.numpy().astype(np.float32)
        distance[~np.isfinite(distance)] = -1
        distance = distance.reshape((self.height, self.width))
        
        return distance
    
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
        
        # Get finite
        mask = distance.isfinite()
        mask = mask.numpy()
        
        # Calculate 3d coordinates
        vertices = rays[:, :3] + rays[:, 3:] * distance.reshape((-1, 1))
        vertices = vertices.numpy()
        
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
        for view_desc in tqdm(views_desc,
                              desc=tqdm_description("mesh_pose.raycaster", "Ray Casting"),
                              disable=(not self.verbose)):
            self.run_view_desc(view_desc)