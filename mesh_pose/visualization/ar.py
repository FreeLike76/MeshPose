import numpy as np
import open3d as o3d

from loguru import logger

from typing import List
from copy import deepcopy

from mesh_pose.visualization.render import SceneRender

class SceneAR:
    def __init__(self, meshes:List[o3d.geometry.TriangleMesh], labels:List[bool],
                 intrinsics:np.ndarray, height:int, width:int,
                 extrinsics:np.ndarray = None, downscale:int = 4) -> None:
        # Save params
        self.intrinsics = intrinsics
        self.height = height
        self.width = width
        self.extrinsics = extrinsics
        
        # Data
        self.meshes = deepcopy(meshes)
        self.labels = labels
        
        # Raycaster
        self.scene = o3d.t.geometry.RaycastingScene()
        for mesh in self.meshes:
            self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            
        # Render
        self.renderer = SceneRender([mesh for i, mesh in enumerate(meshes) if labels[i]], intrinsics, height, width, downscale=downscale)
    
    def set_extrinsics(self, extrinsics:np.ndarray):
        """
        Updates extrinsics of the scene.
        """
        self.extrinsics = extrinsics
        self.renderer.set_extrinsics(extrinsics)
    
    def run(self, frame:np.ndarray) -> np.ndarray:
        """
        Renders AR scene on top of the given frame.
        Parameters:
        --------
        frame: np.ndarray
            Frame to render AR scene on top of.
        
        Returns:
        --------
        frame: np.ndarray
            A copy of the given frame with AR scene rendered on top of it.
        """
        assert self.extrinsics is not None, logger.error("Extrinsics have not been set!")
        
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            self.intrinsics, self.extrinsics, self.width, self.height)
        
        # Cast rays
        ans = self.scene.cast_rays(rays)
        
        # Create mask
        geometry = ans["geometry_ids"].cpu().numpy()
        mask = np.zeros((self.height, self.width), dtype=bool)
        for i in range(len(self.meshes)):
            if self.labels[i]:
                mask |= geometry == i
        # Run renderer
        vis = self.renderer.run()
        frame[mask] = vis[mask]
        return frame