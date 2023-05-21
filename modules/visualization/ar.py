import numpy as np
import open3d as o3d

from loguru import logger

from .render import SceneRender
from ..raycaster import RayCaster

class SceneAR:
    def __init__(self,
                 env_mesh:o3d.geometry.TriangleMesh,
                 ar_mesh:o3d.geometry.TriangleMesh,
                 intrinsics:np.ndarray,
                 height:int, width:int,
                 extrinsics:np.ndarray = None) -> None:
        # Save params
        self.intrinsics = intrinsics
        self.height = height
        self.wight = width
        self.extrinsics = extrinsics
        
        # Init scenes
        self.env_scene = RayCaster(env_mesh, intrinsics, height, width, extrinsics)
        self.ar_scene = RayCaster(ar_mesh, intrinsics, height, width, extrinsics)
        self.ar_render = SceneRender(ar_mesh, intrinsics, height, width, extrinsics)
    
    def set_extrinsics(self, extrinsics:np.ndarray):
        """
        Updates extrinsics of the scene.
        """
        self.extrinsics = extrinsics
        
        # Update scenes
        self.env_scene.set_extrinsics(extrinsics)
        self.ar_scene.set_extrinsics(extrinsics)
        self.ar_render.set_extrinsics(extrinsics)
    
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
        
        env_depth = self.env_scene.get_depth_buffer(center_intrinsics=True)
        ar_depth = self.ar_scene.get_depth_buffer(center_intrinsics=True)
        ar_frame = self.ar_render.run()
        
        mask = (ar_depth < env_depth) & (ar_depth > 0)
        
        _frame = frame.copy()
        _frame[mask] = ar_frame[mask]
        
        return _frame