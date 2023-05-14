import numpy as np
import open3d as o3d

from ..raycaster import RayCaster

class SceneAR:
    def __init__(self,
                 env_mesh:o3d.geometry.TriangleMesh,
                 ar_mesh:o3d.geometry.TriangleMesh) -> None:
        # Create two raycasters
        self.env_scene = RayCaster(env_mesh)
        self.ar_scene = RayCaster(ar_mesh)
        self.ar_render = None # TODO
        
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
    
    def run(self, frame:np.ndarray) -> np.ndarray:
        # TODO:
        # 1. Get depth buffer from AR
        # 2. Get depth buffer from env
        # 3. Cover AR with env
        # 4. Render AR on frame