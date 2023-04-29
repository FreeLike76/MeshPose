import cv2
import numpy as np
import open3d as o3d

from typing import List, Tuple

class RayCaster:
    def __init__(self) -> None:
        pass
    
    def set_mesh(self, mesh:o3d.geometry.TriangleMesh):
        pass
    
    def set_intrinsics(self, intrinsics:np.ndarray, width:int, height:int):
        pass
    
    def set_extrinsics(self, extrinsics:np.ndarray):
        pass
    
    def cast_rays(self, keypoints:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
        pass