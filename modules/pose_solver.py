import cv2
import numpy as np

from loguru import logger

class PoseSolver():
    def __init__(self, intrinsics:np.ndarray, dist_coeffs:np.ndarray) -> None:
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
        
    def __call__(self, pts2d:np.ndarray, pts3d:np.ndarray, **kwargs) -> np.ndarray:
        assert pts2d.shape[0] == pts3d.shape[0], logger.info("Number of 2D and 3D points must be equal!")
        
        # Solve PnP
        _, rvec, tvec = cv2.solvePnP(pts3d, pts2d, self.intrinsics, self.dist_coeffs, **kwargs)
        
        # Convert Rodrigues vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Return pose
        return rmat, tvec

# TODO: video pose solver -> take guess from previous frame
class VideoPoseSolver():
    pass