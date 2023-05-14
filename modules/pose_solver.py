import cv2
import numpy as np

from loguru import logger

class BasePoseSolver():
    def __init__(self, intrinsics:np.ndarray, dist_coeffs:np.ndarray) -> None:
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
    
    def run(self, pts2d:np.ndarray, pts3d:np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError("PoseSolver is an abstract class. Use a concrete implementation instead.")

class ImagePoseSolver(BasePoseSolver):
    def __init__(self, intrinsics:np.ndarray, dist_coeffs:np.ndarray) -> None:
        super().__init__(intrinsics, dist_coeffs)
        
    def run(self, pts2d:np.ndarray, pts3d:np.ndarray, **kwargs) -> np.ndarray:
        assert pts2d.shape[0] == pts3d.shape[0], logger.info("Number of 2D and 3D points must be equal!")
        
        # Solve PnP
        _, rvec, tvec = cv2.solvePnP(pts3d, pts2d, self.intrinsics, self.dist_coeffs, **kwargs)
        
        # Convert Rodrigues vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Return pose
        return rmat, tvec

class VideoPoseSolver(BasePoseSolver):
    def __init__(self, intrinsics:np.ndarray, dist_coeffs:np.ndarray) -> None:
        super().__init__(intrinsics, dist_coeffs)
        
        # Prev frame pose
        self.rvec = None
        self.tvec = None
        
    def run(self, pts2d:np.ndarray, pts3d:np.ndarray, track:bool=False, **kwargs) -> np.ndarray:
        assert pts2d.shape[0] == pts3d.shape[0], logger.info("Number of 2D and 3D points must be equal!")
        
        if track:
            pass # TODO: use guess from previous frame
        
        # Solve PnP
        _, rvec, tvec = cv2.solvePnP(pts3d, pts2d, self.intrinsics, self.dist_coeffs, **kwargs)
        
        # Convert Rodrigues vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Return pose
        return rmat, tvec