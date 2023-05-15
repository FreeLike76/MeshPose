import cv2
import numpy as np

from loguru import logger

from typing import Tuple, List

from .data import ViewMatches

class BasePoseSolver():
    def __init__(self, intrinsics:np.ndarray = None, dist_coeffs:np.ndarray = None) -> None:
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
    
    def run(self, pts2d:np.ndarray, pts3d:np.ndarray, **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        raise NotImplementedError("PoseSolver is an abstract class. Use a concrete implementation instead.")
    
    def run_matches(self, view_matches:List[ViewMatches], **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        raise NotImplementedError("PoseSolver is an abstract class. Use a concrete implementation instead.")

class ImagePoseSolver(BasePoseSolver):
    """
    Basic implementation of a Pose Solver for a single image.
    """
    def __init__(self, intrinsics:np.ndarray = None, dist_coeffs:np.ndarray = None) -> None:
        super().__init__(intrinsics, dist_coeffs)
        
    def run(self, pts2d:np.ndarray, pts3d:np.ndarray, **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        assert pts2d.shape[0] == pts3d.shape[0], logger.info("Number of 2D and 3D points must be equal!")
        assert self.intrinsics is not None, logger.info("Intrinsics must be set!")
        assert self.dist_coeffs is not None, logger.info("Distortion coefficients must be set!")
        
        # Solve PnP
        _, rvec, tvec = cv2.solvePnP(pts3d, pts2d, self.intrinsics, self.dist_coeffs, **kwargs)
        
        # Convert Rodrigues vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Return pose
        return rmat, tvec
    
    def run_matches(self, view_matches:List[ViewMatches], **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        # Find best view
        best_n, best_i = 0, 0
        for view in view_matches:
            if len(view) > best_n:
                best_n = len(view)
                best_i = view_matches.index(view)
        
        # Test if enough matches, min is 4
        if best_n < max(4, kwargs.get("min_matches", 0)):
            return False, None, None
        
        # Run PnP
        return self.run(view_matches[best_i].pts2d, view_matches[best_i].pts3d, **kwargs)

# TODO: video pose solver
class VideoPoseSolver(ImagePoseSolver):
    """
    Pose Solver for a video sequence. Reuses the previous frame pose as a guess for the current frame.
    Make sure to set track=True when calling run() for consecutive frames.
    """
    def __init__(self, intrinsics:np.ndarray = None, dist_coeffs:np.ndarray = None) -> None:
        super().__init__(intrinsics, dist_coeffs)
        
        # Prev frame pose
        self.rvec = None
        self.tvec = None
        
    def run(self, pts2d:np.ndarray, pts3d:np.ndarray, track:bool=False, **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        assert pts2d.shape[0] == pts3d.shape[0], logger.info("Number of 2D and 3D points must be equal!")
        assert self.intrinsics is not None, logger.info("Intrinsics must be set!")
        assert self.dist_coeffs is not None, logger.info("Distortion coefficients must be set!")
        
        if track:
            pass # TODO: use guess from previous frame
        
        # Solve PnP
        _, rvec, tvec = cv2.solvePnP(pts3d, pts2d, self.intrinsics, self.dist_coeffs, **kwargs)
        
        # Convert Rodrigues vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Return pose
        return rmat, tvec
    
    def run_matches(self, view_matches:List[ViewMatches], **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        pass