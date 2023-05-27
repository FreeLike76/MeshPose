import cv2
import numpy as np

from loguru import logger

from typing import Tuple, List

from mesh_pose.data import ViewMatches

class BasePoseSolver:
    def __init__(self, intrinsics:np.ndarray = None, dist_coeffs:np.ndarray = np.zeros((1, 5)),
                 min_matches:int = 4, min_inliers:int = 4, min_inliers_ratio:float = 0.33, verbose:bool = False) -> None:
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.min_inliers_ratio = min_inliers_ratio
        
        self.verbose = verbose
    
    def run(self, pts2d:np.ndarray, pts3d:np.ndarray, **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        raise NotImplementedError("PoseSolver is an abstract class. Use a concrete implementation instead.")
    
    def run_matches(self, view_matches:List[ViewMatches], **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        raise NotImplementedError("PoseSolver is an abstract class. Use a concrete implementation instead.")

    def _validate(self, pts2d:np.ndarray, pts3d:np.ndarray):
        assert pts2d.shape[0] == pts3d.shape[0], logger.info("Number of 2D and 3D points must be equal!")
        assert self.intrinsics is not None, logger.info("Intrinsics must be set!")
        assert self.dist_coeffs is not None, logger.info("Distortion coefficients must be set!")
        
class ImagePoseSolver(BasePoseSolver):
    """
    Basic implementation of a Pose Solver for a single image.
    """
    def __init__(self, intrinsics:np.ndarray = None, dist_coeffs:np.ndarray = np.zeros((1, 5)),
                 min_matches:int = 4, min_inliers:int = 4, min_inliers_ratio:float = 0.33, verbose:bool = False) -> None:
        super().__init__(intrinsics, dist_coeffs, min_matches, min_inliers, min_inliers_ratio, verbose)
        
    def run(self, pts2d:np.ndarray, pts3d:np.ndarray, **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        self._validate(pts2d, pts3d)
        
        # Solve PnP
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d.astype(np.float32), pts2d[:, ::-1].astype(np.float32),
                                                      self.intrinsics, self.dist_coeffs, iterationsCount=1000)
        if ret:
            len_inliers = len(inliers)
            rel_inliers = len_inliers / len(pts3d)
            if self.verbose:
                logger.info(f"Estimated {len_inliers} inliers ({rel_inliers*100:.2f}%).")
                
            if len_inliers < self.min_inliers:
                return False, None, None
            
            if rel_inliers < self.min_inliers_ratio:
                return False, None, None
        
        # Convert Rodrigues vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Return pose
        return ret, rmat, tvec
    
    def run_matches(self, view_matches:List[ViewMatches], **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        # Find best view
        best_n, best_i = 0, 0
        for view in view_matches:
            if len(view) > best_n:
                best_n = len(view)
                best_i = view_matches.index(view)
        
        # Test if enough matches, min is 4
        if best_n < self.min_matches:
            logger.warning(f"Not enough matches {best_n}/{self.min_matches}!")
            return False, None, None
        
        # Run PnP
        return self.run(view_matches[best_i].pts2d, view_matches[best_i].pts3d, **kwargs)

# TODO: video pose solver
class VideoPoseSolver(ImagePoseSolver):
    """
    Pose Solver for a video sequence. Reuses the previous frame pose as a guess for the current frame.
    Make sure to set track=True when calling run() for consecutive frames.
    """
    def __init__(self, intrinsics:np.ndarray = None, dist_coeffs:np.ndarray = np.zeros((1, 5)),
                 min_matches:int = 4, min_inliers:int = 4, min_inliers_ratio:float = 0.33, verbose:bool = False) -> None:
        super().__init__(intrinsics, dist_coeffs, min_matches, min_inliers, min_inliers_ratio, verbose)
        
        # Prev frame pose
        self.rvec = None
        self.tvec = None
        
    def run(self, pts2d:np.ndarray, pts3d:np.ndarray, track:bool=False, **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        # Validate params
        self._validate(pts2d, pts3d)
        
        # Track if possible
        _track = track and self.rvec is not None and self.tvec is not None
        
        # Solve PnP, RANSAC, Guess
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d.astype(np.float32), pts2d[:, ::-1].astype(np.float32),
                                                      self.intrinsics, self.dist_coeffs, iterationsCount=1000,
                                                      useExtrinsicGuess=_track, rvec=self.rvec, tvec=self.tvec)
        # Reset prev position
        self.rvec = None
        self.tvec = None
        
        if not ret:
            return False, None, None
        
        # Test if valid
        len_inliers = len(inliers)
        rel_inliers = len_inliers / len(pts3d)
        if self.verbose:
            logger.info(f"Estimated {len_inliers} inliers ({rel_inliers*100:.2f}%).")
        
        # Abs test
        if len_inliers < self.min_inliers:
            return False, None, None
        # Rel test
        if rel_inliers < self.min_inliers_ratio:
            return False, None, None
        
        # Save prev pose
        self.rvec = rvec
        self.tvec = tvec
        
        # Convert Rodrigues vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Return pose
        return ret, rmat, tvec
    
    def run_matches(self, view_matches:List[ViewMatches], **kwargs) -> Tuple[bool, np.ndarray, np.ndarray]:
        # Find best view
        best_n, best_i = 0, 0
        for view in view_matches:
            if len(view) > best_n:
                best_n = len(view)
                best_i = view_matches.index(view)
        
        # Test if enough matches, min is 4
        if best_n < self.min_matches:
            logger.warning(f"Not enough matches {best_n}/{self.min_matches}!")
            return False, None, None
        
        # Run PnP
        return self.run(view_matches[best_i].pts2d, view_matches[best_i].pts3d, **kwargs)