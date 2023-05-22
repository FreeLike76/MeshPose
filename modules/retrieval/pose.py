import cv2
import numpy as np

from typing import List

from ..data import ViewDescription, PresetView
from .base import BaseImageRetrieval
from ..localization import functional

class PoseRetrieval(BaseImageRetrieval):
    def __init__(self, n:float=0.5):
        self.rvecs = None
        self.tvecs = None
        
        self.last_rmat = None
        self.last_tvec = None
        
        self.n = n
    
    def train(self, views_desc:List[ViewDescription]):
        self.rvecs = []
        self.tvecs = []
        for vd in views_desc:
            rvec, tvec = functional.decompose(vd.view.camera.extrinsics)
            self.rvecs.append(rvec)
            self.tvecs.append(tvec)
        self.rvecs = np.array(self.rvecs, dtype=np.float32)
        self.tvecs = np.array(self.tvecs, dtype=np.float32)
    
    def set_extrinsic(self, extrinsics:np.ndarray):
        self.last_rmat, self.last_tvec = functional.decompose(extrinsics)
    
    def set_pose(self, rmat:np.ndarray, tvec:np.ndarray):
        self.last_rmat = rmat
        self.last_tvec = tvec.reshape((-1))
    
    def query(self, view:PresetView)-> List[int]:
        # Get last extrinsics
        R1, t1 = self.last_rmat, self.last_tvec
        
        # Array of distances
        pose_difference = np.zeros(self.rvecs.shape[0], dtype=np.float32)
        
        for i, (R2, t2) in enumerate(zip(self.rvecs, self.tvecs)):
            radians = (np.linalg.inv(R2) @ R1 @ np.array([0, 0, 1]).T) * np.array([0, 0, 1])
            radians = np.arccos(radians.sum())
            radians = 0 if np.isnan(radians) else radians / np.pi
            distance = np.linalg.norm(t2 - t1)
            
            # Calculate pose difference
            pose_difference[i] = radians * (distance + 1)
        
        # Get n% of the best matches
        n_ret = int(self.n * len(pose_difference))
        return np.argsort(pose_difference)[:n_ret]