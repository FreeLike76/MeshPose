import cv2
import numpy as np

from loguru import logger

from pathlib import Path
from typing import Tuple

from . import io

def rotate_arkit(extrinsics):
    rotation = np.array([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])
    return extrinsics @ rotation

class Camera:
    def __init__(self, intrinsics: np.ndarray, transform: np.ndarray = None) -> None:
        self.intrinsics = intrinsics
        self.transform = transform

    @property
    def extrinsics(self) -> np.ndarray:
        extrinsics = rotate_arkit(self.transform)
        return np.linalg.inv(extrinsics)

    @property
    def R(self) -> np.ndarray:
        return self.extrinsics[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.extrinsics[:3, 3]

class View:
    def __init__(self, id: int, p_image: Path, camera: Camera = None, rotate_img=False):
        # Public
        self.id = id
        self.camera = camera
        
        # Private variables
        self._p_image = p_image
        self._rotate_img = rotate_img
    
    @property
    def image(self) -> np.ndarray:
        image = io.functional.load_image(self._p_image)
        if self._rotate_img:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image

    @property
    def image_gray(self) -> np.ndarray:
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

class PresetView(View):
    def __init__(self, id: int, p_image: Path, camera: Camera = None):
        super().__init__(id, p_image, camera)

class QueryView(View):
    def __init__(self, p_image: Path, camera: Camera = None):
        super().__init__(0, p_image, camera)

class ViewDescription:
    def __init__(self,
                 view: PresetView,
                 keypoints_2d: np.ndarray = None,
                 descriptors: np.ndarray = None,
                 keypoints_3d: np.ndarray = None):
        # Validate
        if not (keypoints_2d is None and descriptors is None):
            assert keypoints_2d.shape[0] == descriptors.shape[0], logger.error(
                f"Number of keypoints: {keypoints_2d.shape[0]} is not equal " +
                f"to the number of descriptors: {descriptors.shape[0]}!")
        
        # Reference to the view, for quick access
        self.view = view
        
        # Features
        self.keypoints_2d = keypoints_2d
        self.descriptors = descriptors
        self.keypoints_3d = keypoints_3d

    def is_valid(self) -> bool:
        return (self.keypoints_2d is not None) and (self.descriptors is not None)
    
    # TODO: this
    def set_keypoints_3d(self, keypoints_3d: np.ndarray, mask: np.ndarray = None):
        if mask is not None:
            self.keypoints_2d = self.keypoints_2d[mask]
            self.descriptors = self.descriptors[mask]
        keypoints_3d = keypoints_3d[mask]

class ViewMatches:
    def __init__(self, query: ViewDescription, preset: ViewDescription, matches) -> None:
        # Store references to views
        self.query = query
        self.preset = preset

        # Store matches
        self.query_matches = np.array([m.queryIdx for m in matches], dtype=np.int32)
        self.preset_matches = np.array([m.trainIdx for m in matches], dtype=np.int32)

    def __len__(self):
        return min(len(self.query_matches), len(self.preset_matches))
    
    @property
    def pts2d(self):
        return self.query.keypoints_2d[self.query_matches]
    
    @property
    def pts3d(self):
        return self.preset.keypoints_3d[self.preset_matches]

#class ARTrajectory:
#    def __init__(self, init_view: View, init_est_pose: np.ndarray) -> None:
#        self._init_arkit: View = init_view
#        self._init_est: np.ndarray = init_est_pose
#        
#        init_pose = self._init_arkit.camera.extrinsics
#        self._delta: np.ndarray = ARTrajectory.delta(init_pose, self._init_est)
#    
#    @staticmethod
#    def delta(arkit_pose: np.ndarray, est_pose: np.ndarray):
#        return np.linalg.inv(arkit_pose) @ est_pose
#
#    def estimate_last(self, last_pose: View, delta: np.ndarray = None):
#        last_pose = last_pose.camera.extrinsics
#        delta = self._delta if delta is None else delta
#        # Long formula
#        # trj_est_pose = init_pose @ delta @ np.linalg.inv(self._init_est) @ last_pose @ delta
#        # Same but shorter
#        trj_est_pose = last_pose @ delta
#
#        return trj_est_pose
#
#    def __str__(self) -> str:
#        return f'ARTrajectory for start-view: {self._init_arkit}'
#