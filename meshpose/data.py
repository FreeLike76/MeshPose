import cv2
import numpy as np

from loguru import logger

from pathlib import Path

from meshpose.io import functional

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
    def __init__(self, id: int, p_image: Path, camera: Camera = None, image: np.ndarray = None):
        # Public
        self.id:int = id
        self.camera:Camera = camera
        
        # Private variables
        self._p_image:Path = p_image
        self._image:np.ndarray = image
    
    @property
    def image(self) -> np.ndarray:
        image = self._image
        if image is None:
            image = functional.load_image(self._p_image)
        return image

    @property
    def image_gray(self) -> np.ndarray:
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

class PresetView(View):
    def __init__(self, id: int, p_image: Path = None, camera: Camera = None, image: np.ndarray = None):
        super().__init__(id, p_image, camera, image)

class QueryView(View):
    def __init__(self, p_image: Path = None, camera: Camera = None, image: np.ndarray = None):
        super().__init__(0, p_image, camera, image)

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

    def is_2d(self) -> bool:
        return (self.keypoints_2d is not None) and (self.descriptors is not None)
    
    def is_3d(self) -> bool:
        return self.keypoints_3d is not None
    
    def set_keypoints_3d(self, keypoints_3d: np.ndarray, mask: np.ndarray = None):
        if mask is not None:
            self.keypoints_2d = self.keypoints_2d[mask]
            self.descriptors = self.descriptors[mask]
        self.keypoints_3d = keypoints_3d[mask]

class ViewMatches:
    def __init__(self, query: ViewDescription, preset: ViewDescription,
                 query_matches:np.ndarray, preset_matches:np.ndarray,
                 score:float = 1.0) -> None:
        # Store references to views
        self.query = query
        self.preset = preset

        # Store score
        self.score = score
        
        # Store matches
        self.query_matches = query_matches
        self.preset_matches = preset_matches
        
    def __len__(self):
        return min(len(self.query_matches), len(self.preset_matches))
    
    @property
    def pts2d(self):
        return self.query.keypoints_2d[self.query_matches]
    
    @property
    def pts3d(self):
        return self.preset.keypoints_3d[self.preset_matches]
