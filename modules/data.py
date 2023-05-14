import cv2
import numpy as np

from pathlib import Path
from typing import List, Tuple, Union

from . import io

def rotate_arkit(extrinsics):
    rotation = np.array([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])
    return extrinsics @ rotation

class Camera:
    def __init__(self, transform: np.ndarray, intrinsics: np.ndarray) -> None:
        self.transform = transform
        self.intrinsics = intrinsics

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
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
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

class FrameDescription:
    def __init__(self,
                 view: PresetView,
                 keypoints_2d_cv: List[cv2.KeyPoint] = None,
                 descriptors: np.ndarray = None,
                 keypoints_3d: np.ndarray = None):
        # Validate
        if not (len(keypoints_2d_cv) == 0 and descriptors is None):
            assert len(keypoints_2d_cv) == len(descriptors)
        
        # Reference to the view, for quick access
        self.view = view
        
        # Features
        self.keypoints_2d_cv = keypoints_2d_cv
        self.descriptors = descriptors
        self.keypoints_3d = keypoints_3d

    @property
    def keypoints_2d_np(self) -> np.ndarray:
        return np.asarray([kp.pt for kp in self.keypoints_2d_cv]).astype(int)

    # TODO: where is it used? change to return all values
    def __getitem__(self, obj) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns keypoint coordinates and the descriptor if integer passed
        or returns corresponding descriptor in case cv2.KeyPoint was passed
        """
        if isinstance(obj, int):
            return self._keypoints_2d_cv[obj].pt.astype(int), self._descriptors[obj].astype(int),

        elif isinstance(obj, cv2.KeyPoint):
            idx = self._keypoints_2d_cv.index(obj)
            return self._descriptors[idx]

        else:
            raise Exception(f'Such type {type(obj)} is not supported.')

    def __str__(self) -> str:
        kp_2d_len = None if self._keypoints_2d_cv is None else len(self._keypoints_2d_cv)
        ds_len = None if self._descriptors is None else len(self._descriptors)
        kp_3d_len = None if self._keypoints_3d is None else len(self._keypoints_3d)
        return f'FrameDescription: points2d-{kp_2d_len}, descriptors2d-{ds_len}, points3d-{kp_3d_len}'

    def __len__(self) -> str:
        return len(self.keypoints_2d_cv)
    
    def __iter__(self):
        return iter((self._keypoints_2d_cv, self._descriptors, self._keypoints_3d))

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            keypoints_2d = (cv2.KeyPoint(angle=k.angle,
                                         class_id=k.class_id,
                                         octave=k.octave,
                                         x=k.pt[0],
                                         y=k.pt[1],
                                         response=k.response,
                                         size=k.size) for k in self._keypoints_2d_cv)
            descriptors = self._descriptors.copy() if self._descriptors is not None else None
            keypoints_3d = self._keypoints_3d.copy() if self._keypoints_3d is not None else None

            _copy = type(self)(tuple(keypoints_2d), descriptors, keypoints_3d)
            memo[id_self] = _copy

        return _copy

class ViewsMatch:
    def __init__(self, query_view: View, preset_view: View) -> None:
        self.query_view = query_view
        self.preset_view = preset_view

        self._query_points2d = None
        self._preset_points2d = None
        self._preset_points3d = None

        self._index = -1

    @property
    def query_points2d(self):
        if self._query_points2d is None:
            raise Exception('Cannot access `query_points2d`: they are not set.')
        return self._query_points2d

    @query_points2d.setter
    def query_points2d(self, points: np.ndarray):
        if points is not None:
            assert isinstance(points, np.ndarray)
            assert points.ndim == 2
            assert points.shape[1] == 2

            if self._preset_points2d is not None:
                assert points.shape[0] == self._preset_points2d.shape[0]

            if self._preset_points3d is not None:
                assert points.shape[0] == self._preset_points3d.shape[0]

        self._query_points2d = points

    @property
    def preset_points2d(self):
        if self._preset_points2d is None:
            raise Exception('Cannot access `preset_points2d`: they are not set.')
        return self._preset_points2d

    @preset_points2d.setter
    def preset_points2d(self, points: np.ndarray):
        if points is not None:
            assert isinstance(points, np.ndarray)
            assert points.ndim == 2
            assert points.shape[1] == 2

            if self._query_points2d is not None:
                assert points.shape[0] == self._query_points2d.shape[0]

            if self._preset_points3d is not None:
                assert points.shape[0] == self._preset_points3d.shape[0]

        self._preset_points2d = points

    @property
    def preset_points3d(self):
        if self._preset_points3d is None:
            raise Exception('Cannot access `preset_points3d`: they are not set.')
        return self._preset_points3d

    @preset_points3d.setter
    def preset_points3d(self, points: np.ndarray):
        if points is not None:
            assert points is None or isinstance(points, np.ndarray)
            assert points.ndim == 2
            assert points.shape[1] == 3

            if self._query_points2d is not None:
                assert points.shape[0] == self._query_points2d.shape[0]

            if self._preset_points2d is not None:
                assert points.shape[0] == self._preset_points2d.shape[0]

        self._preset_points3d = points

    def is_empty(self) -> bool:
        return len(self) == 0

    # TODO: where is it used? check number of returned values
    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        return self.query_view, self.preset_view, self.query_points2d[index], self.preset_points2d[index], None

    def __len__(self):
        return 0 if self._query_points2d is None else len(self._query_points2d)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self.query_points2d) - 1:
            raise StopIteration
        else:
            self._index += 1
            return self[self._index]

    def __str__(self):
        return f'Query-view {self.query_view.index} and preset-view {self.preset_view.index} have {len(self)} corresponding points.'


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