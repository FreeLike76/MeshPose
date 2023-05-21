import cv2
import numpy as np

from typing import List

from ..data import PresetView
from .base import BaseImageRetrieval
from ..localization import functional

class PoseRetrieval(BaseImageRetrieval):
    def __init__(self):
        self.rvecs = None
        self.tvecs = None
    
    def train(self, views:List[PresetView]):
        self.rvecs = []
        self.tvecs = []
        for view in views:
            rvec, tvec = functional.decompose(view.camera.extrinsics)
            self.rvecs.append(rvec)
            self.tvecs.append(tvec)
    
    def query(self, view:PresetView, n:int=50)-> List[int]:
        pass