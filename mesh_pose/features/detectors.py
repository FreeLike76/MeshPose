import cv2
import numpy as np

from copy import deepcopy
from loguru import logger

from mesh_pose.utils import Serializable

DEFINED_DETECTORS = {
    "GFTT": cv2.GFTTDetector,
    "FAST": cv2.FastFeatureDetector,
    "ORB": cv2.ORB,
    "SIFT": cv2.SIFT,
    "KAZE": cv2.KAZE,
    "AKAZE": cv2.AKAZE,
    }

# Check if xfeatures2d is available
try:
    from cv2 import xfeatures2d
    DEFINED_DETECTORS["SURF"] = xfeatures2d.SURF
except:
    logger.info("OpenCV-Contrib not installed. Some feature detectors will not be available.")


class Detector(Serializable):
    def __init__(self, algorithm:str, params:dict={}, verbose:bool=False) -> None:
        assert algorithm in DEFINED_DETECTORS.keys(), logger.error(
            f"Detector algorithm {algorithm} is not defined. " +
            f"Use one of {DEFINED_DETECTORS.keys()}")
        
        self.algorithm = algorithm
        self.params = deepcopy(params)
        self.verbose = verbose
        
        self.detector = DEFINED_DETECTORS[algorithm].create(**params)
        if self.verbose: logger.info(f"Created {self.algorithm} detector.")
    
    def run(self, image:np.ndarray):
        if self.verbose: logger.info(f"Detecting {self.algorithm} features.")
        kp = self.detector.detect(image)
        if self.verbose: logger.info(f"Detected {len(kp)} features.")
        
        return kp
    
    def to_json(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "params": self.params,
            }
    
    @staticmethod
    def from_json(json:dict) -> "Detector":
        return Detector(json["algorithm"], json["params"])