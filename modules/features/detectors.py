import cv2

from copy import deepcopy
from loguru import logger

DETECTORS = {
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
    DETECTORS["SURF"] = xfeatures2d.SURF
except:
    logger.info("OpenCV-Contrib not installed. Some feature detectors will not be available.")

class Detector:
    def __init__(self, algorithm:str, params:dict={}, verbose:bool=False) -> None:
        assert algorithm in DETECTORS.keys(), f"Detector algorithmrithm {algorithm} not available. Use one of {DETECTORS.keys()}"
        
        self.algorithm = algorithm
        self.params = deepcopy(params)
        self.verbose = verbose
        
        self.detector = DETECTORS[algorithm].create(**params)
        if self.verbose: logger.info(f"Created {self.algorithm} detector.")
    
    def detect(self, image):
        if self.verbose: logger.info(f"Detecting {self.algorithm} features.")
        kp = self.detector.detect(image)
        if self.verbose: logger.info(f"Detected {len(kp)} features.")
        
        return kp