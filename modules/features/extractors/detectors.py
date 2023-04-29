import cv2

from copy import deepcopy
from loguru import logger

DETECTORS = {
    "GFFT": cv2.GFTTDetector,
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
    def __init__(self, type:str, params:dict={}, verbose:bool=False):
        assert type in DETECTORS.keys(), f"Detector type {type} not available. Use one of {DETECTORS.keys()}"
        
        self.type = type
        self.params = deepcopy(params)
        self.verbose = verbose
        
        self.detector = DETECTORS[type].create(**params)
        if self.verbose: logger.info(f"Created {self.type} detector.")
    
    def detect(self, image):
        if self.verbose: logger.info(f"Detecting {self.type} features.")
        kp = self.detector.detect(image)
        if self.verbose: logger.info(f"Detected {len(kp)} features.")
        
        return kp