import cv2
import numpy as np

from copy import deepcopy
from loguru import logger

DESCRIPTORS = {
    "ORB": cv2.ORB,
    "BRISK": cv2.BRISK,
    "SIFT": cv2.SIFT,
    "ROOT_SIFT": cv2.SIFT, # Extra version of SIFT
    "KAZE": cv2.KAZE,
    "AKAZE": cv2.AKAZE,
    }

# Check if xfeatures2d is available
try:
    from cv2 import xfeatures2d
    DESCRIPTORS["SURF"] = xfeatures2d.SURF
    DESCRIPTORS["BRIEF"]: xfeatures2d.BriefDescriptorExtractor
    DESCRIPTORS["FREAK"]: xfeatures2d.FREAK
except:
    logger.info("OpenCV-Contrib not installed. Some feature descriptors will not be available.")

class Descriptor:
    def __init__(self, algorithm:str, params:dict={}, verbose:bool=False) -> None:
        assert algorithm in DESCRIPTORS.keys(), f"Descriptor algorithmrithm {algorithm} not available. Use one of {DESCRIPTORS.keys()}"
        
        self.algorithm = algorithm
        self.params = deepcopy(params)
        self.verbose = verbose
        
        self.descriptor = DESCRIPTORS[algorithm].create(**params)
        if self.verbose: logger.info(f"Created {self.algorithm} descriptor.")
    
    def describe(self, image:np.ndarray, keypoints):
        if self.verbose: logger.info(f"Computing {self.algorithm} descriptors.")
        
        # Compute descriptors
        kp, des = self.descriptor.compute(image, keypoints)
        
        # If root sirf -> process
        if self.algorithm == "ROOT_SIFT" and des is not None:
            des = des / (des.sum(axis=1, keepdims=True) + 1e-6)
            des = np.sqrt(des)
  
        if self.verbose: logger.info(f"Computed {len(des)} descriptors")
        return kp, des