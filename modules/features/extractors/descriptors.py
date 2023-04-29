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
    def __init__(self, type:str, params:dict={}, verbose:bool=False):
        assert type in DESCRIPTORS.keys(), f"Descriptor type {type} not available. Use one of {DESCRIPTORS.keys()}"
        
        self.type = type
        self.params = deepcopy(params)
        self.verbose = verbose
        
        self.descriptor = DESCRIPTORS[type].create(**params)
        if self.verbose: logger.info(f"Created {self.type} descriptor.")
    
    def descript(self, image:np.ndarray, keypoints):
        if self.verbose: logger.info(f"Computing {self.type} descriptors.")
        
        # Compute descriptors
        desc = self.descriptor.compute(image, keypoints)
        # If root sirf -> process
        if self.type == "ROOT_SIFT" and desc is not None:
            desc = desc / (desc.sum(axis=1, keepdims=True) + 1e-6)
            desc = np.sqrt(desc)
  
        if self.verbose: logger.info(f"Computed {len(desc)} descriptors")
        return desc