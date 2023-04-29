import cv2
import numpy as np

from loguru import logger

from pathlib import Path
from typing import List, Tuple

import detectors
import descriptors

class FeatureExtractor:
    def __init__(self) -> None:
        pass
    
    def __call__(self, image:np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        raise NotImplementedError("FeatureExtractor is an abstract class. Use a concrete implementation instead.")

class ClassicalExtractor(FeatureExtractor):
    def __init__(self, detector:Tuple[str, dict], descriptor:Tuple[str, dict], verbose:bool=False) -> None:
        self.detector = detectors.Detector(*detector, verbose=verbose)
        self.descriptor = descriptors.Descriptor(*descriptor, verbose=verbose)
        self.verbose = verbose
        
        if self.verbose: logger.info(f"Created feature extractor with {self.detector.type} detector and {self.descriptor.type} descriptor.")
    
    def __call__(self, image:np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        kp = self.detector.detect(image)
        desc = self.descriptor.descript(image, kp)
        
        return kp, desc

# TODO: Implement this
class SilkFeatureExtractor(FeatureExtractor):
    def __init__(self, checkpoints_p:Path, device:str="cpu", verbose:bool=False) -> None:
        assert checkpoints_p.exists() and checkpoints_p.is_file(), f"Checkpoints file {checkpoints_p} does not exist!"
        
        # Load model, etc
    
    def __call__(self, image:np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        pass
        # Do magic