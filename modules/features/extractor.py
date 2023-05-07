import cv2
import numpy as np

from loguru import logger

from pathlib import Path
from typing import List, Tuple

from .detectors import Detector
from .descriptors import Descriptor

# Try to import torch and torchvision
TORCH_AVAILABLE = True
try:
    import torch
    import torchvision
except:
    TORCH_AVAILABLE = False

class FeatureExtractor:
    """
    Base class for feature extractors.
    """
    def __init__(self) -> None:
        pass
    
    def __call__(self, image:np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        raise NotImplementedError("FeatureExtractor is an abstract class. Use a concrete implementation instead.")

class ClassicalExtractor(FeatureExtractor):
    """
    Class for classical feature extractors.
    See `detectors.py` and `descriptors.py` for more info about supported algorithms.
    """
    def __init__(self, detector: str, descriptor: str,
                 detector_params: dict = {}, descriptor_params: dict = {},
                 verbose: bool = False) -> None:
        self.detector = Detector(detector, detector_params, verbose=verbose)
        self.descriptor = Descriptor(descriptor, descriptor_params, verbose=verbose)
        self.verbose = verbose
        
        if self.verbose: logger.info(
            f"Created feature extractor with {self.detector.algorithm} " +
            f"detector and {self.descriptor.algorithm} descriptor.")
    
    def extract(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        kp = self.detector.detect(image)
        kp, des = self.descriptor.describe(image, kp)
        return kp, des

# TODO: Implement this
class SilkFeatureExtractor(FeatureExtractor):
    """
    Class for learning-based feature extractor, named SILK.
    """
    def __init__(self, checkpoints_p: Path,
                 device: str = "cpu", verbose: bool = False) -> None:
        assert TORCH_AVAILABLE, logger.error(
            "PyTorch and TorchVision are not available! Install them to use SilkFeatureExtractor.")
        assert checkpoints_p.exists() and checkpoints_p.is_file(), logger.error(
            f"Checkpoints file {checkpoints_p} does not exist!")
        
        # Load model, etc
    
    def __call__(self, image:np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        pass
        # Do magic
        