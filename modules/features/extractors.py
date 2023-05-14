import cv2
import numpy as np

from loguru import logger

from pathlib import Path
from typing import List, Tuple, Union

from .detectors import Detector
from .descriptors import Descriptor
from ..utils import Serializable

# Try to import torch and torchvision
TORCH_AVAILABLE = True
try:
    import torch
    import torchvision
except:
    TORCH_AVAILABLE = False

# All defined feature extractors
DEFINED_EXTRACTORS = {}

class BaseFeatureExtractor(Serializable):
    """
    Base class for feature extractors.
    """
    def __init__(self, subclass) -> None:
        DEFINED_EXTRACTORS[subclass.__name__] = subclass
    
    def extract(self, image:np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        raise NotImplementedError("FeatureExtractor is an abstract class. Use a concrete implementation instead.")

class ClassicalFeatureExtractor(BaseFeatureExtractor):
    """
    Class for classical feature extractors.
    See `detectors.py` and `descriptors.py` for more info about supported algorithms.
    """
    def __init__(self, detector: Union[str, Detector], descriptor: Union[str, Descriptor],
                 detector_params: dict = {}, descriptor_params: dict = {},
                 verbose: bool = False) -> None:
        super().__init__(ClassicalFeatureExtractor)
        
        # Init detector
        self.detector = detector
        if isinstance(detector, str):
            self.detector = Detector(detector, detector_params, verbose=verbose)
            
        # Init descriptor
        self.descriptor = descriptor
        if isinstance(detector, str):
            self.descriptor = Descriptor(descriptor, descriptor_params, verbose=verbose)
        
        self.verbose = verbose
        if self.verbose: logger.info(
            f"Created feature extractor with {self.detector.algorithm} " +
            f"detector and {self.descriptor.algorithm} descriptor.")
    
    def extract(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        kp = self.detector.detect(image)
        kp, des = self.descriptor.describe(image, kp)
        return kp, des
    
    def to_json(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "detector": self.detector.to_json(),
            "descriptor": self.descriptor.to_json(),
            }
        
    @staticmethod
    def from_json(json: dict) -> "ClassicalFeatureExtractor":
        return ClassicalFeatureExtractor(
            detector=Detector.from_json(json["detector"]),
            descriptor=Descriptor.from_json(json["descriptor"]),
            verbose=json.get("verbose", False),
            )

# TODO: Implement this
class SilkFeatureExtractor(BaseFeatureExtractor):
    """
    Class for learning-based feature extractor, named SILK.
    """
    def __init__(self, checkpoints_p: Path, 
                 device: str = "cpu", verbose: bool = False) -> None:
        super().__init__(SilkFeatureExtractor)
        
        assert TORCH_AVAILABLE, logger.error(
            "PyTorch and TorchVision are not available! Install them to use SilkFeatureExtractor.")
        assert checkpoints_p.exists() and checkpoints_p.is_file(), logger.error(
            f"Checkpoints file {checkpoints_p} does not exist!")
        # Load model, etc
    
    def extract(self, image:np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        pass
        # Do magic