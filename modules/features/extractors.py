import cv2
import numpy as np

from tqdm import tqdm
from loguru import logger

from pathlib import Path
from typing import List, Tuple, Union

from .detectors import Detector
from .descriptors import Descriptor
from ..utils import Serializable, Vebsosity
from ..data import PresetView, FrameDescription

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
    
    def run(self, image:np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Runs feature extractor on image and returns keypoints and descriptors.
        """
        raise NotImplementedError("FeatureExtractor is an abstract class. Use a concrete implementation instead.")
    
    def run_view(self, view:PresetView) -> FrameDescription:
        """
        Runs feature extractor on a view and returns a single FrameDescription.
        """
        raise NotImplementedError("FeatureExtractor is an abstract class. Use a concrete implementation instead.")
    
    def run_all(self, views:List[PresetView]) -> List[FrameDescription]:
        """
        Runs feature extractor on all views and returns a list of FrameDescriptions.
        """
        raise NotImplementedError("FeatureExtractor is an abstract class. Use a concrete implementation instead.")

class ClassicalFeatureExtractor(BaseFeatureExtractor):
    """
    Class for classical feature extractors.
    See `detectors.py` and `descriptors.py` for more info about supported algorithms.
    """
    def __init__(self, detector: Union[str, Detector], descriptor: Union[str, Descriptor],
                 detector_params: dict = {}, descriptor_params: dict = {},
                 verbosity: int = 1) -> None:
        super().__init__(ClassicalFeatureExtractor)
        self.verbosity = verbosity
        
        # Init detector
        self.detector = detector
        if isinstance(detector, str):
            self.detector = Detector(detector, detector_params, verbose=verbosity>1)
            
        # Init descriptor
        self.descriptor = descriptor
        if isinstance(detector, str):
            self.descriptor = Descriptor(descriptor, descriptor_params, verbose=verbosity>1)
        
        if self.verbosity: logger.info(
            f"Created feature extractor with {self.detector.algorithm} " +
            f"detector and {self.descriptor.algorithm} descriptor.")
    
    def run(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        kp = self.detector.run(image)
        kp, des = self.descriptor.run(image, kp)
        return kp, des
    
    def run_view(self, view:PresetView) -> FrameDescription:
        """
        Runs feature extractor on a view and returns a single FrameDescription.
        """
        kp, des = self.run(view.image)
        return FrameDescription(view, keypoints_2d_cv=kp, descriptors=des)
    
    def run_all(self, views:List[PresetView]) -> List[FrameDescription]:
        """
        Runs feature extractor on all views and returns a list of FrameDescriptions.
        """
        descriptions = []
        for view in tqdm(views, disable=self.verbosity>1):
            description = self.run_view(view)
            descriptions.append(description)
        return descriptions
    
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
    
    def run(self, image:np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Runs feature extractor on image and returns keypoints and descriptors.
        """
        raise NotImplementedError("FeatureExtractor is an abstract class. Use a concrete implementation instead.")
    
    def run_view(self, view:PresetView) -> FrameDescription:
        """
        Runs feature extractor on a view and returns a single FrameDescription.
        """
        raise NotImplementedError("FeatureExtractor is an abstract class. Use a concrete implementation instead.")
    
    def run_all(self, views:List[PresetView]) -> List[FrameDescription]:
        """
        Runs feature extractor on all views and returns a list of FrameDescriptions.
        """
        raise NotImplementedError("FeatureExtractor is an abstract class. Use a concrete implementation instead.")