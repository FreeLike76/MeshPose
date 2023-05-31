import cv2
import numpy as np

from tqdm import tqdm
from loguru import logger

from pathlib import Path
from typing import List, Tuple, Union

from mesh_pose.features.detectors import Detector
from mesh_pose.features.descriptors import Descriptor
from mesh_pose.utils import Serializable, tqdm_description
from mesh_pose.data import PresetView, ViewDescription

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
    
    def run(self, image:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs feature extractor on image and returns keypoints and descriptors.
        """
        raise NotImplementedError("FeatureExtractor is an abstract class. Use a concrete implementation instead.")
    
    def run_view(self, view:PresetView) -> ViewDescription:
        """
        Runs feature extractor on a view and returns a single FrameDescription.
        """
        raise NotImplementedError("FeatureExtractor is an abstract class. Use a concrete implementation instead.")
    
    def run_views(self, views:List[PresetView]) -> List[ViewDescription]:
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
    
    def run(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Run detector and descriptor
        kp_cv = self.detector.run(image)
        kp_cv, des = self.descriptor.run(image, kp_cv)
        
        # If no keypoints -> return None
        if kp_cv is None or len(kp_cv) == 0:
            return None, None
        
        # Convert cv2.KeyPoint to np.ndarray
        kp_np = np.array([kp.pt for kp in kp_cv]).astype(np.int32)
        kp_np = np.flip(kp_np, axis=1)
        
        return kp_np, des
    
    def run_view(self, view:PresetView) -> ViewDescription:
        """
        Runs feature extractor on a view and returns a single FrameDescription.
        """
        kp, des = self.run(view.image)
        return ViewDescription(view, keypoints_2d=kp, descriptors=des)
    
    def run_views(self, views:List[PresetView]) -> List[ViewDescription]:
        """
        Runs feature extractor on all views and returns a list of FrameDescriptions.
        """
        descriptions = []
        for view in tqdm(
            views, desc=tqdm_description("mesh_pose.features.extractors", "Feature Extraction"),
            disable=self.verbosity > 1):
            
            description = self.run_view(view)
            descriptions.append(description)
        return descriptions
    
    def to_json(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "detector": self.detector.to_json(),
            "descriptor": self.descriptor.to_json(),
            "verbosity": self.verbosity,
            }
        
    @staticmethod
    def from_json(json: dict) -> "ClassicalFeatureExtractor":
        return ClassicalFeatureExtractor(
            detector=Detector.from_json(json["detector"]),
            descriptor=Descriptor.from_json(json["descriptor"]),
            verbosity=json.get("verbosity", 1),
            )

class SilkFeatureExtractor(BaseFeatureExtractor):
    """
    Class for learning-based feature extractor, named SILK.
    """
    def __init__(self, checkpoints_p: Path, 
                 device: str = "cpu", verbosity: int = 1, top_k:int=1000, down_s:int=4) -> None:
        super().__init__(SilkFeatureExtractor)
        # Assert torch and torchvision are available
        assert TORCH_AVAILABLE, logger.error(
            "PyTorch and TorchVision are not available! Install them to use SilkFeatureExtractor.")
        assert checkpoints_p.exists() and checkpoints_p.is_file(), logger.error(
            f"Checkpoints file {checkpoints_p} does not exist!")
        
        # Params
        self.device = device
        self.down_s = down_s
        self.verbosity = verbosity
        
        # Try to import SILK
        try:
            from ..silk import get_model, preprocess_image, from_feature_coords_to_image_coords
            self.model = get_model(checkpoints_p, device=device, top_k=top_k,
                                   default_outputs=("sparse_positions", "sparse_descriptors"))
            self.preprocess_image = preprocess_image
            self.conv_coords = from_feature_coords_to_image_coords
        except:
            logger.error("Error while loading SILK model!")
            
        if verbosity: logger.info(f"Created feature extractor with SILK model from {checkpoints_p}.")
        
    def run(self, image:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs feature extractor on image and returns keypoints and descriptors.
        """
        # Preprocess image
        image_pt = self.preprocess_image(image, device=self.device, res=self.down_s)
        sparse_positions, sparse_descriptors = self.model(image_pt)
        sparse_positions = self.conv_coords(self.model, sparse_positions)
        
        # Convert to numpy
        kp = (sparse_positions[0].detach().cpu().numpy()[:, :2] * self.down_s).astype(np.int32)
        des = sparse_descriptors[0].detach().cpu().numpy().astype(np.float32)
        
        return kp, des
    
    def run_view(self, view:PresetView) -> ViewDescription:
        """
        Runs feature extractor on a view and returns a single FrameDescription.
        """
        kp, des = self.run(view.image)
        return ViewDescription(view, keypoints_2d=kp, descriptors=des)
    
    def run_views(self, views:List[PresetView]) -> List[ViewDescription]:
        """
        Runs feature extractor on all views and returns a list of FrameDescriptions.
        """
        descriptions = []
        for view in tqdm(
            views, desc=tqdm_description("mesh_pose.features.extractors", "Feature Extraction"),
            disable=self.verbosity > 1):
            
            description = self.run_view(view)
            descriptions.append(description)
        return descriptions