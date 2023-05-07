import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from loguru import logger

from typing import List

from . import io, utils
from .features.extractor import FeatureExtractor
from .project import Project

def preprocess(project: Project,
               feature_extractors: List[FeatureExtractor],
               verbose: bool = False):
    """
    Scan a dataset and extract features from all images.
    """
    
    for extractor_i, extractor in enumerate(feature_extractors):
        # TODO: load intrinsics & mesh
        
        # Create list of preset views
        preset_views: List[preset_views] = []
        for frame_i, frame_name in enumerate(project.keyframe_names):
            # Get paths
            frame_p = project.get_frame_p(frame_name)
            frame_json_p = project.get_frame_json_p(frame_name)
            
            # Load image and json
            frame = io.load_image(frame_p)
            frame_json = io.load_json(frame_json_p)
            
            kp, des = extractor(frame)
            
    return 