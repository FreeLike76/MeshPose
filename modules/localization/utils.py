import cv2
import numpy as np
import open3d as o3d

from tqdm import tqdm
from loguru import logger

from typing import List, Dict

from .. import io
from ..features.extractors import BaseFeatureExtractor
from ..raycaster import RayCaster
from ..data import PresetView, ViewDescription

def preprocess(data: io.DataIOBase,
               feature_extractors: Dict[str, BaseFeatureExtractor],
               verbose: bool = False) -> Dict[str, List[ViewDescription]]:
    """
    Init project and extract features from all images.
    """
    # Load project data
    views: List[PresetView] = data.load_views()
    
    # Create raycaster
    mesh: o3d.geometry.TriangleMesh = io.functional.load_mesh(data.get_mesh_p())
    raycaster = RayCaster(mesh, verbose=verbose)
    
    # Run preprocessing
    result = {}
    for extractor_name, extractor_pipeline in feature_extractors.items():
        # Describe frames
        views_desc = extractor_pipeline.run_views(views)
        
        # Filter valid
        views_desc = [vd for vd in views_desc if vd.is_valid()]
        
        # Run raycaster
        raycaster.run_views_desc(views_desc)
        
        # Save pipeline
        result[extractor_name] = views_desc
    
    return result