import cv2
import numpy as np
import open3d as o3d

from tqdm import tqdm
from loguru import logger

from typing import List

from . import io, utils
from .features.extractors import BaseFeatureExtractor
from .raycaster import RayCaster
from .data import PresetView, ViewDescription
from .localization import Localization

def preprocess(data: io.DataIOBase,
               feature_extractors: List[BaseFeatureExtractor],
               verbose: bool = False):
    """
    Init project and extract features from all images.
    """
    # Load project data
    views: List[PresetView] = data.load_views()
    # Create raycaster
    mesh: o3d.geometry.TriangleMesh = io.functional.load_mesh(data.get_mesh_p())
    raycaster = RayCaster(mesh)
    
    for extractor_i, extractor in enumerate(feature_extractors):
        views_desc = extractor.run_views(views)
        views_desc = [vd for vd in views_desc if vd.is_valid()]
        
        raycaster.run_views_desc(views_desc)
        
        #save_as_id = project.add_processed_features_meta(extractor.to_json())
        #io.save_features()
        
    return 