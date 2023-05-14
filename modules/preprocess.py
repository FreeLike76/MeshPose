import cv2
import numpy as np
import open3d as o3d

from tqdm import tqdm
from loguru import logger

from typing import List

from . import io, utils
from .features.extractors import BaseFeatureExtractor
from .raycaster import RayCaster
from .data import PresetView, FrameDescription
from .localization import Localization

def preprocess(data: io.DataIOBase,
               feature_extractors: List[BaseFeatureExtractor],
               verbose: bool = False):
    """
    Init project and extract features from all images.
    """
    # Load project data
    preset_views: List[PresetView] = data.load_views()
    mesh: o3d.geometry.TriangleMesh = io.functional.load_mesh(data.get_mesh_p())
    raycaster = RayCaster(mesh)
    
    for extractor_i, extractor in enumerate(feature_extractors):
        KP2D, KP3D, DES = [], [], []
        
        descriptions = extractor.run_all(preset_views)
        
        for pv in preset_views:
            # Delete old features
            #pv.reset()
            
            # Raycast
            raycaster.set_view(pv)
            kp3d = raycaster.cast(kp_2d)
            
            # Save
            KP2D.append(kp_2d)
            KP3D.append(kp3d)
            DES.append(des)
            
        #save_as_id = project.add_processed_features_meta(extractor.to_json())
        #io.save_features()
        
    return 