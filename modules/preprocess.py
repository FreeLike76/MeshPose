import cv2
import numpy as np
import open3d as o3d

from tqdm import tqdm
from loguru import logger

from typing import List

from . import core, io, utils
from .features.extractors import FeatureExtractor
from .project import ProjectMeta
from .raycaster import RayCaster
from .data import PresetARView, FrameDescription
from .localization import Localization

def preprocess(project: ProjectMeta,
               feature_extractors: List[FeatureExtractor],
               verbose: bool = False):
    """
    Init project and extract features from all images.
    """
    # Load project data
    preset_views: List[PresetARView] = Localization.load_preset_views(project)
    mesh: o3d.geometry.TriangleMesh = io.load_mesh(project.get_mesh_p())
    raycaster = RayCaster(mesh)
    
    for extractor_i, extractor in enumerate(feature_extractors):
        KP2D, KP3D, DES = [], [], []
        for pv in preset_views:
            # Delete old features
            #pv.reset()
            
            # Get new features
            kp_cv, des = extractor(pv.image)
            
            # To numpy
            kp_2d = np.asarray([k.pt for k in kp_cv])
            des = np.asarray(des)
            
            # Raycast
            raycaster.set_view(pv)
            kp3d = raycaster.cast(kp_2d)
            
            # Save
            KP2D.append(kp_2d)
            KP3D.append(kp3d)
            DES.append(des)
            
        save_as_id = project.add_processed_features_meta(extractor.to_json())
        io.save_features()
        
    return 