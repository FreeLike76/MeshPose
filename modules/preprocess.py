import cv2
import numpy as np
import open3d as o3d

from tqdm import tqdm
from loguru import logger

from typing import List

from . import core, io, utils
from .features.extractor import FeatureExtractor
from .project import ProjectMeta
from .raycaster import RayCaster
from .data import PresetARView, FrameDescription

def preprocess(project: ProjectMeta,
               feature_extractors: List[FeatureExtractor],
               verbose: bool = False):
    """
    Init project and extract features from all images.
    """
    # Load project data
    preset_views: List[PresetARView] = core.load_preset_views(project)
    mesh: o3d.geometry.TriangleMesh = io.load_mesh(project.get_mesh_p())
    raycaster = RayCaster(mesh)
    
    for extractor_i, extractor in enumerate(feature_extractors):
        for pv in preset_views:
            # Delete old features
            pv.reset()
            
            kp, des = extractor(pv.image)
            frame_des = FrameDescription(keypoints_2d_cv=kp, descriptors=des)
            pv.description = frame_des
    return 