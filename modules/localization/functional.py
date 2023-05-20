import cv2
import numpy as np
import open3d as o3d

from typing import List, Dict, Tuple

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
        views_desc = [vd for vd in views_desc if vd.is_2d()]
        
        # Run raycaster
        raycaster.run_views_desc(views_desc)
        
        # Filter valid
        views_desc = [vd for vd in views_desc if vd.is_3d()]
        
        # Save pipeline
        result[extractor_name] = views_desc
    
    return result

def compose(rmat: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    extrinsics = np.zeros([3, 4], dtype=np.float32)
    extrinsics[:3, :3] = rmat
    extrinsics[:3, 3] = tvec.reshape((3,))
    extrinsics = np.vstack((extrinsics, [0, 0, 0, 1]))
    
    return extrinsics

def decompose(extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rmat = extrinsics[:3, :3]
    tvec = extrinsics[:3, 3]
    
    return rmat, tvec

def compare(pose1: np.ndarray, pose2: np.ndarray) -> Tuple[float, float]:
    R1, t1 = decompose(pose1)
    R2, t2 = decompose(pose2)
    value = (np.linalg.inv(R2) @ R1 @ np.array([0, 0, 1]).T) * np.array([0, 0, 1])
    radians = np.arccos(value.sum())
    radians = 0 if np.isnan(radians) else radians
    distance = np.linalg.norm(t2 - t1)
    return radians * 180 / np.pi, distance