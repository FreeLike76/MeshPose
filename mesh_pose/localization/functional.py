import cv2
import numpy as np
import open3d as o3d

from tqdm import tqdm

from typing import List, Dict, Tuple


from .image_localization import ImageLocalization
from mesh_pose import pose_solver, io
from mesh_pose.raycaster import RayCaster
from mesh_pose.data import PresetView, ViewDescription
from mesh_pose.features import extractors, matchers
from mesh_pose.utils import tqdm_description

def preprocess(data: io.DataIOBase,
               feature_extractors: Dict[str, extractors.BaseFeatureExtractor],
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
    value = value.sum()
    
    radians = 0
    if -1 < value < 1:
        radians = np.arccos(value) 
    
    distance = np.linalg.norm(t2 - t1)
    return radians * 180 / np.pi, distance


def eval_helper(fe, mat, views_desc, val: float = 0.25, desc="FE"):
    # Result table
    result = {"TP": 0, "TN": 0, "FP": 0}
    # Run
    ps = pose_solver.ImagePoseSolver(views_desc[0].view.camera.intrinsics, min_inliers=20)
    image_loc = ImageLocalization(views_desc=views_desc, feature_extractor=fe,
                                               matcher=mat, pose_solver=ps)
    
    step = max(1, int(1 / val))
    for i in tqdm(range(0, len(views_desc) - 1, step), desc=tqdm_description("localization.eval", desc)):
        query_desc = views_desc[i]
        
        # Run on image
        status, rmat, tvec = image_loc.run(query_desc.view, drop=i)
        
        # Next if failed
        if not status:
            result["TN"] += 1
            continue
        
        # Eval
        extrinsics = compose(rmat, tvec)
        ang, dist = compare(query_desc.view.camera.extrinsics, extrinsics)
        if ang < 20 and dist < 1:
            result["TP"] += 1
        else:
            result["FP"] += 1
    
    # Convert to %
    total = sum(result.values())
    for k, v in result.items():
        result[k] = v / total * 100
    
    return result

def eval(data:io.DataIOBase,
         feature_extractors:Dict[str, extractors.BaseFeatureExtractor],
         mat: matchers.BaseMatcher = None,
         verbosity:int = 1) -> dict:
    # Run evaluation
    result = {}
    for name, fe in feature_extractors.items():
        precomputed_features = preprocess(data, {name: fe}, verbose=verbosity)
        views_desc = precomputed_features[name]
        
        # Choose norm type
        normType = cv2.NORM_L2
        if isinstance(fe, extractors.ClassicalFeatureExtractor) \
            and fe.descriptor.algorithm in ["ORB", "BRIEF"]:
            normType = cv2.NORM_HAMMING
        
        # Create matcher or use provided
        _mat = mat
        if _mat is None:
            _mat = matchers.BruteForceMatcher(params={"normType": normType,
                                                      "crossCheck": False},
                                              test_ratio=True,
                                              test_ratio_th=0.7)
        
        result[name] = eval_helper(fe, _mat, views_desc, desc=name)
        # Free memory
        del fe, views_desc
    
    return result