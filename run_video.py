import cv2
import numpy as np

from loguru import logger

import argparse
from typing import Dict
from pathlib import Path
from time import time

from modules import io, localization, visualization, retrieval
from modules.data import QueryView, Camera
from modules.features import extractors, matchers
from modules import pose_solver

def main(paths: Dict[str, Path], verbosity: int = False):
    # Load data
    data = io.DataIO3DSA(paths["data"], verbose=verbosity)
    views = data.load_views()
    views_desc = data.load_view_descriptions("ORB", views)
    mesh = io.functional.load_mesh(data.get_mesh_p())
    
    # Define camera params
    intrinsics = views_desc[0].view.camera.intrinsics
    h, w = views_desc[0].view.image.shape[:2]
    
    # Load AR if provided
    ar_mesh = io.functional.load_mesh(paths["ar"]) if "ar" in paths.keys() else None
    
    # Create image retrieval
    ret = retrieval.PoseRetrieval()
    ret.train(views_desc)
    
    # Create feature extractor
    fe = extractors.ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbosity=1)
    
    # Create matcher
    mat = matchers.BruteForceMatcher(
        params={"normType": cv2.NORM_HAMMING, "crossCheck": False},
        test_ratio=True, test_ratio_th=0.7,
        test_symmetry=False, verbose=False)
    
    # Create pose solver
    ps = pose_solver.VideoPoseSolver(intrinsics,
                                     min_inliers=20, verbose=True)
    
    # Create localization pipeline
    video_loc = localization.VideoLocalization(
        views_desc=views_desc,
        feature_extractor=fe,
        matcher=mat,
        pose_solver=ps,
        track_image_retrieval=ret,
        verbose=True)
    
    # Create visualization
    scene_ar = visualization.SceneAR(mesh, ar_mesh,
                                     intrinsics, h, w)
    
    for i in range(0, len(views_desc)):
        query_view = views_desc[i].view
        query_image = query_view.image
        
        # Perfrome localization
        ts = time()
        status, rmat, tvec = video_loc.run(query_view, drop=i)
        if verbosity: logger.info(f"Localization took {round((time() - ts) * 1000, 2)} ms.")
        
        # Check if valid
        if not status:
            logger.warning("Localization failed!")
            continue
        
        # Get extrinsics
        extrinsics = localization.functional.compose(rmat, tvec)
        
        # Render
        scene_ar.set_extrinsics(extrinsics)
        display = scene_ar.run(query_image)
        display = cv2.resize(display, (query_image.shape[1] // 2, query_image.shape[0] // 2))
        display = np.rot90(display, 3)
        cv2.imshow("image", display)
        cv2.waitKey(1)
    
    else:
        logger.error("Video reading failed!")
    
def args_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset with a set of feature extractors.")
    
    parser.add_argument(
        "--data", type=str, required=False, default="data/first_room/data",
        help="Path to a dataset folder. Default is data/first_room/data")
    
    parser.add_argument(
        "--ar", type=str, required=False, default="data/first_room/ar/plane.obj",
        help="Path to AR-Mesh.")
    
    parser.add_argument(
        "--verbosity", type=int, choices=[0, 1, 2], default=1)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    
    # Get paths from args
    verbosity = args.verbosity
    
    paths = {}
    paths["data"] = args.data
    if args.ar is not None: paths["ar"] = args.ar
    
    # Validate
    for k, v in paths.items():
        p = Path(v)
        assert p.exists(), logger.error(f"Path {v} for '{k}' does not exist!")
        paths[k] = p
    
    # Run main
    main(paths, verbosity=verbosity)