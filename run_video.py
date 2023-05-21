import cv2
import numpy as np

from loguru import logger

import argparse
from typing import Dict
from pathlib import Path
from time import time

from modules import io, localization, visualization
from modules.data import QueryView, Camera
from modules.features import extractors, matchers
from modules import pose_solver

def main(paths: Dict[str, Path], verbosity: int = False):
    # Load data
    data = io.DataIO3DSA(paths["data"], verbose=verbosity)
    views = data.load_views()
    views_desc = data.load_view_descriptions("ORB", views)
    mesh = io.functional.load_mesh(data.get_mesh_p())
    
    # Load video
    video = cv2.VideoCapture(str(paths["video"]))
    camera = Camera(views[0].camera.intrinsics)
    
    # Load AR if provided
    ar_mesh = io.functional.load_mesh(paths["ar"]) if "ar" in paths.keys() else None
    
    # Init video localization pipeline
    fe = extractors.ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbosity=1)
    
    mat = matchers.BruteForceMatcher(
        params={"normType": cv2.NORM_HAMMING, "crossCheck": False},
        test_ratio=True, test_ratio_th=0.7,
        test_symmetry=False, verbose=False)
    
    ps = pose_solver.VideoPoseSolver(camera.intrinsics, min_inliers=20, verbose=True)
    
    video_loc = localization.VideoLocalization(
        views_desc=views_desc,
        feature_extractor=fe,
        matcher=mat,
        pose_solver=ps,
        verbose=True)
    
    every = 10
    counter = 0
    while video.isOpened():
        counter += 1
        # Read video
        ret, frame = video.read()
        if not ret: break
        
        if counter % every != 0: continue
        counter = 0
        
        # Frame time
        ts = time()
        
        query_view = QueryView(None, camera, frame)
        
        # Run on image
        status, rmat, tvec = video_loc.run(query_view)
        if not status:
            logger.warning("Localization failed!")
            continue
        
        # Get extrinsics
        extrinsics = localization.functional.compose(rmat, tvec)
        
        # Render
        #scene_render = visualization.SceneRender(mesh, query_view.camera.intrinsics,
        #                                         frame.shape[0], frame.shape[1],
        #                                         extrinsics=extrinsics)
        scene_ar = visualization.SceneAR(mesh, ar_mesh,
                                         query_view.camera.intrinsics,
                                         frame.shape[0], frame.shape[1],
                                         extrinsics=extrinsics)
        display = scene_ar.run(frame)
        display = cv2.resize(display, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.imshow("image", display)
        cv2.waitKey(1)
    
        te = time()
        if verbosity: logger.info(f"Localization took {round((te - ts) * 1000, 2)} ms.")
    else:
        logger.error("Video reading failed!")
    
def args_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset with a set of feature extractors.")
    
    parser.add_argument(
        "--data", type=str, required=False, default="data/first_room/data",
        help="Path to a dataset folder. Default is data/first_room/data")
    
    parser.add_argument(
        "--video", type=str, required=False, default="data/first_room/video1.MOV",
        help="Path to a video. Default is data/first_room/video1.MOV")
    
    parser.add_argument(
        "--ar", type=str, required=False, default="data/first_room/ar/plane.obj", # TODO: none
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
    paths["video"] = args.video
    if args.ar is not None: paths["ar"] = args.ar
    
    # Validate
    for k, v in paths.items():
        p = Path(v)
        assert p.exists(), logger.error(f"Path {v} for '{k}' does not exist!")
        paths[k] = p
    
    # Run main
    main(paths, verbosity=verbosity)