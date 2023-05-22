import cv2
import numpy as np

from loguru import logger

import argparse
from pathlib import Path

from modules import io, localization, visualization
from modules.features import extractors, matchers
from modules import pose_solver

def main(data_p: Path, n:int=25, verbosity: int = False):
    # Load project
    data = io.DataIO3DSA(data_p, verbose=verbosity)

    # Load data
    views = data.load_views()
    views_desc = data.load_view_descriptions("ORB", views)
    mesh = io.functional.load_mesh(data.get_mesh_p())
  
    # Create feature extractor
    fe = extractors.ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbosity=1)
    
    # Create matcher
    mat = matchers.BruteForceMatcher(
        params={"normType": cv2.NORM_HAMMING, "crossCheck": False},
        test_ratio=True, test_ratio_th=0.7,
        test_symmetry=False, verbose=False)
    
    # Create pose solver
    ps = pose_solver.ImagePoseSolver(views_desc[0].view.camera.intrinsics, min_inliers=20, verbose=True)
    
    # Create localization pipeline
    image_loc = localization.ImageLocalization(views_desc=views_desc, feature_extractor=fe,
                                               matcher=mat, pose_solver=ps, verbose=True)
    # Run 25 photos
    step = int(len(views_desc) // n) + 1
    for i in range(0, len(views_desc), step):
        query_desc = views_desc[i]
        
        # Run on image
        status, rmat, tvec = image_loc.run(query_desc.view, drop=i)
        
        # Next if failed
        if not status:
            logger.warning("Localization failed!")
            continue
        
        # Eval
        extrinsics = localization.functional.compose(rmat, tvec)
        ang, dist = localization.functional.compare(query_desc.view.camera.extrinsics, extrinsics)
        logger.info(f"Localization successful! Error: {round(ang, 3)} deg, {round(dist, 2)} m.")
        
        # Render
        image = query_desc.view.image
        scene_render = visualization.SceneRender(mesh, query_desc.view.camera.intrinsics, image.shape[0], image.shape[1],
                                                 extrinsics=extrinsics)
        render = scene_render.run()
        
        # Show
        display = visualization.functional.compose(image, render, max_dim=1440)
        
        cv2.imshow("image", display)
        cv2.waitKey(1)
    
def args_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset with a set of feature extractors.")
    
    parser.add_argument(
        "--data", type=str, required=False, default="data/first_room/data",
        help="Path to a dataset folder. Default is data/first_room/data")
    
    parser.add_argument(
        "-n", type=int, required=False, default=25,
        help="Number of images to localize. Default is 25.")
    
    parser.add_argument(
        "--verbosity", type=int, choices=[0, 1, 2], default=1)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    
    # Get params
    data_p = Path(args.data)
    n = args.n
    verbosity = args.verbosity
    
    # Verify path
    assert data_p.exists() and data_p.is_dir(), logger.error(
        f"Directory {str(data_p)} does not exist!")
    assert n > 0, logger.error(f"Number of images to localize must be positive!")
    
    # Run main
    main(data_p, n=n, verbosity=verbosity)