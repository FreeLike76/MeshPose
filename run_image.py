import cv2
import numpy as np

from loguru import logger

import argparse
from pathlib import Path

from modules import io, localization, visualization
from modules.features import extractors, matchers
from modules import pose_solver

def main(data_p: Path, verbosity: int = False):
    # Load project
    data = io.DataIO3DSA(data_p, verbose=verbosity)

    # Load data
    views = data.load_views()
    views_desc = data.load_view_descriptions("orb_orb", views)
    mesh = io.functional.load_mesh(data.get_mesh_p())
  
    # Init feature extractor
    fe = extractors.ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbosity=1)

    # Init matcher
    mat = matchers.BruteForceMatcher(
        params={"normType": cv2.NORM_HAMMING, "crossCheck": False},
        test_ratio=True, test_ratio_th=0.7,
        test_symmetry=False, verbose=False)
        
    for i in range(0, len(views_desc) - 1, 25):
        query_desc = views_desc[i]
        
        # Init pose solver
        ps = pose_solver.ImagePoseSolver(query_desc.view.camera.intrinsics, min_inliers=20)

        image_loc = localization.ImageLocalization(
            views_desc=views_desc[:i] + views_desc[i + 1:],
            feature_extractor=fe,
            matcher=mat,
            pose_solver=ps,
            verbose=True)
        #image_loc.mesh = io.functional.load_mesh(data.get_mesh_p())
        # Run on image
        status, rmat, tvec = image_loc.run(query_desc.view)
        
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
        display = visualization.functional.compose(image, render, max_dim=1080)
        cv2.imshow("image", display)
        cv2.waitKey(0)
        
        # AR
        #scene_ar = visualization.SceneAR()
    
    
def args_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset with a set of feature extractors.")
    
    parser.add_argument(
        "--data", type=str, required=False, default="data/office_model_1/",
        help="Path to a dataset folder. Default is data/office_model_1/")
    
    parser.add_argument(
        "--verbosity", type=int, choices=[0, 1, 2], default=1)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    
    # Get params
    data_p = Path(args.data)
    verbosity = args.verbosity
    
    # Verify path
    assert data_p.exists() and data_p.is_dir(), logger.error(
        f"Directory {str(data_p)} does not exist!")
    
    # Run main
    main(data_p, verbosity=verbosity)