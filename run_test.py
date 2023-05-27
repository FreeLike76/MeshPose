import cv2
import numpy as np

from loguru import logger

import argparse
from pathlib import Path

from mesh_pose import io, localization, visualization, retrieval
from mesh_pose.features import extractors, matchers
from mesh_pose import pose_solver

def main(data_p: Path, n:int=25, verbosity: int = False):    
    # Load project
    data = io.DataIO3DSA(data_p, verbose=verbosity)

    # Load data
    views = data.load_views()
    views_desc = data.load_view_descriptions("ORB", views)
    mesh = io.functional.load_mesh(data.get_mesh_p())
    
    # Define camera params
    def_intrinsics = views_desc[0].view.camera.intrinsics
    def_h, def_w = views_desc[0].view.image.shape[:2]
    
    # Create feature extractor
    fe = extractors.ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbosity=1)
    fe_norm = cv2.NORM_HAMMING
    #fe = extractors.SilkFeatureExtractor(checkpoints_p=Path("checkpoints/coco-rgb-aug.ckpt"), device="cuda:0", top_k=500, verbosity=verbosity)
    #fe_norm = cv2.NORM_L2
    
    # Create matcher
    #mat = matchers.BruteForceMatcher(
    #    params={"normType": fe_norm, "crossCheck": False},
    #    test_ratio=True, test_ratio_th=0.7,
    #    test_symmetry=False, verbose=False)
    
    mat = matchers.PytorchL2Matcher(device="cuda")
    #mat = matchers.BatchedPytorchL2Matcher(device="cuda")
    
    # Create pose solver
    ps = pose_solver.ImagePoseSolver(def_intrinsics, min_inliers=20, verbose=True)
    
    # Image retrieval
    #ret = None
    ret = retrieval.DLRetrieval(n=0.25)
    ret.train(views_desc)
    
    # Create localization pipeline
    image_loc = localization.ImageLocalization(views_desc=views_desc, feature_extractor=fe,
                                               matcher=mat, pose_solver=ps, verbose=True,
                                               image_retrieval=ret)
    
    # Render object
    scene_render = visualization.SceneRender(mesh, def_intrinsics, def_h, def_w)
    pose_render = visualization.ScenePose(mesh)
    
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
        pose_render.add_pose(extrinsics)
        scene_render.set_extrinsics(extrinsics)
        render = scene_render.run()
        
        # Show
        image = query_desc.view.image
        display = visualization.functional.compose(image, render, max_dim=1440)
        
        cv2.imshow("image", display)
        cv2.waitKey(1)
    
    pose_render.run(scale=0.35)
    
def args_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset with a set of feature extractors.")
    
    parser.add_argument(
        "--data", type=str, required=False, default="data/second_room/data",
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