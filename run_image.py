import cv2
import numpy as np

from loguru import logger

import argparse
from pathlib import Path

from modules import io, localization
from modules.localization import preprocess
from modules.features import extractors, matchers
from modules import pose_solver

def main(data_p: Path, verbosity: int = False):
    # Load project
    data = io.DataIO3DSA(data_p, verbose=verbosity)
    
    # Define feature extractors
    #feature_extractors = {
    #    "orb_orb": ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbosity=1),
    #}
    
    # Preprocess dataset
    #precomputed_features = preprocess(data, feature_extractors, verbose=verbosity)
    
    # Save precomputed features
    #for extractor_name, views_desc in precomputed_features.items():
    #    data.save_view_descriptions(extractor_name, views_desc)

    # Read views
    views = data.load_views()
    
    # Read view descriptions
    views_desc = data.load_view_descriptions("orb_orb", views)
    
    # TEST
    query_desc = views_desc.pop(0)
    # TEST
        
    # Init feature extractor
    fe = extractors.ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbosity=1)
    
    # Init matcher
    mat = matchers.BruteForceMatcher(
        params={"normType": cv2.NORM_HAMMING, "crossCheck": False},
        test_ratio=True, test_ratio_th=0.7,
        test_symmetry=False, verbose=True)
    
    # Init pose solver
    ps = pose_solver.ImagePoseSolver(query_desc.view.camera.intrinsics)
    
    image_loc = localization.ImageLocalization(
        views_desc=views_desc,
        feature_extractor=fe,
        matcher=mat,
        pose_solver=ps,
        verbose=True)
    
    # Run on image
    status, rmat, tvec = image_loc.run(query_desc.view)
    
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