import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from loguru import logger

import argparse
from pathlib import Path

from modules import io, localization, pose_solver, utils
from modules.localization.functional import preprocess
from modules.features import extractors, matchers

def eval(fe, mat, views_desc, val: float = 0.25, desc="FE"):
    # Result table
    result = {"TP": 0, "TN": 0, "FP": 0}
    # Run
    step = max(1, int(1 / val))
    for i in tqdm(range(0, len(views_desc) - 1, step), desc=utils.tqdm_description("eval", desc)):
        query_desc = views_desc[i]
        # Init image localization
        ps = pose_solver.ImagePoseSolver(query_desc.view.camera.intrinsics, min_inliers=20)
        image_loc = localization.ImageLocalization(
            views_desc=views_desc[:i] + views_desc[i + 1:],
            feature_extractor=fe, matcher=mat, pose_solver=ps)
        
        # Run on image
        status, rmat, tvec = image_loc.run(query_desc.view)
        
        # Next if failed
        if not status:
            result["TN"] += 1
            continue
        
        # Eval
        extrinsics = localization.functional.compose(rmat, tvec)
        ang, dist = localization.functional.compare(query_desc.view.camera.extrinsics, extrinsics)
        if ang < 20 and dist < 1:
            result["TP"] += 1
        else:
            result["FP"] += 1
    
    # Convert to %
    total = sum(result.values())
    for k, v in result.items():
        result[k] = v / total * 100
    
    return result
        
def main(data_p: Path, verbosity: int = False):
    # Load project
    data = io.DataIO3DSA(data_p, verbose=verbosity)
    
    # Define feature extractors
    feature_extractors = {
        "ORB": extractors.ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbosity=1),
        #"SIFT": extractors.ClassicalFeatureExtractor(detector="SIFT", descriptor="SIFT", verbosity=1),
        #"ROOT_SIFT": extractors.ClassicalFeatureExtractor(detector="SIFT", descriptor="ROOT_SIFT", verbosity=1),
        #"GFTT_SIFT": extractors.ClassicalFeatureExtractor(detector="GFTT", descriptor="SIFT", verbosity=1),
        # SilkFeatureExtractor
    }
    
    # Preprocess dataset
    precomputed_features = preprocess(data, feature_extractors, verbose=verbosity)
    
    # Run evaluation
    result = {}
    for extractor_name, views_desc in precomputed_features.items():
        fe = feature_extractors[extractor_name]
        mat = matchers.BruteForceMatcher(
            params={
                "normType": cv2.NORM_HAMMING if fe.descriptor.algorithm in ["ORB", "BRIEF"] else cv2.NORM_L2,
                "crossCheck": False},
            test_ratio=True,
            test_ratio_th=0.7)
        result[extractor_name] = eval(fe, mat, views_desc, desc=extractor_name)
    
    # Print results
    result = pd.DataFrame(result)
    print(result)

def args_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset with a set of feature extractors.")
    
    parser.add_argument(
        "--data", type=str, required=False, default="data/first_room/data",
        help="Path to a dataset folder. Default is 'data/first_room/data'.")
    
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