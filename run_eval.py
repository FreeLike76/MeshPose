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
    ps = pose_solver.ImagePoseSolver(views_desc[0].view.camera.intrinsics, min_inliers=20)
    image_loc = localization.ImageLocalization(views_desc=views_desc, feature_extractor=fe,
                                               matcher=mat, pose_solver=ps)
    
    step = max(1, int(1 / val))
    for i in tqdm(range(0, len(views_desc) - 1, step), desc=utils.tqdm_description("eval", desc)):
        query_desc = views_desc[i]
        
        # Run on image
        status, rmat, tvec = image_loc.run(query_desc.view, drop=i)
        
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
        # Classical
        #"ORB": extractors.ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbosity=1),
        "SIFT": extractors.ClassicalFeatureExtractor(detector="SIFT", descriptor="SIFT", verbosity=1),
        #"ROOT_SIFT": extractors.ClassicalFeatureExtractor(detector="SIFT", descriptor="ROOT_SIFT", verbosity=1),
        #"GFTT_SIFT": extractors.ClassicalFeatureExtractor(detector="GFTT", descriptor="SIFT", verbosity=1),
        # SilkFeatureExtractor
        #"SILK": extractors.SilkFeatureExtractor(checkpoints_p=Path("checkpoints/coco-rgb-aug.ckpt"), device="cuda:0", top_k=500, verbosity=verbosity),
    }
    # Run evaluation
    result = {}
    for name, fe in feature_extractors.items():
        precomputed_features = preprocess(data, {name: fe}, verbose=verbosity)
        views_desc = precomputed_features[name]
        # Norm type
        normType = cv2.NORM_L2
        if isinstance(fe, extractors.ClassicalFeatureExtractor) \
            and fe.descriptor.algorithm in ["ORB", "BRIEF"]:
            normType = cv2.NORM_HAMMING
        # Create matcher
        mat = matchers.BruteForceMatcher(
            params={
                "normType": normType,
                "crossCheck": False},
            test_ratio=True,
            test_ratio_th=0.7)
        result[name] = eval(fe, mat, views_desc, desc=name)
        # Free memory
        del fe, views_desc, mat
    
    # Print results
    result = pd.DataFrame(result)
    print(result)

def args_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset with a set of feature extractors.")
    
    parser.add_argument(
        "--data", type=str, required=False, default="data/second_room/data",
        help="Path to a dataset folder. Default is 'data/second_room/data'.")
    
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