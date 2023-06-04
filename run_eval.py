import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from loguru import logger

import argparse
from pathlib import Path

from meshpose import io, localization
from meshpose.features import extractors
        
def main(data_p: Path, verbosity: int = 1):
    # Load project
    data = io.DataIO3DSA(data_p, verbose=verbosity)
    
    # Define feature extractors
    feature_extractors = {
        # Classical
        "ORB": extractors.ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbosity=1),
        "SIFT": extractors.ClassicalFeatureExtractor(detector="SIFT", descriptor="SIFT", verbosity=1),
        "ROOT_SIFT": extractors.ClassicalFeatureExtractor(detector="GFTT", descriptor="ROOT_SIFT", verbosity=1),
        # SilkFeatureExtractor
        "SILK": extractors.SilkFeatureExtractor(checkpoints_p=Path("checkpoints/coco-rgb-aug.ckpt"),
                                                device="cuda:0", top_k=500, verbosity=verbosity)
    }
    
    # Run eval
    results = localization.functional.eval(data, feature_extractors, verbosity=verbosity)
    
    # Print results
    results = pd.DataFrame(results)
    print(results)

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