import numpy as np

from loguru import logger

import argparse
from pathlib import Path

from modules import io
from modules.preprocess import preprocess
from modules.features.extractors import ClassicalFeatureExtractor, SilkFeatureExtractor

def main(data_p: Path, verbose: bool = False):
    # Load project
    data = io.DataIO3DSA(data_p, verbose=verbose)
    
    # Create feature extractors
    feature_extractors = [
        ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbose=verbose),
        ClassicalFeatureExtractor(detector="GFTT", descriptor="SIFT", verbose=verbose),
        ClassicalFeatureExtractor(detector="SIFT", descriptor="SIFT", verbose=verbose),
        # SilkFeatureExtractor
    ]
    # Preprocess dataset
    preprocess(data, feature_extractors, verbose=verbose)

def args_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset with a set of feature extractors.")
    
    parser.add_argument(
        "--data", type=str, required=False, default="data/office_model_1/",
        help="Path to a dataset folder. Default is data/office_model_1/")
    
    parser.add_argument(
        "--verbose", action="store_true", required=False, default=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    
    # Get params
    data_p = Path(args.data)
    verbose = args.verbose
    
    # Assert params
    assert data_p.exists() and data_p.is_dir(), logger.error(
        f"Directory {str(data_p)} does not exist!")
    
    # Run main
    main(data_p, verbose=verbose)