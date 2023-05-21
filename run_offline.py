import numpy as np

from loguru import logger

import argparse
from pathlib import Path

from modules import io
from modules.localization import preprocess
from modules.features.extractors import ClassicalFeatureExtractor, SilkFeatureExtractor

def main(data_p: Path, verbosity: int = False):
    # Load project
    data = io.DataIO3DSA(data_p, verbose=verbosity)
    
    # Define feature extractors
    feature_extractors = {
        "ORB": ClassicalFeatureExtractor(detector="ORB", descriptor="ORB", verbosity=1),
        "SIFT": ClassicalFeatureExtractor(detector="SIFT", descriptor="SIFT", verbosity=1),
        "ROOT_SIFT": ClassicalFeatureExtractor(detector="SIFT", descriptor="ROOT_SIFT", verbosity=1),
        "GFTT_SIFT": ClassicalFeatureExtractor(detector="GFTT", descriptor="SIFT", verbosity=1),
        #"gftt_rootsift": ClassicalFeatureExtractor(detector="GFTT", descriptor="ROOT_SIFT", verbosity=1),
        #"sift_sift": ClassicalFeatureExtractor(detector="SIFT", descriptor="SIFT", verbosity=1),
        # SilkFeatureExtractor
    }
    
    # Preprocess dataset
    precomputed_features = preprocess(data, feature_extractors, verbose=verbosity)
    
    # Save precomputed features
    for extractor_name, views_desc in precomputed_features.items():
        data.save_view_descriptions(extractor_name, views_desc)

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