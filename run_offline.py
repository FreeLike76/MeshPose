import numpy as np

from loguru import logger

from pathlib import Path

from modules.project import ProjectMeta
from modules.preprocess import preprocess
from modules.features.extractor import FeatureExtractor

def main(data_p: Path, verbose: bool = False):
    project_meta = ProjectMeta(data_p, verbose=verbose)
    
    feature_extractors = [
        FeatureExtractor(detector="ORB", descriptor="ORB", verbose=verbose),
        FeatureExtractor(detector="GFTT", descriptor="SIFT", verbose=verbose),
        FeatureExtractor(detector="SIFT", descriptor="SIFT", verbose=verbose),
    ]
    
    preprocess(project_meta, feature_extractors, verbose=verbose)

if __name__ == "__main__":
    # TODO: args    
    data_p = Path("data/office_model_1/")
    verbose = True
    
    # Assert params
    assert data_p.exists() and data_p.is_dir(), logger.error(
        f"Directory {str(data_p)} does not exist!")
    
    # Run main
    main(data_p, verbose=verbose)