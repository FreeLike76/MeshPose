import numpy as np

from loguru import logger

from typing import List, Tuple

from ..data import ViewDescription, QueryView
from ..features.matchers import BaseMatcher
from ..features.extractors import BaseFeatureExtractor
from ..pose_solver import BasePoseSolver

# TODO: add retrieval
class ImageLocalization:
    def __init__(self,
                 view_desc: List[ViewDescription],
                 feature_extractor: BaseFeatureExtractor,
                 matcher: BaseMatcher,
                 pose_solver: BasePoseSolver,
                 verbose: bool = False):
        """
        Image localization pipeline.
        
        Parameters:
        --------
        view_desc: List[ViewDescription]
            List of ViewDescription objects.
        feature_extractor: BaseFeatureExtractor
            Feature extractor to use.
        matcher: BaseMatcher
            Matcher to use.
        pose_solver: BasePoseSolver
            Pose solving algorithm to use.
        verbose: bool, optional
            Whether to print additional information.
        """
        # Params
        self.verbose = verbose
        
        # Data
        self.view_desc = view_desc
        
        # Algorithms
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.pose_solver = pose_solver
        
    def run(self, query: QueryView) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Run localization on a single query view.
        
        Parameters:
        --------
        query: QueryView
            Query view.
        
        Returns:
        --------
        status: bool
            'True' if localization was successful. 'False' otherwise.
        rmat: np.ndarray
            Rotation matrix.
        tvec: np.ndarray
            Translation vector.
        """
        
        # Extract features
        if self.verbose: logger.info(f"Extracting features from query view.")
        query_desc = self.feature_extractor.run_view(query)
        
        # Check if valid
        if not query_desc.is_valid():
            logger.warning(f"Query view is not valid!")
            return False, None, None
        
        # Run retrieval
        if self.verbose: logger.info(f"Running retrieval.")
        # TODO: retrieval
        
        # Match features
        if self.verbose: logger.info(f"Matching descriptors.")
        matches = self.matcher.run(query_desc, self.view_desc)
        
        # Solve pose
        pose = self.pose_solver.run(query_desc, self.view_desc, matches)
        
        return pose