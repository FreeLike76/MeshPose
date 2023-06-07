import numpy as np

from loguru import logger

from time import time
from typing import List, Tuple

from ..data import ViewDescription, QueryView
from meshpose.features.matchers import BaseMatcher
from meshpose.features.extractors import BaseFeatureExtractor
from meshpose.pose_solver import BasePoseSolver
from meshpose.retrieval import BaseImageRetrieval

class ImageLocalization:
    def __init__(self,
                 views_desc: List[ViewDescription],
                 feature_extractor: BaseFeatureExtractor,
                 matcher: BaseMatcher,
                 pose_solver: BasePoseSolver,
                 image_retrieval: BaseImageRetrieval = None,
                 verbose: bool = False):
        """
        Image localization pipeline.
        
        Parameters:
        --------
        views_desc: List[ViewDescription]
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
        self.views_desc = views_desc
        
        # Algorithms
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.pose_solver = pose_solver
        
        self.image_retrieval = image_retrieval
        
    def run(self, query: QueryView, drop:int=None) -> Tuple[bool, np.ndarray, np.ndarray]:
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
        ts = time()
        if self.verbose: logger.info(f"Extracting features from query view.")
        query_desc = self.feature_extractor.run_view(query)
        if self.verbose: logger.info(f"Feature extraction took {round((time() - ts) * 1000, 2)} ms.")
        
        # Check if valid
        if not query_desc.is_2d():
            logger.warning(f"Query view is not valid!")
            return False, None, None
        
        # Run retrieval
        indices = np.arange(len(self.views_desc))
        if self.image_retrieval is not None:
            if self.verbose: logger.info(f"Running retrieval.")
            indices = self.image_retrieval.query(query_desc)
        
        # Get retrieved
        match_views = [self.views_desc[i] for i in indices if i != drop]
            
        # Match features
        ts = time()
        if self.verbose: logger.info(f"Matching descriptors.")
        matches = self.matcher.run_views_desc(query_desc, match_views)
        if self.verbose: logger.info(f"Matching took {round((time() - ts) * 1000, 2)} ms.")
        
        # Solve pose
        ts = time()
        if self.verbose: logger.info(f"Solving pose.")
        pose = self.pose_solver.run_matches(matches)
        if self.verbose: logger.info(f"Pose solving took {round((time() - ts) * 1000, 2)} ms.")
        
        return pose