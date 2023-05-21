import numpy as np

from loguru import logger

from typing import List, Tuple

from ..retrieval import BaseImageRetrieval
from .image_localization import ImageLocalization
from ..data import ViewDescription, QueryView
from ..features.matchers import BaseMatcher
from ..features.extractors import BaseFeatureExtractor
from ..pose_solver import BasePoseSolver

# TODO: video localization
class VideoLocalization(ImageLocalization):
    def __init__(self,
                 # Params for pose initialization
                 views_desc: List[ViewDescription],
                 feature_extractor: BaseFeatureExtractor,
                 matcher: BaseMatcher,
                 pose_solver: BasePoseSolver,
                 image_retrieval: BaseImageRetrieval = None,
                 
                 # Params for pose tracking
                 track_views_desc: List[ViewDescription] = None,
                 track_feature_extractor: BaseFeatureExtractor = None,
                 track_matcher: BaseMatcher = None,
                 track_image_retrieval: BaseImageRetrieval = None,
                 
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
        # Init base class
        super().__init__(views_desc, feature_extractor, matcher, pose_solver, image_retrieval, verbose)
        
        # Track algorithms
        self.track_views_desc = self.views_desc if track_views_desc is None else track_views_desc
        self.track_feature_extractor = self.feature_extractor if track_feature_extractor is None else track_feature_extractor
        self.track_matcher = self.matcher if track_matcher is None else track_matcher
        self.track_image_retrieval = self.image_retrieval if track_image_retrieval is None else track_image_retrieval
        
        # Start with init
        self._track = False
    
    def is_tracking(self) -> bool:
        """
        Returns whether the localization pipeline is in tracking mode.
        """
        return self._track
    
    def reinit(self):
        """
        Force reinitialization of the localization pipeline.
        """
        self._track = False
    
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
        
        if not self._track:
            ret, rmat, tvec = super().run(query)
        else:
            # Extract features
            if self.verbose: logger.info(f"Extracting features from query view.")
            query_desc = self.track_feature_extractor.run_view(query)
            
            # Check if valid
            if not query_desc.is_2d():
                logger.warning(f"Query view is not valid!")
                return False, None, None
            
            # Run retrieval
            if self.verbose: logger.info(f"Running retrieval.")
            # TODO: retrieval
            
            # Match features
            if self.verbose: logger.info(f"Matching descriptors.")
            matches = self.track_matcher.run_views_desc(query_desc, self.track_views_desc)
            
            # Solve pose
            if self.verbose: logger.info(f"Solving pose.")
            ret, rmat, tvec = self.pose_solver.run_matches(matches, track=True)
        
        # Update tracking status
        self._track = ret
        return ret, rmat, tvec