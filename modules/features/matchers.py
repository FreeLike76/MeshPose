import cv2
import numpy as np

from copy import deepcopy
from loguru import logger

from typing import List

from ..data import ViewDescription, ViewMatches

class BaseMatcher:
    def __init__(self,
                 test_ratio:bool=False, test_ratio_th:float=0.7,
                 test_symmetry:bool=False,
                 verbose:bool=False) -> None:
        # Verbosity
        self.verbose = verbose
        
        # Additional mathing parameters
        self.test_ratio = test_ratio
        self.test_ratio_th = test_ratio_th
        self.test_symmetry = test_symmetry
        
        # Empty matcher
        self.matcher = None
    
    def _test_ratio(self, matches):
        good_matches = []
        for match in matches:
            if len(match) < 2: continue
            m, n = match
            if m.distance < self.test_ratio_th * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def _test_symmetry(self, matches12, matches21):
        good_matches = []
        for match12 in matches12:
            for match21 in matches21:
                if match12.queryIdx == match21.trainIdx \
                    and match12.trainIdx == match21.queryIdx:
                        good_matches.append(match12)
        return good_matches
    
    def run(self, desc1:np.ndarray, desc2:np.ndarray) -> list:
        """
        Performs matching of two sets of descriptors.
        
        Parameters:
        --------
        desc1: np.ndarray
            First set of descriptors.
        desc2: np.ndarray
            Second set of descriptors.
        
        Returns:
        --------
        matches: list
            List of matches.
        """
        assert self.matcher is not None, logger.error("Matcher is not initialized! Use BFMatcher or FLANNMatcher.")
        matches = []
        if self.test_symmetry:
            if self.test_ratio:
                matches12 = self.matcher.knnMatch(desc1, desc2, k=2)
                matches21 = self.matcher.knnMatch(desc2, desc1, k=2)
                matches12 = self._test_ratio(matches12)
                matches21 = self._test_ratio(matches21)
            else:
                matches12 = self.matcher.match(desc1, desc2)
                matches21 = self.matcher.match(desc2, desc1)
            matches = self._test_symmetry(matches12, matches21)
        else:
            if self.test_ratio:
                matches = self.matcher.knnMatch(desc1, desc2, k=2)
                matches = self._test_ratio(matches)
            else:
                matches = self.matcher.match(desc1, desc2)

        if self.verbose: logger.info(f"Detected {len(matches)} matches.")
        return matches

    def run_view_desc(self, query: ViewDescription, preset: ViewDescription) -> ViewMatches:
        """
        Performs pairwise matching of two ViewDescription objects.
        
        Parameters:
        --------
        query: ViewDescription
            Query view.
        preset: ViewDescription
            Preset view.
        
        Returns:
        --------
        view_matches: ViewMatches
            Object containing information about the matches.
        """
        # Match descriptors
        matches = self.run(query.descriptors, preset.descriptors)
        if len(matches) == 0:
            return None
        view_matches = ViewMatches(query, preset, matches)
        return view_matches

    def run_views_desc(self, query: ViewDescription, preset: List[ViewDescription]) -> List[ViewMatches]:
        """
        Performs pairwise matching of a query ViewDescription object and a list of preset ViewDescription objects.
        
        Parameters:
        --------
        query: ViewDescription
            Query view.
        preset: List[ViewDescription]
            List of preset views.
        
        Returns:
        --------
        view_matches: List[ViewMatches]
            List of objects containing information about the matches.
        """
        view_matches: List[ViewMatches] = []
        for view in preset:
            matches = self.run_view_desc(query, view)
            if matches is None: continue
            view_matches.append(matches)
        return view_matches
    
class BruteForceMatcher(BaseMatcher):
    def __init__(self, params:dict={},
                 test_ratio:bool=False, test_ratio_th:float=0.7,
                 test_symmetry:bool=False,
                 verbose:bool=False) -> None:
        # Init base class
        super().__init__(test_ratio=test_ratio, test_ratio_th=test_ratio_th,
                         test_symmetry=test_symmetry, verbose=verbose)
        
        # Init matcher
        self.params = deepcopy(params)
        self.matcher = cv2.BFMatcher_create(**self.params)

class FlannMatcher(BaseMatcher):
    def __init__(self,
                 index_params:dict={"algorithm": 1, "trees": 5},
                 search_params:dict={"algorithm": 6, "table_number": 12, "key_size": 20, "multi_probe_level": 2},
                 test_ratio:bool=False, test_ratio_th:float=0.7,
                 test_symmetry:bool=False,
                 verbose:bool=False) -> None:
        # Init base class
        super().__init__(test_ratio=test_ratio, test_ratio_th=test_ratio_th,
                         test_symmetry=test_symmetry, verbose=verbose)
        
        # Init matcher
        self.index_params = deepcopy(index_params)
        self.search_params = deepcopy(search_params)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)