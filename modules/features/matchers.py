import cv2
import numpy as np

from copy import deepcopy
from loguru import logger

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
    
    def __call__(self, desc1:np.ndarray, desc2:np.ndarray) -> np.ndarray:
        assert self.matcher is not None, logger.error("Matcher is not initialized! Use BFMatcher or FLANNMatcher.")
        
        matches = []
        if self.test_symmetry:
            matches12 = self.matcher.knnMatch(desc1, desc2, k=2)
            matches21 = self.matcher.knnMatch(desc2, desc1, k=2)
            if self.test_ratio:
                matches12 = self._test_ratio(matches12)
                matches21 = self._test_ratio(matches21)
            matches = self._test_symmetry(matches12, matches21)
        else:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            if self.test_ratio:
                matches = self._test_ratio(matches)
        
        if self.verbose: logger.info(f"Detected {len(matches)} matches.")
        return matches
    
    def _test_ratio(self, matches):
        good_matches = []
        for m, n in matches:
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
        self.matcher = cv2.BFMatcher.create(**self.params)


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