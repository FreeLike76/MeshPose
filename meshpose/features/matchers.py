import cv2
import numpy as np

TORCH_AVAILABLE = True
try:
    import torch
except:
    TORCH_AVAILABLE = False

from copy import deepcopy
from loguru import logger

from typing import List

from meshpose.data import ViewDescription, ViewMatches

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
        
        # To numpy
        query_matches = np.array([m.queryIdx for m in matches], dtype=np.int32)
        preset_matches = np.array([m.trainIdx for m in matches], dtype=np.int32)
        
        view_matches = ViewMatches(query, preset, query_matches, preset_matches)
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

class PytorchL2Matcher(BaseMatcher):
    def __init__(self, eps:float=1e-6, device="cuda"):
        assert TORCH_AVAILABLE, logger.error("Pytorch is not installed!")
        # Params
        self.eps = eps
        self.device = device
    
    def run(self, desc1:torch.Tensor, desc2:torch.Tensor, distance_matrix=None):
        # Get distance matrix
        if distance_matrix is None:
            distance_matrix = torch.cdist(desc1, desc2)
        
        ms = min(distance_matrix.size(0), distance_matrix.size(1))
        match_dists, idxs_in_2 = torch.min(distance_matrix, dim=1)
        match_dists2, idxs_in_1 = torch.min(distance_matrix, dim=0)
        minsize_idxs = torch.arange(ms, device=distance_matrix.device)

        if distance_matrix.size(0) <= distance_matrix.size(1):
            mutual_nns = minsize_idxs == idxs_in_1[idxs_in_2][:ms]
            matches_idxs = torch.cat([minsize_idxs.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)[mutual_nns]
            match_dists = match_dists[mutual_nns]
        else:
            mutual_nns = minsize_idxs == idxs_in_2[idxs_in_1][:ms]
            matches_idxs = torch.cat([idxs_in_1.view(-1, 1), minsize_idxs.view(-1, 1)], dim=1)[mutual_nns]
            match_dists = match_dists2[mutual_nns]
        
        return match_dists.view(-1, 1).cpu().numpy(), matches_idxs.view(-1, 2).cpu().numpy()
    
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
        query_desc = torch.from_numpy(query.descriptors).float().to(self.device)
        preset_desc = torch.from_numpy(preset.descriptors).float().to(self.device)
        
        # Match descriptors
        match_dists, matches_idxs = self.run(query_desc, preset_desc)
        if len(matches_idxs) == 0:
            return None
        
        # Calculate score
        dist_sum = match_dists.sum()
        score = 1. / (dist_sum + self.eps)
        
        # To numpy
        query_matches = matches_idxs[:, 0]
        preset_matches = matches_idxs[:, 1]
        
        view_matches = ViewMatches(query, preset, query_matches, preset_matches, score=score)
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
        # Get query desc
        query_desc = torch.from_numpy(query.descriptors).float().to(self.device)
        
        views_matches = []
        for i, preset_view in enumerate(preset):
            # Get preset desc
            preset_desc = torch.from_numpy(preset_view.descriptors).float().to(self.device)
            
            # Match descriptors
            match_dists, matches_idxs = self.run(query_desc, preset_desc)
            
            if len(matches_idxs) == 0:
                return None

            # Calculate score
            dist_sum = match_dists.sum()
            score = 1. / (dist_sum + self.eps)

            # To numpy
            query_matches = matches_idxs[:, 0]
            preset_matches = matches_idxs[:, 1]
            
            # Save
            matches = ViewMatches(query, preset_view, query_matches, preset_matches, score=score)
            views_matches.append(matches)
        return views_matches
    
    
class BatchedPytorchL2Matcher(PytorchL2Matcher):
    def __init__(self, eps:float=1e-6, device="cuda"):
        assert TORCH_AVAILABLE, logger.error("Pytorch is not installed!")
        super().__init__(eps=eps, device=device)
    
    def run_views_desc(self, query: ViewDescription, preset: List[ViewDescription]) -> List[ViewMatches]:
        """
        Performs batched pairwise matching of a query ViewDescription object and a list of preset ViewDescription objects.
        
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
        # Get dimentions
        b = len(preset)
        n = np.max([p.descriptors.shape[0] for p in preset])
        
        # Stack query in a batch
        query_desc = np.stack([query.descriptors for _ in range(b)])
        query_desc = torch.from_numpy(query_desc).float().to(self.device)
        
        # Pad and batch all preset descriptors
        preset_desc_all = np.stack([
            np.pad(p.descriptors, ((0, n - p.descriptors.shape[0]), (0, 0)), mode="constant", constant_values=np.inf) for p in preset])
        preset_desc_all = torch.from_numpy(preset_desc_all).float().to(self.device)
        
        # Find distance all-to-all
        cdist = torch.cdist(query_desc, preset_desc_all)
        
        # Free up memory
        del query_desc, preset_desc_all
        
        views_matches = []
        for i, preset_view in enumerate(preset):
            # Match descriptors
            pd_size = preset_view.descriptors.shape[0]
            match_dists, matches_idxs = self.run(None, None,
                                                 distance_matrix=cdist[i, :, :pd_size])
            
            if len(matches_idxs) == 0:
                return None

            # Calculate score
            dist_sum = match_dists.sum()
            score = 1. / (dist_sum + self.eps)

            # To numpy
            query_matches = matches_idxs[:, 0]
            preset_matches = matches_idxs[:, 1]
            
            # Save
            matches = ViewMatches(query, preset_view, query_matches, preset_matches, score=score)
            views_matches.append(matches)
        return views_matches