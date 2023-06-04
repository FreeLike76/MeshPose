import cv2
import numpy as np
from PIL import Image

from sklearn import cluster

from tqdm import tqdm
from loguru import logger

from typing import List

from .base import BaseImageRetrieval
from meshpose.data import ViewDescription, QueryView
from meshpose.utils import tqdm_description

class BovwRetrieval(BaseImageRetrieval):
    def __init__(self, desc_size:int=256, max_iter:int=500, n:float=0.2):
        # Init params
        self.n = n
        self.desc_size = desc_size
        
        self.model = cluster.KMeans(n_clusters=desc_size, n_init="auto", random_state=0, max_iter=500)
        
        self.descriptions = None

    def _reshape(self, desc:np.ndarray):
        return desc.reshape(-1, desc.shape[-1]).astype(np.float32)
    
    def _labels_to_hist(self, labels:np.ndarray):
        hist = np.histogram(labels, bins=self.desc_size, range=(0, self.desc_size-1), density=True)[0]
        return hist
    
    def train(self, views_desc:List[ViewDescription]):    
        self.descriptions = []
        raw_desc = None
        
        # Stack all descriptors together
        for vd in tqdm(views_desc, desc=tqdm_description("mesh_pose.retrieval.dbow", "Preprocessing features")):
            if raw_desc is None:
                raw_desc = self._reshape(vd.descriptors)
                continue
            raw_desc = np.concatenate((raw_desc, self._reshape(vd.descriptors)), axis=0)
        
        # Fit model
        self.model.fit(raw_desc)
        del raw_desc
        
        # Encode images
        for vd in tqdm(views_desc, desc=tqdm_description("mesh_pose.retrieval.dbow", "Encoding images")):
            desc = self._reshape(vd.descriptors)
            labels = self.model.predict(desc)
            hist = self._labels_to_hist(labels)
            self.descriptions.append(hist)
        
        # Finished
        self.descriptions = np.array(self.descriptions)
            
    def query(self, query_desc:ViewDescription)-> List[int]:
        assert self.descriptions is not None, logger.error("You must train the model first")
        # Encode query
        desc = self._reshape(query_desc.descriptors)
        labels = self.model.predict(desc)
        hist = self._labels_to_hist(labels)
        
        # Get dist
        dist = np.linalg.norm(self.descriptions - hist, axis=1)
        
        # Return n% of the best matches
        n_ret = int(len(self.descriptions) * self.n)
        idx = np.argsort(dist)[:n_ret]
        return idx