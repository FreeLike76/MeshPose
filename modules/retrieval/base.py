from typing import List

from ..data import ViewDescription, PresetView

class BaseImageRetrieval:
    def __init__(self, n:float=0.2):
        raise NotImplementedError("BaseImageRetrieval is an abstract class. Use a concrete implementation instead.")
    
    def train(self, views_desc:List[ViewDescription]):
        raise NotImplementedError("BaseImageRetrieval is an abstract class. Use a concrete implementation instead.")
    
    def query(self, view:PresetView) -> List[int]:
        raise NotImplementedError("BaseImageRetrieval is an abstract class. Use a concrete implementation instead.")