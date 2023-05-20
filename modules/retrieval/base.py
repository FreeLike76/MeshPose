from typing import List

from ..data import PresetView

class BaseImageRetrieval:
    def __init__(self):
        raise NotImplementedError("BaseImageRetrieval is an abstract class. Use a concrete implementation instead.")
    
    def train(self, views:List[PresetView]):
        raise NotImplementedError("BaseImageRetrieval is an abstract class. Use a concrete implementation instead.")
    
    def query(self, view:PresetView, n:int=50) -> List[int]:
        raise NotImplementedError("BaseImageRetrieval is an abstract class. Use a concrete implementation instead.")