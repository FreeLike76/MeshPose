from typing import List

from meshpose.data import ViewDescription

class BaseImageRetrieval:
    def __init__(self, n:float=0.2):
        raise NotImplementedError("BaseImageRetrieval is an abstract class. Use a concrete implementation instead.")
    
    def train(self, views_desc:List[ViewDescription]):
        raise NotImplementedError("BaseImageRetrieval is an abstract class. Use a concrete implementation instead.")
    
    def query(self, query_desc:ViewDescription) -> List[int]:
        raise NotImplementedError("BaseImageRetrieval is an abstract class. Use a concrete implementation instead.")