import open3d as o3d

from typing import List

from . import io
from .data import PresetView, Camera
from .features import extractors

class Localization:
    def __init__(self, verbose: bool = False):
        pass
    """
    def init
        - load ProjectMeta
        
        - set parameters
        - create algorithms
    
    def setup
        - set data
    
    def run: image
        - run localization on image
        - return pose
    """