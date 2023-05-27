from loguru import logger

from pathlib import Path
from typing import List

from mesh_pose.data import PresetView, ViewDescription

class DataIOBase:
    def __init__(self, root_p: Path, verbose: bool = False) -> None:
        self.root_p = root_p
        self.verbose = verbose
    
    def get_project_p(self) -> Path:
        """
        Return a path to the project directory.
        """
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")
    
    def get_mesh_p(self) -> Path:
        """
        Return a path to the .OBJ file (3D Reconstruction).
        """
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")
        
    def load_views(self) -> List[PresetView]:
        """
        Return a list of all PresetView objects and their Camera data.
        """
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")
    
    def save_view_descriptions(self, name:str, views_desc: List[ViewDescription]):
        """
        Saves a list of ViewDescription under a given name.
        """
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")
    
    def load_view_descriptions(self, name:str, views:List[PresetView]) -> List[ViewDescription]:
        """
        Loads and returns a list of ViewDescription from a given name.
        """
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")
    