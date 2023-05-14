from loguru import logger

import json
from glob import glob
from pathlib import Path
from typing import List

from . import functional
from ..data import PresetView, FrameDescription

class DataIOBase:
    def __init__(self, root_p: Path, verbose: bool = False) -> None:
        self.root_p = root_p
        self.verbose = verbose
    
    @staticmethod
    def scan(self):
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")

    def get_project_p(self) -> Path:
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")
    
    def get_mesh_p(self) -> Path:
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")
        
    def read(self) -> List[PresetView]:
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")

    def save_frame_descriptions(self, name:str, frame_descriptions: List[FrameDescription]):
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")
    
    def load_frame_descriptions(self, name:str) -> List[FrameDescription]:
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")
    