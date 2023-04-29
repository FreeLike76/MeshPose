from typing import Any


import cv2
import numpy as np

from loguru import logger

class Vebsosity:
    def __init__(self, level:int=0) -> None:
        self.level = level
    def __call__(self, req:int) -> bool:
        return self.level >= req