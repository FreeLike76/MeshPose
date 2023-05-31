# Fundamental third-party imports
import cv2
import numpy as np

# Built-in modules
import sys
from enum import Enum
from datetime import datetime

# Convenience imports
import logging
from loguru import logger

class Serializable:
    def to_json(self) -> dict:
        raise NotImplementedError("Serializable is an abstract class. Use a concrete implementation instead.")
    
    @staticmethod
    def from_json(json:dict):
        raise NotImplementedError("Serializable is an abstract class. Use a concrete implementation instead.")


class Vebsosity:
    def __init__(self, level:int=0) -> None:
        self.level = level
    
    def __call__(self, req:int) -> bool:
        return self.level >= req


def resize_image(img: np.ndarray, size: int):
    # Get current dimensions
    h, w = img.shape[:2]
    
    # If donwscaling, use INTER_AREA, else INTER_CUBIC
    interpolation = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC
    
    # Compute new dimensions
    new_h, new_w = size, size
    if h > w:
        new_w = int(w * size / h)
    elif w > h:
        new_h = int(h * size / w)
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation)
    return img_resized

class BColors:
    GREEN = '\033[32m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'
    CYAN = '\033[36m'

def tqdm_description(file_name, process_name):
    """
    Convert tqdm appearance to loguru format
    """
    return (f"{BColors.GREEN}{datetime.now().strftime('%F %T.%f')[:-3]}{BColors.ENDC} | " +
            f"{BColors.BOLD}INFO{BColors.ENDC}     | " +
            f"{BColors.CYAN}{file_name}{BColors.ENDC} - " +
            f"{BColors.BOLD}{process_name}{BColors.ENDC}")
