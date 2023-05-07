import cv2
import numpy as np

from loguru import logger

import json
from pathlib import Path

def validate_file(path:Path):
    assert path.exists() and path.is_file(), logger.error(f"No file: {str(path)}!")

def load_image(path:Path) -> np.ndarray:
    validate_file(path)
    
    image = cv2.imread(str(path))
    return image

def load_np(path:Path) -> np.ndarray:
    validate_file(path)
    
    array = np.load(str(path))
    return array

def load_json(path:Path) -> dict:
    validate_file(path)
    
    with open(path, "r") as f:
        data = json.load(f)
    return data