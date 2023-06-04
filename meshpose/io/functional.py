import cv2
import numpy as np
import open3d as o3d

from loguru import logger

import json
from pathlib import Path
from typing import Tuple, List

def validate_file(path:Path):
    assert path.exists() and path.is_file(), logger.warning(f"No file: {str(path)}!")

def validate_create_dir(path:Path):
    assert path.is_dir(), logger.warning(f"Path: {str(path)} is not a directory!")
    path.mkdir(parents=True, exist_ok=True)

def load_image(path:Path) -> np.ndarray:
    validate_file(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
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

def load_mesh(path:Path) -> o3d.geometry.TriangleMesh:
    validate_file(path)
    
    mesh = o3d.io.read_triangle_mesh(str(path), True)
    return mesh