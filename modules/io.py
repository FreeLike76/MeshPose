import cv2
import numpy as np
import open3d as o3d

from loguru import logger

import json
from pathlib import Path
from typing import Tuple

def validate_file(path:Path):
    assert path.exists() and path.is_file(), logger.error(f"No file: {str(path)}!")

def load_image(path:Path) -> np.ndarray:
    # TODO: low, rewrite with PIL
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

def load_camera_meta(path:Path) -> Tuple[np.ndarray, np.ndarray]:
    frame_json = load_json(path)
    # Convert to numpy
    intrinsics = np.asarray(frame_json["intrinsics"]).reshape((3, 3)).astype(np.float32)
    extrinsics = np.asarray(frame_json["cameraPoseARFrame"]).reshape((4, 4)).astype(np.float32)
    return intrinsics, extrinsics

def load_mesh(path:Path) -> o3d.geometry.TriangleMesh:
    validate_file(path)
    
    mesh = o3d.io.read_triangle_mesh(str(path), True)
    return mesh