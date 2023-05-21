import cv2
import numpy as np
import open3d as o3d

from PIL import Image

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
    # TODO: low, rewrite with PIL
    validate_file(path)
    # Load
    image = Image.open(path)
    # RGB -> BGR
    image = np.flip(np.array(image), axis=2)
    #image = cv2.imread(str(path))
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

def save_features(path:Path, names:List[str], kp2d:np.ndarray, kp3d:np.ndarray, des:np.ndarray):
    validate_create_dir(path)
    
    # Create subdirs
    path_2d = path / "kp2d"
    path_2d.mkdir(parents=True, exist_ok=True)
    
    path_3d = path / "kp3d"
    path_3d.mkdir(parents=True, exist_ok=True)
    
    path_des = path / "des"
    path_des.mkdir(parents=True, exist_ok=True)
    
    # Save all
    for n in names:
        np.save(path_2d / f"{n}.npy", kp2d)
        np.save(path_3d / f"{n}.npy", kp3d)
        np.save(path_des / f"{n}.npy", des)

def load_features(path:Path, names:List[str],
                  kp2d:List[np.ndarray], kp3d:List[np.ndarray], des:List[np.ndarray]):
    validate_create_dir(path)
    
    # Create subdirs
    path_2d = path / "kp2d"
    path_2d.mkdir(parents=True, exist_ok=True)
    
    path_3d = path / "kp3d"
    path_3d.mkdir(parents=True, exist_ok=True)
    
    path_des = path / "des"
    path_des.mkdir(parents=True, exist_ok=True)
    
    # Save all
    for i, n in enumerate(names):
        np.save(path_2d / f"{n}.npy", kp2d[i])
        np.save(path_3d / f"{n}.npy", kp3d[i])
        np.save(path_des / f"{n}.npy", des[i])
        
def load_features(path:Path, names:List[str]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    # Create subdirs
    path_2d = path / "kp2d"
    path_3d = path / "kp3d"
    path_des = path / "des"
    
    # Load all
    kp2d, kp3d, des = [], [], []
    for n in names:
        kp2d_i  = np.load(path_2d / f"{n}.npy")
        kp2d.append(kp2d_i)
        
        kp3d_i = np.load(path_3d / f"{n}.npy")
        kp3d.append(kp3d_i)
        
        des_i = np.load(path_des / f"{n}.npy")
        des.append(des_i)
        
    return kp2d, kp3d, des