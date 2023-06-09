import cv2
import numpy as np

def compose(image:np.ndarray, render:np.ndarray, max_dim:int=720, rot:bool=True):
    _image = image.copy()
    _render = render.copy()
    
    if rot:
        _image = np.rot90(_image, 3)
        _render = np.rot90(_render, 3)
    
    # Resize to match
    _render = cv2.resize(_render, (_image.shape[1], _image.shape[0]), interpolation=cv2.INTER_LINEAR)
    _image = np.concatenate([_image, _render], axis=1)
    
    # Resize to max_dim
    scale = max_dim / np.max(_image.shape[:2])
    w, h = int(_image.shape[1] * scale), int(_image.shape[0] * scale)
    _image = cv2.resize(_image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return _image