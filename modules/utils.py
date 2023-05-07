# Fundamental third-party imports
import cv2
import numpy as np

# Built-in modules
import sys
import warnings
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

#class bcolors:
#    GREEN = '\033[32m'
#    BOLD = '\033[1m'
#    ENDC = '\033[0m'
#    CYAN = '\033[36m'
#
#class InterceptHandler(logging.Handler):
#    """Redirect logging output to loguru"""
#    def emit(self, record):
#        # Get corresponding Loguru level if it exists
#        try:
#            level = logger.level(record.levelname).name
#        except ValueError:
#            level = record.levelno
#
#        # Find caller from where originated the logged message
#        frame, depth = logging.currentframe(), 1
#        while frame.f_code.co_filename == logging.__file__:
#            frame = frame.f_back
#            depth += 1
#
#        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
#
#def tqdm_description(file_name, process_name):
#    """Convert tqdm appearance to loguru format"""
#    return \
#        f"{bcolors.GREEN}{datetime.now().strftime('%F %T.%f')[:-3]}{bcolors.ENDC}" + \
#        f" | {bcolors.BOLD}INFO{bcolors.ENDC} |" + \
#        f" {bcolors.CYAN}{file_name}{bcolors.ENDC} -" + \
#        f" {bcolors.BOLD}{process_name}{bcolors.ENDC}"
#
#def configure_stdout():
#    """Set the logs to the desired appearance"""
#    logger.remove(0)
#    logger.add(sys.stderr, format= \
#        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " + \
#        "<level>{level: <8}</level> | " + \
#        "<cyan>{module}</cyan> - <level>{message}</level>")
#
#    logging.disable('DEBUG')
#    logging.basicConfig(handlers=[InterceptHandler()], level=50)
#
#    warnings.filterwarnings("ignore")
#