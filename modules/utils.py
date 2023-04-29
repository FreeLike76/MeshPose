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

class Vebsosity:
    def __init__(self, level:int=0) -> None:
        self.level = level
    def __call__(self, req:int) -> bool:
        return self.level >= req
    
class bcolors:
    GREEN = '\033[32m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'
    CYAN = '\033[36m'

class InterceptHandler(logging.Handler):
    """Redirect logging output to loguru"""
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 1
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def tqdm_description(file_name, process_name):
    """Convert tqdm appearance to loguru format"""
    return \
        f"{bcolors.GREEN}{datetime.now().strftime('%F %T.%f')[:-3]}{bcolors.ENDC}" + \
        f" | {bcolors.BOLD}INFO{bcolors.ENDC} |" + \
        f" {bcolors.CYAN}{file_name}{bcolors.ENDC} -" + \
        f" {bcolors.BOLD}{process_name}{bcolors.ENDC}"

def configure_stdout():
    """Set the logs to the desired appearance"""
    logger.remove(0)
    logger.add(sys.stderr, format= \
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " + \
        "<level>{level: <8}</level> | " + \
        "<cyan>{module}</cyan> - <level>{message}</level>")

    logging.disable('DEBUG')
    logging.basicConfig(handlers=[InterceptHandler()], level=50)

    warnings.filterwarnings("ignore")
