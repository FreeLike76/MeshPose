import numpy as np

from tqdm import tqdm
from loguru import logger

import json
from glob import glob
from pathlib import Path
from typing import List

from . import functional
from .data_io_base import DataIOBase
from ..data import PresetView, Camera, FrameDescription

class DataIO3DSA(DataIOBase):
    def __init__(self, root_p: Path, verbose: bool = False) -> None:
        """
        Data IO implementation for 3D Scannear App data.
        """
        # Validate root path
        assert root_p.exists() and root_p.is_dir(), logger.error(
            f"Directory {str(root_p)} does not exist!")
        
        # Init base class
        super().__init__(root_p, verbose=verbose)
        self._meta: List[str] = []
        
        # Create project path just in case
        project_p = self.get_project_p()
        project_p.mkdir(parents=True, exist_ok=True)
        
        # Try loading the meta data
        if self._load_meta():
            if self.verbose: logger.info(f"Loaded project metadata from {root_p.name}.")
        else:
            self._scan_meta()
            self._save_meta()
            if verbose: logger.info(f"Initialized project metadata for {root_p.name}.")
        
        # Print n of keyframes
        if self.verbose: logger.info(f"Indexed {len(self._meta)} keyframes.")
    
    def get_frame_p_template(self) -> str:
        return str(self.root_p / "frame_{}.jpg")
    
    def get_frame_p(self, name:str) -> Path:
        return Path(self.get_frame_p_template().format(name))

    def get_frame_json_p_template(self) -> str:
        return str(self.root_p / "frame_{}.json")
    
    def get_frame_json_p(self, name:str) -> Path:
        return Path(self.get_frame_json_p_template().format(name))
    
    def get_project_p(self) -> Path:
        return self.root_p / "mesh_pose"
    
    def get_mesh_p(self) -> Path:
        return self.root_p / "textured_output.obj"
    
    def _parse_code_name(self, path: str) -> str:
        """
        Get codename of the frame (json).
        
        Parameters:
        --------
        path: str
            Path to the json file.
        
        Returns:
        --------
        codename: str
            Codename of the frame.
        """
        filename = Path(path).stem
        codename = filename.split("_")[-1]
        return codename
    
    def _scan_meta(self):
        """
        Scans the project folder for keyframes to index them.
        """
        # Template paths
        template_frame_p = self.get_frame_p_template()
        template_frame_json_p = self.get_frame_json_p_template().format("*") 
        
        # Locate all json files
        frame_json_p_list = glob(template_frame_json_p)
        
        # Get ids
        self._meta: List[str] = []
        for json_p in frame_json_p_list:
            try:
                # Extract keyframe name
                codename = self._parse_code_name(json_p)
                # Get matching keyframe
                keyframe_p = Path(template_frame_p.format(codename))
                # Validate existance
                if keyframe_p.exists() and keyframe_p.is_file():
                    self._meta.append(codename)
            except Exception as e:
                logger.warning(f"Failed to parse frame id from {json_p}. Exceotion message: {e}")
    
    def _save_meta(self):
        """
        Saves meta data to the project folder.
        """
        # Init save dir if not present
        save_p = self.get_project_p() / "meta.json"
        with open(save_p, "w") as f:
            json.dump(self._meta, f)
    
    def _load_meta(self) -> bool:
        """
        Loads meta data from the project folder.
        
        Returns:
        --------
        status: bool
            'True' if meta data was loaded successfully, 'False' otherwise.
        """
        meta_p = self.get_project_p() / "meta.json"
        try:
            meta = functional.load_json(meta_p)
            self._meta = meta
        except:
            return False
        return True
    
    def load_views(self) -> List[PresetView]:
        """
        Loads all preset views from the project folder.
        """
        preset_views = []
        for i, codename in enumerate(tqdm(self._meta, desc="Loading views", disable=(not self.verbose))):
            # Get paths
            frame_p = self.get_frame_p(codename)
            frame_json_p = self.get_frame_json_p(codename)

            # Load & Create camera
            frame_json = functional.load_json(frame_json_p)
            intrinsics = np.asarray(frame_json["intrinsics"]).reshape((3, 3)).astype(np.float32)
            extrinsics = np.asarray(frame_json["cameraPoseARFrame"]).reshape((4, 4)).astype(np.float32)
            camera = Camera(intrinsics, extrinsics)

            # Create & store view
            view = PresetView(i, frame_p, camera)
            preset_views.append(view)
        
        return preset_views
    
    def save_frame_descriptions(self, name:str, frame_descriptions: List[FrameDescription]):
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")
    
    def load_frame_descriptions(self, name:str) -> List[FrameDescription]:
        raise NotImplementedError("DataIOBase is an abstract class. Use a concrete implementation instead.")