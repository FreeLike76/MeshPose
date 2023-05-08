from loguru import logger

import json
from glob import glob
from pathlib import Path
from typing import List

from . import io

class ProjectMeta:
    def __init__(self, data_p: Path, verbose: bool = False) -> None:
        # Init params
        self.data_p: Path = data_p
        self.keyframe_names: List[str] = []
        self.processed_features_meta: List[dict] = []

        self.verbose: bool = verbose
        
        # Try load, if not - init
        if self.load():
            if self.verbose: logger.info(f"Loaded project metadata from {self.data_p.name}.")
        else:
            if self.verbose: logger.info(f"Initializing project metadata for {self.data_p.name}.")
            # Scan frames
            self.keyframes_list = self.scan_keyframe_names(self.data_p, verbose=self.verbose)
            # Create project folder
            self.get_project_p().mkdir(parents=True, exist_ok=True)
    
    def save(self):
        # Create meta dict
        meta_dict = {
            "processed_features": self.processed_features_meta,
            "keyframe_names": self.keyframe_names,
        }
        
        # Init save dir if not present
        save_dir = self.get_project_p()
        save_dir.mkdir(parents=True, exist_ok=True)
        save_p = save_dir / "meta.json"
        
        # Save to json
        with open(save_p, "w") as f:
            json.dump(meta_dict, f)
    
    def load(self) -> int:
        # Validate project folder
        meta_p = self.get_project_p() / "meta.json"
        if not meta_p.exists() or not meta_p.is_file():
            return False
        
        # Try loading if present
        try:
            meta_dict = io.load_json(meta_p)
            # Parse
            self.processed_features_meta = meta_dict["processed_features"]
            self.keyframe_names = meta_dict["keyframe_names"]
        except Exception as e:
            logger.warning(f"Failed to load project metadata. Exception message: {e}!")
            return False
        
        return True
    
    def get_frame_p_template(self) -> str:
        return str(self.data_p / "frame_{}.jpg")
    
    def get_frame_p(self, name:str) -> Path:
        return Path(self.get_frame_p_template().format(name))

    def get_frame_json_p_template(self) -> str:
        return str(self.data_p / "frame_{}.json")
    
    def get_frame_json_p(self, name:str) -> Path:
        return Path(self.get_frame_json_p_template().format(name))
    
    def get_mesh_p(self) -> Path:
        return self.data_p / "textured_output.obj"
    
    def get_project_p(self) -> Path:
        return self.data_p / "mesh_pose"
    
    def _parse_frame_name(self, path: str) -> str:
        """
        Get frame name from json path.
        """
        filename = Path(path).stem
        keyframe_name = filename.split("_")[-1]
        return keyframe_name

    def scan_keyframe_names(self, keyframes_p: Path, verbose: bool = False) -> List[str]:
        """
        Scan data folder to get a list of all valid keyframes.
        """
        assert keyframes_p.exists() and keyframes_p.is_dir(), logger.error(
            f"Directory {str(keyframes_p)} does not exist!")

        # Template paths
        #template_frame_json_p = str(keyframes_p / "frame_*.json")
        #template_frame_p = str(keyframes_p / "frame_{}.jpg") # (jpg|png|jpeg)
        
        template_frame_p = self.get_frame_json_p_template()
        template_frame_json_p = self.get_frame_json_p_template().format("*") 
        
        # Locate json files
        frame_json_p_list = glob(template_frame_json_p)
        if verbose: logger.info(f"Found {len(frame_json_p_list)} json files.")

        # Get ids
        keyframe_names: List[str] = []
        for json_p in frame_json_p_list:
            try:
                # Extract keyframe name
                keyframe_name = self._parse_frame_name(json_p)
                # Get matching keyframe
                keyframe_p = template_frame_p.format(keyframe_name)
                # Assert keyframe existance
                keyframe_p = Path(keyframe_p)
                if keyframe_p.exists() and keyframe_p.is_file():
                    keyframe_names.append(keyframe_name)
            except Exception as e:
                logger.warning(f"Failed to parse frame id from {json_p}. Exceotion message: {e}")
        return keyframe_names
    
    def add_processed_features_meta(self, feature_extractors:dict) -> int:
        self.processed_features_meta.append(feature_extractors)
        return len(self.processed_features_meta) - 1

    def get_processed_features_meta_init(self) -> dict:
        # TODO: go over all feature extractors and get the init features
        pass
    
    def get_processed_features_meta_track(self) -> dict:
        # TODO: go over all feature extractors and get the track
        pass