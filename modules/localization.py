import open3d as o3d

from typing import List

from . import io
from .data import PresetARView, ARCamera
from .project import ProjectMeta
from .features import extractors

class Localization:
    def __init__(self, project_meta: ProjectMeta, verbose: bool = False):
        self.project_meta = project_meta
        feature_extractor_meta = project_meta.get_processed_features_meta_init()
        self.feature_extractor = extractors.DEFINED_EXTRACTORS[feature_extractor_meta["class"]].from_json(feature_extractor_meta["params"])
    
    @staticmethod
    def load_preset_views(project: ProjectMeta) -> List[PresetARView]:
        """
        Loads all preset views, listed in the project meta data.
        """
        preset_views: List[PresetARView] = []
        for frame_i, frame_name in enumerate(project.keyframe_names):
            # Get paths
            frame_p = project.get_frame_p(frame_name)
            frame_json_p = project.get_frame_json_p(frame_name)

            # Load & Create camera
            intrinsics, extrinsics = io.load_camera_meta(frame_json_p)
            camera = ARCamera(intrinsics, extrinsics)

            # Create & store view
            view = PresetARView(frame_i, frame_p, camera)
            preset_views.append(view)

        return preset_views

    """
    def init
        - load ProjectMeta
        
        - set parameters
        - create algorithms
    
    def setup
        - set data
    
    def run: image
        - run localization on image
        - return pose
    """