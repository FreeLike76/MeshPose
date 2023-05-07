from typing import List

from . import io
from .project import ProjectMeta
from .data import ARCamera, PresetARView

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