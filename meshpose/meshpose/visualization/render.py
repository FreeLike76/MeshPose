import cv2
import numpy as np
import open3d as o3d

from loguru import logger

from typing import List

class SceneRender:
    """
    Renders mesh from the given view.
    """
    def __init__(self, meshes:List[o3d.geometry.TriangleMesh],
                 intrinsics:np.ndarray, height:int, width:int,
                 extrinsics:np.ndarray = None, downscale:float=2.0) -> None:
        self.meshes = meshes if isinstance(meshes, list) else [meshes]
        self.intrinsics = intrinsics
        self.height = height
        self.width = width
        self.extrinsics = extrinsics
        self.downscale = downscale
        
        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="render", visible=False,
                               width=int(self.width // downscale),
                               height=int(self.height // downscale))
        # Add meshes
        for mesh in self.meshes:
            self.vis.add_geometry(mesh)

    def set_extrinsics(self, extrinsics):
        self.extrinsics = extrinsics

    def run(self):
        # Ensure extrinsics has been set
        assert self.extrinsics is not None, logger.error("Extrinsics have not been set!")
        
        # Parse intrinsics
        fx = self.intrinsics[0, 0] // self.downscale
        fy = self.intrinsics[1, 1] // self.downscale
        cx = self.intrinsics[0, 2] // self.downscale
        cy = self.intrinsics[1, 2] // self.downscale
        
        # Create PinholeCameraIntrinsic object
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(int(self.width // self.downscale),
                                                                     int(self.height // self.downscale),
                                                                     fx, fy, cx, cy)

        # Create PinholeCameraParameters and set its intrinsic and extrinsic
        pinhole_camera_parameters = o3d.camera.PinholeCameraParameters()
        pinhole_camera_parameters.intrinsic = pinhole_camera_intrinsic
        pinhole_camera_parameters.extrinsic = self.extrinsics
        
        # Set view control
        ctr = self.vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(pinhole_camera_parameters, allow_arbitrary=True)
        
        # Update
        self.vis.poll_events()
        self.vis.update_renderer()
        
        # Render
        image = self.vis.capture_screen_float_buffer(do_render=True)
        image = np.array(image)
        
        # Transform to BGR
        image = (image[:, :, ::-1] * 255).astype(np.uint8)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        
        return image