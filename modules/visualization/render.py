import numpy as np
import open3d as o3d

from loguru import logger

class SceneRender:
    """
    Renders mesh from the given view.
    """
    def __init__(self, mesh:o3d.geometry.TriangleMesh,
                 intrinsics:np.ndarray, height:int, width:int,
                 extrinsics:np.ndarray = None) -> None:
        self.mesh = mesh
        self.intrinsics = intrinsics
        self.height = height
        self.width = width
        self.extrinsics = extrinsics

    def set_extrinsics(self, extrinsics):
        self.extrinsics = extrinsics

    def run(self):
        # Ensure extrinsics has been set
        assert self.extrinsics is not None, logger.error("Extrinsics have not been set!")
        
        # Parse intrinsics
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]
        
        # Create PinholeCameraIntrinsic object
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, fx, fy, cx, cy)

        # Create PinholeCameraParameters and set its intrinsic and extrinsic
        pinhole_camera_parameters = o3d.camera.PinholeCameraParameters()
        pinhole_camera_parameters.intrinsic = pinhole_camera_intrinsic
        pinhole_camera_parameters.extrinsic = self.extrinsics

        # Render image
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(self.mesh)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(pinhole_camera_parameters)
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        return image
