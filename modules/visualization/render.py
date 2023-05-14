import numpy as np
import open3d as o3d

class SceneRender:
    """
    Renders mesh from the given view.
    """
    def __init__(self, mesh, intrinsics, height, width):
        self.mesh = mesh
        self.intrinsics = intrinsics
        self.height = height
        self.width = width
        self.extrinsics = None

    def set_extrinsics(self, extrinsics):
        self.extrinsics = extrinsics

    def run(self):
        # Ensure extrinsics has been set
        if self.extrinsics is None:
            print("Extrinsics not set")
            return None

        # Create PinholeCameraIntrinsic object
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.intrinsics[0, 0], self.intrinsics[1, 1], self.intrinsics[0, 2], self.intrinsics[1, 2])

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
