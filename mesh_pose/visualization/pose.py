import numpy as np
import open3d as o3d

from copy import deepcopy

class ScenePose:
    def __init__(self, env_mesh:o3d.geometry.TriangleMesh) -> None:
        # Save params
        self.env_mesh = deepcopy(env_mesh)
        self.poses = []
    
    def add_pose(self, pose:np.ndarray) -> None:
        """
        Adds a pose to the scene.
        Parameters:
        --------
        pose: np.ndarray
            Pose to add to the scene.
        """
        self.poses.append(pose)
    
    def run(self, scale:float=1.0) -> np.ndarray:
        obj_3d = [self.env_mesh]
        for pose in self.poses:
            pose_inv = np.linalg.inv(pose)
            cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
            cam.transform(pose_inv)
            obj_3d.append(cam)
        
        o3d.visualization.draw_geometries(obj_3d)