import math
import numpy as np

from utils.rotation import pose_rotation, yaw_pitch_roll


class WorldGazeController:
    """Hold a world-frame (pitch, yaw) orientation with zero world roll."""

    def __init__(self, model, prefix='eye0_'):
        self.model = model
        self.base_id = model.body('base_link').id
        self.eye_id = model.body(f'{prefix}roll_frame').id
        self.actuators = np.array([
            model.actuator(f'{prefix}{axis}_servo').id
            for axis in ('yaw', 'pitch', 'roll')
        ])
        self.joints = np.array([
            model.joint(f'{prefix}{axis}_joint').id
            for axis in ('yaw', 'pitch', 'roll')
        ])
        self.target = np.zeros(2)

    def set_target(self, pitch_rad, yaw_rad):
        """Set persistent world pitch and yaw in radians."""
        self.target[:] = (pitch_rad, yaw_rad)
        pitch_lim = math.radians(40.0)
        yaw_lim = math.radians(50.0)
        radius = math.hypot(self.target[0] / pitch_lim, self.target[1] / yaw_lim)
        if radius > 1.0:
            self.target /= radius

    def update(self, data):
        pitch, yaw = self.target
        desired_world = pose_rotation(0.0, pitch, yaw)
        base_world = data.xmat[self.base_id].reshape(3, 3)
        relative = base_world.T @ desired_world
        desired_joints = yaw_pitch_roll(relative)
        limits = self.model.jnt_range[self.joints]
        data.ctrl[self.actuators] = np.clip(desired_joints, limits[:, 0], limits[:, 1])

    def measured_world_pose(self, data):
        rotation = data.xmat[self.eye_id].reshape(3, 3)
        return yaw_pitch_roll(rotation)
