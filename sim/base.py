import mujoco
import numpy as np

from utils.rotation import pose_rotation


class MocapBase:
    """Applies a generated pose directly to the mocap base_link."""

    def __init__(self, model, home_position, motion):
        base_id = model.body('base_link').id
        self.mocap_id = int(model.body_mocapid[base_id])
        self.home = np.asarray(home_position, dtype=float)
        self.motion = motion

    def update(self, data, time_s):
        offset = self.motion.sample(time_s)
        roll, pitch, yaw = offset[3:]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, pose_rotation(roll, pitch, yaw).ravel())
        data.mocap_pos[self.mocap_id] = self.home + offset[:3]
        data.mocap_quat[self.mocap_id] = quaternion
