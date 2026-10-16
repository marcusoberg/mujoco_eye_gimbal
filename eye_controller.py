from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, dtype=float)
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=float)


def mat_to_ypr_for_gimbal(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose R = Rz(yaw) @ Ry(-pitch) @ Rx(roll).

    Returns (yaw, pitch, roll), all radians.
    This is nonsingular for the intended pitch range (< 90 deg).
    """
    r20 = float(np.clip(R[2, 0], -1.0, 1.0))
    pitch = math.asin(r20)
    yaw = math.atan2(R[1, 0], R[0, 0])
    roll = math.atan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


def clamp_ellipse(pitch: float, yaw: float, pitch_limit: float, yaw_limit: float) -> tuple[float, float]:
    u = pitch / pitch_limit
    v = yaw / yaw_limit
    r2 = u*u + v*v
    if r2 <= 1.0:
        return pitch, yaw
    scale = 1.0 / math.sqrt(r2)
    return pitch * scale, yaw * scale


def slew_2d(current: np.ndarray, target: np.ndarray, max_speed: float, dt: float) -> np.ndarray:
    delta = target - current
    mag = float(np.linalg.norm(delta))
    max_step = max_speed * dt
    if mag <= max_step or mag < 1e-12:
        return target.copy()
    return current + delta * (max_step / mag)


@dataclass
class EyeConfig:
    pitch_limit_deg: float = 40.0
    yaw_limit_deg: float = 50.0
    max_gaze_speed_deg_s: float = 550.0
    command_timeout_s: float = 0.075


class EyeController:
    """World-stabilized 3-axis eye/gimbal controller.

    Command semantics:
      submit_command(delta_pitch, delta_yaw) is an ABSOLUTE gaze offset from the
      neutral world +X direction, not an incremental integration command.

    The sender must refresh the command. If it stops refreshing for command_timeout_s,
    the desired gaze automatically returns to true neutral (0, 0).

    The roll target is not commanded by the client. It is solved automatically so
    that the eye body has zero world roll for the requested pitch/yaw direction.
    """

    def __init__(self, mujoco, model, config: EyeConfig = EyeConfig()):
        self.mujoco = mujoco
        self.model = model
        self.cfg = config

        self.pitch_lim = math.radians(config.pitch_limit_deg)
        self.yaw_lim = math.radians(config.yaw_limit_deg)
        self.max_speed = math.radians(config.max_gaze_speed_deg_s)

        self.base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_base")
        self.eye_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eye")
        self.yaw_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_servo")
        self.pitch_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo")
        self.roll_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll_servo")

        self._command = np.zeros(2, dtype=float)      # pitch, yaw [rad]
        self._gaze = np.zeros(2, dtype=float)         # slew-limited pitch, yaw [rad]
        self._last_command_wall = -1e9
        self.last_joint_target = np.zeros(3, dtype=float)  # yaw, pitch, roll

    @property
    def command_alive(self) -> bool:
        return (time.monotonic() - self._last_command_wall) <= self.cfg.command_timeout_s

    def submit_command(self, delta_pitch: float, delta_yaw: float, *, radians: bool = False) -> None:
        """Refresh desired gaze offset. Call repeatedly to keep the command alive."""
        if not radians:
            delta_pitch = math.radians(delta_pitch)
            delta_yaw = math.radians(delta_yaw)
        delta_pitch, delta_yaw = clamp_ellipse(
            delta_pitch, delta_yaw, self.pitch_lim, self.yaw_lim
        )
        self._command[:] = (delta_pitch, delta_yaw)
        self._last_command_wall = time.monotonic()

    def step(self, data) -> None:
        dt = float(self.model.opt.timestep)
        active = (time.monotonic() - self._last_command_wall) <= self.cfg.command_timeout_s
        requested = self._command if active else np.zeros(2)

        # Saccade-like slew limit on the intended gaze direction.
        self._gaze = slew_2d(self._gaze, requested, self.max_speed, dt)
        pitch_world, yaw_world = float(self._gaze[0]), float(self._gaze[1])

        # Desired eye body attitude in the WORLD frame. Body +X is optical forward;
        # body +Z remains 'up', i.e. zero roll.
        R_des_world = rot_z(yaw_world) @ rot_y(-pitch_world)

        # This exact simulated attitude stands in for the attitude quaternion that a
        # real IMU fusion filter (gyro + accel + optional magnetometer) would provide.
        R_base_world = quat_to_mat(data.xquat[self.base_body])
        R_rel = R_base_world.T @ R_des_world

        yaw_j, pitch_j, roll_j = mat_to_ypr_for_gimbal(R_rel)

        # Respect mechanical limits. If head motion makes the required compensation
        # unreachable, these clamps naturally model saturation.
        yaw_j = float(np.clip(yaw_j, -self.yaw_lim, self.yaw_lim))
        pitch_j = float(np.clip(pitch_j, -self.pitch_lim, self.pitch_lim))
        roll_lim = math.radians(35.0)
        roll_j = float(np.clip(roll_j, -roll_lim, roll_lim))

        data.ctrl[self.yaw_act] = yaw_j
        data.ctrl[self.pitch_act] = pitch_j
        data.ctrl[self.roll_act] = roll_j
        self.last_joint_target[:] = (yaw_j, pitch_j, roll_j)

    def actual_world_gaze(self, data) -> tuple[float, float, float]:
        """Return eye world yaw, pitch, roll [rad] using the same convention."""
        R = np.asarray(data.xmat[self.eye_body], dtype=float).reshape(3, 3)
        return mat_to_ypr_for_gimbal(R)
