import math
import numpy as np


def rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose_rotation(roll, pitch, yaw):
    return rz(yaw) @ ry(-pitch) @ rx(roll)


def yaw_pitch_roll(rotation):
    return np.array([
        math.atan2(rotation[1, 0], rotation[0, 0]),
        math.asin(float(np.clip(rotation[2, 0], -1.0, 1.0))),
        math.atan2(rotation[2, 1], rotation[2, 2]),
    ])
