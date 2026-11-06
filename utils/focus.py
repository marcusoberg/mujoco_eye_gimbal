import cv2
import numpy as np


class SoftwareFocus:
    """Depth-aware approximation of defocus for a pinhole MuJoCo camera."""

    def __init__(self, model, settings):
        self.distance_m = settings.distance_m
        self.strength = settings.blur_strength
        self.max_sigma_px = settings.max_sigma_px
        self.near = model.vis.map.znear * model.stat.extent
        self.far = model.vis.map.zfar * model.stat.extent

    def set_focus_distance(self, distance_m):
        self.distance_m = distance_m

    def apply(self, rgb, depth):
        distance = self.near / (1.0 - depth * (1.0 - self.near / self.far))
        blur_amount = self.strength * np.abs(
            1.0 / distance - 1.0 / self.distance_m
        )
        blend = np.clip(blur_amount, 0.0, 1.0)[..., None]
        blurred = cv2.GaussianBlur(rgb, (0, 0), self.max_sigma_px)
        return np.ascontiguousarray(
            (rgb * (1.0 - blend) + blurred * blend).astype(np.uint8)
        )
