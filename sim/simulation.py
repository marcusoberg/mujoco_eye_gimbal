import time
from pathlib import Path

import mujoco

from utils.focus import SoftwareFocus
from utils.motion import BaseMotion
from sim.base import MocapBase
from sim.eye import WorldGazeController
from sim.viewer import SplitViewer


class Simulation:
    """Owns simulation time; controller, disturbances and viewer stay independent."""

    def __init__(self, settings):
        scene_path = Path(__file__).resolve().parent.parent / 'model' / 'scene.xml'
        print("Scene path:", scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        motion = BaseMotion(settings.noise, settings.jerks)
        self.base = MocapBase(self.model, settings.home_position, motion)
        self.eye = WorldGazeController(self.model)
        self.focus = SoftwareFocus(self.model, settings.focus)
        self.viewer = SplitViewer(self.model, settings.window_size, self.focus)
        self.frame_interval = 1.0 / settings.render_fps
        self.eye.set_target(settings.target_pitch_rad, settings.target_yaw_rad)
        self.base.update(self.data, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        self.base.update(self.data, self.data.time)
        mujoco.mj_forward(self.model, self.data)
        self.eye.update(self.data)
        mujoco.mj_step(self.model, self.data)

    def run(self):
        step_dt = float(self.model.opt.timestep)
        accumulator = 0.0
        last_time = time.perf_counter()
        last_render = last_time - self.frame_interval

        while self.viewer.is_open():
            now = time.perf_counter()
            accumulator += min(now - last_time, 0.05)
            last_time = now

            while accumulator >= step_dt:
                self.step()
                accumulator -= step_dt

            if now - last_render >= self.frame_interval:
                self.viewer.render(self.data)
                last_render = now

            time.sleep(0.001)

        self.viewer.close()
