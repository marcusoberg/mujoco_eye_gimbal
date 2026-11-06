import glfw
import mujoco
import numpy as np


class SplitViewer:
    """Render an external camera and an eye camera into one GLFW window."""

    def __init__(self, model, window_size, focus):
        self.model = model
        self.focus = focus
        glfw.init()
        self.window = glfw.create_window(
            *window_size, 'MuJoCo | External view + stabilized eye', None, None
        )
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        self.options = mujoco.MjvOption()
        self.external_scene = mujoco.MjvScene(model, maxgeom=1000)
        self.eye_scene = mujoco.MjvScene(model, maxgeom=1000)

        self.external_camera = mujoco.MjvCamera()
        self.external_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.external_camera.lookat[:] = [0.8, 0.0, 1.2]
        self.external_camera.distance = 2.5
        self.external_camera.azimuth = 140.0
        self.external_camera.elevation = -16.0

        self.eye_camera = mujoco.MjvCamera()
        self.eye_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.eye_camera.fixedcamid = model.camera('eye0_eye_cam').id

    def is_open(self):
        return not glfw.window_should_close(self.window)

    def draw_scene(self, data, camera, scene, viewport):
        mujoco.mjv_updateScene(
            self.model, data, self.options, None, camera,
            mujoco.mjtCatBit.mjCAT_ALL, scene,
        )
        mujoco.mjr_render(viewport, scene, self.context)

    def render(self, data):
        width, height = glfw.get_framebuffer_size(self.window)
        split = width // 2
        left = mujoco.MjrRect(0, 0, split, height)
        right = mujoco.MjrRect(split, 0, width - split, height)

        self.draw_scene(data, self.external_camera, self.external_scene, left)
        self.draw_scene(data, self.eye_camera, self.eye_scene, right)

        # Read the right pane, apply depth-based focus, and draw it back.
        rgb = np.empty((right.height, right.width, 3), dtype=np.uint8)
        depth = np.empty((right.height, right.width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, right, self.context)
        # mujoco.mjr_drawPixels(self.focus.apply(rgb, depth), None, right, self.context)
        focused_rgb = self.focus.apply(rgb, depth)
        mujoco.mjr_drawPixels(focused_rgb.reshape(-1, 1), None, right, self.context)

        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            left, 'EXTERNAL', 'Base motion + gimbal', self.context,
        )
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            right, 'EYE CAMERA', f'Focus: {self.focus.distance_m:.2f} m', self.context,
        )
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        glfw.destroy_window(self.window)
        glfw.terminate()
