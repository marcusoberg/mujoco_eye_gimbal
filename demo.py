from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pygame
import mujoco

from eye_controller import EyeConfig, EyeController, rot_x, rot_y, rot_z

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "eye_gimbal.xml"

# One native GUI window only: Pygame owns the macOS main thread.
# MuJoCo renders offscreen into an RGB array which is blitted into this window.
WINDOW = (1220, 650)
CONTROL_CENTER = (285, 325)
CONTROL_RADII = (235, 180)
RENDER_POS = (610, 105)
RENDER_SIZE = (576, 432)  # width, height
UI_FPS = 120
RENDER_FPS = 60
BASE_POS = np.array([0.0, 0.0, 1.2])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> MuJoCo wxyz quaternion."""
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def set_base_pose(model, data, base_body_id: int, roll: float, pitch: float, yaw: float) -> None:
    mocap_id = int(model.body_mocapid[base_body_id])
    # Same convention as the eye: +pitch raises local +X.
    R = rot_z(yaw) @ rot_y(-pitch) @ rot_x(roll)
    data.mocap_pos[mocap_id] = BASE_POS
    data.mocap_quat[mocap_id] = mat_to_quat(R)


def clamp_mouse_to_ellipse(mx, my, cx, cy, rx, ry):
    nx = (mx - cx) / rx
    ny = (my - cy) / ry
    r = math.hypot(nx, ny)
    if r > 1.0:
        nx /= r
        ny /= r
    return nx, ny


def rgb_to_surface(rgb: np.ndarray) -> pygame.Surface:
    """Convert MuJoCo HxWx3 RGB uint8 image to a Pygame surface."""
    # pygame.surfarray uses (width, height, channels), unlike image arrays.
    return pygame.surfarray.make_surface(np.ascontiguousarray(np.swapaxes(rgb, 0, 1)))


def make_external_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.70, 0.0, 1.20])
    cam.distance = 2.7
    cam.azimuth = 145
    cam.elevation = -18
    return cam


def render_mujoco(renderer, data, eye_view: bool, external_cam) -> pygame.Surface:
    if eye_view:
        renderer.update_scene(data, camera="eye_cam")
    else:
        renderer.update_scene(data, camera=external_cam)
    rgb = renderer.render()
    return rgb_to_surface(rgb)


def draw_ui(
    screen,
    font,
    small,
    controller,
    mouse_down,
    cmd_deg,
    actual_rad,
    base_angles,
    eye_view,
    sim_surface,
):
    screen.fill((25, 27, 31))
    cx, cy = CONTROL_CENTER
    rx, ry = CONTROL_RADII

    # Left: command surface.
    pygame.draw.ellipse(screen, (150, 156, 168), pygame.Rect(cx-rx, cy-ry, 2*rx, 2*ry), 2)
    pygame.draw.line(screen, (75, 80, 90), (cx-rx, cy), (cx+rx, cy), 1)
    pygame.draw.line(screen, (75, 80, 90), (cx, cy-ry), (cx, cy+ry), 1)

    # Command point.
    cp, cyaw = cmd_deg
    px = cx + (cyaw / controller.cfg.yaw_limit_deg) * rx
    py = cy - (cp / controller.cfg.pitch_limit_deg) * ry
    pygame.draw.circle(screen, (235, 174, 65), (int(px), int(py)), 9, 0 if mouse_down else 2)

    # Actual world gaze point.
    ayaw, apitch, aroll = actual_rad
    ax = cx + (math.degrees(ayaw) / controller.cfg.yaw_limit_deg) * rx
    ay = cy - (math.degrees(apitch) / controller.cfg.pitch_limit_deg) * ry
    pygame.draw.circle(screen, (80, 205, 235), (int(ax), int(ay)), 6, 2)

    # Right: MuJoCo render.
    render_rect = pygame.Rect(RENDER_POS, RENDER_SIZE)
    pygame.draw.rect(screen, (8, 9, 11), render_rect)
    if sim_surface is not None:
        screen.blit(sim_surface, RENDER_POS)
    pygame.draw.rect(screen, (150, 156, 168), render_rect, 2)

    def text(s, x, y, f=None):
        if f is None:
            f = small
        if f is not None:
            screen.blit(f.render(s, True, (220, 223, 230)), (x, y))

    text("Eye-gimbal command surface", 18, 14, font)
    text("click + hold = continuously refresh (pitch, yaw)", 18, 47)
    text("release = watchdog expires -> gaze returns to world (0, 0)", 18, 72)

    status = "LIVE" if controller.command_alive else "neutral watchdog"
    text(f"command: pitch {cp:+5.1f} deg, yaw {cyaw:+5.1f} deg   [{status}]", 18, 101)
    text(
        f"actual:  pitch {math.degrees(apitch):+5.1f} deg, "
        f"yaw {math.degrees(ayaw):+5.1f} deg, roll {math.degrees(aroll):+5.1f} deg",
        18,
        125,
    )
    br, bp, by = [math.degrees(v) for v in base_angles]
    text(f"head:    pitch {bp:+5.1f} deg, yaw {by:+5.1f} deg, roll {br:+5.1f} deg", 18, 149)

    text(f"yaw +/-{controller.cfg.yaw_limit_deg:.0f} deg", cx + rx - 85, cy + 10)
    text(f"pitch +/-{controller.cfg.pitch_limit_deg:.0f} deg", cx + 10, cy - ry - 24)
    text("orange = command", 18, 539)
    text("cyan = actual world gaze", 18, 562)

    text("MuJoCo: eye camera" if eye_view else "MuJoCo: external camera", 610, 68, font)
    text("V: toggle camera", 610, 552)
    text("A/D: head yaw    W/S: head pitch    Q/E: head roll    C: center head", 610, 576)
    text("Esc: quit", 610, 600)

    pygame.display.flip()


def main():
    # Pygame must own the Cocoa/NSWindow main thread on macOS.
    # Do NOT run this file with mjpython; normal `python demo.py` is intended.
    pygame.init()
    pygame.display.set_caption("MuJoCo eye gimbal")
    screen = pygame.display.set_mode(WINDOW)

    try:
        font = pygame.font.Font(None, 28)
        small = pygame.font.Font(None, 22)
    except (ImportError, NotImplementedError, AttributeError) as exc:
        print(f"WARNING: pygame font unavailable; continuing without UI text: {exc}")
        font = None
        small = None

    clock = pygame.time.Clock()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    cfg = EyeConfig()
    controller = EyeController(mujoco, model, cfg)

    base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_base")
    base_angles = np.zeros(3, dtype=float)  # roll, pitch, yaw
    set_base_pose(model, data, base_body, *base_angles)
    mujoco.mj_forward(model, data)

    # MuJoCo's offscreen renderer has no native interactive window. On macOS,
    # current MuJoCo uses an offscreen CGL context, so it does not compete with
    # Pygame for NSWindow/main-thread ownership.
    renderer = mujoco.Renderer(model, height=RENDER_SIZE[1], width=RENDER_SIZE[0])
    external_cam = make_external_camera()

    mouse_down = False
    command_deg = np.zeros(2, dtype=float)  # pitch, yaw
    eye_view = False
    running = True

    accumulator = 0.0
    render_accumulator = 1.0 / RENDER_FPS  # render immediately
    last_wall = time.perf_counter()
    sim_surface = None

    try:
        while running:
            now = time.perf_counter()
            frame_dt = min(now - last_wall, 0.05)
            last_wall = now
            accumulator += frame_dt
            render_accumulator += frame_dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        base_angles[:] = 0.0
                    elif event.key == pygame.K_v:
                        eye_view = not eye_view
                        # Force a fresh render on the next iteration.
                        render_accumulator = 1.0 / RENDER_FPS
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_down = True
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    mouse_down = False

            # Move the simulated head/body to demonstrate stabilization.
            keys = pygame.key.get_pressed()
            head_rate = math.radians(55.0)
            if keys[pygame.K_q]:
                base_angles[0] += head_rate * frame_dt
            if keys[pygame.K_e]:
                base_angles[0] -= head_rate * frame_dt
            if keys[pygame.K_w]:
                base_angles[1] += head_rate * frame_dt
            if keys[pygame.K_s]:
                base_angles[1] -= head_rate * frame_dt
            if keys[pygame.K_a]:
                base_angles[2] += head_rate * frame_dt
            if keys[pygame.K_d]:
                base_angles[2] -= head_rate * frame_dt
            base_angles[:] = np.clip(base_angles, math.radians(-30), math.radians(30))

            if mouse_down:
                mx, my = pygame.mouse.get_pos()
                nx, ny = clamp_mouse_to_ellipse(
                    mx, my,
                    CONTROL_CENTER[0], CONTROL_CENTER[1],
                    CONTROL_RADII[0], CONTROL_RADII[1],
                )
                command_deg[1] = nx * cfg.yaw_limit_deg
                command_deg[0] = -ny * cfg.pitch_limit_deg
                # Heartbeat semantics: this must be refreshed while the command is active.
                controller.submit_command(command_deg[0], command_deg[1])
            elif not controller.command_alive:
                command_deg[:] = 0.0

            # Fixed-step 1 kHz physics, independent of UI/render frame rate.
            while accumulator >= model.opt.timestep:
                set_base_pose(model, data, base_body, *base_angles)
                # Refresh body poses so the controller gets the current simulated IMU attitude.
                mujoco.mj_forward(model, data)
                controller.step(data)
                mujoco.mj_step(model, data)
                accumulator -= model.opt.timestep

            # Rendering is intentionally slower than control/physics.
            if render_accumulator >= 1.0 / RENDER_FPS:
                sim_surface = render_mujoco(renderer, data, eye_view, external_cam)
                render_accumulator %= 1.0 / RENDER_FPS

            actual = controller.actual_world_gaze(data)  # yaw, pitch, roll
            draw_ui(
                screen,
                font,
                small,
                controller,
                mouse_down,
                command_deg,
                actual,
                base_angles,
                eye_view,
                sim_surface,
            )
            clock.tick(UI_FPS)
    finally:
        renderer.close()
        pygame.quit()


if __name__ == "__main__":
    main()
