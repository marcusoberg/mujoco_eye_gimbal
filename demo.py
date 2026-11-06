
"""Eye-gimbal demo using MuJoCo's native viewer (no Pygame).

Run:
    macOS:          mjpython demo_mujoco.py
    Linux/Windows:  python demo_mujoco.py

Controls:
    I / K       Gaze up / down
    J / L       Gaze left / right
    R           Release gaze command (watchdog returns to neutral)
    W / S       Head pitch
    A / D       Head yaw
    Q / E       Head roll
    C           Center head
    V           Toggle external / eye camera

Each key press adjusts the angle by 5 degrees.
"""

from __future__ import annotations

import math
import queue
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from eye_controller import (
    EyeConfig,
    EyeController,
    clamp_ellipse,
    rot_x,
    rot_y,
    rot_z,
)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "eye_gimbal.xml"

BASE_POS = np.array([0.0, 0.0, 1.2])

GAZE_STEP_RAD = math.radians(5.0)
HEAD_STEP_RAD = math.radians(5.0)
HEAD_LIMIT_RAD = math.radians(30.0)

VIEWER_FPS = 60.0


def set_head_pose(data, mocap_id, angles):
    """Set head orientation using the mocap body."""
    roll, pitch, yaw = angles

    R = rot_z(yaw) @ rot_y(-pitch) @ rot_x(roll)

    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R.reshape(9))

    data.mocap_pos[mocap_id] = BASE_POS
    data.mocap_quat[mocap_id] = quat


def set_camera(viewer, model, eye_view):
    """Switch between eye camera and external camera."""
    with viewer.lock():
        if eye_view:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                "eye_cam",
            )
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [0.70, 0.0, 1.20]
            viewer.cam.distance = 2.7
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -18


def main():
    # Load MuJoCo model and simulation state.
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Existing world-stabilized eye controller.
    cfg = EyeConfig()
    controller = EyeController(mujoco, model, cfg)

    # Find the mocap head body.
    head_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "head_base",
    )
    mocap_id = int(model.body_mocapid[head_id])

    # Head angles: roll, pitch, yaw.
    head_angles = np.zeros(3)

    # Desired world gaze: pitch, yaw.
    gaze_command = np.zeros(2)

    command_active = False
    eye_view = False

    set_head_pose(data, mocap_id, head_angles)
    mujoco.mj_forward(model, data)

    # Queue keyboard events from MuJoCo's viewer thread.
    pressed_keys = queue.SimpleQueue()

    def on_key(keycode):
        pressed_keys.put(keycode)

    print(__doc__)

    # Launch the native MuJoCo viewer.
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=on_key,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:

        set_camera(viewer, model, eye_view=False)
        viewer.sync()

        last_wall = time.perf_counter()
        last_sync = last_wall
        last_report = last_wall

        accumulated = 0.0
        timestep = float(model.opt.timestep)

        while viewer.is_running():
            now = time.perf_counter()

            frame_dt = min(now - last_wall, 0.05)
            accumulated += frame_dt
            last_wall = now

            # Process keyboard commands.
            while True:
                try:
                    key = pressed_keys.get_nowait()
                except queue.Empty:
                    break

                # Eye gaze commands.
                if key == ord("I"):
                    gaze_command[0] += GAZE_STEP_RAD
                    command_active = True

                elif key == ord("K"):
                    gaze_command[0] -= GAZE_STEP_RAD
                    command_active = True

                elif key == ord("J"):
                    gaze_command[1] += GAZE_STEP_RAD
                    command_active = True

                elif key == ord("L"):
                    gaze_command[1] -= GAZE_STEP_RAD
                    command_active = True

                elif key == ord("R"):
                    # Stop refreshing the gaze command.
                    # Watchdog returns the eye to world neutral.
                    command_active = False

                # Head motion commands.
                elif key == ord("Q"):
                    head_angles[0] += HEAD_STEP_RAD

                elif key == ord("E"):
                    head_angles[0] -= HEAD_STEP_RAD

                elif key == ord("W"):
                    head_angles[1] += HEAD_STEP_RAD

                elif key == ord("S"):
                    head_angles[1] -= HEAD_STEP_RAD

                elif key == ord("A"):
                    head_angles[2] += HEAD_STEP_RAD

                elif key == ord("D"):
                    head_angles[2] -= HEAD_STEP_RAD

                elif key == ord("C"):
                    head_angles[:] = 0.0

                elif key == ord("V"):
                    eye_view = not eye_view
                    set_camera(viewer, model, eye_view)

            # Enforce elliptical gaze limits.
            gaze_command[0], gaze_command[1] = clamp_ellipse(
                gaze_command[0],
                gaze_command[1],
                controller.pitch_lim,
                controller.yaw_lim,
            )

            # Limit head rotations.
            head_angles[:] = np.clip(
                head_angles,
                -HEAD_LIMIT_RAD,
                HEAD_LIMIT_RAD,
            )

            # Continuously refresh active gaze commands.
            if command_active:
                controller.submit_command(
                    *gaze_command,
                    radians=True,
                )
            elif not controller.command_alive:
                gaze_command[:] = 0.0

            # Run the simulation at its fixed 1 kHz timestep.
            while accumulated >= timestep:
                set_head_pose(data, mocap_id, head_angles)

                # Update body poses for IMU attitude feedback.
                mujoco.mj_forward(model, data)

                # Compute gimbal motor targets.
                controller.step(data)

                # Advance MuJoCo physics.
                mujoco.mj_step(model, data)

                accumulated -= timestep

            # Update the viewer at approximately 60 FPS.
            if now - last_sync >= 1.0 / VIEWER_FPS:
                viewer.sync()
                last_sync = now

            # Print current gaze orientation.
            if now - last_report >= 0.5:
                yaw, pitch, roll = controller.actual_world_gaze(data)

                status = (
                    "LIVE"
                    if controller.command_alive
                    else "neutral watchdog"
                )

                print(
                    "\r"
                    f"target (pitch,yaw): "
                    f"({math.degrees(gaze_command[0]):+5.1f}, "
                    f"{math.degrees(gaze_command[1]):+5.1f}) deg | "
                    f"actual (pitch,yaw,roll): "
                    f"({math.degrees(pitch):+5.1f}, "
                    f"{math.degrees(yaw):+5.1f}, "
                    f"{math.degrees(roll):+5.1f}) deg | "
                    f"{status:<18}",
                    end="",
                    flush=True,
                )

                last_report = now

            time.sleep(0.002)

    print()


if __name__ == "__main__":
    main()
