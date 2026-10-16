# MuJoCo 3-axis eye gimbal demo

A small MuJoCo + Pygame model of a camera/robot eye with coincident yaw, pitch, and roll axes.

## macOS note

Run the demo with **normal Python**, not `mjpython`:

```bash
python demo.py
```

`mujoco.viewer.launch_passive()` requires `mjpython` on macOS, while Pygame requires its `NSWindow` to be created on the main thread. Those requirements conflict if both native windows are used in one process.

This version therefore does **not** use `mujoco.viewer`. It uses `mujoco.Renderer` for offscreen MuJoCo rendering and copies the RGB image into the Pygame window. Pygame is the only native GUI window.

## Install

```bash
python -m pip install -r requirements.txt
python demo.py
```

For Python 3.14, `pygame-ce` is used because it ships current macOS wheels and keeps the normal `import pygame` API.

## Controls

- Hold left mouse button: continuously send the pitch/yaw command indicated by the cursor.
- Release left mouse button: stop sending the command. After the watchdog timeout, desired gaze returns to world `(pitch=0, yaw=0)`.
- `A / D`: rotate simulated head yaw.
- `W / S`: rotate simulated head pitch.
- `Q / E`: rotate simulated head roll.
- `C`: center simulated head.
- `V`: switch the embedded MuJoCo image between external camera and the moving eye camera.
- `Esc`: quit.

## Controller semantics

`EyeController.submit_command(delta_pitch, delta_yaw)` takes an **absolute gaze offset from world neutral**, in degrees by default. The caller must repeatedly refresh the command. The default timeout is 75 ms.

The controller:

1. clamps requested pitch/yaw to an ellipse,
2. slew-limits gaze motion to 550 deg/s,
3. constructs the desired camera orientation in the world frame with zero world roll,
4. compares that orientation with the simulated head/IMU attitude,
5. solves the required relative yaw/pitch/roll gimbal angles,
6. sends those targets to MuJoCo position actuators.

All three gimbal hinge axes intersect at the same model-space point.

## Files

- `eye_gimbal.xml` — MuJoCo model.
- `eye_controller.py` — watchdog, gaze slew limiter, stabilization and inverse orientation calculation.
- `demo.py` — Pygame command UI, simulated head motion, physics loop and embedded offscreen MuJoCo rendering.
