"""Generate scripted or keyboard-controlled UR5e/RG2 demonstrations from the local USD.

The generated file follows the HDF5 layout consumed by
``isaaclab_arena_gr00t/lerobot/convert_hdf5_to_native_lerobot.py``.  The
standalone Kit application also starts the repository WebRTC experience, so
the motion can be watched while the file is being generated.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd-path", type=Path, default=Path(__file__).resolve().parents[1] / "ur5ewithcam2.usd")
parser.add_argument("--output", type=Path, default=Path("/datasets/isaaclab_arena/ur5e_rg2/ur5e_rg2_dataset_generated.hdf5"))
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--duration-scale", type=float, default=1.0, help="Scale waypoint dwell times for smoke tests.")
parser.add_argument(
    "--keyboard",
    action="store_true",
    help="Record demonstrations interactively with the UR5e keyboard controller.",
)
parser.add_argument(
    "--window",
    action="store_true",
    help="Open a local Kit window for keyboard input instead of running headless.",
)
parser.add_argument("--host", default="100.65.219.19")
parser.add_argument("--signal-port", type=int, default=49100)
parser.add_argument("--image-camera", default="/World/org2/Camera")
parser.add_argument("--view-eye", nargs=3, type=float, default=(2.0, -2.0, 1.5))
parser.add_argument("--view-target", nargs=3, type=float, default=(0.0, 0.0, 0.5))
parser.add_argument("--no-webrtc", action="store_true")
args = parser.parse_args()

if not args.usd_path.is_file():
    raise FileNotFoundError(f"USD file not found: {args.usd_path}")
if args.episodes < 1 or args.fps < 1 or args.duration_scale <= 0:
    raise ValueError("--episodes, --fps, and --duration-scale must be positive")

_native_library_paths = (
    "/isaac-sim/kit/python/lib/python3.12/site-packages/isaacteleop/cloudxr/native",
    "/isaac-sim/extscache/omni.kit.livestream.webrtc-10.7.0+110.1.2.lx64.r.cp312/bin",
)
if os.environ.get("ARENA_WEBRTC_LIBS_READY") != "1":
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        (*_native_library_paths, os.environ.get("LD_LIBRARY_PATH", ""))
    ).rstrip(":")
    os.environ["ARENA_WEBRTC_LIBS_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

from isaacsim import SimulationApp

experience = str(Path(__file__).resolve().parent / "isaacsim_webrtc.kit")
simulation_app = SimulationApp(
    {
        "headless": not args.window,
        **({} if args.window else {"experience": experience}),
        "/exts/omni.kit.livestream.app/primaryStream/publicIp": args.host,
        "/exts/omni.kit.livestream.app/primaryStream/signalPort": args.signal_port,
        "/exts/omni.kit.livestream.app/primaryStream/streamPort": 47998,
        "/exts/omni.kit.livestream.app/primaryStream/streamType": "webrtc",
        "extra_args": [
            "--ext-folder",
            "/isaac-sim/extscache",
            "--ext-folder",
            "/isaac-sim/extsDeprecated",
        ],
    }
)

import omni.usd
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
JOINT_PATHS = [Sdf.Path(f"/World/ur5e/joints/{name}") for name in ARM_JOINTS]
FINGER_JOINT_PATH = Sdf.Path("/World/org2/onrobot_rg2_base_link/finger_joint")
IMAGE_SIZE = (640, 480)


def add_task_geometry(stage: Usd.Stage) -> None:
    """Add a dynamic object and a static work surface for the USD gripper."""
    material = UsdShade.Material.Define(stage, "/World/ScriptedTask/GraspMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(1.2)
    physics_material.CreateDynamicFrictionAttr(1.0)
    physics_material.CreateRestitutionAttr(0.0)

    cube = UsdGeom.Cube.Define(stage, "/World/ScriptedTask/RubiksCube")
    cube.CreateSizeAttr(0.06)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.66, 0.165, 0.08))
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.15, 0.8)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.05)
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)

    bowl = UsdGeom.Cylinder.Define(stage, "/World/ScriptedTask/Bowl")
    bowl.CreateRadiusAttr(0.16)
    bowl.CreateHeightAttr(0.08)
    bowl.AddTranslateOp().Set(Gf.Vec3d(0.49, 0.36, 0.04))
    bowl.CreateDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.45, 0.05)])
    UsdPhysics.CollisionAPI.Apply(bowl.GetPrim())

    floor = UsdGeom.Cube.Define(stage, "/World/ScriptedTask/Floor")
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
    floor.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.02))
    floor.CreateDisplayColorAttr().Set([Gf.Vec3f(0.18, 0.18, 0.18)])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    UsdShade.MaterialBindingAPI(floor.GetPrim()).Bind(material)


def ensure_lighting(stage: Usd.Stage) -> None:
    """Provide lighting for USD assets that contain no usable scene lights."""
    lights = [prim for prim in stage.Traverse() if prim.IsA(UsdLux.LightAPI)]
    if lights:
        return

    dome = UsdLux.DomeLight.Define(stage, "/World/ScriptedTask/DomeLight")
    dome.CreateIntensityAttr(1200.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    area = UsdLux.RectLight.Define(stage, "/World/ScriptedTask/KeyLight")
    area.CreateIntensityAttr(25000.0)
    area.CreateWidthAttr(5.0)
    area.CreateHeightAttr(5.0)
    area.AddTranslateOp().Set(Gf.Vec3d(1.5, -1.5, 3.0))
    area.AddRotateXYZOp().Set(Gf.Vec3f(25.0, 0.0, 40.0))


def set_joint_target(stage: Usd.Stage, joint_path: Sdf.Path, radians: float) -> None:
    joint = stage.GetPrimAtPath(joint_path)
    if not joint:
        raise RuntimeError(f"Missing expected joint prim: {joint_path}")
    drive = UsdPhysics.DriveAPI.Get(joint, "angular")
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
    drive.GetTargetPositionAttr().Set(math.degrees(radians))


def set_targets(stage: Usd.Stage, arm: np.ndarray, finger: float) -> np.ndarray:
    for path, value in zip(JOINT_PATHS, arm):
        set_joint_target(stage, path, float(value))
    set_joint_target(stage, FINGER_JOINT_PATH, finger)
    return np.concatenate((arm.astype(np.float32), np.asarray([finger], dtype=np.float32)))


def make_waypoints(episode: int) -> list[tuple[np.ndarray, float, int]]:
    """Return joint-space waypoints for approach, grasp, lift, place, release."""
    phase = (episode % 5) * 0.025
    home = np.array([0.0, -1.2, 1.7, -2.0, -1.57, 0.0], dtype=np.float32)
    approach = home + np.array([0.10 + phase, 0.12, -0.10, -0.02, 0.0, 0.0], dtype=np.float32)
    descend = approach + np.array([0.0, 0.18, -0.25, 0.07, 0.0, 0.0], dtype=np.float32)
    lift = approach
    place = home + np.array([0.45 + phase, 0.05, -0.20, 0.15, 0.0, 0.0], dtype=np.float32)
    return [
        (home, 0.0, 50),
        (approach, 0.0, 50),
        (descend, 0.0, 35),
        (descend, 0.785, 25),
        (lift, 0.785, 45),
        (place, 0.785, 60),
        (place, 0.0, 30),
        (home, 0.0, 60),
    ]


def capture_image(viewport, path: Path) -> np.ndarray:
    from omni.kit.viewport.utility import capture_viewport_to_file

    capture_viewport_to_file(viewport, file_path=str(path), is_hdr=False)
    for _ in range(120):
        simulation_app.update()
        if path.is_file():
            break
    if not path.is_file():
        raise RuntimeError(f"Viewport image capture did not complete: {path}")
    with Image.open(path) as image:
        frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if frame.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise RuntimeError(f"Unexpected viewport image shape {frame.shape}; expected {(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)}")
    return frame


def write_episode(trajectory, states: list[np.ndarray], actions: list[np.ndarray], frames: list[np.ndarray]) -> None:
    """Write one terminal-sample-padded demonstration to the HDF5 trajectory."""
    if not states:
        raise ValueError("Cannot write an empty demonstration.")
    observations = trajectory.create_group("obs")
    cameras = trajectory.create_group("camera_obs")
    terminal_state = states[-1].copy()
    terminal_frame = frames[-1].copy()
    observations.create_dataset(
        "robot_joint_pos", data=np.stack([*states, terminal_state]), compression="gzip"
    )
    trajectory.create_dataset(
        "processed_actions", data=np.stack([*actions, terminal_state]), compression="gzip"
    )
    camera_data = np.stack([*frames, terminal_frame])
    for camera_name in ("external_camera_rgb", "robot_head_cam", "wrist_camera_rgb"):
        cameras.create_dataset(camera_name, data=camera_data, compression="gzip")


def record_keyboard_demos(
    stage,
    viewport,
    frame_dir: Path,
    data,
) -> None:
    """Record UR5e keyboard demonstrations until the requested count is reached."""
    from isaaclab.devices import DeviceCfg

    from isaaclab_arena.assets.device_library import Ur5eKeyboard

    device = Ur5eKeyboard(DeviceCfg(class_type=Ur5eKeyboard))
    state = {"finish": False, "reset": False, "quit": False}

    def finish_episode() -> None:
        state["finish"] = True

    def reset_episode() -> None:
        state["reset"] = True

    def quit_recording() -> None:
        state["quit"] = True

    device.add_callback("N", finish_episode)
    device.add_callback("R", reset_episode)
    device.add_callback("X", quit_recording)
    print(
        "Keyboard recording: Q/A, W/S, E/D, T/G, Y/H, U/J move joints; "
        "K toggles RG2; L resets pose; N saves; R discards/resets; X quits.",
        flush=True,
    )

    current_episode = 0
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    current = np.zeros(7, dtype=np.float32)
    step_updates = max(1, round(60 / args.fps))
    while current_episode < args.episodes and not state["quit"]:
        for _ in range(step_updates):
            simulation_app.update()
        command = device.advance().detach().cpu().numpy().astype(np.float32)
        action = set_targets(stage, command[:6], float(command[6]))
        image_path = frame_dir / f"keyboard_{current_episode:04d}_{len(frames):06d}.png"
        frames.append(capture_image(viewport, image_path))
        states.append(action.copy())
        actions.append(action.copy())
        current = action

        if state["reset"]:
            state["reset"] = False
            device.reset()
            states.clear()
            actions.clear()
            frames.clear()
            current = np.zeros(7, dtype=np.float32)
            print("Discarded current episode; pose reset.", flush=True)
        elif state["finish"]:
            state["finish"] = False
            trajectory = data.create_group(f"demo_{current_episode:04d}")
            write_episode(trajectory, states, actions, frames)
            current_episode += 1
            print(f"Recorded episode {current_episode}/{args.episodes} ({len(frames)} frames)", flush=True)
            device.reset()
            states.clear()
            actions.clear()
            frames.clear()
            current = np.zeros(7, dtype=np.float32)

    if state["quit"]:
        print(f"Stopped after recording {current_episode}/{args.episodes} episodes.", flush=True)


def main() -> None:
    import omni.kit.app
    from isaacsim.core.rendering_manager import ViewportManager
    from omni.kit.viewport.utility import get_active_viewport
    import omni.timeline

    omni.usd.get_context().open_stage(str(args.usd_path.resolve()))
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {args.usd_path}")
    add_task_geometry(stage)
    ensure_lighting(stage)

    if not stage.GetPrimAtPath(Sdf.Path(args.image_camera)):
        raise RuntimeError(f"Camera prim not found: {args.image_camera}")
    ViewportManager.set_camera_view(
        camera="/OmniverseKit_Persp",
        eye=args.view_eye,
        target=args.view_target,
    )
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Isaac Sim did not create an active viewport for image capture")
    viewport.set_texture_resolution(IMAGE_SIZE)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Opened {args.usd_path}", flush=True)
    if not args.no_webrtc:
        print(f"Connect an Isaac Sim WebRTC client to {args.host}:{args.signal_port}.", flush=True)

    with h5py.File(args.output, "w") as output:
        data = output.create_group("data")
        output.attrs["language_instruction"] = "Pick up the Rubik's cube and place it in the bowl."
        output.attrs["fps"] = args.fps
        with tempfile.TemporaryDirectory(prefix="ur5e_rg2_frames_") as frame_dir:
            for episode in range(args.episodes):
                if args.keyboard:
                    record_keyboard_demos(stage, viewport, Path(frame_dir), data)
                    break

                trajectory = data.create_group(f"demo_{episode:04d}")
                states: list[np.ndarray] = []
                actions: list[np.ndarray] = []
                frames: list[np.ndarray] = []
                current = np.zeros(7, dtype=np.float32)
                for waypoint, finger, duration in make_waypoints(episode):
                    start = current[:6].copy()
                    start_finger = float(current[6])
                    step_count = max(1, round(duration * args.duration_scale))
                    for step in range(step_count):
                        alpha = (step + 1) / step_count
                        arm = start + alpha * (waypoint - start)
                        finger_value = start_finger + alpha * (finger - start_finger)
                        action = set_targets(stage, arm, finger_value)
                        for _ in range(max(1, round(60 / args.fps))):
                            simulation_app.update()
                        image_path = Path(frame_dir) / f"{episode:04d}_{len(frames):06d}.png"
                        frames.append(capture_image(viewport, image_path))
                        states.append(action.copy())
                        actions.append(action.copy())
                        current = action
                write_episode(trajectory, states, actions, frames)
                print(f"Recorded episode {episode + 1}/{args.episodes} ({len(frames) - 1} frames)", flush=True)
    print(f"HDF5 demonstrations written to {args.output}", flush=True)


try:
    main()
finally:
    simulation_app.close()
