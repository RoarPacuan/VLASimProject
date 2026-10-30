# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Open a USD stage and expose its Kit viewport through Isaac Sim WebRTC."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd-path", type=Path, required=True)
parser.add_argument("--host", default="100.65.219.19")
parser.add_argument("--signal-port", type=int, default=49100)
parser.add_argument("--camera-eye", nargs=3, type=float, default=(2.0, -2.0, 1.5))
parser.add_argument("--camera-target", nargs=3, type=float, default=(0.0, 0.0, 0.5))
args = parser.parse_args()

assert args.usd_path.is_file(), f"USD file not found: {args.usd_path}"

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

simulation_app = SimulationApp(
    {
        "headless": True,
        "experience": str(Path(__file__).resolve().parent / "isaacsim_webrtc.kit"),
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
from isaacsim.core.rendering_manager import ViewportManager
from pxr import UsdLux

omni.usd.get_context().open_stage(str(args.usd_path.resolve()))
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError(f"Failed to open USD stage: {args.usd_path}")

if not any(prim.IsA(UsdLux.LightAPI) for prim in stage.Traverse()):
    dome_light = UsdLux.DomeLight.Define(stage, "/ArenaWebRTCFallbackDomeLight")
    dome_light.CreateIntensityAttr(1000.0)

ViewportManager.set_camera_view(
    camera="/OmniverseKit_Persp",
    eye=args.camera_eye,
    target=args.camera_target,
)
print(f"Opened {args.usd_path}", flush=True)
print("Connect an Isaac Sim WebRTC client to this host on signaling port 49100.", flush=True)

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()