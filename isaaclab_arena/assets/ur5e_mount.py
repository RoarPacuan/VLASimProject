"""Rotate the RG2 gripper (and the wrist camera mounted on it) about the UR5e flange axis.

With the original sim mount, the same joint angles that make the real wrist camera look forward
over the table made the sim camera look sideways. UR5E_TOOL_YAW_DEG (default 90) rotates the fixed
joint wrist_3_link -> onrobot_rg2_base_link about the tool axis (z of wrist_3_link; the gripper
centre sits at +0.24 m along it) before physics starts. UR5E_TOOL_YAW_DEG=0 keeps the old mount.
"""

import math
import os

JOINT = "Robot/org2/onrobot_rg2_base_link/AssemblerFixedJoint"


def rotate_tool_mount(env, env_ids, deg):
    import omni.usd
    from pxr import Gf

    stage = omni.usd.get_context().get_stage()
    half = math.radians(deg) / 2.0
    q = Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half)))
    for e in range(env.num_envs):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/{JOINT}")
        if not prim.IsValid():
            print(f"[UR5e] tool yaw: joint not found in env {e}", flush=True)
            continue
        attr = prim.GetAttribute("physics:localRot0")
        old = attr.Get() or Gf.Quatf(1.0)
        attr.Set(q * old)
    print(f"[UR5e] gripper + wrist camera rotated {deg:+.0f} deg about the flange axis", flush=True)


def add_tool_yaw(env_cfg):
    deg = float(os.environ.get("UR5E_TOOL_YAW_DEG", "90"))
    if deg == 0.0:
        return
    from isaaclab.managers import EventTermCfg

    env_cfg.events.ur5e_tool_yaw = EventTermCfg(func=rotate_tool_mount, mode="prestartup", params={"deg": deg})
