"""Automatic (scripted expert) driver for the UR5e lift task: synthetic demonstrations.

Enable with UR5E_AUTO=1. It replaces the keys of the Cartesian teleop device. Every control step it
reads the target box pose from the simulator (privileged information that the trained policy never
sees), turns the fingers onto two box faces, moves above the box, descends, closes, lifts and holds.
It uses the same IK and absolute joint-position actions as keyboard teleop, so record_demos records,
checks success and discards failures exactly as for human demos.

Retry: once the gripper is CHECK_LIFT up, the box must have risen BOX_RISE; if not (or if it slips
while held), the expert opens, waits, measures the box again, re-aligns and tries again, up to
UR5E_AUTO_RETRIES times. If all tries fail, or IK cannot reach a target for STUCK_STEPS, it gives up
and the episode is discarded at once (env._ur5e_abort, read by ur5e_lift.episode_too_long).

Tunables: UR5E_AUTO_GRASP_DZ (grasp point above box centre, m, default 0.045),
UR5E_AUTO_HOVER (min,max m above the grasp height, default 0.10,0.16), UR5E_AUTO_LIFT (m, 0.15),
UR5E_AUTO_SPEED (min,max m/s, 0.12,0.20), UR5E_AUTO_RETRIES (default 2).
"""

import math
import os
import random

import numpy as np

from isaaclab_arena.assets.ur5e_cartesian import START_DEFAULT, fk, ik

GRASP_DZ = float(os.environ.get("UR5E_AUTO_GRASP_DZ", "0.045"))
HOVER = tuple(float(x) for x in os.environ.get("UR5E_AUTO_HOVER", "0.10,0.16").split(","))
LIFT = float(os.environ.get("UR5E_AUTO_LIFT", "0.15"))
SPEED = tuple(float(x) for x in os.environ.get("UR5E_AUTO_SPEED", "0.12,0.20").split(","))
RETRIES = int(os.environ.get("UR5E_AUTO_RETRIES", "2"))
VZ = 0.08           # m/s, descend and lift
ROT_RATE = 1.0      # rad/s
ZMIN = 0.01         # lowest grasp point above the robot base, as in keyboard teleop
CHECK_LIFT = 0.04   # after lifting the gripper this far ...
BOX_RISE = 0.02     # ... the box must have risen at least this much
STUCK_STEPS = 100   # 2 s without a reachable IK target -> give up

ENV_REF = []        # set by ur5e_lift.choose_target


def _np(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import warp as wp

        return wp.to_torch(x).detach().cpu().numpy()
    except Exception:
        return np.asarray(x)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw_xyzw(q):
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_quarter(a):
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def _give_up(st, why):
    print(f"[UR5e auto] {why} -> giving up", flush=True)
    st["env"]._ur5e_abort = True
    st["phase"] = "done"


def _move(dev, goal, R_goal, speed, dt):
    """Step the grasp point toward goal and the yaw toward R_goal; True when both are reached."""
    st = dev._auto
    d = goal - dev._cp
    dist = float(np.linalg.norm(d))
    step = min(dist, speed * dt)
    p_new = dev._cp + (d / dist * step if dist > 1e-9 else 0.0)
    p_new[2] = max(p_new[2], ZMIN)
    Rrel = R_goal @ dev._cR.T
    ang = math.atan2(Rrel[1, 0], Rrel[0, 0])
    a = max(-ROT_RATE * dt, min(ROT_RATE * dt, ang))
    R_new = _rz(a) @ dev._cR
    q, ep, eo = ik(dev._cq, p_new, R_new)
    if ep < 0.005 and eo < 0.02:
        dev._cq, dev._cp, dev._cR = q, p_new, R_new
        st["stuck"] = 0
        return dist - step < 0.002 and abs(ang - a) < 0.01
    st["stuck"] += 1
    if st["stuck"] >= STUCK_STEPS:
        _give_up(st, f"stuck in {st['phase']}: IK cannot reach {np.round(goal, 3)} "
                     f"(pos err {ep:.3f} m, rot err {eo:.3f})")
    return False


def _align(dev, st, env, name, world_to_base):
    tp = _np(env.scene["ee_frame"].data.target_pos_w)[0]
    fd = world_to_base @ (tp[2] - tp[1])
    finger = math.atan2(fd[1], fd[0])
    base_rot = math.atan2(world_to_base[1, 0], world_to_base[0, 0])
    box = _yaw_xyzw(_np(env.scene[name].data.root_quat_w)[0]) + base_rot
    turn = _wrap_quarter(box - finger)
    st["R_goal"] = _rz(turn) @ dev._cR
    print(f"[UR5e auto] {name} try {st['attempt'] + 1}: box yaw {math.degrees(box):.0f}, "
          f"finger yaw {math.degrees(finger):.0f}, turn {math.degrees(turn):.0f} deg, "
          f"speed {st['speed']:.2f} m/s", flush=True)


def _new_episode(dev, env):
    q = np.array([float(x) for x in os.environ.get("UR5E_START", START_DEFAULT).split(",")], dtype=float)
    dev._cq = q
    dev._cp, dev._cR = fk(q)
    dev._closed = False
    print("[UR5e auto] new episode: moving to start pose", flush=True)
    return {
        "env": env,
        "last_ep": -1,
        "phase": "settle",
        "n": random.randint(60, 90),
        "speed": random.uniform(*SPEED),
        "hover": random.uniform(*HOVER),
        "jitter": np.array([random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), 0.0]),
        "close_steps": random.randint(20, 35),
        "stuck": 0,
        "attempt": 0,
    }


def _grasp_failed(st, why):
    st["attempt"] += 1
    if st["attempt"] > RETRIES:
        _give_up(st, f"{why}, no tries left")
        return
    print(f"[UR5e auto] {why} -> opening and retrying (try {st['attempt'] + 1})", flush=True)
    st["phase"], st["n"] = "reopen", 15


def auto_step(dev, world_to_base):
    if not ENV_REF:
        return
    env = ENV_REF[0]
    dt = float(getattr(env, "step_dt", 0.02))
    ep = int(_np(env.episode_length_buf).reshape(-1)[0])
    st = getattr(dev, "_auto", None)
    if st is None or st["env"] is not env or ep < st["last_ep"]:
        st = _new_episode(dev, env)
        dev._auto = st
    st["last_ep"] = ep

    names = getattr(env, "_ur5e_names", None)
    if not names:
        return
    name = names[int(_np(env._ur5e_target).reshape(-1)[0])]
    base = _np(env.scene["robot"].data.root_pos_w)[0]
    pw = _np(env.scene[name].data.root_pos_w)[0]
    pb = world_to_base @ (pw - base)

    ph = st["phase"]
    if ph == "settle":
        st["n"] -= 1
        if st["n"] <= 0:
            _align(dev, st, env, name, world_to_base)
            st["phase"] = "hover"
    elif ph == "hover":
        goal = pb + st["jitter"] + np.array([0.0, 0.0, GRASP_DZ + st["hover"]])
        if _move(dev, goal, st["R_goal"], st["speed"], dt):
            st["phase"] = "descend"
    elif ph == "descend":
        goal = pb + np.array([0.0, 0.0, GRASP_DZ])
        if _move(dev, goal, st["R_goal"], VZ, dt):
            st["phase"], st["n"] = "close", st["close_steps"]
    elif ph == "close":
        dev._closed = True
        st["n"] -= 1
        if st["n"] <= 0:
            st["box_z0"], st["grasp_z"], st["checked"] = float(pw[2]), float(dev._cp[2]), False
            st["lift_goal"] = dev._cp + np.array([0.0, 0.0, LIFT])
            st["phase"] = "lift"
    elif ph == "lift":
        reached = _move(dev, st["lift_goal"], st["R_goal"], VZ, dt)
        if not st["checked"] and dev._cp[2] - st["grasp_z"] >= CHECK_LIFT:
            st["checked"] = True
            if pw[2] - st["box_z0"] < BOX_RISE:
                _grasp_failed(st, "box did not come up")
                return
        if reached and st["phase"] == "lift":
            st["phase"] = "hold"
    elif ph == "hold":
        if pw[2] - st["box_z0"] < BOX_RISE:
            _grasp_failed(st, "box slipped out")
    elif ph == "reopen":
        dev._closed = False
        st["n"] -= 1
        if st["n"] <= 0:
            st["phase"], st["n"] = "settle", 15
