"""Cartesian (toolhead) keyboard teleop for the UR5e + RG2.

The keys move the gripper (forward/back, left/right, up/down, rotate) and inverse
kinematics turns that into the 6 arm joint angles. The robot is still commanded
with ABSOLUTE JOINT TARGETS, so the recorded action format is unchanged.

Keys (hold to move):
    W / S   forward / back   (away from / toward the robot base)
    A / D   left / right
    Up / Down arrow   up / down
    Z / X   rotate gripper
    Space   open / close gripper
    L       reset to start pose

Environment variables:
    UR5E_TELEOP=joint   use the old per-joint keys instead
    UR5E_XY_SPEED       m/s for W/A/S/D (default 0.20)
    UR5E_LIN_SPEED      m/s for up/down (default 0.10)
    UR5E_ROT_SPEED      rad/s     (default 0.8)
    UR5E_TCP            flange -> grasp point in metres (default 0.20)
    UR5E_ZMIN           lowest grasp-point height above robot base (default 0.01)
    UR5E_BASE_YAW       robot base yaw in world, radians (default pi = rot (0,0,0,1))
"""

import math
import os
import time

import numpy as np

# Standard UR5e DH parameters
_D = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
_A = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)


def _env_f(name, default):
    return float(os.environ.get(name, default))


TCP = _env_f("UR5E_TCP", 0.20)
START_DEFAULT = "-0.56,-2.29,2.25,-1.53,-1.57,-0.56"


def fk_frames(q):
    T = np.eye(4)
    frames = [T.copy()]
    for i in range(6):
        ct, st = math.cos(q[i]), math.sin(q[i])
        ca, sa = math.cos(_ALPHA[i]), math.sin(_ALPHA[i])
        A = np.array([
            [ct, -st * ca, st * sa, _A[i] * ct],
            [st, ct * ca, -ct * sa, _A[i] * st],
            [0.0, sa, ca, _D[i]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        T = T @ A
        frames.append(T.copy())
    tool = np.eye(4)
    tool[2, 3] = TCP
    frames.append(T @ tool)
    return frames


def fk(q):
    f = fk_frames(q)[-1]
    return f[:3, 3].copy(), f[:3, :3].copy()


def _rot_err(R, Rt):
    return 0.5 * (np.cross(R[:, 0], Rt[:, 0]) + np.cross(R[:, 1], Rt[:, 1]) + np.cross(R[:, 2], Rt[:, 2]))


def ik(q, p_t, R_t, iters=10, lam=0.03):
    """Damped least squares IK starting from q. Returns (q, pos_err_m, rot_err)."""
    q = np.array(q, dtype=float)
    for _ in range(iters):
        f = fk_frames(q)
        pe, R = f[-1][:3, 3], f[-1][:3, :3]
        e = np.concatenate([p_t - pe, _rot_err(R, R_t)])
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
            break
        J = np.zeros((6, 6))
        for i in range(6):
            z, o = f[i][:3, 2], f[i][:3, 3]
            J[:3, i] = np.cross(z, pe - o)
            J[3:, i] = z
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        m = np.max(np.abs(dq))
        if m > 0.2:
            dq *= 0.2 / m
        q += dq
    q[2] = float(np.clip(q[2], -math.pi, math.pi))
    p, R = fk(q)
    return q, float(np.linalg.norm(p_t - p)), float(np.linalg.norm(_rot_err(R, R_t)))


_KEYS = {
    "W": (0, +1), "S": (0, -1),
    "A": (1, +1), "D": (1, -1),
    "UP": (2, +1), "DOWN": (2, -1),
    "Z": (3, +1), "X": (3, -1),
}


def install(cls):
    """Turn the existing Ur5eKeyboard class into a Cartesian teleop device."""
    if os.environ.get("UR5E_TELEOP", "cartesian") != "cartesian":
        return cls
    if getattr(cls, "_cartesian_installed", False):
        return cls

    start = [float(x) for x in os.environ.get("UR5E_START", START_DEFAULT).split(",")]
    lin = _env_f("UR5E_LIN_SPEED", 0.10)
    xy = _env_f("UR5E_XY_SPEED", 0.20)
    rot = _env_f("UR5E_ROT_SPEED", 0.8)
    zmin = _env_f("UR5E_ZMIN", 0.01)
    yaw = _env_f("UR5E_BASE_YAW", math.pi)
    c, s = math.cos(-yaw), math.sin(-yaw)
    world_to_base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    orig_init = cls.__init__
    orig_advance = cls.advance
    orig_reset = getattr(cls, "reset", None)

    def _cart_reset(self):
        self._cq = np.array(start, dtype=float)
        self._cp, self._cR = fk(self._cq)
        self._held = set()
        self._ct = None
        self._closed = False

    def _on_key(self, event):
        import carb

        T = carb.input.KeyboardEventType
        if event.type not in (T.KEY_PRESS, T.KEY_RELEASE):
            return True  # CHAR / KEY_REPEAT events: nothing to do
        inp = event.input
        name = inp if isinstance(inp, str) else getattr(inp, "name", str(inp))
        if event.type == T.KEY_PRESS:
            if name in _KEYS:
                self._held.add(name)
            elif name == "SPACE":
                self._closed = not self._closed
            elif name == "L":
                _cart_reset(self)
        else:
            self._held.discard(name)
        return True

    def _disable_space_hotkey():
        try:
            import omni.kit.hotkeys.core as hk

            reg = hk.get_hotkey_registry()
            for h in list(reg.get_all_hotkeys()):
                kc = getattr(h, "key_combination", None)
                txt = str(getattr(kc, "as_string", "")) + " " + str(getattr(kc, "key", ""))
                if "SPACE" in txt.upper() and not getattr(kc, "modifiers", 0):
                    reg.deregister_hotkey(h)
                    print("[UR5e] disabled Isaac Sim Space hotkey:", getattr(h, "action_text", h))
        except Exception as e:
            print("[UR5e] could not check Space hotkey (%s)" % e)

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _cart_reset(self)
        import carb
        import omni.appwindow

        self._cart_input = carb.input.acquire_input_interface()
        kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._cart_sub = self._cart_input.subscribe_to_keyboard_events(kb, lambda ev, *a: _on_key(self, ev))
        _disable_space_hotkey()
        p = world_to_base.T @ self._cp
        print("[UR5e] CARTESIAN teleop ON: W/S fwd/back  A/D left/right  arrows up/down  Z/X rotate  Space gripper  L reset")
        print("[UR5e] start grasp point (relative to robot base, world axes): x=%.3f y=%.3f z=%.3f" % tuple(p))

    def advance(self, *args, **kwargs):
        out = orig_advance(self, *args, **kwargs)
        if not hasattr(self, "_cq"):
            _cart_reset(self)
        now = time.monotonic()
        dt = 0.02 if self._ct is None else min(now - self._ct, 0.05)
        self._ct = now

        if os.environ.get("UR5E_AUTO", "0") == "1":
            from isaaclab_arena.assets.ur5e_auto import auto_step

            self._held = set()
            auto_step(self, world_to_base)

        v = np.zeros(3)
        w = 0.0
        for name in self._held:
            ax, sg = _KEYS[name]
            if ax < 3:
                v[ax] += sg
            else:
                w += sg
        if v.any() or w:
            p_new = self._cp + world_to_base @ (v * np.array([xy, xy, lin]) * dt)
            p_new[2] = max(p_new[2], zmin)
            a = w * rot * dt
            Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
            R_new = Rz @ self._cR
            q, ep, eo = ik(self._cq, p_new, R_new)
            if ep < 0.005 and eo < 0.02:  # only accept reachable targets
                self._cq, self._cp, self._cR = q, p_new, R_new

        if hasattr(out, "clone"):
            import torch

            out = out.clone()
            flat = out.reshape(-1)
            flat[:6] = torch.tensor(self._cq, dtype=flat.dtype, device=flat.device)
        else:
            out = np.array(out, dtype=float, copy=True)
            flat = out.reshape(-1)
            flat[:6] = self._cq
        if getattr(self, "_open_val", None) is None:
            v0 = float(flat[-1])
            self._open_val = v0 if v0 != 0.0 else 1.0
        flat[-1] = -self._open_val if self._closed else self._open_val
        return out

    def reset(self, *args, **kwargs):
        if orig_reset is not None:
            orig_reset(self, *args, **kwargs)
        _cart_reset(self)

    cls.__init__ = __init__
    cls.advance = advance
    cls.reset = reset
    cls._cartesian_installed = True
    return cls


if __name__ == "__main__":
    # Offline self-test of the maths (no simulator needed)
    start = [float(x) for x in os.environ.get("UR5E_START", START_DEFAULT).split(",")]
    q = np.array(start)
    p0, R0 = fk(q)
    print("START grasp point in base frame: %s  (flange z=%.3f)" % (np.round(p0, 3), fk_frames(q)[6][2, 3]))
    print("START gripper pointing (tool z axis): %s" % np.round(R0[:, 2], 3))
    worst = 0.0
    tests = [("down 25cm", np.array([0, 0, -0.25]), 0.0), ("up 10cm", np.array([0, 0, 0.10]), 0.0),
             ("fwd 20cm", np.array([0.20, 0, 0]), 0.0), ("back 15cm", np.array([-0.15, 0, 0]), 0.0),
             ("left 25cm", np.array([0, 0.25, 0]), 0.0), ("right 25cm", np.array([0, -0.25, 0]), 0.0),
             ("rotate 90deg", np.zeros(3), math.pi / 2)]
    W2B = np.diag([-1.0, -1.0, 1.0])
    for label, d, ang in tests:
        qq, p, R = q.copy(), p0.copy(), R0.copy()
        n = 100
        ok = True
        for k in range(n):
            pn = p + W2B @ (d / n)
            a = ang / n
            Rn = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]) @ R
            qn, ep, eo = ik(qq, pn, Rn)
            if ep < 0.005 and eo < 0.02:
                qq, p, R = qn, pn, Rn
            else:
                ok = False
                break
        pf, Rf = fk(qq)
        moved = W2B.T @ (pf - p0)
        worst = max(worst, ep)
        print("TEST %-12s ok=%s moved(world)=%s pointing=%s joints=%s" % (
            label, ok, np.round(moved, 3), np.round(Rf[:, 2], 2), np.round(qq, 2)))
    print("DONE worst step error %.5f m" % worst)
