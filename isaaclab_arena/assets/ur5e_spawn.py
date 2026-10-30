"""Spawn region for the UR5e maple-table task: keep the cube and the bowl where the arm can work.

Runs AFTER Arena's own placement_reset and moves the objects into a rectangle in front of the robot,
a minimum distance apart, each with a random yaw.

Environment variables (metres, env frame, robot base at x=0.3, y=0):
    UR5E_SPAWN_X          x range, default "0.52,0.82"
    UR5E_SPAWN_Y          y range, default "-0.30,0.30"
    UR5E_SPAWN_MIN_DIST   minimum distance between object centres, default 0.18
    UR5E_SPAWN=off        keep Arena's placement
"""

import math
import os

import torch


def _range(name, default):
    a, b = (float(v) for v in os.environ.get(name, default).split(","))
    return (min(a, b), max(a, b))


def sample_positions(n, k, x_range, y_range, min_dist, max_tries=200):
    lo = torch.tensor([x_range[0], y_range[0]])
    hi = torch.tensor([x_range[1], y_range[1]])
    xy = torch.zeros(n, k, 2)
    for i in range(k):
        cand = lo + (hi - lo) * torch.rand(n, 2)
        if i > 0:
            for _ in range(max_tries):
                d = torch.linalg.norm(xy[:, :i] - cand[:, None], dim=-1).min(dim=1).values
                bad = d < min_dist
                if not bad.any():
                    break
                cand[bad] = lo + (hi - lo) * torch.rand(int(bad.sum()), 2)
        xy[:, i] = cand
    return xy


def _default_xy(env, name):
    asset = env.scene[name]
    try:
        d = asset.data.default_root_state
        if not isinstance(d, torch.Tensor):
            import warp as wp

            d = wp.to_torch(d)
        return d[:, 0:2].detach().cpu()
    except Exception:
        p = asset.cfg.init_state.pos
        return torch.tensor([[float(p[0]), float(p[1])]]).repeat(env.num_envs, 1)


_printed = {"done": False}


def _put(env, name, e, x, y, yaw):
    import warp as wp

    asset = env.scene[name]
    cur = asset.data.root_pos_w
    cur = cur if isinstance(cur, torch.Tensor) else wp.to_torch(cur)
    org = env.scene.env_origins
    org = org if isinstance(org, torch.Tensor) else wp.to_torch(org)
    pose = torch.tensor([[float(org[e, 0]) + x, float(org[e, 1]) + y, float(cur[e, 2]), 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]], device=cur.device)
    ids = torch.tensor([e], device=cur.device, dtype=torch.int32)
    err = None
    for name_fn in ("write_root_pose_to_sim", "write_root_pose_to_sim_index"):
        fn = getattr(asset, name_fn, None)
        if fn is None:
            continue
        for conv in (lambda q: q, wp.from_torch, lambda q: wp.from_torch(q, dtype=wp.transformf)):
            try:
                fn(conv(pose), env_ids=ids)
                return
            except Exception as ex:
                err = ex
    raise RuntimeError("could not write pose: %r" % (err,))


def reset_objects_in_region(env, env_ids, asset_names, x_range, y_range, min_dist):
    from isaaclab.envs import mdp
    from isaaclab.managers import SceneEntityCfg

    if env_ids is None:
        ids = list(range(env.num_envs))
    elif isinstance(env_ids, torch.Tensor):
        ids = env_ids.detach().cpu().tolist()
    else:
        ids = list(env_ids)
    if not ids:
        return
    xy = sample_positions(len(ids), len(asset_names), x_range, y_range, min_dist)
    for k, name in enumerate(asset_names):
        dflt = _default_xy(env, name)
        for j, e in enumerate(ids):
            dx = float(xy[j, k, 0] - dflt[e, 0])
            dy = float(xy[j, k, 1] - dflt[e, 1])
            yaw = (2.0 * float(torch.rand(1)) - 1.0) * math.pi
            _put(env, name, e, float(xy[j, k, 0]), float(xy[j, k, 1]), yaw)
    if not _printed["done"]:
        _printed["done"] = True
        where = ", ".join("%s at (%.2f, %.2f)" % (n, xy[0, k, 0], xy[0, k, 1]) for k, n in enumerate(asset_names))
        print("[UR5e] spawn region x=%s y=%s min_dist=%.2f: %s" % (x_range, y_range, min_dist, where))


def add_spawn_region(env_cfg, asset_names):
    if os.environ.get("UR5E_SPAWN", "on") == "off":
        return env_cfg
    from isaaclab.managers import EventTermCfg

    params = {
        "asset_names": list(asset_names),
        "x_range": _range("UR5E_SPAWN_X", "0.52,0.82"),
        "y_range": _range("UR5E_SPAWN_Y", "-0.30,0.30"),
        "min_dist": float(os.environ.get("UR5E_SPAWN_MIN_DIST", "0.18")),
    }
    setattr(env_cfg.events, "ur5e_spawn_region", EventTermCfg(func=reset_objects_in_region, mode="reset", params=params))
    return env_cfg
