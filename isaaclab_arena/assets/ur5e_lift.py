"""UR5e lift task: two boxes with random colours; a random one is the target on every reset.

Enable with UR5E_TASK=lift. On every reset the two boxes get two different colours from the palette
(restrict with UR5E_COLORS=red,green,...) and one box is chosen as target. The target is printed to
the terminal and stored per step as policy observations:
  target_object_id  0 = pick_up_object, 1 = destination_location
  target_color_id   index into COLOR_NAMES
  box_color_ids     colour index of each box, in object order
"""

import os
from functools import partial

import torch

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import EventTermCfg, ObservationTermCfg, SceneEntityCfg, TerminationTermCfg

from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.tasks.predicates.object_settling import objects_settled
from isaaclab_arena.tasks.predicates.spatial import object_is_above_height
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

LIFT_HEIGHT = float(os.environ.get("UR5E_LIFT_HEIGHT", "0.10"))  # m above resting height for success
WRONG_HEIGHT = 0.03  # lifting the other box this far ends the episode as a failure
HOLD_STEPS = int(os.environ.get("UR5E_HOLD_STEPS", "50"))  # control steps (1/50 s each at 50 Hz) it must stay up
NUM_BOXES = 2
MAX_EPISODE_S = float(os.environ.get("UR5E_MAX_EPISODE_S", "30"))  # longer episodes are discarded

COLORS = {
    "red": (0.80, 0.05, 0.05),
    "green": (0.05, 0.55, 0.10),
    "blue": (0.05, 0.20, 0.85),
    "yellow": (0.95, 0.80, 0.05),
    "orange": (0.95, 0.40, 0.02),
    "purple": (0.45, 0.10, 0.65),
    "pink": (0.95, 0.40, 0.70),
}
COLOR_NAMES = list(COLORS)


def _palette():
    raw = os.environ.get("UR5E_COLORS", ",".join(COLOR_NAMES))
    names = [n.strip() for n in raw.split(",") if n.strip() in COLORS]
    return [COLOR_NAMES.index(n) for n in names] or list(range(len(COLOR_NAMES)))


def _state(env):
    if getattr(env, "_ur5e_target", None) is None:
        env._ur5e_target = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._ur5e_colors = torch.zeros(env.num_envs, NUM_BOXES, dtype=torch.long, device=env.device)
    return env._ur5e_target, env._ur5e_colors


def _ids(env, env_ids, device):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=device)
    if not isinstance(env_ids, torch.Tensor):
        try:
            import warp as wp

            env_ids = wp.to_torch(env_ids)
        except Exception:
            env_ids = torch.as_tensor(env_ids)
    return env_ids.to(device=device, dtype=torch.long)


def _color_input(env, e, name):
    """Bind our own UsdPreviewSurface material to the box once; return its diffuseColor input."""
    cache = getattr(env, "_ur5e_color_inputs", None)
    if cache is None:
        cache = {}
        env._ur5e_color_inputs = cache
    if (e, name) in cache:
        return cache[(e, name)]
    inp = None
    try:
        import omni.usd
        from pxr import Sdf, UsdShade

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/{name}")
        if prim.IsValid():
            mat_path = f"/World/Looks/ur5e_box_color_{name}_env{e}"
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            inp = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
        else:
            print(f"[UR5e] colour: prim for {name} not found", flush=True)
    except Exception as err:
        print(f"[UR5e] colour setup failed for {name}: {err}", flush=True)
    cache[(e, name)] = inp
    return inp


def setup_box_materials(env, env_ids, object_names):
    """Bind colour + physics materials before physics starts (prestartup).

    Binding them later, at the first reset, makes PhysX rebuild the boxes mid-run, and in the
    first episode the collision no longer matches what is rendered.
    """
    for e in range(env.num_envs):
        for name in object_names:
            _color_input(env, e, name)
    print("[UR5e] box materials bound before physics start", flush=True)


def choose_target(env, env_ids, object_names):
    t, c = _state(env)
    ids = _ids(env, env_ids, t.device)
    env._ur5e_names = list(object_names)
    try:
        from isaaclab_arena.assets import ur5e_auto

        ur5e_auto.ENV_REF[:] = [env]
    except Exception as err:
        print(f"[UR5e] auto driver not available: {err}", flush=True)
    if ids.numel() == 0:
        return
    pal = _palette()
    k = len(object_names)
    for e in ids.tolist():
        pick = torch.randperm(len(pal))[:k].tolist()
        for j, name in enumerate(object_names):
            cid = pal[pick[j]] if j < len(pick) else pal[0]
            c[e, j] = cid
            inp = _color_input(env, e, name)
            if inp is not None:
                from pxr import Gf

                inp.Set(Gf.Vec3f(*COLORS[COLOR_NAMES[cid]]))
        t[e] = int(torch.randint(0, k, (1,)).item())
    e = int(ids[0])
    tgt = COLOR_NAMES[int(c[e, int(t[e])])]
    other = [COLOR_NAMES[int(c[e, j])] for j in range(k) if j != int(t[e])]
    print(f"\n[UR5e] ===> TARGET: pick up the {tgt.upper()} BOX <===   (other: {', '.join(other)})\n", flush=True)


def _lifted(env, object_names, distance):
    cols = [object_is_above_height(env, object_name=n, use_settled_state=True, distance=distance) for n in object_names]
    return torch.stack([col.to(torch.bool) for col in cols], dim=1)


def target_lifted(env, object_names, distance=LIFT_HEIGHT):
    lifted = _lifted(env, object_names, distance)
    t = _state(env)[0].to(lifted.device)
    return lifted.gather(1, t.unsqueeze(1)).squeeze(1)


def other_lifted(env, object_names, distance=WRONG_HEIGHT):
    lifted = _lifted(env, object_names, distance)
    t = _state(env)[0].to(lifted.device)
    mask = torch.ones_like(lifted)
    mask.scatter_(1, t.unsqueeze(1), False)
    return (lifted & mask).any(dim=1)


def episode_too_long(env, max_s=MAX_EPISODE_S):
    """Discard episodes longer than max_s. record_demos removes the normal time_out, and at 50 Hz
    with two cameras a long episode keeps growing in RAM until the process is killed."""
    done = env.episode_length_buf * env.step_dt >= max_s
    if getattr(env, "_ur5e_abort", False):
        env._ur5e_abort = False
        print("\n[UR5e] auto driver gave up -> episode discarded and reset\n", flush=True)
        return torch.ones_like(done, dtype=torch.bool)
    if bool(done.any()):
        print(f"\n[UR5e] episode longer than {max_s:.0f} s -> discarded and reset\n", flush=True)
    return done


def target_obs(env):
    return _state(env)[0].to(torch.float32).unsqueeze(1)


def target_color_obs(env):
    t, c = _state(env)
    return c.gather(1, t.unsqueeze(1)).to(torch.float32)


def box_colors_obs(env):
    return _state(env)[1].to(torch.float32)


class Ur5eLiftTask(PickAndPlaceTask):
    """Lift whichever of the two boxes was chosen as target at reset."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("task_description", "Pick up the box with the requested colour")
        super().__init__(*args, **kwargs)
        self.object_names = [self.pick_up_object.name, self.destination_location.name]

    def get_termination_cfg(self) -> TaskTerminationCfg:
        names = self.object_names
        dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={"minimum_height": self.background_scene.object_min_z, "asset_cfg": SceneEntityCfg(names[0])},
        )
        wrong = TerminationTermCfg(func=other_lifted, params={"object_names": names})
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="lift_target",
                    predicate_sequence=[
                        partial(objects_settled, object_names=names),
                        TrueForConsecutiveStepsCfg(
                            predicate=partial(target_lifted, object_names=names),
                            required_steps=HOLD_STEPS,
                        ),
                    ],
                ),
            ],
            failures={"object_dropped": dropped, "wrong_box_lifted": wrong, "too_long": TerminationTermCfg(func=episode_too_long)},
        )


def add_target_choice(env_cfg, object_names):
    names = list(object_names)
    env_cfg.events.ur5e_box_materials = EventTermCfg(func=setup_box_materials, mode="prestartup", params={"object_names": names})
    env_cfg.events.ur5e_target_choice = EventTermCfg(func=choose_target, mode="reset", params={"object_names": names})
    env_cfg.observations.policy.target_object_id = ObservationTermCfg(func=target_obs)
    env_cfg.observations.policy.target_color_id = ObservationTermCfg(func=target_color_obs)
    env_cfg.observations.policy.box_color_ids = ObservationTermCfg(func=box_colors_obs)
    # Control/recording rate: 50 Hz = 200 Hz physics / decimation 4; one camera render per control step.
    control_hz = float(os.environ.get("UR5E_CONTROL_HZ", "50"))
    physics_hz = float(os.environ.get("UR5E_PHYSICS_HZ", "200"))
    env_cfg.decimation = max(1, round(physics_hz / control_hz))
    env_cfg.sim.dt = 1.0 / physics_hz
    env_cfg.sim.render_interval = env_cfg.decimation
    print(f"[UR5e] control {physics_hz / env_cfg.decimation:.1f} Hz (physics {physics_hz:.0f} Hz, decimation {env_cfg.decimation})", flush=True)
    if getattr(env_cfg, "num_rerenders_on_reset", 0) < 1:
        env_cfg.num_rerenders_on_reset = 1  # camera frames after reset show the new colours
    seed = os.environ.get("UR5E_SEED")
    env_cfg.seed = int(seed) if seed else int.from_bytes(os.urandom(4), "little") % (2**31)
    print(f"[UR5e] env seed {env_cfg.seed} (set UR5E_SEED=<n> to repeat a run)", flush=True)
    pal = [COLOR_NAMES[i] for i in _palette()]
    print(f"[UR5e] lift task: random target from {names}, colours {pal}, lift {LIFT_HEIGHT} m, hold {HOLD_STEPS} steps", flush=True)
