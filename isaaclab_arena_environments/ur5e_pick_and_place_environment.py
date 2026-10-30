# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class Ur5ePickAndPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the local UR5e/RG2 pick-and-place environment."""

    embodiment: str = "ur5e_rg2_abs_joint_pos"
    pick_up_object: str = "rubiks_cube_hot3d_robolab"
    destination_location: str = "bowl_ycb_robolab"
    hdr: str | None = "home_office_robolab"
    light_intensity: float = 500.0
    additional_table_objects: list[str] = field(default_factory=list)
    episode_length_s: float = 70.0


@register_environment
class Ur5ePickAndPlaceEnvironment(ArenaEnvironmentFactory[Ur5ePickAndPlaceEnvironmentCfg]):
    """Registered Arena environment for keyboard-controlled UR5e demonstrations."""

    name = "ur5e_pick_and_place"
    _legacy_argparse_cfg_type = Ur5ePickAndPlaceEnvironmentCfg

    def build(self, cfg: Ur5ePickAndPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the UR5e/RG2 pick-and-place environment."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.pick_up_object)()
        destination_location = self.asset_registry.get_asset_by_name(cfg.destination_location)()
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())
        pick_up_object.add_relation(On(table_reference))
        destination_location.add_relation(On(table_reference))

        additional_table_objects = [
            self.asset_registry.get_asset_by_name(name)() for name in cfg.additional_table_objects
        ]
        for obj in additional_table_objects:
            obj.add_relation(On(table_reference))

        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
        )
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(0.0, 0.0, 0.35),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        scene = Scene(
            assets=[
                background,
                light,
                directional_light,
                pick_up_object,
                destination_location,
                table_reference,
                *additional_table_objects,
            ]
        )
        task = PickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background,
            episode_length_s=cfg.episode_length_s,
        )

        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=_set_viewer_cfg,
        )

    @staticmethod
    def _add_legacy_cli_only_args(parser) -> None:
        parser.add_argument("--teleop_device", type=str, default=None)
