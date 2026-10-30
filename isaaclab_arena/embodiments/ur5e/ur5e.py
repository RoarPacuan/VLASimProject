# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.managers import ActionTermCfg, ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.gripper import RobotiqGripper
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose


_UR5E_USD_PATH = str(Path(__file__).resolve().parents[3] / "ur5ewithcam2_rg2_mimic.usda")
_UR5E_ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
_RG2_DRIVER_JOINT = "finger_joint"
_START = [float(x) for x in __import__("os").environ.get("UR5E_START", "-0.56,-2.29,2.25,-1.53,-1.57,-0.56").split(",")]


@configclass
class Ur5eSceneCfg:
    """Scene additions for the local UR5E plus OnRobot RG2 USD."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        articulation_root_prim_path="/ur5e/root_joint",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_UR5E_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(x) for x in __import__("os").environ.get("UR5E_POS", "0.3,0.0,0.003").split(",")),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan_joint": _START[0],
                "shoulder_lift_joint": _START[1],
                "elbow_joint": _START[2],
                "wrist_1_joint": _START[3],
                "wrist_2_joint": _START[4],
                "wrist_3_joint": _START[5],
                _RG2_DRIVER_JOINT: 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(_UR5E_ARM_JOINTS),
                effort_limit=150.0,
                velocity_limit=3.14,
                stiffness=800.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit=20.0, velocity_limit=2.0, stiffness=200.0, damping=10.0),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["left_inner_knuckle_joint", "left_inner_finger_joint",
                              "right_outer_knuckle_joint", "right_inner_knuckle_joint", "right_inner_finger_joint"],
            effort_limit=20.0, velocity_limit=2.0, stiffness=0.0, damping=0.0),
        },
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ur5e/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/org2/onrobot_rg2_base_link",
                name="end_effector",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/org2/left_inner_finger",
                name="tool_leftfinger",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/org2/right_inner_finger",
                name="tool_rightfinger",
            ),
        ],
    )


@configclass
class Ur5eCameraCfg(ArenaCameraCfg):
    """Camera observations mounted on the local UR5E asset."""

    external_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_camera",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=1.0),
        offset=CameraCfg.OffsetCfg(pos=(1.2, -0.6, 0.7), rot=(-0.2991, 0.1269, 0.8707, 0.3693), convention="world"),
    )
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/org2/onrobot_rg2_base_link/wrist_camera",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, focus_distance=0.3),
        offset=CameraCfg.OffsetCfg(pos=(0.08, -0.0006, 0.05), rot=(0.6969, 0.6913, 0.1345, 0.1355), convention="opengl"),
    )


@configclass
class Ur5eObservationsCfg:
    """Policy state terms used by custom UR5E VLA adapters."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        robot_joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        gripper_pos = ObsTerm(
            func=mdp_isaac_lab.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class Ur5eAbsoluteJointPositionActionsCfg:
    """Absolute joint-position actions: six arm joints plus RG2 finger driver."""

    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_UR5E_ARM_JOINTS),
        preserve_order=True,
        use_default_offset=False,
    )
    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.55},
    )


@register_asset
class Ur5eRg2AbsoluteJointPositionEmbodiment(EmbodimentBase):
    """Local UR5E + OnRobot RG2 embodiment for custom VLA training."""

    name = "ur5e_rg2_abs_joint_pos"
    tags = ["embodiment"]
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
        )
        self.scene_config = Ur5eSceneCfg()
        self.camera_config = Ur5eCameraCfg()
        self.observation_config = Ur5eObservationsCfg()
        self.action_config = Ur5eAbsoluteJointPositionActionsCfg()
        self.gripper = RobotiqGripper(
            driver_joint_name="finger_joint",
            frame_transformer_name="ee_frame",
            target_frame_name="end_effector",
            left_finger_frame_name="tool_leftfinger",
            right_finger_frame_name="tool_rightfinger",
        )

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"