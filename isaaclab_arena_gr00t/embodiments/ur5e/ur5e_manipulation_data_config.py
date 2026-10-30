# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T modality contract for the local UR5E/RG2 embodiment."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig


ur5e_manipulation_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["external_camera_rgb", "wrist_camera_rgb"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_position", "gripper_position"],
        sin_cos_embedding_keys=["joint_position"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["joint_position", "gripper_position"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(ur5e_manipulation_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)