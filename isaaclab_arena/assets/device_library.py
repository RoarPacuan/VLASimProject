# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections.abc import Callable

from isaaclab.devices import DeviceBase, DeviceCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab_teleop import IsaacTeleopCfg, XrCfg

from isaaclab_arena.assets.register import register_device


class TeleopDeviceBase(ABC):

    name: str | None = None

    def __init__(self, sim_device: str | None = None):
        self.sim_device = sim_device

    @abstractmethod
    def get_device_cfg(self, pipeline_builder: Callable | None = None, embodiment: object | None = None):
        raise NotImplementedError


@register_device
class OpenXRCfg(TeleopDeviceBase):
    name = "openxr"

    def __init__(self, sim_device: str | None = None):
        super().__init__(sim_device=sim_device)

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> IsaacTeleopCfg:
        if pipeline_builder is None:
            raise ValueError("OpenXRCfg requires a pipeline_builder (got None)")
        xr_cfg = embodiment.get_xr_cfg() if embodiment is not None else XrCfg()
        target_frame_prim_path = embodiment.get_teleop_target_frame_prim_path()
        return IsaacTeleopCfg(
            pipeline_builder=pipeline_builder,
            sim_device=self.sim_device,
            xr_cfg=xr_cfg,
            target_frame_prim_path=target_frame_prim_path,
        )


@register_device
class KeyboardCfg(TeleopDeviceBase):
    name = "keyboard"

    def __init__(self, sim_device: str | None = None, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.05):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> Se3KeyboardCfg:
        return Se3KeyboardCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
        )


@register_device
class Ur5eKeyboardCfg(TeleopDeviceBase):
    """Keyboard device that emits absolute UR5e joint targets and an RG2 command."""

    name = "ur5e_keyboard"

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> DeviceCfg:
        return DeviceCfg(class_type=Ur5eKeyboard)


class Ur5eKeyboard(DeviceBase):
    """Incremental keyboard controller for the six UR5e joints and RG2 driver."""

    _INITIAL_JOINTS = tuple(float(x) for x in __import__("os").environ.get("UR5E_START", "-0.56,-2.29,2.25,-1.53,-1.57,-0.56").split(","))
    _JOINT_KEYS = {
        "A": (0, -1),
        "D": (0, 1),
        "W": (1, -1),
        "S": (1, 1),
        "Q": (2, -1),
        "E": (2, 1),
        "T": (3, 1),
        "G": (3, -1),
        "Y": (4, 1),
        "H": (4, -1),
        "U": (5, 1),
        "J": (5, -1),
    }
    _STEP = float(__import__("os").environ.get("UR5E_STEP", "0.2"))

    def __init__(self, cfg: DeviceCfg):
        import carb
        import omni

        super().__init__()
        self._sim_device = cfg.sim_device
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self._joint_deltas = [0.0] * 6
        self._close_gripper = False
        self._additional_callbacks = {}

    def __del__(self):
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)

    def reset(self):
        self._joint_deltas = [0.0] * 6
        self._close_gripper = False

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self):
        import torch

        command = [initial + delta for initial, delta in zip(self._INITIAL_JOINTS, self._joint_deltas)]
        command.append(-1.0 if self._close_gripper else 1.0)
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def _on_keyboard_event(self, event, *args, **kwargs):
        import carb

        if event.type == carb.input.KeyboardEventType.KEY_REPEAT and event.input.name in self._JOINT_KEYS:
            joint_index, direction = self._JOINT_KEYS[event.input.name]
            self._joint_deltas[joint_index] += direction * self._STEP
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "K":
                self._close_gripper = not self._close_gripper
            elif event.input.name == "L":
                self.reset()
            elif event.input.name in self._JOINT_KEYS:
                joint_index, direction = self._JOINT_KEYS[event.input.name]
                self._joint_deltas[joint_index] += direction * self._STEP
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()
        return True


@register_device
class SpaceMouseCfg(TeleopDeviceBase):
    name = "spacemouse"

    def __init__(self, sim_device: str | None = None, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.05):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> Se3SpaceMouseCfg:
        return Se3SpaceMouseCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
        )

# --- UR5e Cartesian (toolhead) teleop. Set UR5E_TELEOP=joint for the old per-joint keys ---
from isaaclab_arena.assets.ur5e_cartesian import install as _ur5e_cartesian_install

_ur5e_cartesian_install(Ur5eKeyboard)
