# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert Arena HDF5 demonstration datasets to native LeRobot format.

The output uses the same ``LeRobotDataset.create`` and ``add_frame`` API as
the ``vr-teleop-kit`` data saver.  State and action joint ordering is
therefore intentionally taken from the HDF5 file without GR00T-specific
remapping; the configured ordering must match the real-robot dataset.

``hdf5_name`` may be a glob (e.g. ``lift_boxes/*.hdf5``) to merge several recording
runs. ``instruction_template`` builds a per-episode instruction from an observation
(e.g. ``pick up the {color} box`` from ``target_color_id``). ``extra_cameras`` adds
more camera streams next to ``pov_cam_name_sim``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

DEFAULT_COLOR_NAMES = ["red", "green", "blue", "yellow", "orange", "purple"]


@dataclass
class NativeLeRobotConfig:
    """Configuration for converting Arena HDF5 demonstrations."""

    data_root: Path
    hdf5_name: str
    language_instruction: str
    output_dir: Path | None = None
    state_name_sim: str = "robot_joint_pos"
    action_name_sim: str = "processed_actions"
    pov_cam_name_sim: str = "robot_head_cam"
    state_name_lerobot: str = "observation.state"
    action_name_lerobot: str = "action"
    video_name_lerobot: str = "observation.images.wrist"
    extra_cameras: dict[str, str] = field(default_factory=dict)
    instruction_template: str | None = None
    instruction_obs_key: str = "target_color_id"
    color_names: list[str] = field(default_factory=lambda: list(DEFAULT_COLOR_NAMES))
    fps: int = 50
    image_height: int = 480
    image_width: int = 640
    image_channels: int = 3
    repo_id: str = "local/arena_dataset"
    hdf5_file_path: Path = field(init=False)
    hdf5_file_paths: list[Path] = field(init=False)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        if any(ch in self.hdf5_name for ch in "*?["):
            self.hdf5_file_paths = sorted(self.data_root.glob(self.hdf5_name))
            parts = [p for p in Path(self.hdf5_name).with_suffix("").parts if not any(ch in p for ch in "*?[")]
            stem = "_".join(parts) or "merged"
        else:
            self.hdf5_file_paths = [self.data_root / self.hdf5_name]
            stem = Path(self.hdf5_name).stem
        assert self.hdf5_file_paths, f"no HDF5 files match {self.data_root / self.hdf5_name}"
        for path in self.hdf5_file_paths:
            assert path.exists(), f"{path} does not exist"
        self.hdf5_file_path = self.hdf5_file_paths[0]
        if self.output_dir is None:
            self.output_dir = self.data_root / f"{stem}_lerobot"
        else:
            self.output_dir = Path(self.output_dir)
        assert self.fps > 0, "fps must be positive"

    @property
    def cameras(self) -> dict[str, str]:
        """Sim camera name -> LeRobot video key, main camera first."""
        return {self.pov_cam_name_sim: self.video_name_lerobot, **self.extra_cameras}


def _feature_config(
    config: NativeLeRobotConfig, state_shape: tuple[int, ...], action_shape: tuple[int, ...]
) -> dict[str, dict[str, Any]]:
    """Return the feature declaration shared with ``vr-teleop-kit``."""
    features: dict[str, dict[str, Any]] = {
        config.state_name_lerobot: {"dtype": "float32", "shape": state_shape},
        config.action_name_lerobot: {"dtype": "float32", "shape": action_shape},
    }
    for video_key in config.cameras.values():
        features[video_key] = {
            "dtype": "video",
            "shape": (config.image_channels, config.image_height, config.image_width),
            "names": ["c", "h", "w"],
        }
    features["language_instruction"] = {"dtype": "string", "shape": (1,)}
    return features


def _episode_instruction(trajectory: h5py.Group, config: NativeLeRobotConfig) -> str:
    """Fixed instruction, or one built from an observation via ``instruction_template``."""
    if not config.instruction_template:
        return config.language_instruction
    key = config.instruction_obs_key
    assert key in trajectory["obs"], f"missing obs {key!r} needed by instruction_template"
    idx = int(np.asarray(trajectory["obs"][key][0]).reshape(-1)[0])
    return config.instruction_template.format(color=config.color_names[idx])


def _episode_frames(
    trajectory: h5py.Group, config: NativeLeRobotConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read one episode and remove Isaac Lab's terminal observation/action."""
    assert "obs" in trajectory, "episode is missing the obs group"
    assert config.state_name_sim in trajectory["obs"], f"missing state {config.state_name_sim!r}"
    assert config.action_name_sim in trajectory, f"missing action {config.action_name_sim!r}"
    assert "camera_obs" in trajectory, "episode is missing the camera_obs group"

    state = np.asarray(trajectory["obs"][config.state_name_sim][:-1], dtype=np.float32)
    action = np.asarray(trajectory[config.action_name_sim][:-1], dtype=np.float32)
    assert state.ndim == 2 and action.ndim == 2, "state and action must be two-dimensional"
    assert len(state) == len(action), "state and action lengths differ"
    assert len(state) > 0, "episode has no frames after removing the terminal sample"

    images: dict[str, np.ndarray] = {}
    for sim_name, video_key in config.cameras.items():
        assert sim_name in trajectory["camera_obs"], f"missing camera {sim_name!r}"
        frames = np.asarray(trajectory["camera_obs"][sim_name][:-1], dtype=np.uint8)
        assert frames.ndim == 4 and frames.shape[1:] == (
            config.image_height,
            config.image_width,
            config.image_channels,
        ), f"unexpected camera shape {frames.shape} for {sim_name!r}"
        assert len(frames) == len(state), f"camera {sim_name!r} length differs from state"
        images[video_key] = frames
    return state, action, images


def _demo_sort_key(name: str) -> tuple[int, Any]:
    tail = name.rsplit("_", 1)[-1]
    return (0, int(tail)) if tail.isdigit() else (1, name)


def convert_hdf5_to_native_lerobot(config: NativeLeRobotConfig) -> Path:
    """Convert Arena HDF5 episodes (one or more files) to a native LeRobot dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig

    assert config.output_dir is not None
    dataset = None
    first_state_shape = first_action_shape = None
    counts: dict[str, int] = {}
    for path in config.hdf5_file_paths:
        with h5py.File(path, "r") as hdf5_file:
            assert "data" in hdf5_file, f"{path} is missing the data group"
            episode_ids = sorted(hdf5_file["data"].keys(), key=_demo_sort_key)
            print(f"{path}: {len(episode_ids)} episodes")
            for episode_id in episode_ids:
                trajectory = hdf5_file["data"][episode_id]
                state, action, images = _episode_frames(trajectory, config)
                instruction = _episode_instruction(trajectory, config)
                if dataset is None:
                    first_state_shape, first_action_shape = state.shape[1:], action.shape[1:]
                    dataset = LeRobotDataset.create(
                        repo_id=config.repo_id,
                        fps=config.fps,
                        features=_feature_config(config, first_state_shape, first_action_shape),
                        root=config.output_dir,
                        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
                    )
                assert state.shape[1:] == first_state_shape, "state shape differs between episodes"
                assert action.shape[1:] == first_action_shape, "action shape differs between episodes"
                for i in range(len(state)):
                    frame = {
                        config.state_name_lerobot: state[i],
                        config.action_name_lerobot: action[i],
                        "language_instruction": instruction,
                        "task": instruction,
                    }
                    for video_key, frames in images.items():
                        frame[video_key] = frames[i]
                    dataset.add_frame(frame)
                dataset.save_episode()
                counts[instruction] = counts.get(instruction, 0) + 1
                print(f"  {episode_id}: {len(state)} frames, '{instruction}'")
    assert dataset is not None, "no episodes found"
    dataset.finalize()
    print("Episodes per instruction:", counts)
    return config.output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml_file", type=Path, required=True, help="Path to a NativeLeRobotConfig YAML file.")
    args = parser.parse_args()
    with args.yaml_file.open(encoding="utf-8") as yaml_file:
        config = NativeLeRobotConfig(**yaml.safe_load(yaml_file))
    output_dir = convert_hdf5_to_native_lerobot(config)
    print(f"Native LeRobot dataset written to {output_dir}")


if __name__ == "__main__":
    main()
