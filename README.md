# VLASimProject

UR5e + OnRobot RG2 simulation setup for our VLA project at KU, built on NVIDIA IsaacLab-Arena.
This repo only contains our own files and the IsaacLab-Arena files we changed, at the same paths as upstream.

- Upstream: https://github.com/isaac-sim/IsaacLab-Arena
- Base commit: 8619fd7fd365436330694876800bf734e43cbe7e

To use it: check out IsaacLab-Arena at the base commit, copy these files on top,
and run git apply isaaclab_changes.patch inside submodules/IsaacLab.
