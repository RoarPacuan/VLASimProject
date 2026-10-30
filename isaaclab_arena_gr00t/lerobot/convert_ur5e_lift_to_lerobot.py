"""Convert UR5e lift-task HDF5 demos (Isaac Lab Arena, record_demos) to native LeRobot v3,
matching the real UR5 teleop dataset (UR5_data_merged, vr-teleop-kit, lerobot 0.6.1):

  observation.state         float32 [7]  6 absolute arm joints (rad) + gripper in [0, 1], 0 = closed
  action                    float32 [7]  6 absolute joint targets + gripper command, 1 = open, 0 = closed
  observation.images.wrist  video 3x480x640, h264, 50 fps
  language_instruction/task "Pick up the {color} object"

Gripper: the sim finger_joint q goes from 0 (open) to 0.55 rad (closed). State is
GRIPPER_STATE_OPEN * (1 - q / 0.55); the real robot reads about 0.905 when fully open. The sim
gripper command is binary: raw action >= 0 is open (1.0), < 0 is closed (0.0).

Run with the lerobot of the real dataset (teleop-data venv) plus h5py, e.g.
  PYTHONPATH=/tmp/h5py_pkg python convert_ur5e_lift_to_lerobot.py \
      --input "/home/loever/datasets/lift_boxes/*.hdf5" --output /mnt/big/teleop_data/UR5_sim_lift_boxes
"""

from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

COLOR_NAMES = ["red", "green", "blue", "yellow", "orange", "purple", "pink"]  # same order as ur5e_lift.py
FINGER_CLOSED = 0.55
H, W = 480, 640
START_POSE = np.array([-0.147, -1.788, 1.17, -0.996, -1.683, -3.222], dtype=np.float32)  # real mean start


def features():
    return {
        "observation.state": {"dtype": "float32", "shape": (7,)},
        "action": {"dtype": "float32", "shape": (7,)},
        "observation.images.wrist": {"dtype": "video", "shape": (3, H, W), "names": ["c", "h", "w"]},
        "language_instruction": {"dtype": "string", "shape": (1,)},
    }


def read_episode(g, gripper_open, trim_tol=0.05):
    obs = g["obs"]
    q = np.asarray(obs["robot_joint_pos"], dtype=np.float32)
    finger = np.asarray(obs["gripper_pos"], dtype=np.float32).reshape(-1) if "gripper_pos" in obs else q[:, 6]
    g_state = np.clip(1.0 - finger / FINGER_CLOSED, 0.0, 1.0) * gripper_open
    state = np.concatenate([q[:, :6], g_state[:, None]], axis=1).astype(np.float32)
    a = np.asarray(g["actions"], dtype=np.float32)
    a_g = (a[:, 6] >= 0.0).astype(np.float32)
    action = np.concatenate([a[:, :6], a_g[:, None]], axis=1).astype(np.float32)
    frames = np.asarray(g["camera_obs"]["wrist_camera_rgb"], dtype=np.uint8)
    assert frames.shape[1:] == (H, W, 3), f"unexpected camera shape {frames.shape}"
    cid = int(np.asarray(obs["target_color_id"][0]).reshape(-1)[0])
    n = min(len(state), len(action), len(frames)) - 1  # drop Isaac Lab's terminal sample
    assert n > 0, "empty episode"
    # The sim resets the arm away from the start pose and the expert drives it there in the first
    # frames. The real episodes start at the start pose, so drop frames until the arm is there.
    k = 0
    if trim_tol > 0:
        near = np.abs(state[:n, :6] - START_POSE).max(axis=1) < trim_tol
        k = int(np.argmax(near)) if near.any() else 0
    return state[k:n], action[k:n], frames[k:n], f"Pick up the {COLOR_NAMES[cid]} object", k


def demo_key(k):
    tail = k.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="/home/loever/datasets/lift_boxes/*.hdf5")
    ap.add_argument("--output", required=True)
    ap.add_argument("--repo-id", default="local/ur5_sim_lift_boxes")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--gripper-state-open", type=float, default=0.905)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--trim-start-tol", type=float, default=0.05,
                    help="drop leading frames until all arm joints are within this many rad of the start pose (0 = off)")
    args = ap.parse_args()

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.output)
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists; use --overwrite")
        shutil.rmtree(out)
    files = sorted(glob.glob(args.input))
    if not files:
        sys.exit(f"no files match {args.input}")

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=features(), root=out,
                               rgb_encoder=RGBEncoderConfig(vcodec="h264"))
    counts, n_ep, n_fr = {}, 0, 0
    for p in files:
        try:
            f = h5py.File(p, "r")
        except Exception as e:
            print("SKIP (unreadable)", p, type(e).__name__, flush=True)
            continue
        with f:
            demos = sorted(f["data"].keys(), key=demo_key) if "data" in f else []
            print(f"{p}: {len(demos)} demos", flush=True)
            for k in demos:
                state, action, frames, text, cut = read_episode(f["data"][k], args.gripper_state_open, args.trim_start_tol)
                for i in range(len(state)):
                    ds.add_frame({
                        "observation.state": state[i],
                        "action": action[i],
                        "observation.images.wrist": frames[i],
                        "language_instruction": text,
                        "task": text,
                    })
                ds.save_episode()
                n_ep += 1
                n_fr += len(state)
                counts[text] = counts.get(text, 0) + 1
                print(f"  {k}: {len(state)} frames (cut {cut} at start), '{text}', gripper closed in "
                      f"{100 * (1 - action[:, 6].mean()):.0f}% of frames", flush=True)
                if args.max_episodes and n_ep >= args.max_episodes:
                    break
        if args.max_episodes and n_ep >= args.max_episodes:
            break
    ds.finalize()
    print("EPISODES", n_ep, "FRAMES", n_fr)
    print("PER INSTRUCTION", counts)
    print("WRITTEN", out)


if __name__ == "__main__":
    main()
