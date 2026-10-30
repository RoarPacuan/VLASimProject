# Pi0 and GR00T workflows

Arena runs both policies as closed-loop clients in the maintained DROID
pick-and-place Experiment. The policy model stays in its own inference server;
Isaac Sim runs in the Arena process.

Run the simulation launcher inside the Arena Docker container, not from the host
`.venv`, because Isaac Lab is provided by the container's `/isaac-sim/python.sh`.
From the repository root, start or attach to the container with `./docker/run_docker.sh`,
then run the commands below from `/workspaces/isaaclab_arena`.

The Arena container does not need the Docker CLI. Start the OpenPI server from a separate
host terminal, then pass `--external-server` to the simulation launcher inside Arena:

```bash
# Host terminal
./isaaclab_arena_openpi/docker/run_openpi_server.sh -v pi0 -p 8000

# Arena container terminal
./tools/run_vla_simulation.sh --policy pi0 --pi0-variant pi0 \
  --episodes 3 --viz kit --livestream 2 --external-server \
  --server-host 100.75.62.1
```

## Simulation

Run Pi0 (the `pi0` OpenPI checkpoint):

```bash
./tools/run_vla_simulation.sh --policy pi0 --episodes 3 --viz kit
```

The launcher defaults to Pi05. Select the original Pi0 checkpoint explicitly with
`--pi0-variant pi0`.

The local `ur5ewithgripper2.usd` is registered as `ur5e_rg2_abs_joint_pos` and is
intended for custom VLA checkpoints, not the stock DROID checkpoint. Start a custom
GR00T server from the host first:

```bash
cd submodules/Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /models/ur5e/gr00t/checkpoint-30000 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$OLDPWD/isaaclab_arena_gr00t/embodiments/ur5e/ur5e_manipulation_data_config.py" \
  --device cuda --host 0.0.0.0 --port 5555
```

Then evaluate it in Arena:

```bash
./tools/run_vla_simulation.sh \
  --policy gr00t \
  --experiment-config isaaclab_arena_environments/experiment_configs/ur5e_pnp_gr00t_experiment.yaml \
  --run-name ur5e_pnp_gr00t \
  --model-path /models/ur5e/gr00t/checkpoint-30000 \
  --server-host 100.75.62.1 \
  --port 5555 \
  --episodes 3 \
  --viz kit \
  --livestream 2 \
  --external-server
```

To expose the Isaac Sim viewport through WebRTC, use private/local-network mode:

```bash
./tools/run_vla_simulation.sh \
  --policy pi0 \
  --pi0-variant pi0 \
  --episodes 3 \
  --viz kit \
  --livestream 2
```

Isaac Sim's WebRTC signaling uses TCP port `49100`. The Arena Docker launcher uses host
networking, so that port is available directly on the host. Use `--livestream 1` only when
you specifically need the public-network WebRTC mode.

Run GR00T N1.6 DROID:

```bash
git submodule update --init submodules/Isaac-GR00T
./tools/run_vla_simulation.sh --policy gr00t --episodes 3 --viz kit
```

The script starts the matching local server, waits for its startup signal, runs
the Arena Experiment Runner, and stops only the server it started. Use
`--model-path /absolute/path/to/checkpoint` for a fine-tuned GR00T model. For a
custom Experiment YAML, pass both `--experiment-config` and its Run name with
`--run-name`.

For production or repeated evaluations, the lower-level two-terminal workflow
in `docs/pages/quickstart/running_a_real_policy/openpi.rst` and `gr00t.rst`
remains the canonical option.

## GR00T fine-tuning

Arena's HDF5 demonstrations must first be converted to GR00T-flavored LeRobot
v2 data. The repository's conversion configs and examples are documented in
`docs/pages/example_workflows/static_manipulation/step_4_policy_training.rst`.
Then run, from the host and from the GR00T checkout:

```bash
./tools/finetune_gr00t.sh \
  --dataset-path /datasets/arena/task/lerobot \
  --output-dir /models/arena/gr00t_task \
  --embodiment-tag GR1 \
  --modality-config-path "$PWD/isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py" \
  --max-steps 30000 --num-gpus 1
```

For eight GPUs, the wrapper automatically uses `torch.distributed.run`:

```bash
./tools/finetune_gr00t.sh \
  --dataset-path /datasets/arena/task/lerobot \
  --output-dir /models/arena/gr00t_task \
  --num-gpus 8 --global-batch-size 24
```

Use `NEW_EMBODIMENT` plus a custom modality config for a new controller or
action/state layout. After training, point `--model-path` in the simulation
launcher at the saved `checkpoint-*` directory.

For the local UR5E, use the supplied modality and joint-space definitions:

```bash
./tools/finetune_gr00t.sh \
  --dataset-path /datasets/ur5e/task/lerobot \
  --output-dir /models/ur5e/gr00t \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$PWD/isaaclab_arena_gr00t/embodiments/ur5e/ur5e_manipulation_data_config.py" \
  --max-steps 30000 --num-gpus 1
```

## Pi0 fine-tuning

Pi0 fine-tuning is not implemented in Arena. It must be run in an OpenPI
checkout with an OpenPI-supported dataset and training config. The wrapper
passes the current upstream training arguments through unchanged:

```bash
OPENPI_ROOT=/path/to/openpi ./tools/finetune_pi0.sh \
  --config <current-openpi-pi0-config> \
  --exp-name arena_pi0_task -- \
  <dataset-and-checkpoint-arguments-for-that-openpi-config>
```

After training, deploy the resulting checkpoint with OpenPI's server command
and update the Arena policy adapter/config if the checkpoint uses an
embodiment or action representation other than DROID absolute joint position.
The stock Pi05 DROID server cannot consume the UR5E 6-arm-joint-plus-RG2 action
space; Pi05 requires a matching OpenPI model/config and a UR5E adapter before it
can be evaluated on this embodiment.