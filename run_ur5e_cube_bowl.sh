#!/bin/bash
# Record UR5e + RG2 keyboard demos in the maple-table task.
# Usage: ./run_ur5e_record.sh                              (start pose from code)
#        ./run_ur5e_record.sh 0 -1.57 1.57 -1.57 -1.57 0   (custom start pose, radians)
# Step size per keypress: UR5E_STEP=0.1 ./run_ur5e_record.sh
# Robot position (x,y,z m): UR5E_POS=0.3,0.0,0.003 ./run_ur5e_record.sh
cd /workspaces/isaaclab_arena
if [ $# -eq 6 ]; then export UR5E_START="$1,$2,$3,$4,$5,$6"; fi
echo "Start pose: ${UR5E_START:-code default}   Step: ${UR5E_STEP:-code default}   Pos: ${UR5E_POS:-code default}"
export CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export WEBRTC_LIB=/isaac-sim/extscache/omni.kit.livestream.webrtc-10.7.0+110.1.2.lx64.r.cp312/bin
export TELEOP_LIB=/isaac-sim/kit/python/lib/python3.12/site-packages/isaacteleop/cloudxr/native
export CUDA_LIB=/isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH="$WEBRTC_LIB:$TELEOP_LIB:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p /datasets
/isaac-sim/python.sh submodules/IsaacLab/scripts/tools/record_demos.py \
  --device cpu --viz kit --livestream 2 \
  --kit_args "--enable omni.kit.livestream.webrtc --/app/livestream/publicEndpointAddress=100.75.62.1 --/app/livestream/port=49100" \
  --task pick_and_place_maple_table --embodiment ur5e_rg2_abs_joint_pos \
  --arena_teleop_device ur5e_keyboard --teleop_device ur5e_keyboard \
  --step_hz 50 --num_demos 5 --num_success_steps 2 \
  --enable_cameras --dataset_file /datasets/ur5e_rg2_dataset_recorded.hdf5 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  2>&1 | tee run.log
