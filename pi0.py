import argparse
import torch

# 1. Launch the app before importing Isaac Lab modules
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run Pi0 with a custom arm and OnRobot RG2.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Isaac Lab imports
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from omni.isaac.lab.managers import ObservationGroupCfg, ObservationTermCfg, EventTermCfg
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.scene import InteractiveSceneCfg

# 3. OpenPI / Arena imports
# (Ensure you are running this inside the IsaacLab-Arena container)
from openpi.policies.policy_config import PolicyConfig
from openpi.policies.openpi_policy import OpenPIPolicy


# --- CONFIGURATION ---

class CustomRobotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="YOUR_CUSTOM_ARM_AND_RG2.usd", # <--- UPDATE THIS
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos={".*": 0.0}, 
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["arm_joint_.*"], # <--- UPDATE THIS
                stiffness=400.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["left_finger_joint", "right_finger_joint"], # <--- UPDATE THIS
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )

    wrist_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/hand_base_link/wrist_camera", # <--- UPDATE THIS
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    scene_camera = CameraCfg(
        prim_path="/World/envs/env_.*/scene_camera",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 1.0), rot=(0.707, 0.0, 0.707, 0.0)),
    )


class CustomRobotActionsCfg:
    arm_action = mdp.DifferentialIKActionCfg(
        asset_name="robot",
        joint_names=["arm_joint_.*"], 
        body_name="hand_base_link",   
        controller=mdp.DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
    )
    
    gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_finger_joint", "right_finger_joint"], 
        scale=[1.0, -1.0], 
        offset=[0.0, 0.0],
        use_default_offset=False,
    )


# Isaac Lab requires Observations and Events defined in the config
class CustomObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # We manually extract images and proprio in the loop, but IsaacLab needs at least a dummy config
        # to initialize the ObservationManager without crashing.
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_name": "robot"})
    
    policy: PolicyCfg = PolicyCfg()


class CustomEventCfg:
    # Handles environment resets
    reset_robot = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_name": "robot",
            "position_range": (0.0, 0.0), # Resets to default init_state
            "velocity_range": (0.0, 0.0),
        },
    )


class Pi0EnvCfg(ManagerBasedRLEnvCfg):
    scene: CustomRobotSceneCfg = CustomRobotSceneCfg(num_envs=1, env_spacing=2.5)
    actions: CustomRobotActionsCfg = CustomRobotActionsCfg()
    observations: CustomObservationsCfg = CustomObservationsCfg()
    events: CustomEventCfg = CustomEventCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.decimation = 2 


# --- EXECUTION LOOP ---

def main():
    env_cfg = Pi0EnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    
    print("[INFO] Loading Pi0 Policy...")
    policy_cfg = PolicyConfig(model_name="pi0", device=device)
    pi0_policy = OpenPIPolicy(policy_cfg)
    
    language_instruction = "pick up the object" 
    
    obs, _ = env.reset()
    print(f"[INFO] Executing task: '{language_instruction}'")
    
    while simulation_app.is_running():
        # 1. Gather sensors to tensors
        # Arena models usually expect tensors normalized [0, 1] mapped to BxCxHxW
        img_primary = env.scene.sensors["scene_camera"].data.output["rgb"]
        img_wrist = env.scene.sensors["wrist_camera"].data.output["rgb"]
        proprio = env.scene.articulations["robot"].data.joint_pos
        
        pi0_obs = {
            "image_primary": img_primary.to(device),
            "image_wrist": img_wrist.to(device),
            "proprio": proprio.to(device),
            "instruction": language_instruction,
        }
        
        # 2. Query Policy (Action Chunking)
        with torch.no_grad():
            # Expected shape: [batch, chunk_size, action_dim]
            action_chunk = pi0_policy.compute_action(pi0_obs)
        
        chunk_size = action_chunk.shape[1]
        
        # 3. Execute the chunk sequentially in the physics step
        for i in range(chunk_size):
            # Extract the single action step from the chunk -> shape: [batch, action_dim]
            current_action = action_chunk[:, i, :]
            
            # Step the Isaac Lab physics environment
            env.step(current_action)
            
            # Check if environment needs reset or app closed
            if not simulation_app.is_running():
                break

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()