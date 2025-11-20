# simulation.py
import mujoco
from mujoco import viewer
import time
import numpy as np
import torch
import pygame

from config_loader.config_loader import load_config, load_actor_network
from planner_policy.planning_policy import PlanningPolicy

# Utils
from utils.robot_utils import swap_legs, clip_torques_in_groups, joint_linear_interpolation
from utils.kinematics import get_feet_positions, update_swing_foot_targets
from utils.quaternion_utils import quat_to_yaw
from utils.obs_utils import compute_observation
from utils.visualization_utils import add_environment_visuals
from utils.plot_utils import init_plot, plot_body_trajectory

from planner_policy.flax_to_torch_net import ActorMLP


# ----------------------
# Constants
# ----------------------
INCLINATION_THRESHOLD = 45  # degrees
DEFAULT_MID = [0.0, 0.7, -1.5]
HL_MODEL_PATH = "planner_policy/actor_torch_full.pt"

# Flags
ENABLE_PLOTTING = False   
INTERACTIVE_PLOTS = False    
# Initialize plotting backend
if ENABLE_PLOTTING:
    init_plot(interactive=INTERACTIVE_PLOTS)

# ----------------------
# RobotSimulation Class
# ----------------------
class RobotSimulation:
    def __init__(self, config_path='config.yaml'):
        # Load config and actor
        self.config = load_config(config_path)
        self.actor_network = load_actor_network(self.config)

        self.timestep = self.config['simulation']['timestep_simulation']
        self.timestep_policy = self.config['simulation']['timestep_policy']
        self.decimation = self.config['simulation']['decimation']
        self.default_joint_angles = np.array(self.config['robot']['default_joint_angles'])
        self.kp_custom = np.array(self.config['robot']['kp_custom'])
        self.kd_custom = np.array(self.config['robot']['kd_custom'])
        self.scaling_factors = self.config['scaling']
        self.min_mu_v, self.max_mu_v = self.config["robot"]["mu_vRange"]
        self.min_Fs, self.max_Fs = self.config["robot"]["FsRange"]

        # High-level policy
        self.hl_policy = PlanningPolicy(HL_MODEL_PATH)

        # MuJoCo setup
        self.model = mujoco.MjModel.from_xml_path("aliengo_description/aliengo.xml")
        self.model.opt.timestep = self.timestep
        self.data = mujoco.MjData(self.model)
        self.data.qpos = np.array([-6., 0., 0.38, 1., 0., 0., 0.] + list(self.default_joint_angles))
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model)

        # Joystick
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Detected joystick: {self.joystick.get_name()}")

        # Initialize states
        self.hl_state = torch.zeros((1, 19), dtype=torch.float32)
        self.hl_state[0, 4:6] = torch.tensor(self.data.qpos[:2])        # Base position
        self.hl_state[0, 10:12] = torch.tensor([-3, 0.35])              # Fixed obstacle position
        self.hl_state[0, 12:14] = torch.tensor([1, 5])                  # Pos wall position
        self.hl_state[0, 14:16] = self.hl_state[0, 12:14] - 8           # Width = 8m - can be modified
        self.hl_state[0, 16] = torch.norm(self.hl_state[0, 4:6] - self.hl_state[0, 10:12]) # Distance to obstacle
        self.hl_state[0, 17] = torch.norm(self.hl_state[0, 4:6])      

        # copy of hl state - send to cuda:0
        self.hl_state_old = self.hl_state.clone().to(self.hl_state.device)

        self.foot_targets_full = get_feet_positions(self.model, self.data)
        self.gait_time_left = 0
        self.contact_pair = 0
        self.hl_dt = torch.tensor(0.)
        self.hl_actions = torch.zeros((1, 7), dtype=torch.float32)
        self.grav_tens = torch.tensor([[0.,0.,-1.]],device="cuda:0" , dtype=torch.double)

        # History & simulation params
        self.body_target_history = []
        self.body_target_raw_history = []
        self.current_actions = np.zeros(12)
        self.disable_torques = False
        self.timecounter = 0
        self.step_counter = 0
        self.time_init = 1.0
        self.time_standup = 4.0

        # Initialize base_pos_target, body_vel_targets to initial values
        self.base_pos_target = self.hl_state[:, 4:6]
        self.body_vel_targets = self.hl_state[:, 6:8]


    # ----------------------
    # Helper functions
    # ----------------------
    def get_timestep(self, gait_time, gait_time_left):
        return gait_time - gait_time_left + self.timestep_policy

    def set_intermediate_reference(self, hl_state, hl_actions):
        time_ref = self.get_timestep(self.hl_dt, self.gait_time_left)
        hl_state_intermediate = self.hl_policy.step_intermediate(hl_state, hl_actions, time_ref)
        self.base_pos_target = hl_state_intermediate[:, :2]
        self.body_vel_targets = hl_state_intermediate[:, 2:4]

    # ----------------------
    # Main simulation loop
    # ----------------------
    def run(self, simulation_duration=40.0):
        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            while v.is_running() and np.linalg.norm(self.data.qpos[:2].copy()) > 0.2:
                step_start = time.time()
                self.timecounter += 1

                # Update random friction
                motor_mu_v = np.random.uniform(self.min_mu_v, self.max_mu_v, (12,))
                motor_Fs = np.random.uniform(self.min_Fs, self.max_Fs, (12,))

                # Check inclination
                body_quat = self.data.qpos[3:7].copy()
                inclination = 2 * np.arcsin(np.sqrt(body_quat[1]**2 + body_quat[2]**2)) * (180/np.pi)
                if inclination > INCLINATION_THRESHOLD:
                    self.disable_torques = True

                if self.disable_torques:
                    torques = np.zeros_like(self.data.ctrl)
                else:
                    joint_angles = swap_legs(self.data.qpos[7:].copy())
                    joint_velocities = swap_legs(self.data.qvel[6:].copy())

                    if self.step_counter % self.decimation == 0 and self.timecounter >= (self.time_init+self.time_standup)*(1/self.timestep):
                        # Step high-level policy
                        if self.gait_time_left <= self.timestep_policy:
                            self.hl_state_old = self.hl_state.clone().to(self.hl_state.device)
                            self.hl_state, self.hl_dt, self.hl_actions = self.hl_policy.step(self.hl_state)
                            self.gait_time_left = self.hl_dt.item()
                            self.foot_targets_full = update_swing_foot_targets(
                                contact_pair=int(self.hl_state[:,-1].item()),
                                hl_state=self.hl_state[0,:4].cpu().numpy(),
                                foot_targets_full=self.foot_targets_full,
                                base_pos=self.hl_state[:,4:6].cpu().numpy(),
                                base_yaw=0.
                            )
                            self.contact_pair = 1 - self.hl_state[:,-1].item()
                            self.body_target_raw_history.append(self.hl_state[0, 4:6].cpu().numpy())

                        self.set_intermediate_reference(self.hl_state_old.to("cuda:0"), self.hl_actions)
                        self.body_target_history.append(self.base_pos_target[0,:].cpu().numpy())
            
                        # Compute observation
                        obs = compute_observation(
                            data=self.data,
                            joint_angles=joint_angles,
                            joint_velocities=joint_velocities,
                            current_actions=self.current_actions,
                            hl_state={'foot_targets': self.foot_targets_full},
                            base_pos_target=self.base_pos_target[0,:].cpu().numpy(),
                            body_vel_targets=self.body_vel_targets[0,:].cpu().numpy(),
                            contact_pair=self.contact_pair,
                            gait_time_left=self.gait_time_left,
                            grav_tens=self.grav_tens,
                            scaling_factors=self.scaling_factors,
                            actor_network=self.actor_network
                        )

                        # Get actions
                        with torch.no_grad():
                            self.current_actions = self.actor_network(obs).numpy()
                        self.gait_time_left -= self.timestep_policy

                    # Compute desired joints & torques
                    if self.timecounter < self.time_init*(1/self.timestep):
                        qInit = joint_angles
                        torques1 = np.zeros_like(self.current_actions)
                        rate_count = 0
                    elif self.time_init*(1/self.timestep) <= self.timecounter < (self.time_init+self.time_standup)*(1/self.timestep):
                        rate_count += 1
                        rate = rate_count / (3*(1/self.timestep))
                        qDes = [joint_linear_interpolation(qInit[i], DEFAULT_MID[i % 3], rate) for i in range(12)]
                        torques1 = (self.kp_custom * (qDes - joint_angles) - self.kd_custom * joint_velocities)
                    else:
                        qDes = 0.5*self.current_actions + self.default_joint_angles
                        torques1 = (self.kp_custom * (qDes - joint_angles) - self.kd_custom * joint_velocities)

                    # Optional friction
                    tau_sticktion = motor_Fs * np.tanh(joint_velocities / 0.1)
                    tau_viscose = motor_mu_v * joint_velocities
                    # torques1 -= (tau_sticktion + tau_viscose)

                    torques = swap_legs(torques1)
                    torques = clip_torques_in_groups(torques)

                # Step simulation
                self.data.ctrl = torques
                mujoco.mj_step(self.model, self.data)

                # Viewer update
                with v.lock():
                    v.cam.lookat[:] = self.data.qpos[:3]
                if self.step_counter == 0:
                    add_environment_visuals(v, self.hl_state)
                v.sync()

                self.step_counter += 1
                # Maintain real-time
                dt_sleep = self.timestep - (time.time() - step_start)
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
        
        if ENABLE_PLOTTING:
            plot_body_trajectory(
                history=self.body_target_history,
                raw_history=self.body_target_raw_history,
                show=INTERACTIVE_PLOTS,
                save_path='plots/body_target_trajectory.png' if not INTERACTIVE_PLOTS else None
            )


# ----------------------
# Main
# ----------------------
if __name__ == '__main__':
    sim = RobotSimulation()
    sim.run()
