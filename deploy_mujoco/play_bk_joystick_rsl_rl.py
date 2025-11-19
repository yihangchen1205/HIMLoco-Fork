# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""使用北通游戏手柄控制的PyTorch策略部署到C MuJoCo - 安全版本."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import sys
import os
import threading
import time
import multiprocessing
import queue
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Dict, Optional

LEGGED_GYM_SRC = "/root/Documents/HIMLoco-main/legged_gym"
if LEGGED_GYM_SRC not in sys.path:
    sys.path.insert(0, LEGGED_GYM_SRC)

from legged_gym.envs.aliengo.aliengo_config import AlienGoRoughCfg

ALIENGO_XML_PATH = epath.Path(
    "/root/Documents/HIMLoco-main/deploy_mujoco/aliengo_mj_description-master/xml/scene_mjx_flat_terrain.xml"
)
ALIENGO_POLICY_PATH = epath.Path(
    "/root/Documents/HIMLoco-main/legged_gym/logs/rough_aliengo/exported/policies/policy_model.pt"
)

ALIENGO_CFG = AlienGoRoughCfg()
ALIENGO_COMMAND_RANGES = np.array(
    [
        ALIENGO_CFG.commands.ranges.lin_vel_x,
        ALIENGO_CFG.commands.ranges.lin_vel_y,
        ALIENGO_CFG.commands.ranges.ang_vel_yaw,
    ],
    dtype=np.float32,
)
ALIENGO_COMMAND_SCALE = np.array(
    [
        ALIENGO_CFG.normalization.obs_scales.lin_vel,
        ALIENGO_CFG.normalization.obs_scales.lin_vel,
        ALIENGO_CFG.normalization.obs_scales.ang_vel,
    ],
    dtype=np.float32,
)
ALIENGO_OBS_SCALES = {
    "ang_vel": float(ALIENGO_CFG.normalization.obs_scales.ang_vel),
    "dof_pos": float(ALIENGO_CFG.normalization.obs_scales.dof_pos),
    "dof_vel": float(ALIENGO_CFG.normalization.obs_scales.dof_vel),
}
ALIENGO_DEFAULT_JOINT_DICT = dict(ALIENGO_CFG.init_state.default_joint_angles)

_HERE = epath.Path(__file__).parent


def joystick_process(command_queue, status_queue):
    """在独立进程中运行游戏手柄控制器"""
    try:
        # 在独立进程中导入pygame相关模块
        sys.path.append('/Users/cyh/Documents/mujoco_playground_fork/mujoco_playground/experimental/sim2sim')
        from beitong_game import BeitongJoystickController
        
        print("独立进程: 正在初始化北通游戏手柄控制器...")
        controller = BeitongJoystickController(wait_timeout=10.0)
        print("独立进程: 北通游戏手柄初始化成功!")
        
        status_queue.put("initialized")
        
        while True:
            try:
                controller.update()
                cmd = controller.get_command()
                
                # 发送命令到主进程
                try:
                    command_queue.put(cmd, timeout=0.001)
                except queue.Full:
                    # 如果队列满了，跳过这次更新
                    pass
                
                time.sleep(0.02)  # 50Hz
                
            except Exception as e:
                print(f"独立进程游戏手柄更新错误: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        print(f"独立进程游戏手柄初始化失败: {e}")
        status_queue.put(f"error: {e}")


def compute_aliengo_joint_metadata(model: mujoco.MjModel):
    """从AlienGo配置中构造默认关节角以及qpos/qvel索引."""
    joint_defaults = []
    joint_qpos_indices = []
    joint_qvel_indices = []
    actuator_joint_names = []
    missing = []

    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        actuator_joint_names.append(joint_name)
        joint_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        joint_qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        if joint_name in ALIENGO_DEFAULT_JOINT_DICT:
            joint_defaults.append(float(ALIENGO_DEFAULT_JOINT_DICT[joint_name]))
        else:
            missing.append(joint_name)
            joint_defaults.append(0.0)

    if missing:
        raise KeyError(
            f"AlienGo配置缺少以下关节的默认角度: {', '.join(sorted(set(filter(None, missing))))}"
        )

    return (
        np.array(joint_defaults, dtype=np.float32),
        np.array(joint_qpos_indices, dtype=np.int32),
        np.array(joint_qvel_indices, dtype=np.int32),
        actuator_joint_names,
    )


class PyTorchControllerWithJoystick:
    """带北通游戏手柄控制的PyTorch控制器 - 安全版本."""

    def __init__(
        self,
        model: mujoco.MjModel,
        policy_path: str,
        default_angles: np.ndarray,
        joint_qpos_indices: np.ndarray,
        joint_qvel_indices: np.ndarray,
        n_substeps: int,
        action_scale: float = 0.5,
        vel_scale_x: float = 1.5,
        vel_scale_y: float = 0.8,
        vel_scale_rot: float = 2 * np.pi,
        noise_config: dict = None,
        command_alpha: float = 0.8,
        device: str = "cpu",
        command_ranges: Optional[np.ndarray] = None,
        command_scale: Optional[np.ndarray] = None,
        obs_scales: Optional[Dict[str, float]] = None,
    ):
        self._model = model
        self._device = device
        self._imu_site_id = mujoco.mj_name2id(self._model,
                                              mujoco.mjtObj.mjOBJ_SITE,
                                              "imu")

        self._action_scale = action_scale

        # 使用训练时相同的默认关节角
        self._default_angles = default_angles.astype(np.float32)
        self._joint_qpos_indices = joint_qpos_indices
        self._joint_qvel_indices = joint_qvel_indices
        self._num_dofs = self._default_angles.shape[0]
        assert self._joint_qpos_indices.shape[0] == self._num_dofs
        assert self._joint_qvel_indices.shape[0] == self._num_dofs
        self._obs_scales = obs_scales or ALIENGO_OBS_SCALES
        self._command_ranges = command_ranges or ALIENGO_COMMAND_RANGES
        self._command_scale = command_scale or ALIENGO_COMMAND_SCALE
        self._command_lower = self._command_ranges[:, 0]
        self._command_upper = self._command_ranges[:, 1]
        self._command_target_scale = np.maximum(
            np.abs(self._command_lower), np.abs(self._command_upper)
        )

        # 动作记录
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)

        self._counter = 0
        self._n_substeps = n_substeps

        # 速度缩放参数
        self._vel_scale_x = vel_scale_x
        self._vel_scale_y = vel_scale_y
        self._vel_scale_rot = vel_scale_rot

        # 控制命令 - 初始为静止状态
        self.command = np.zeros(3, dtype=np.float32)  # [x_vel, y_vel, angular_vel]
        self.raw_command = np.zeros(3, dtype=np.float32)  # 原始未平滑的命令
        self.is_locked = False
        self.motor_state = 1
        
        # 命令平滑参数
        self._command_alpha = command_alpha  # 平滑系数，越大越平滑 (0.0-1.0)

        # 游戏手柄进程间通信
        self._command_queue = None
        self._status_queue = None
        self._joystick_process = None
        self._joystick_initialized = False

        # 启动游戏手柄进程
        self._init_joystick_process()

        # 噪声配置初始化
        self._noise_config = noise_config
        if self._noise_config is None:
            self._noise_config = {
                'level': 0.0,
                'scales': {
                    'linvel': 0,
                    'gyro': 0,
                    'gravity': 0,
                    'joint_pos': 0,
                    'joint_vel': 0
                }
            }

        # 加载PyTorch策略
        self._load_pytorch_policy(policy_path)

    def _load_pytorch_policy(self, policy_path: str):
        """加载RSL-RL策略模型 - 使用固定网络参数的简化版本"""
        print(f"加载PyTorch策略模型: {policy_path}")
        
        # 检查模型文件是否存在
        checkpoint_path = epath.Path(policy_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"策略文件不存在: {checkpoint_path}")
        
        # 导入ActorCritic模块
        from rsl_rl.modules.actor_critic import ActorCritic
        
        # 使用固定的网络参数（根据您的训练配置）
        num_actor_obs = 45    # 根据您的模型调整
        num_critic_obs = 120  # 根据您的模型调整  
        num_actions = 12      # BK机器人的关节数
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        
        print(f"使用固定网络结构:")
        print(f"  Actor观测维度: {num_actor_obs}")
        print(f"  Critic观测维度: {num_critic_obs}")
        print(f"  动作维度: {num_actions}")
        print(f"  Actor隐藏层: {actor_hidden_dims}")
        print(f"  Critic隐藏层: {critic_hidden_dims}")
        
        # 创建观测字典和观测组配置（新版本RSL-RL需要）
        # 创建模拟的观测数据用于初始化网络，使用BK环境的观测键名
        obs = TensorDict({
            "state": torch.zeros(1, num_actor_obs),             # 策略网络观测（45维）
            "privileged_state": torch.zeros(1, num_critic_obs)  # 价值网络观测（120维）
        }, batch_size=[1])
        
        # 观测组配置（根据BK环境的实际观测键名配置）
        obs_groups = {
            "policy": ["state"],           # 策略网络使用"state"观测（45维）
            "critic": ["privileged_state"] # 价值网络使用"privileged_state"观测（120维）
        }
        
        print(f"观测组配置: {obs_groups}")
        
        # 创建ActorCritic网络（使用新版本RSL-RL的方式）
        # 🎯 关键修复：使用与训练时相同的激活函数
        self._policy_network = ActorCritic(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation='swish',  # 🔥 修复：使用与rsl_rl_brax_matched_config一致的swish激活函数
            init_noise_std=1.0,
        )
        
        # 加载检查点
        print(f"从检查点加载模型: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=torch.device('cpu'))
        
        # 获取模型状态字典
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint
        
        # 分离网络权重和标准化器参数
        network_state_dict = {}
        actor_normalizer_params = {}
        critic_normalizer_params = {}
        
        for key, value in model_state_dict.items():
            if 'actor_obs_normalizer' in key:
                # 提取actor标准化器参数
                norm_key = key.replace('actor_obs_normalizer.', '')
                actor_normalizer_params[norm_key] = value
            elif 'critic_obs_normalizer' in key:
                # 提取critic标准化器参数
                norm_key = key.replace('critic_obs_normalizer.', '')
                critic_normalizer_params[norm_key] = value
            else:
                # 网络权重
                network_state_dict[key] = value
        
        # 存储标准化器参数用于推理时应用
        if actor_normalizer_params:
            print("检测到Actor观测标准化器参数，将在推理时应用")
            self._actor_obs_mean = actor_normalizer_params.get('_mean', None)
            self._actor_obs_var = actor_normalizer_params.get('_var', None)
            self._actor_obs_std = actor_normalizer_params.get('_std', None)
            if self._actor_obs_mean is not None:
                self._actor_obs_mean = self._actor_obs_mean.flatten()
            if self._actor_obs_std is not None:
                self._actor_obs_std = self._actor_obs_std.flatten()
            print(f"Actor标准化器: mean.shape={self._actor_obs_mean.shape if self._actor_obs_mean is not None else 'None'}, "
                  f"std.shape={self._actor_obs_std.shape if self._actor_obs_std is not None else 'None'}")
        else:
            self._actor_obs_mean = None
            self._actor_obs_std = None
            
        if critic_normalizer_params:
            print("检测到Critic观测标准化器参数")
            
        # 加载网络权重
        try:
            self._policy_network.load_state_dict(network_state_dict, strict=True)
            print("成功加载模型权重（已分离标准化器参数）")
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            print("尝试使用strict=False模式...")
            self._policy_network.load_state_dict(network_state_dict, strict=False)
            print("使用非严格模式加载完成")
        
        # 移动到指定设备
        if self._device == "cuda" and torch.cuda.is_available():
            self._policy_network = self._policy_network.cuda()
            print("模型已移动到GPU")
        else:
            self._policy_network = self._policy_network.to(self._device)
            print(f"模型已移动到设备: {self._device}")
        
        # 设置为评估模式
        self._policy_network.eval()
        
        # 创建推理函数
        def inference_policy(obs_tensor):
            with torch.no_grad():
                # 应用观测标准化（如果存在）
                normalized_obs = obs_tensor
                if self._actor_obs_mean is not None and self._actor_obs_std is not None:
                    # 移动标准化参数到相同设备
                    obs_mean = self._actor_obs_mean.to(obs_tensor.device)
                    obs_std = self._actor_obs_std.to(obs_tensor.device)
                    # 应用标准化: (obs - mean) / std
                    normalized_obs = (obs_tensor - obs_mean) / (obs_std + 1e-8)
                
                # 创建TensorDict格式的观测数据用于推理，使用正确的键名
                obs_dict = TensorDict({
                    "state": normalized_obs  # 使用标准化后的观测
                }, batch_size=normalized_obs.shape[:1])
                
                # 使用act_inference方法进行推理
                actions = self._policy_network.act_inference(obs_dict)
                return actions
        
        self._policy = inference_policy
        
        # 设置观测维度（使用固定值）
        self._obs_dim = num_actor_obs
        
        print(f"PyTorch策略加载成功!")
        print(f"观测维度: {self._obs_dim}")
        print(f"动作维度: {num_actions}")

    def _init_joystick_process(self):
        """初始化游戏手柄进程"""
        try:
            print("启动独立的游戏手柄进程...")
            self._command_queue = multiprocessing.Queue(maxsize=10)
            self._status_queue = multiprocessing.Queue()
            
            self._joystick_process = multiprocessing.Process(
                target=joystick_process,
                args=(self._command_queue, self._status_queue)
            )
            self._joystick_process.start()
            
            # 等待初始化状态
            try:
                status = self._status_queue.get(timeout=15.0)
                if status == "initialized":
                    self._joystick_initialized = True
                    print("游戏手柄进程初始化成功!")
                else:
                    print(f"游戏手柄进程初始化失败: {status}")
                    self._joystick_initialized = False
            except queue.Empty:
                print("游戏手柄进程初始化超时")
                self._joystick_initialized = False
                
        except Exception as e:
            print(f"启动游戏手柄进程失败: {e}")
            self._joystick_initialized = False

    def _update_joystick_command(self):
        """更新游戏手柄命令"""
        if not self._joystick_initialized or self._command_queue is None:
            return
        
        try:
            # 获取最新的命令（非阻塞）
            while not self._command_queue.empty():
                cmd = self._command_queue.get_nowait()
                # 更新原始命令（-1..1）
                self.raw_command[0] = cmd['x_velocity']
                self.raw_command[1] = cmd['y_velocity']
                self.raw_command[2] = cmd['angular_velocity']
                self.is_locked = cmd['is_locked']
                self.motor_state = cmd['motor_state']
            
            # 目标命令（按训练幅值缩放到物理单位）
            target_cmd = self.raw_command * self._command_target_scale
            
            # 平滑处理（指数移动平均）
            self.command = (
                self._command_alpha * self.command
                + (1.0 - self._command_alpha) * target_cmd
            )
            
            # 截断命令到训练时范围
            self.command = np.clip(self.command, self._command_lower, self._command_upper)
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"更新游戏手柄命令错误: {e}")

    def _add_noise(self, value: np.ndarray, scale: float) -> np.ndarray:
        """根据配置为数据添加噪声"""
        if self._noise_config['level'] == 0.0:
            return value
        noise = (2 * np.random.uniform(size=value.shape) - 
                 1) * self._noise_config['level'] * scale
        return value + noise

    def get_obs(self, data: mujoco.MjData) -> np.ndarray:
        """构建与训练时完全一致的观测向量 - 固定45维版本."""
        # 传感器数据
        gyro = data.sensor("gyro").data * self._obs_scales["ang_vel"]
        gravity = data.site_xmat[self._imu_site_id].reshape(3, 3).T @ np.array(
            [0, 0, -1])
        joint_angles = (
            data.qpos[self._joint_qpos_indices] - self._default_angles
        ) * self._obs_scales["dof_pos"]
        joint_velocities = data.qvel[self._joint_qvel_indices] * self._obs_scales[
            "dof_vel"
        ]

        # 使用游戏手柄的控制命令
        command = np.clip(self.command, self._command_lower, self._command_upper)
        command_scaled = command * self._command_scale
        
        # 调试信息：显示原始命令和平滑后命令的对比
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 1
            
        if self._debug_counter % 50 == 0:  # 每50帧打印一次
            raw_cmd_scaled = self.raw_command * self._command_target_scale
            print(
                "原始命令(未平滑,已按训练幅值缩放): "
                f"[{raw_cmd_scaled[0]:.3f}, {raw_cmd_scaled[1]:.3f}, {raw_cmd_scaled[2]:.3f}] "
                f"-> 平滑+截断后: [{command[0]:.3f}, {command[1]:.3f}, {command[2]:.3f}] "
                f"(范围: x[{self._command_lower[0]}, {self._command_upper[0]}], "
                f"y[{self._command_lower[1]}, {self._command_upper[1]}], "
                f"ω[{self._command_lower[2]}, {self._command_upper[2]}])"
            )

        # 构建观测向量 (固定45维: 3+3+3+12+12+12=45)
        obs_list = [
            command_scaled,           # 3维
            gyro,                     # 3维
            gravity,                  # 3维
            joint_angles,             # 12维
            joint_velocities,         # 12维
            self._last_action,        # 12维
        ]
        obs = np.concatenate(obs_list).astype(np.float32)

        # 如果这是第一次调用，打印各组件维度用于调试
        if not hasattr(self, '_dims_printed'):
            print(f"观测组件维度:")
            print(f"  command_scaled: {command_scaled.shape}")
            print(f"  gyro: {gyro.shape}")
            print(f"  gravity: {gravity.shape}")
            print(f"  joint_angles: {joint_angles.shape}")
            print(f"  joint_velocities: {joint_velocities.shape}")
            print(f"  last_action: {self._last_action.shape}")
            print(f"  总观测维度: {obs.shape}")
            self._dims_printed = True
            
        # 验证观测维度
        assert obs.shape[0] == 45, f"观测维度应该是45，但得到了{obs.shape[0]}"

        return obs

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """获取控制信号"""
        # 更新游戏手柄命令
        self._update_joystick_command()
        
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(data)

            if self._counter == self._n_substeps:
                print(f"观测向量维度: {obs.shape}")
                assert obs.shape[0] == self._obs_dim, (
                    f"观测维度不匹配！期望 {self._obs_dim}, "
                    f"得到 {obs.shape[0]}")

            try:
                # 检查是否锁定或电机关闭
                if self.is_locked or self.motor_state == 0:
                    # 机器人锁定或电机关闭时，保持默认姿态
                    data.ctrl[:] = self._default_angles
                else:
                    # 正常控制 - 使用PyTorch策略（参考 train_rsl_rl.py 的推理方式）
                    obs_torch = torch.from_numpy(obs).float().to(self._device).unsqueeze(0)
                    
                    # 使用标准的 RSL-RL 推理方式：act_inference 方法
                    actions = self._policy(obs_torch)
                    actions = torch.clip(actions, -1.0, 1.0)  # pytype: disable=attribute-error
                    # 转换为numpy并应用到控制器
                    actions_np = actions.cpu().numpy().flatten()
                    self._last_action = actions_np.copy()
                    data.ctrl[:] = actions_np * self._action_scale + self._default_angles

            except Exception as e:
                print(f"PyTorch推理错误: {e}")
                data.ctrl[:] = self._default_angles

    def cleanup(self):
        """清理资源"""
        if self._joystick_process and self._joystick_process.is_alive():
            self._joystick_process.terminate()
            self._joystick_process.join(timeout=2.0)
        print("控制器资源已清理")


def load_callback(model=None, data=None):
    """加载回调函数"""
    mujoco.set_mjcb_control(None)

    # 环境配置（无需依赖mujoco_playground）
    class EnvConfig:
        Kd = 1.0
        Kp = 35.0
        ctrl_dt = ALIENGO_CFG.control.decimation * ALIENGO_CFG.sim.dt
        sim_dt = ALIENGO_CFG.sim.dt
        action_scale = ALIENGO_CFG.control.action_scale
        history_len = 1

    env_config = EnvConfig()

    # 加载模型
    if not ALIENGO_XML_PATH.exists():
        raise FileNotFoundError(
            f"找不到指定的XML场景文件: {ALIENGO_XML_PATH.as_posix()}"
        )

    model = mujoco.MjModel.from_xml_path(ALIENGO_XML_PATH.as_posix())
    # 可选择粗糙地形
    # model = mujoco.MjModel.from_xml_path(
    #     bk_constants.FEET_ONLY_ROUGH_TERRAIN_XML.as_posix(),
    #     assets=get_assets(),
    # )
    
    # model.dof_damping[6:] = env_config.Kd
    # model.actuator_gainprm[:, 0] = env_config.Kp
    # model.actuator_biasprm[:, 1] = -env_config.Kp

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = env_config.ctrl_dt
    sim_dt = env_config.sim_dt
    n_substeps = int(round(ctrl_dt / sim_dt))

    # 创建带游戏手柄的PyTorch控制器
    default_angles, joint_qpos_indices, joint_qvel_indices, actuator_joint_names = (
        compute_aliengo_joint_metadata(model)
    )
    print("使用AlienGo配置默认关节角:")
    for name, value in zip(actuator_joint_names, default_angles):
        print(f"  {name}: {value:.3f} rad")

    policy = PyTorchControllerWithJoystick(
        model=model,
        policy_path=ALIENGO_POLICY_PATH.as_posix(),  # PyTorch .pt模型文件
        default_angles=default_angles,
        joint_qpos_indices=joint_qpos_indices,
        joint_qvel_indices=joint_qvel_indices,
        n_substeps=n_substeps,
        action_scale=env_config.action_scale,
        command_alpha=0.95,  # 命令平滑系数，可调整 (0.0=无平滑, 1.0=最大平滑)
        device="cpu",  # 可以改为"cuda"如果有GPU
        command_ranges=ALIENGO_COMMAND_RANGES,
        command_scale=ALIENGO_COMMAND_SCALE,
        obs_scales=ALIENGO_OBS_SCALES,
    )

    mujoco.set_mjcb_control(policy.get_control)

    return model, data


if __name__ == "__main__":
    print("=== 北通游戏手柄控制的机器人仿真 (PyTorch版本) ===")
    print("控制说明:")
    print("  左摇杆: 控制机器人前后左右移动")
    print("  右摇杆X轴: 控制机器人左右旋转")
    print("  请确保北通游戏手柄已连接")
    print("  注意: 游戏手柄在独立进程中运行，避免资源冲突")
    print("  新功能: 命令平滑 - 减少手柄抖动，提供更流畅的控制体验")
    print("  平滑系数: 0.95 (可在代码中调整，0.0=无平滑, 1.0=最大平滑)")
    print(
        "  命令截断: "
        f"x[{ALIENGO_COMMAND_RANGES[0][0]}, {ALIENGO_COMMAND_RANGES[0][1]}], "
        f"y[{ALIENGO_COMMAND_RANGES[1][0]}, {ALIENGO_COMMAND_RANGES[1][1]}], "
        f"yaw[{ALIENGO_COMMAND_RANGES[2][0]}, {ALIENGO_COMMAND_RANGES[2][1]}]"
    )
    print("  模型类型: PyTorch (.pt) 模型 - 自动推断网络结构")
    print("-" * 50)

    policy_controller = None
    
    try:
        # 启动仿真
        viewer.launch(loader=load_callback)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        print("清理资源...")
        # 注意：由于viewer.launch的限制，我们无法直接访问policy对象
        # 清理工作会在程序退出时自动进行 