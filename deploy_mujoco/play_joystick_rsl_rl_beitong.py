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
"""使用北通遥控器命令的PyTorch策略部署到C MuJoCo."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import sys
import os
import traceback
import torch
from typing import Dict, Optional, Sequence
import multiprocessing
import queue
import time
import argparse
import yaml
from pathlib import Path

_HERE = epath.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_DEFAULT_LEGGED_GYM_SRC = (_REPO_ROOT / "legged_gym").as_posix()
LEGGED_GYM_SRC = os.environ.get("LEGGED_GYM_SRC", _DEFAULT_LEGGED_GYM_SRC)
LEGGED_GYM_SRC_PATH = epath.Path(LEGGED_GYM_SRC).expanduser().resolve().as_posix()
if LEGGED_GYM_SRC_PATH not in sys.path:
    sys.path.insert(0, LEGGED_GYM_SRC_PATH)

_DEFAULT_RSL_RL_SRC = (_REPO_ROOT / "rsl_rl").as_posix()
RSL_RL_SRC = os.environ.get("RSL_RL_SRC", _DEFAULT_RSL_RL_SRC)
RSL_RL_SRC_PATH = epath.Path(RSL_RL_SRC).expanduser().resolve().as_posix()
if RSL_RL_SRC_PATH not in sys.path:
    sys.path.insert(0, RSL_RL_SRC_PATH)

from legged_gym.envs.aliengo.aliengo_config import AlienGoRoughCfg, AlienGoRoughCfgPPO
from rsl_rl.modules.him_actor_critic import HIMActorCritic
# ALIENGO_XML_PATH = epath.Path(
#     os.environ.get(
#         "ALIENGO_XML_PATH",
#         (_HERE / "aliengo_mj_description-master/xml/scene_mjx_flat_terrain.xml").as_posix(),
#     )
ALIENGO_XML_PATH = epath.Path(
    os.environ.get(
        "ALIENGO_XML_PATH",
        (_HERE / "aliengo_description/scene_mjx_flat_terrain.xml").as_posix(),
    )
).expanduser()
ALIENGO_POLICY_PATH = epath.Path(
    os.environ.get(
        "ALIENGO_POLICY_PATH",
        (_REPO_ROOT / "legged_gym/logs/Nov19_15-36-27_/model_2020.pt").as_posix(),
    )
).expanduser()

ALIENGO_CFG = AlienGoRoughCfg()
ALIENGO_PPO_CFG = AlienGoRoughCfgPPO()
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

# 全局变量用于存储配置
_GLOBAL_CONFIG = None
_GLOBAL_POLICY_PATH = None
_GLOBAL_XML_PATH = None


def get_gravity_orientation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """四元数 [w,x,y,z] -> 机体坐标系下的重力方向 (3,)"""
    qw, qx, qy, qz = float(quaternion_wxyz[0]), float(quaternion_wxyz[1]), float(quaternion_wxyz[2]), float(quaternion_wxyz[3])
    g = np.zeros(3, dtype=np.float32)
    g[0] = 2.0 * (-qz * qx + qw * qy)
    g[1] = -2.0 * (qz * qy + qw * qx)
    g[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return g


def joystick_process(command_queue, status_queue):
    """在独立进程中运行北通遥控器控制器."""
    try:
        sys.path.append('/Users/cyh/Documents/mujoco_playground_fork/mujoco_playground/experimental/sim2sim')
        from beitong_game import BeitongJoystickController

        print("独立进程: 初始化北通遥控器控制器...")
        controller = BeitongJoystickController(wait_timeout=10.0)
        print("独立进程: 北通遥控器初始化成功!")

        status_queue.put("initialized")

        while True:
            try:
                controller.update()
                cmd = controller.get_command()

                try:
                    command_queue.put(cmd, timeout=0.001)
                except queue.Full:
                    pass

                time.sleep(0.02)

            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"独立进程遥控器更新错误: {exc}")
                time.sleep(0.1)

    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"独立进程遥控器初始化失败: {exc}")
        status_queue.put(f"error: {exc}")




_MUJOCO_LEG_ORDER = ("FR", "FL", "RR", "RL")
_POLICY_LEG_ORDER = ("FL", "FR", "RL", "RR")
_LEG_DOF = 3


def _reorder_leg_groups(
    values: np.ndarray,
    source_order: Sequence[str],
    target_order: Sequence[str],
) -> np.ndarray:
    """Reorder leg joint triplets from source ordering to target ordering."""
    arr = np.asarray(values)
    expected_dim = _LEG_DOF * len(source_order)
    if arr.shape[-1] != expected_dim:
        raise ValueError(
            f"关节向量维度应为{expected_dim}，得到形状: {arr.shape}"
        )

    reordered = np.empty_like(arr)
    for dst_leg_idx, leg_name in enumerate(target_order):
        try:
            src_leg_idx = source_order.index(leg_name)
        except ValueError as exc:
            raise ValueError(f"未知的腿部名称: {leg_name}") from exc

        dst_slice = slice(dst_leg_idx * _LEG_DOF, (dst_leg_idx + 1) * _LEG_DOF)
        src_slice = slice(src_leg_idx * _LEG_DOF, (src_leg_idx + 1) * _LEG_DOF)
        reordered[..., dst_slice] = arr[..., src_slice]
    return reordered


def _mujoco_to_policy_order(values: np.ndarray) -> np.ndarray:
    """Convert MuJoCo joint ordering (FR, FL, RR, RL) to policy ordering (FL, FR, RL, RR)."""
    return _reorder_leg_groups(values, _MUJOCO_LEG_ORDER, _POLICY_LEG_ORDER)


def _policy_to_mujoco_order(values: np.ndarray) -> np.ndarray:
    """Convert policy ordering (FL, FR, RL, RR) back to MuJoCo ordering (FR, FL, RR, RL)."""
    return _reorder_leg_groups(values, _POLICY_LEG_ORDER, _MUJOCO_LEG_ORDER)


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
    """使用北通遥控器命令的PyTorch控制器."""

    def __init__(
        self,
        model: mujoco.MjModel,
        policy_path: str,
        default_angles: np.ndarray,
        joint_qpos_indices: np.ndarray,
        joint_qvel_indices: np.ndarray,
        actuator_joint_names: Sequence[str],
        n_substeps: int,
        action_scale: float = 0.5,
        vel_scale_x: float = 1.5,
        vel_scale_y: float = 0.8,
        vel_scale_rot: float = 2 * np.pi,
        noise_config: dict = None,
        device: str = "cpu",
        command_ranges: Optional[np.ndarray] = None,
        command_scale: Optional[np.ndarray] = None,
        obs_scales: Optional[Dict[str, float]] = None,
        kp: float = 40.0,
        kd: float = 2.0,
        hip_reduction: float = 1.0,
        torque_limit: Optional[float] = None,
        command_alpha: float = 0.8,
        dof_pos_scale: float = 1.0,
        dof_vel_scale: float = 0.05,
        ang_vel_scale: float = 0.25,
        action_smoothing: float = 0.0,
        joint2motor_idx: Optional[np.ndarray] = None,
    ):
        self._model = model
        self._device = device
        self._base_body_id = self._resolve_mj_name(
            mujoco.mjtObj.mjOBJ_BODY, ("trunk", "base", "base_link")
        )

        self._action_scale = action_scale

        # 使用训练时相同的默认关节角（MuJoCo顺序）及Isaac顺序
        self._default_angles = default_angles.astype(np.float32)
        self._default_angles_policy = _mujoco_to_policy_order(self._default_angles)
        self._joint_qpos_indices = joint_qpos_indices
        self._joint_qvel_indices = joint_qvel_indices
        self._actuator_joint_names = list(actuator_joint_names)
        self._num_dofs = self._default_angles.shape[0]
        assert self._joint_qpos_indices.shape[0] == self._num_dofs
        assert self._joint_qvel_indices.shape[0] == self._num_dofs
        self._obs_scales = (
            obs_scales if obs_scales is not None else ALIENGO_OBS_SCALES
        )
        self._command_ranges = (
            command_ranges if command_ranges is not None else ALIENGO_COMMAND_RANGES
        )
        self._command_scale = (
            command_scale if command_scale is not None else ALIENGO_COMMAND_SCALE
        )
        self._command_lower = self._command_ranges[:, 0]
        self._command_upper = self._command_ranges[:, 1]
        self._command_span = self._command_upper - self._command_lower
        self._command_mid = (self._command_upper + self._command_lower) * 0.5

        # 观测历史维度（与训练时一致，历史长度=6，单步维度=45）
        self._single_obs_dim = int(ALIENGO_CFG.env.num_one_step_observations)
        history_len = max(1, int(ALIENGO_CFG.env.num_observations // self._single_obs_dim))
        self._history_len = history_len
        self._obs_dim = self._single_obs_dim * self._history_len
        self._obs_history = np.zeros((self._history_len, self._single_obs_dim), dtype=np.float32)
        self._clip_obs = float(ALIENGO_CFG.normalization.clip_observations)
        self._clip_actions = float(ALIENGO_CFG.normalization.clip_actions)

        # 动作记录（使用Isaac Gym的关节顺序，供观测使用）
        self._current_action_policy = np.zeros_like(default_angles, dtype=np.float32)
        self._last_action_policy = np.zeros_like(default_angles, dtype=np.float32)
        self._action_smoothed = np.zeros_like(default_angles, dtype=np.float32)
        
        # 观测缩放参数（从 yaml 配置加载）
        self._dof_pos_scale = dof_pos_scale
        self._dof_vel_scale = dof_vel_scale
        self._ang_vel_scale = ang_vel_scale
        self._action_smoothing = action_smoothing
        
        # 关节到电机的映射（用于观测中的顺序转换）
        if joint2motor_idx is not None:
            self._joint2motor_idx = np.array(joint2motor_idx, dtype=np.int32)
            # 创建反向映射：从电机索引到关节索引
            # 如果 joint2motor_idx[joint_idx] = motor_idx，则 motor2joint_idx[motor_idx] = joint_idx
            # 这样可以将按照关节索引顺序的数据转换为按照电机索引顺序
            self._motor2joint_idx = np.zeros(len(self._joint2motor_idx), dtype=np.int32)
            for joint_idx, motor_idx in enumerate(self._joint2motor_idx):
                if 0 <= motor_idx < len(self._motor2joint_idx):
                    self._motor2joint_idx[motor_idx] = joint_idx
            print(f"关节到电机映射: {self._joint2motor_idx}")
            print(f"电机到关节映射: {self._motor2joint_idx}")
            # 重新排列 default_angles 以匹配观测中的顺序
            # default_angles 是从 yaml 加载的，可能是按照关节顺序的
            # 我们需要将其转换为按照电机索引顺序（与 qj 的顺序一致）
            self._default_angles = self._default_angles[self._motor2joint_idx].copy()
            print(f"已重新排列 default_angles 以匹配观测顺序")
        else:
            # 如果没有提供映射，使用恒等映射
            self._joint2motor_idx = np.arange(self._num_dofs, dtype=np.int32)
            self._motor2joint_idx = np.arange(self._num_dofs, dtype=np.int32)

        self._counter = 0
        self._n_substeps = n_substeps

        # 速度缩放参数
        self._vel_scale_x = vel_scale_x
        self._vel_scale_y = vel_scale_y
        self._vel_scale_rot = vel_scale_rot

        # 控制命令 - 由遥控器驱动
        self.command = np.zeros(3, dtype=np.float32)
        self.raw_command = np.zeros(3, dtype=np.float32)
        self.is_locked = False
        self.motor_state = 1
        self._command_alpha = float(np.clip(command_alpha, 0.0, 1.0))

        # 关节目标与力矩相关参数
        self._hip_reduction = hip_reduction
        # 根据关节名称找到hip关节的索引（对应Isaac Gym中的[0, 3, 6, 9]）
        # 假设关节顺序为: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
        hip_indices = []
        for i, name in enumerate(self._actuator_joint_names):
            if "hip" in name.lower() and ("FL" in name or "FR" in name or "RL" in name or "RR" in name):
                hip_indices.append(i)
        # 确保找到4个hip关节，且索引为[0, 3, 6, 9]（或类似模式）
        self._hip_indices = np.array(hip_indices, dtype=np.int32) if len(hip_indices) == 4 else np.array([0, 3, 6, 9], dtype=np.int32)
        self._target_joint_pos = self._default_angles.copy()
        self._last_torque = np.zeros(self._num_dofs, dtype=np.float32)
        self._kp = np.full(self._num_dofs, kp, dtype=np.float32)
        self._kd = np.full(self._num_dofs, kd, dtype=np.float32)
        if torque_limit is not None:
            self._torque_limit = np.full(self._num_dofs, torque_limit, dtype=np.float32)
        else:
            self._torque_limit = None

        # 遥控器进程
        self._command_queue = None
        self._status_queue = None
        self._joystick_process = None
        self._joystick_initialized = False

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
        """加载与训练完全一致的RSL-RL策略 (HIMActorCritic)."""
        print(f"加载PyTorch策略模型: {policy_path}")

        checkpoint_path = epath.Path(policy_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"策略文件不存在: {checkpoint_path}")

        policy_cfg = ALIENGO_PPO_CFG.policy
        num_actor_obs = int(self._obs_dim)
        num_critic_obs = int(
            getattr(ALIENGO_CFG.env, "num_privileged_obs", 0) or num_actor_obs
        )
        num_one_step_obs = int(self._single_obs_dim)
        num_actions = int(ALIENGO_CFG.env.num_actions)
        actor_hidden_dims = list(policy_cfg.actor_hidden_dims)
        critic_hidden_dims = list(policy_cfg.critic_hidden_dims)

        print("使用与 legged_gym/scripts/play.py 相同的网络结构:")
        print(
            f"  Actor观测维度: {num_actor_obs} "
            f"(单步 {num_one_step_obs} × 历史 {self._history_len})"
        )
        print(f"  Critic观测维度: {num_critic_obs}")
        print(f"  动作维度: {num_actions}")
        print(f"  Actor隐藏层: {actor_hidden_dims}")
        print(f"  Critic隐藏层: {critic_hidden_dims}")
        print(f"  激活函数: {policy_cfg.activation}")

        self._policy_network = HIMActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_one_step_obs=num_one_step_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=policy_cfg.activation,
            init_noise_std=policy_cfg.init_noise_std,
        )

        print(f"从检查点加载模型: {checkpoint_path}")
        try:
            checkpoint = torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

        model_state_dict = checkpoint.get("model_state_dict", checkpoint)

        try:
            self._policy_network.load_state_dict(model_state_dict, strict=True)
            print("成功加载模型权重 (strict=True)")
        except RuntimeError as exc:
            print(f"严格加载失败: {exc}")
            missing, unexpected = self._policy_network.load_state_dict(
                model_state_dict, strict=False
            )
            if missing:
                print(f"缺失权重: {missing}")
            if unexpected:
                print(f"额外权重: {unexpected}")
            print("使用strict=False模式加载完成（请确认权重与配置一致）")

        target_device = self._device
        if (
            isinstance(target_device, str)
            and target_device.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            print("CUDA不可用，自动回退到CPU")
            target_device = "cpu"
            self._device = "cpu"

        self._policy_network = self._policy_network.to(target_device)
        self._policy_network.eval()
        print(f"模型已移动到设备: {target_device}")

        def inference_policy(obs_tensor: torch.Tensor):
            with torch.no_grad():
                return self._policy_network.act_inference(obs_tensor)

        self._policy = inference_policy
        print(f"PyTorch策略加载成功! 观测维度: {self._obs_dim}, 动作维度: {num_actions}")

    def _resolve_mj_name(self, obj_type: int, candidate_names: Sequence[str]) -> int:
        """Return first matching mujoco name id for given type."""
        for name in candidate_names:
            try:
                name_id = mujoco.mj_name2id(self._model, obj_type, name)
            except Exception:
                name_id = -1
            if name_id != -1:
                return name_id
        raise KeyError(
            f"无法在模型中找到名称，已尝试: {', '.join(candidate_names)}"
        )

    def _scale_actions_to_targets(self, actions_policy: np.ndarray) -> None:
        """将策略输出转换为目标关节角."""
        actions_mujoco = _policy_to_mujoco_order(actions_policy)
        actions_scaled = actions_mujoco * self._action_scale
        # 应用hip_reduction，与Isaac Gym一致：只对hip关节索引[0, 3, 6, 9]应用
        if self._hip_reduction != 1.0 and len(self._hip_indices) > 0:
            actions_scaled[self._hip_indices] *= self._hip_reduction
        
        # 在 refactor 代码中，目标姿态是动作和默认姿态的叠加
        self._target_joint_pos = self._default_angles + actions_scaled

    def _compute_torque_command(self, data: mujoco.MjData) -> np.ndarray:
        """根据当前目标姿态与实际状态计算力矩 (参考 refactor 代码风格)."""
        joint_pos = data.qpos[self._joint_qpos_indices].astype(np.float32)
        joint_vel = data.qvel[self._joint_qvel_indices].astype(np.float32)
        
        # PD 控制: Kp * (q_desired - q_current) - Kd * q_dot_current
        # 在 refactor 代码中，qDes = 0.5 * self.current_actions + self.default_joint_angles
        # 这里的 self._target_joint_pos 已经包含了 self.default_angles，所以逻辑一致
        torques = self._kp * (self._target_joint_pos - joint_pos) - self._kd * joint_vel

        if self._torque_limit is not None:
            torques = np.clip(torques, -self._torque_limit, self._torque_limit)
        
        self._last_torque = torques
        return torques

    def _add_noise(self, value: np.ndarray, scale: float) -> np.ndarray:
        """根据配置为数据添加噪声"""
        if self._noise_config['level'] == 0.0:
            return value
        noise = (2 * np.random.uniform(size=value.shape) - 
                 1) * self._noise_config['level'] * scale
        return value + noise

    def _init_joystick_process(self):
        """启动独立的北通遥控器进程."""
        try:
            print("启动独立的北通遥控器进程...")
            self._command_queue = multiprocessing.Queue(maxsize=10)
            self._status_queue = multiprocessing.Queue()

            self._joystick_process = multiprocessing.Process(
                target=joystick_process,
                args=(self._command_queue, self._status_queue),
            )
            self._joystick_process.start()

            try:
                status = self._status_queue.get(timeout=15.0)
                if status == "initialized":
                    self._joystick_initialized = True
                    print("北通遥控器进程初始化成功!")
                else:
                    print(f"北通遥控器进程初始化失败: {status}")
            except queue.Empty:
                print("北通遥控器进程初始化超时")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"启动北通遥控器进程失败: {exc}")

    def _update_joystick_command(self):
        """更新遥控器命令并映射至训练范围."""
        if not self._joystick_initialized or self._command_queue is None:
            return

        try:
            while not self._command_queue.empty():
                cmd = self._command_queue.get_nowait()
                self.raw_command[0] = -cmd["x_velocity"]
                self.raw_command[1] = -cmd["y_velocity"]
                self.raw_command[2] = cmd["angular_velocity"]
                self.is_locked = cmd["is_locked"]
                self.motor_state = cmd["motor_state"]

            raw = np.clip(self.raw_command, -1.0, 1.0)
            target_cmd = self._command_mid + 0.5 * raw * self._command_span
            self.command = (
                self._command_alpha * self.command
                + (1.0 - self._command_alpha) * target_cmd
            )
            self.command = np.clip(self.command, self._command_lower, self._command_upper)
        except queue.Empty:
            pass
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"更新遥控器命令错误: {exc}")

    def get_obs(self, data: mujoco.MjData) -> np.ndarray:
        """构建观测向量 - 与 deploy_mujoco_controller.py 完全一致的计算方法"""
        try:
            # 提取基本状态 - 直接从 qpos/qvel 索引，与 deploy 一致
            qj_raw = data.qpos[7:7+self._num_dofs].copy()
            dqj_raw = data.qvel[6:6+self._num_dofs].copy()
            quat = data.qpos[3:7].copy()
            omega = data.qvel[3:6].copy()
            
            # 根据 joint2motor_idx 重新排列关节角度和速度
            # 从 MuJoCo 的关节顺序转换为策略期望的顺序
            # motor2joint_idx[j] 表示电机 j 对应的关节索引
            qj = qj_raw[self._motor2joint_idx].copy()
            dqj = dqj_raw[self._motor2joint_idx].copy()
            
            # 归一化/缩放 - 使用 yaml 配置中的缩放因子
            qj_n = (qj - self._default_angles) * self._dof_pos_scale
            dqj_n = dqj * self._dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega_n = omega * self._ang_vel_scale
            
            # 处理命令
            command_clipped = np.clip(self.command, self._command_lower, self._command_upper)
            command_scaled = command_clipped * self._command_scale
            
            # 确保 action_smoothed 长度正确
            if self._action_smoothed.size != self._num_dofs:
                self._action_smoothed = np.zeros(self._num_dofs, dtype=np.float32)
            
            # 组装单步观测 (45维: 3+3+3+12+12+12)
            single_obs = np.concatenate([
                command_scaled,         # 3维
                omega_n,                # 3维
                gravity_orientation,     # 3维
                qj_n,                   # 12维
                dqj_n,                  # 12维
                self._action_smoothed,  # 12维
            ]).astype(np.float32)
            
            if single_obs.size != self._single_obs_dim:
                raise ValueError(f"观测维度错误: single_obs.size={single_obs.size} != {self._single_obs_dim}")
            
            # 维护历史观测缓冲
            self._obs_history[1:] = self._obs_history[:-1]
            self._obs_history[0] = single_obs
            obs = self._obs_history.reshape(-1).astype(np.float32).copy()
            np.clip(obs, -self._clip_obs, self._clip_obs, out=obs)
            
            if obs.size != self._obs_dim:
                raise ValueError(f"历史观测维度错误: obs.size={obs.size} != {self._obs_dim}")
            
            # 如果这是第一次调用，打印各组件维度用于调试
            if not hasattr(self, '_dims_printed'):
                print("观测组件维度:")
                print(f"  单步 command_scaled: {command_scaled.shape}")
                print(f"  单步 omega_n: {omega_n.shape}")
                print(f"  单步 gravity_orientation: {gravity_orientation.shape}")
                print(f"  单步 qj_n: {qj_n.shape}")
                print(f"  单步 dqj_n: {dqj_n.shape}")
                print(f"  单步 action_smoothed: {self._action_smoothed.shape}")
                print(f"  单步总维度: {single_obs.shape}")
                print(f"  历史长度: {self._history_len}")
                print(f"  拼接后观测维度: {obs.shape}")
                self._dims_printed = True
            
            return obs
            
        except Exception as e:
            print(f"构建观测失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回零观测作为后备
            return np.zeros(self._obs_dim, dtype=np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """获取控制信号"""
        self._update_joystick_command()
        self._counter += 1
        run_policy = (self._counter % self._n_substeps) == 0

        try:
            if run_policy:
                obs = self.get_obs(data)

                if self._counter == self._n_substeps:
                    assert obs.shape[0] == self._obs_dim, (
                        f"观测维度不匹配！期望 {self._obs_dim}, "
                        f"得到 {obs.shape[0]}"
                    )

                if self.is_locked or self.motor_state == 0:
                    self._target_joint_pos = self._default_angles.copy()
                else:
                    obs_torch = (
                        torch.from_numpy(obs).float().to(self._device).unsqueeze(0)
                    )

                    # 使用标准的 RSL-RL 推理方式：act_inference 方法
                    actions = self._policy(obs_torch)
                    actions = torch.clip(
                        actions,
                        -self._clip_actions,
                        self._clip_actions,
                    )  # pytype: disable=attribute-error
                    actions_np = actions.cpu().numpy().flatten()
                    # 动作平滑
                    if self._action_smoothing > 0.0:
                        self._action_smoothed = self._action_smoothing * self._action_smoothed + (1.0 - self._action_smoothing) * actions_np
                    else:
                        self._action_smoothed = actions_np.copy()
                    # 更新动作历史：当前动作 -> last_action
                    self._last_action_policy = self._current_action_policy.copy()
                    self._current_action_policy = actions_np.copy()
                    self._scale_actions_to_targets(self._action_smoothed)
            else:
                if self.is_locked or self.motor_state == 0:
                    self._target_joint_pos = self._default_angles.copy()

            torque_cmd = self._compute_torque_command(data)

            data.ctrl[:] = torque_cmd

        except Exception as e:
            print(f"PyTorch推理或力矩计算错误: {e}")
            self._target_joint_pos = self._default_angles.copy()
            torque_cmd = self._compute_torque_command(data)
            data.ctrl[:] = torque_cmd

    def cleanup(self):
        """清理资源"""
        if self._joystick_process and self._joystick_process.is_alive():
            self._joystick_process.terminate()
            self._joystick_process.join(timeout=2.0)
        print("控制器资源已清理")


def load_callback(model=None, data=None):
    """加载回调函数 - 使用 yaml 配置"""
    global _GLOBAL_CONFIG, _GLOBAL_POLICY_PATH, _GLOBAL_XML_PATH  # noqa: PLW0603
    
    try:
        mujoco.set_mjcb_control(None)

        # 从 yaml 配置加载参数
        config = _GLOBAL_CONFIG
        policy_path = _GLOBAL_POLICY_PATH
        xml_path = _GLOBAL_XML_PATH
        
        simulation_dt = float(config["simulation_dt"])
        control_decimation = int(config["control_decimation"])
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        dof_pos_scale = float(config.get("dof_pos_scale", 1.0))
        dof_vel_scale = float(config.get("dof_vel_scale", 0.05))
        ang_vel_scale = float(config.get("ang_vel_scale", 0.25))
        action_scale = float(config["action_scale"])
        action_smoothing = float(config.get("action_smoothing", 0.0))
        torque_limit = float(config.get("torque_limit", 100.0))
        joint2motor_idx = config.get("joint2motor_idx", None)
        if joint2motor_idx is not None:
            joint2motor_idx = np.array(joint2motor_idx, dtype=np.int32)
        
        # 从 yaml 配置读取 cmd_scale，如果不存在则使用默认值
        cmd_scale = config.get("cmd_scale", None)
        if cmd_scale is not None:
            cmd_scale = np.array(cmd_scale, dtype=np.float32)
            print(f"使用 yaml 配置中的 cmd_scale: {cmd_scale}")
        else:
            cmd_scale = ALIENGO_COMMAND_SCALE
            print(f"使用默认的 ALIENGO_COMMAND_SCALE: {cmd_scale}")
        
        print(f"配置参数:")
        print(f"  simulation_dt: {simulation_dt}")
        print(f"  control_decimation: {control_decimation}")
        print(f"  action_scale: {action_scale}")
        print(f"  dof_pos_scale: {dof_pos_scale}")
        print(f"  dof_vel_scale: {dof_vel_scale}")
        print(f"  ang_vel_scale: {ang_vel_scale}")
        print(f"  cmd_scale: {cmd_scale}")
        
        # 加载模型
        if not Path(xml_path).exists():
            raise FileNotFoundError(f"找不到指定的XML场景文件: {xml_path}")

        model = mujoco.MjModel.from_xml_path(xml_path)
        model.opt.timestep = simulation_dt

        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)

        ctrl_dt = simulation_dt * control_decimation
        n_substeps = control_decimation

        # 创建PyTorch控制器
        default_angles_from_model, joint_qpos_indices, joint_qvel_indices, actuator_joint_names = (
            compute_aliengo_joint_metadata(model)
        )
        
        # 如果 yaml 中提供了 default_angles，使用 yaml 的；否则使用模型计算的
        if default_angles.size == len(default_angles_from_model):
            print("使用 yaml 配置中的默认关节角")
            default_angles_to_use = default_angles
        else:
            print("使用模型计算的默认关节角")
            default_angles_to_use = default_angles_from_model
        
        print("使用默认关节角:")
        for name, value in zip(actuator_joint_names, default_angles_to_use):
            print(f"  {name}: {value:.3f} rad")

        # 计算平均 kp 和 kd（如果配置中是数组）
        kp_mean = float(np.mean(kps)) if kps.size > 0 else 40.0
        kd_mean = float(np.mean(kds)) if kds.size > 0 else 2.0

        policy = PyTorchControllerWithJoystick(
            model=model,
            policy_path=policy_path,
            default_angles=default_angles_to_use,
            joint_qpos_indices=joint_qpos_indices,
            joint_qvel_indices=joint_qvel_indices,
            actuator_joint_names=actuator_joint_names,
            n_substeps=n_substeps,
            action_scale=action_scale,
            device="cpu",
            command_ranges=ALIENGO_COMMAND_RANGES,
            command_scale=cmd_scale,
            obs_scales=ALIENGO_OBS_SCALES,
            kp=kp_mean,
            kd=kd_mean,
            hip_reduction=ALIENGO_CFG.control.hip_reduction,
            torque_limit=torque_limit,
            command_alpha=0.95,
            dof_pos_scale=dof_pos_scale,
            dof_vel_scale=dof_vel_scale,
            ang_vel_scale=ang_vel_scale,
            action_smoothing=action_smoothing,
            joint2motor_idx=joint2motor_idx,
        )

        mujoco.set_mjcb_control(policy.get_control)

        return model, data
    except Exception as exc:
        print("加载MuJoCo环境时发生错误:", exc)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # global _GLOBAL_CONFIG, _GLOBAL_POLICY_PATH, _GLOBAL_XML_PATH  # noqa: PLW0603
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, nargs="?", default="go2.yaml", 
                       help="配置文件名（默认: go2.yaml，会在脚本目录查找）")
    args = parser.parse_args()
    
    # 处理配置文件路径
    cfg_path = args.config_file
    if not os.path.isabs(cfg_path):
        # 如果不是绝对路径，先在当前目录查找，再在脚本目录查找
        if not os.path.exists(cfg_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path_in_script_dir = os.path.join(script_dir, cfg_path)
            if os.path.exists(cfg_path_in_script_dir):
                cfg_path = cfg_path_in_script_dir
            else:
                raise FileNotFoundError(
                    f"配置文件未找到: {args.config_file}\n"
                    f"  已尝试: {os.path.abspath(args.config_file)}\n"
                    f"  已尝试: {cfg_path_in_script_dir}"
                )
        else:
            cfg_path = os.path.abspath(cfg_path)
    
    print(f"加载配置文件: {cfg_path}")
    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 从配置中提取路径（支持相对路径和绝对路径）
    policy_path = config["policy_path"]
    xml_path = config["xml_path"]
    
    # 处理相对路径
    if not os.path.isabs(policy_path):
        policy_path = os.path.join(_REPO_ROOT.as_posix(), policy_path)
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(_REPO_ROOT.as_posix(), xml_path)
    
    policy_path = os.path.abspath(policy_path)
    xml_path = os.path.abspath(xml_path)
    
    # 设置全局变量
    _GLOBAL_CONFIG = config
    _GLOBAL_POLICY_PATH = policy_path
    _GLOBAL_XML_PATH = xml_path
    
    print("=== 北通遥控器控制的机器人仿真 (PyTorch版本) ===")
    print("说明:")
    print("  使用北通遥控器控制机器人，命令由用户输入")
    print("  遥控器逻辑运行在独立进程中，避免阻塞仿真")
    print(
        "  命令范围: "
        f"x[{ALIENGO_COMMAND_RANGES[0][0]}, {ALIENGO_COMMAND_RANGES[0][1]}], "
        f"y[{ALIENGO_COMMAND_RANGES[1][0]}, {ALIENGO_COMMAND_RANGES[1][1]}], "
        f"yaw[{ALIENGO_COMMAND_RANGES[2][0]}, {ALIENGO_COMMAND_RANGES[2][1]}]"
    )
    print("  模型类型: PyTorch (.pt) 模型 - 自动推断网络结构")
    print(f"  策略路径: {policy_path}")
    print(f"  XML路径: {xml_path}")
    print("-" * 50)

    policy_controller = None
    
    try:
        # 启动仿真
        viewer.launch(loader=load_callback)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()
    finally:
        print("清理资源...")
        # 注意：由于viewer.launch的限制，我们无法直接访问policy对象
        # 清理工作会在程序退出时自动进行