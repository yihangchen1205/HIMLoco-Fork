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
"""使用固定命令的PyTorch策略部署到C MuJoCo."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import sys
import os
import traceback
import torch
from typing import Dict, Optional, Sequence

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

ALIENGO_XML_PATH = epath.Path(
    os.environ.get(
        "ALIENGO_XML_PATH",
        (_HERE / "aliengo_mj_description-master/xml/scene_mjx_flat_terrain.xml").as_posix(),
    )
).expanduser()
ALIENGO_POLICY_PATH = epath.Path(
    os.environ.get(
        "ALIENGO_POLICY_PATH",
        (_REPO_ROOT / "legged_gym/logs/Nov19_15-36-27_/model_2040.pt").as_posix(),
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
_GLOBAL_GRAVITY_VECTOR = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _project_gravity_to_body(body_xmat_row: np.ndarray) -> np.ndarray:
    """Project world gravity into the body frame using MuJoCo's xmat slice."""
    rot = np.asarray(body_xmat_row, dtype=np.float64).reshape(3, 3)
    return rot.T @ _GLOBAL_GRAVITY_VECTOR


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
    """使用固定命令的PyTorch控制器."""

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
        fixed_command: Optional[np.ndarray] = None,
    ):
        self._model = model
        self._device = device
        self._base_body_id = self._resolve_mj_name(
            mujoco.mjtObj.mjOBJ_BODY, ("trunk", "base", "base_link")
        )

        self._action_scale = action_scale

        # 使用训练时相同的默认关节角
        self._default_angles = default_angles.astype(np.float32)
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

        # 动作记录（当前动作和上一步动作）
        self._current_action = np.zeros_like(default_angles, dtype=np.float32)
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)

        self._counter = 0
        self._n_substeps = n_substeps

        # 速度缩放参数
        self._vel_scale_x = vel_scale_x
        self._vel_scale_y = vel_scale_y
        self._vel_scale_rot = vel_scale_rot

        # 控制命令 - 使用固定命令
        if fixed_command is not None:
            fixed_cmd = np.asarray(fixed_command, dtype=np.float32)
            if fixed_cmd.shape != (3,):
                raise ValueError(f"fixed_command必须是3维数组 [x_vel, y_vel, angular_vel], 得到形状: {fixed_cmd.shape}")
            # 截断到有效范围
            self.command = np.clip(fixed_cmd, self._command_lower, self._command_upper)
            print(f"使用固定命令: [{self.command[0]:.3f}, {self.command[1]:.3f}, {self.command[2]:.3f}]")
        else:
            # 默认静止状态
            self.command = np.zeros(3, dtype=np.float32)
            print("使用默认静止命令: [0.0, 0.0, 0.0]")
        
        self.is_locked = False
        self.motor_state = 1

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

    def _scale_actions_to_targets(self, actions_np: np.ndarray) -> None:
        """将策略输出转换为目标关节角."""
        actions_scaled = actions_np * self._action_scale
        # 应用hip_reduction，与Isaac Gym一致：只对hip关节索引[0, 3, 6, 9]应用
        if self._hip_reduction != 1.0 and len(self._hip_indices) > 0:
            actions_scaled[self._hip_indices] *= self._hip_reduction
        self._target_joint_pos = self._default_angles + actions_scaled

    def _compute_torque_command(self, data: mujoco.MjData) -> np.ndarray:
        """根据当前目标姿态与实际状态计算力矩."""
        joint_pos = data.qpos[self._joint_qpos_indices].astype(np.float32)
        joint_vel = data.qvel[self._joint_qvel_indices].astype(np.float32)
        pos_err = self._target_joint_pos - joint_pos
        vel_err = -joint_vel  # 目标速度为0
        torques = self._kp * pos_err + self._kd * vel_err
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

    def get_obs(self, data: mujoco.MjData) -> np.ndarray:
        """构建与训练时完全一致的观测向量 - 固定45维版本."""
        base_ang_vel = np.asarray(
            data.cvel[self._base_body_id, 3:], dtype=np.float64
        )
        gyro = base_ang_vel.astype(np.float32) * self._obs_scales["ang_vel"]
        gravity = _project_gravity_to_body(
            data.xmat[self._base_body_id]
        ).astype(np.float32)
        # joint_angles = (
        #     data.qpos[self._joint_qpos_indices] - self._default_angles
        # ) * self._obs_scales["dof_pos"]
        joint_angles = (
            data.qpos[self._joint_qpos_indices] - self._default_angles
        ) * self._obs_scales["dof_pos"]

        joint_velocities = data.qvel[self._joint_qvel_indices] * self._obs_scales[
            "dof_vel"
        ]

        # 使用固定的控制命令
        command = np.clip(self.command, self._command_lower, self._command_upper)
        command_scaled = command * self._command_scale

        # 构建单步观测向量 (45维: 3+3+3+12+12+12)
        # 注意：与Isaac Gym一致，使用当前动作（self._current_action）而不是last_action
        obs_list = [
            command_scaled,           # 3维
            gyro,                     # 3维
            gravity,                  # 3维
            joint_angles,             # 12维
            joint_velocities,         # 12维
            self._current_action,     # 12维 - 使用当前动作，与训练时一致
        ]
        single_obs = np.concatenate(obs_list).astype(np.float32)

        # 维护历史观测缓冲，与训练时相同：最新观测在最前面
        # 注意：与Isaac Gym和pdandrl.py一致，历史缓冲区初始化为全零
        # 第一次调用时，历史缓冲区仍然是全零，只有当前观测会被更新
        # 正常更新：将历史向后移动，新观测放在最前面
        self._obs_history[1:] = self._obs_history[:-1]
        self._obs_history[0] = single_obs
        obs = self._obs_history.reshape(-1).astype(np.float32).copy()
        np.clip(obs, -self._clip_obs, self._clip_obs, out=obs)

        # 如果这是第一次调用，打印各组件维度用于调试
        if not hasattr(self, '_dims_printed'):
            print("观测组件维度:")
            print(f"  单步 command_scaled: {command_scaled.shape}")
            print(f"  单步 gyro: {gyro.shape}")
            print(f"  单步 gravity: {gravity.shape}")
            print(f"  单步 joint_angles: {joint_angles.shape}")
            print(f"  单步 joint_velocities: {joint_velocities.shape}")
            print(f"  单步 current_action: {self._current_action.shape}")
            print(f"  单步总维度: {single_obs.shape}")
            print(f"  历史长度: {self._history_len}")
            print(f"  拼接后观测维度: {obs.shape}")
            self._dims_printed = True
            
        # 验证观测维度
        assert obs.shape[0] == self._obs_dim, (
            f"观测维度应该是{self._obs_dim}，但得到了{obs.shape[0]}"
        )

        return obs

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """获取控制信号"""
        self._counter += 1
        run_policy = (self._counter % self._n_substeps) == 0

        try:
            if run_policy:
                obs = self.get_obs(data)

                if self._counter == self._n_substeps:
                    print(f"观测向量维度: {obs.shape}")
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
                    # 更新动作历史：当前动作 -> last_action
                    self._last_action = self._current_action.copy()
                    self._current_action = actions_np.copy()
                    self._scale_actions_to_targets(actions_np)
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
        print("控制器资源已清理")


def load_callback(model=None, data=None):
    """加载回调函数"""
    try:
        mujoco.set_mjcb_control(None)

        # 环境配置（无需依赖mujoco_playground）
        class EnvConfig:
            Kd = 2.0
            Kp = 40.0
            ctrl_dt = ALIENGO_CFG.control.decimation * ALIENGO_CFG.sim.dt
            sim_dt = ALIENGO_CFG.sim.dt
            action_scale = ALIENGO_CFG.control.action_scale
            history_len = 6
            torque_limit = 24.0


        env_config = EnvConfig()
        print(f"env_config.Kp: {env_config.Kp}")
        print(f"env_config.Kd: {env_config.Kd}")
        print(f"env_config.action_scale: {env_config.action_scale}")
        print(f"env_config.history_len: {env_config.history_len}")
        print(f"env_config.ctrl_dt: {env_config.ctrl_dt}")
        print(f"env_config.sim_dt: {env_config.sim_dt}")
        # exit(0)
        # print(f"env_config.n_substeps: {env_config.n_substeps}")
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

        # 创建PyTorch控制器
        default_angles, joint_qpos_indices, joint_qvel_indices, actuator_joint_names = (
            compute_aliengo_joint_metadata(model)
        )
        print("使用AlienGo配置默认关节角:")
        for name, value in zip(actuator_joint_names, default_angles):
            print(f"  {name}: {value:.3f} rad")

        # 设置固定命令 [x_vel, y_vel, angular_vel]
        # 例如: [1.0, 0.0, 0.0] 表示向前移动，速度为1.0 m/s
        fixed_command = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 静止状态
        
        policy = PyTorchControllerWithJoystick(
            model=model,
            policy_path=ALIENGO_POLICY_PATH.as_posix(),  # PyTorch .pt模型文件
            default_angles=default_angles,
            joint_qpos_indices=joint_qpos_indices,
            joint_qvel_indices=joint_qvel_indices,
             actuator_joint_names=actuator_joint_names,
            n_substeps=n_substeps,
            action_scale=env_config.action_scale,
            device="cpu",  # 可以改为"cuda"如果有GPU
            command_ranges=ALIENGO_COMMAND_RANGES,
            command_scale=ALIENGO_COMMAND_SCALE,
            obs_scales=ALIENGO_OBS_SCALES,
            kp=env_config.Kp,
            kd=env_config.Kd,
            hip_reduction=ALIENGO_CFG.control.hip_reduction,
            torque_limit=env_config.torque_limit,
            fixed_command=fixed_command,  # 使用固定命令
        )

        mujoco.set_mjcb_control(policy.get_control)

        return model, data
    except Exception as exc:
        print("加载MuJoCo环境时发生错误:", exc)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("=== 使用固定命令的机器人仿真 (PyTorch版本) ===")
    print("说明:")
    print("  使用固定命令控制机器人，命令在代码中设置")
    print("  可在 load_callback 函数中修改 fixed_command 参数来改变命令")
    print(
        "  命令范围: "
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