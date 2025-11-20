#!/usr/bin/env python3
# Copyright 2025.
#
# 在 MuJoCo 中运行与 legged_gym/scripts/play.py 行为一致的测试脚本。
# 通过直接复用 deploy_mujoco/play_joystick_rsl_rl_fixed.py 中的控制器，
# 读取与训练时一致的策略，按 play.py 的流程执行推理、记录日志并绘制状态。
# pylint: disable=import-error
# pyright: reportMissingImports=false

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import mujoco
from mujoco import viewer
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str((_REPO_ROOT / "legged_gym")) not in sys.path:
    sys.path.insert(0, str((_REPO_ROOT / "legged_gym")))
if str((_REPO_ROOT / "rsl_rl")) not in sys.path:
    sys.path.insert(0, str((_REPO_ROOT / "rsl_rl")))

from legged_gym.utils import get_args, Logger, export_policy_as_jit, task_registry  # pylint: disable=import-error

import play_joystick_rsl_rl_fixed as mujoco_ctrl  # noqa: E402

EXPORT_POLICY = True
RECORD_FRAMES = False
MOVE_CAMERA = False


def _ensure_mjpython_on_macos() -> None:
    """Re-exec the script with `mjpython` on macOS if needed."""
    if sys.platform != "darwin":
        return
    exe_name = Path(sys.executable).name
    if exe_name == "mjpython" or os.environ.get("HIMLOCO_SKIP_MJPYTHON") == "1":
        return

    mjpython_path = shutil.which("mjpython")
    if mjpython_path is None:
        raise RuntimeError(
            "`mujoco.viewer` requires the script to run under `mjpython` on macOS, "
            "but `mjpython` was not found in PATH. Install MuJoCo's mjpython wrapper "
            "or run this script via `python -m mujoco.viewer` compatible environment."
        )

    print("MuJoCo viewer on macOS requires `mjpython`. Re-launching via:", mjpython_path)
    env = os.environ.copy()
    env["HIMLOCO_SKIP_MJPYTHON"] = "1"
    cmd = [mjpython_path, *sys.argv]
    raise SystemExit(subprocess.call(cmd, env=env))


def _set_camera(mj_viewer: Any, position: Sequence[float], lookat: Sequence[float]) -> None:
    mj_viewer.cam.lookat[:] = lookat
    delta = np.array(lookat) - np.array(position)
    distance = np.linalg.norm(delta)
    if distance > 1e-6:
        mj_viewer.cam.distance = distance
    mj_viewer.cam.azimuth = np.degrees(np.arctan2(delta[1], delta[0]))
    mj_viewer.cam.elevation = np.degrees(np.arctan2(delta[2], np.linalg.norm(delta[:2])))


def _get_joint_packets(
    controller: mujoco_ctrl.PyTorchControllerWithJoystick,
    data: mujoco.MjData,
    joint_qpos_indices: np.ndarray,
    joint_qvel_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    joint_pos = data.qpos[joint_qpos_indices].astype(np.float32)
    joint_vel = data.qvel[joint_qvel_indices].astype(np.float32)
    joint_pos_policy = mujoco_ctrl._swap_leg_groups(joint_pos)  # pylint: disable=protected-access
    joint_vel_policy = mujoco_ctrl._swap_leg_groups(joint_vel)  # pylint: disable=protected-access
    target_policy = mujoco_ctrl._swap_leg_groups(controller._target_joint_pos)  # pylint: disable=protected-access
    torque_policy = mujoco_ctrl._swap_leg_groups(controller._last_torque)  # pylint: disable=protected-access
    return {
        "pos": joint_pos_policy,
        "vel": joint_vel_policy,
        "target": target_policy,
        "torque": torque_policy,
    }


def _get_base_states(controller: mujoco_ctrl.PyTorchControllerWithJoystick, data: mujoco.MjData) -> Dict[str, np.ndarray]:
    base_id = controller._base_body_id  # pylint: disable=protected-access
    base_cvel = data.cvel[base_id]
    lin_vel = base_cvel[3:].astype(np.float32)
    ang_vel = base_cvel[:3].astype(np.float32)
    return {"lin": lin_vel, "ang": ang_vel}


def _get_contact_forces(foot_body_ids: Sequence[int], data: mujoco.MjData) -> np.ndarray:
    forces = []
    for bid in foot_body_ids:
        forces.append(data.cfrc_ext[bid, :3].astype(np.float32))
    return np.stack(forces, axis=0)


def _resolve_feet(controller: mujoco_ctrl.PyTorchControllerWithJoystick) -> List[int]:
    foot_name_candidates = {
        "FR": ("FR_foot",),
        "FL": ("FL_foot",),
        "RR": ("RR_foot",),
        "RL": ("RL_foot",),
    }
    ids = []
    for _, names in foot_name_candidates.items():
        ids.append(
            controller._resolve_mj_name(  # pylint: disable=protected-access
                mujoco.mjtObj.mjOBJ_BODY, names
            )
        )
    return ids


def _reset_sim(model: mujoco.MjModel, data: mujoco.MjData, controller: mujoco_ctrl.PyTorchControllerWithJoystick) -> None:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    controller._obs_history[:] = 0.0  # pylint: disable=protected-access
    controller._current_action_policy[:] = 0.0  # pylint: disable=protected-access
    controller._last_action_policy[:] = 0.0  # pylint: disable=protected-access
    controller.command[:] = 0.0
    controller._counter = 0  # pylint: disable=protected-access


def play_mujoco(args, x_vel: float = 1.0, y_vel: float = 0.0, yaw_vel: float = 0.0) -> None:
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False

    if not mujoco_ctrl.ALIENGO_XML_PATH.exists():
        raise FileNotFoundError(f"找不到 MuJoCo XML: {mujoco_ctrl.ALIENGO_XML_PATH}")

    model = mujoco.MjModel.from_xml_path(mujoco_ctrl.ALIENGO_XML_PATH.as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = env_cfg.control.decimation * env_cfg.sim.dt
    sim_dt = env_cfg.sim.dt
    n_substeps = int(round(ctrl_dt / sim_dt))

    default_angles, joint_qpos_indices, joint_qvel_indices, actuator_joint_names = (
        mujoco_ctrl.compute_aliengo_joint_metadata(model)
    )

    controller = mujoco_ctrl.PyTorchControllerWithJoystick(
        model=model,
        policy_path=mujoco_ctrl.ALIENGO_POLICY_PATH.as_posix(),
        default_angles=default_angles,
        joint_qpos_indices=joint_qpos_indices,
        joint_qvel_indices=joint_qvel_indices,
        actuator_joint_names=actuator_joint_names,
        n_substeps=n_substeps,
        action_scale=env_cfg.control.action_scale,
        device=args.rl_device,
        command_ranges=mujoco_ctrl.ALIENGO_COMMAND_RANGES,
        command_scale=mujoco_ctrl.ALIENGO_COMMAND_SCALE,
        obs_scales=mujoco_ctrl.ALIENGO_OBS_SCALES,
        kp=env_cfg.control.stiffness.get("joint", 40.0),
        kd=env_cfg.control.damping.get("joint", 2.0),
        hip_reduction=env_cfg.control.hip_reduction,
        torque_limit=24.0,
        fixed_command=None,
    )

    mujoco.set_mjcb_control(None)

    foot_body_ids = _resolve_feet(controller)
    logger = Logger(ctrl_dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = int(np.ceil(env_cfg.env.episode_length_s / ctrl_dt)) + 1
    img_idx = 0

    if EXPORT_POLICY:
        export_dir = _REPO_ROOT / "legged_gym" / "logs" / train_cfg.runner.experiment_name / "exported" / "policies"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_policy_as_jit(controller._policy_network, export_dir.as_posix())  # pylint: disable=protected-access

    total_steps = 10 * int(np.ceil(env_cfg.env.episode_length_s / ctrl_dt))
    episode_step = 0

    with viewer.launch_passive(model, data) as mj_viewer:
        _set_camera(mj_viewer, env_cfg.viewer.pos, env_cfg.viewer.lookat)
        camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
        camera_velocity = np.array([1.0, 1.0, 0.0])

        for step in range(total_steps):
            controller.command[:] = np.array([x_vel, y_vel, yaw_vel], dtype=np.float32)
            obs = controller.get_obs(data)
            obs_torch = torch.from_numpy(obs).float().to(controller._device).unsqueeze(0)  # pylint: disable=protected-access
            actions = controller._policy(obs_torch)  # pylint: disable=protected-access
            actions = torch.clip(
                actions, -controller._clip_actions, controller._clip_actions  # pylint: disable=protected-access
            )
            actions_np = actions.detach().cpu().numpy().flatten()
            controller._last_action_policy = controller._current_action_policy.copy()  # pylint: disable=protected-access
            controller._current_action_policy = actions_np.copy()  # pylint: disable=protected-access
            controller._scale_actions_to_targets(actions_np)  # pylint: disable=protected-access
            torque_cmd = controller._compute_torque_command(data)  # pylint: disable=protected-access

            for _ in range(n_substeps):
                data.ctrl[:] = torque_cmd
                mujoco.mj_step(model, data)
                mj_viewer.sync()

            joint_packets = _get_joint_packets(controller, data, joint_qpos_indices, joint_qvel_indices)
            base_states = _get_base_states(controller, data)
            contact_forces = _get_contact_forces(foot_body_ids, data)

            if RECORD_FRAMES and step % 2 == 0:
                export_frames = _REPO_ROOT / "legged_gym" / "logs" / train_cfg.runner.experiment_name / "exported" / "frames"
                export_frames.mkdir(parents=True, exist_ok=True)
                filename = export_frames / f"{img_idx}.png"
                mj_viewer.save_snapshot(str(filename))
                img_idx += 1

            if MOVE_CAMERA:
                camera_position += camera_velocity * ctrl_dt
                _set_camera(mj_viewer, camera_position, camera_position + camera_direction)

            if step < stop_state_log:
                logger.log_states(
                    {
                        "dof_pos_target": joint_packets["target"][joint_index].item(),
                        "dof_pos": joint_packets["pos"][joint_index].item(),
                        "dof_vel": joint_packets["vel"][joint_index].item(),
                        "dof_torque": joint_packets["torque"][joint_index].item(),
                        "command_x": x_vel,
                        "command_y": y_vel,
                        "command_yaw": yaw_vel,
                        "base_vel_x": base_states["lin"][0].item(),
                        "base_vel_y": base_states["lin"][1].item(),
                        "base_vel_z": base_states["lin"][2].item(),
                        "base_vel_yaw": base_states["ang"][2].item(),
                        "contact_forces_z": contact_forces[:, 2].copy(),
                    }
                )
            elif step == stop_state_log:
                logger.plot_states()

            infos = {"episode": {}}
            if 0 < step < stop_rew_log and infos["episode"]:
                logger.log_rewards(infos["episode"], 1)
            elif step == stop_rew_log:
                logger.print_rewards()

            episode_step += 1
            if episode_step >= int(np.ceil(env_cfg.env.episode_length_s / ctrl_dt)):
                _reset_sim(model, data, controller)
                episode_step = 0


if __name__ == "__main__":
    _ensure_mjpython_on_macos()
    args = get_args()
    play_mujoco(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0)

