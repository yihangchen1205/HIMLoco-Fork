# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import warnings

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

# NOTE:
# Importing the training environments pulls in isaacgym, which is not strictly
# required when we only need configuration objects (e.g. for MuJoCo runtime
# deployment).  To keep config imports usable without isaacgym we gate the
# heavy legged_robot import behind a try/except block.
try:
    from .base.legged_robot import LeggedRobot
    from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
    from .go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO
    from .aliengo.aliengo_config import AlienGoRoughCfg, AlienGoRoughCfgPPO
    from legged_gym.utils.task_registry import task_registry

    task_registry.register("a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO())
    task_registry.register("go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO())
    task_registry.register("aliengo", LeggedRobot, AlienGoRoughCfg(), AlienGoRoughCfgPPO())
except ModuleNotFoundError as exc:
    # Skip task registration if isaacgym (or another hard dependency) is absent.
    # Config modules remain importable so downstream MuJoCo-only tooling works.
    warnings.warn(
        f"Skipping legged_gym task registration because dependency "
        f"'{exc.name}' is missing. Config modules can still be imported."
    )
    LeggedRobot = None