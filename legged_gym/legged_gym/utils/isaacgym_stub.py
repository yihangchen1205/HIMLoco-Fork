"""Lightweight stub for the Isaac Gym API used by helper utilities.

This allows importing legged_gym configuration and logging utilities on
machines where the proprietary Isaac Gym package is not installed (for example
when running MuJoCo-only deployments).  The stub only implements the small
subset of functionality accessed by legged_gym.utils.helpers:

* gymapi.SimParams along with the SIM_PHYSX/SIM_FLEX constants.
* gymutil.parse_arguments for CLI parsing with custom parameters.
* gymutil.parse_sim_config for mapping a dictionary onto SimParams objects.

The real Isaac Gym bindings expose many more capabilities.  This stub is not
intended to run Isaac environments; it simply prevents import-time failures
when we only need configuration objects and loggers.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any, Dict


class _PhysxParams(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(use_gpu=False, num_subscenes=1, num_threads=0)


class SimParams(SimpleNamespace):
    """Minimal drop-in replacement for gymapi.SimParams."""

    def __init__(self) -> None:
        super().__init__()
        self.physx = _PhysxParams()
        self.use_gpu_pipeline = False


SIM_PHYSX = "physx"
SIM_FLEX = "flex"


def _deep_update(namespace: SimpleNamespace, cfg: Dict[str, Any]) -> None:
    """Recursively copy values from cfg into namespace attributes."""
    for key, value in cfg.items():
        if isinstance(value, dict):
            child = getattr(namespace, key, None)
            if child is None or not isinstance(child, SimpleNamespace):
                child = SimpleNamespace()
                setattr(namespace, key, child)
            _deep_update(child, value)
        else:
            setattr(namespace, key, value)


def parse_sim_config(cfg: Dict[str, Any], sim_params: SimParams) -> None:
    """Best-effort replacement for gymutil.parse_sim_config."""
    if not isinstance(cfg, dict):
        return
    _deep_update(sim_params, cfg)


def parse_arguments(description: str, custom_parameters=None):
    """Simplified CLI parser mirroring isaacgym.gymutil.parse_arguments."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--physics_engine", default=SIM_PHYSX, choices=[SIM_PHYSX, SIM_FLEX])
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--compute_device_id", type=int, default=0)
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--use_gpu_pipeline", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--subscenes", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=0)

    custom_parameters = custom_parameters or []
    for param in custom_parameters:
        # param dicts originate from legged_gym helper definitions and follow
        # argparse.add_argument naming.
        names = param["name"]
        if isinstance(names, (list, tuple)):
            option_strings = list(names)
        else:
            option_strings = [names]
        # Avoid re-registering the same CLI option (e.g. --headless).
        if any(opt in parser._option_string_actions for opt in option_strings):
            continue
        kwargs = {k: v for k, v in param.items() if k != "name"}
        parser.add_argument(*option_strings, **kwargs)

    args, _ = parser.parse_known_args()
    return args


class _GymUtilNamespace(SimpleNamespace):
    parse_arguments = staticmethod(parse_arguments)
    parse_sim_config = staticmethod(parse_sim_config)


class _GymApiNamespace(SimpleNamespace):
    SimParams = SimParams
    SIM_PHYSX = SIM_PHYSX
    SIM_FLEX = SIM_FLEX


gymapi = _GymApiNamespace()
gymutil = _GymUtilNamespace()

