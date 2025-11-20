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

import os
import copy
import torch
import numpy as np
import random
try:
    from isaacgym import gymapi  # type: ignore
    from isaacgym import gymutil  # type: ignore
except ModuleNotFoundError:
    from . import isaacgym_stub as _isaacgym_stub

    gymapi = _isaacgym_stub.gymapi  # type: ignore
    gymutil = _isaacgym_stub.gymutil  # type: ignore
import torch.nn.functional as F

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "aliengo", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    # args.sim_device_id = args.compute_device_id
    args.sim_device = args.rl_device
    # if args.sim_device=='cuda':
    #     args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'estimator'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterHIM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

# class PolicyExporterLSTM(torch.nn.Module):
#     def __init__(self, actor_critic):
#         super().__init__()
#         self.actor = copy.deepcopy(actor_critic.actor)
#         self.is_recurrent = actor_critic.is_recurrent
#         self.memory = copy.deepcopy(actor_critic.memory.rnn)
#         self.memory.cpu()
#         self.hidden_encoder = copy.deepcopy(actor_critic.hidden_encoder)
#         self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
#         self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

#     def forward(self, x):
#         out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
#         self.hidden_state[:] = h
#         self.cell_state[:] = c
#         latent = self.hidden_encoder(out.squeeze(0))
#         return self.actor(torch.cat((x, latent), dim=1))

#     @torch.jit.export
#     def reset_memory(self):
#         self.hidden_state[:] = 0.
#         self.cell_state[:] = 0.

#     def export(self, path):
#         os.makedirs(path, exist_ok=True)
#         path = os.path.join(path, 'policy_lstm.pt')
#         self.to('cpu')
#         traced_script_module = torch.jit.script(self)
#         traced_script_module.save(path)

class PolicyExporterHIM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)

    def forward(self, obs_history):
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        # Use manual normalization instead of F.normalize for better JIT compatibility
        z_norm = torch.norm(z, p=2.0, dim=-1, keepdim=True)
        z = z / (z_norm + 1e-8)  # Add small epsilon to avoid division by zero
        return self.actor(torch.cat((obs_history[:, 0:45], vel, z), dim=1))

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        self.eval()
        
        # Create a dummy input for tracing
        dummy_input = torch.zeros(1, 270)
        
        # Test if the model works first
        try:
            with torch.no_grad():
                test_output = self(dummy_input)
            print(f"Model test forward pass successful, output shape: {test_output.shape}")
        except Exception as e:
            print(f"Warning: Model forward pass failed: {e}")
            raise
        
        # Since PyTorch 1.10.0 has issues with JIT, we'll use a workaround:
        # Save the model state and create a simple inference wrapper
        # For C++ deployment, you may need to use torch::load() instead of torch::jit::load()
        
        # Try to export using torch.save as fallback
        try:
            # First, try the standard JIT export
            print("Attempting torch.jit.trace...")
            with torch.no_grad():
                traced_module = torch.jit.trace(self, dummy_input, strict=False)
            
            # Check if we got a ScriptModule
            if isinstance(traced_module, torch.jit.ScriptModule):
                traced_module.save(path)
                print(f"Successfully exported policy using JIT trace to: {path}")
                return
            else:
                print(f"Warning: trace() returned {type(traced_module)}, not ScriptModule")
                print("This appears to be a PyTorch 1.10.0 compatibility issue.")
                print("Falling back to torch.save()...")
        except Exception as e:
            print(f"JIT trace failed: {e}")
            print("Falling back to torch.save()...")
        
        # Fallback: Save model state dict and architecture
        # Note: This requires Python runtime for loading, not pure C++
        try:
            # Save the full model (requires Python to load)
            model_path = os.path.join(os.path.dirname(path), 'policy_model.pt')
            torch.save(self.state_dict(), model_path)
            print(f"Saved model state dict to: {model_path}")
            
            # Also try to save the entire model
            full_model_path = os.path.join(os.path.dirname(path), 'policy_full.pt')
            torch.save(self, full_model_path)
            print(f"Saved full model to: {full_model_path}")
            
            # For C++ deployment, you'll need to reconstruct the model architecture
            # and load the state dict, or use a different export method
            print("\n" + "="*60)
            print("WARNING: JIT export failed due to PyTorch 1.10.0 compatibility issue.")
            print("The model has been saved using torch.save() instead.")
            print("For C++ deployment, you may need to:")
            print("1. Upgrade PyTorch to a newer version, OR")
            print("2. Manually reconstruct the model in C++ and load the state dict, OR")
            print("3. Use ONNX export instead")
            print("="*60)
            
            # Still try to create a minimal JIT export by exporting submodules
            self._export_separate_modules(path)
            
        except Exception as e2:
            raise RuntimeError(f"All export methods failed. JIT: {e if 'e' in locals() else 'Unknown'}, Fallback: {e2}")
    
    def _export_separate_modules(self, base_path):
        """Try to export actor and estimator as separate modules"""
        try:
            base_dir = os.path.dirname(base_path)
            
            # Export actor
            actor_path = os.path.join(base_dir, 'actor.pt')
            actor_dummy = torch.zeros(1, 64)
            
            class ActorWrapper(torch.nn.Module):
                def __init__(self, actor):
                    super().__init__()
                    self.actor = actor
                def forward(self, x):
                    return self.actor(x)
            
            actor_wrapper = ActorWrapper(self.actor)
            actor_wrapper.to('cpu').eval()
            
            with torch.no_grad():
                actor_traced = torch.jit.trace(actor_wrapper, actor_dummy, strict=False)
            
            if isinstance(actor_traced, torch.jit.ScriptModule):
                actor_traced.save(actor_path)
                print(f"Exported actor to: {actor_path}")
            else:
                torch.save(actor_wrapper.state_dict(), actor_path)
                print(f"Saved actor state dict to: {actor_path}")
            
            # Export estimator
            estimator_path = os.path.join(base_dir, 'estimator.pt')
            estimator_dummy = torch.zeros(1, 270)
            
            class EstimatorWrapper(torch.nn.Module):
                def __init__(self, estimator):
                    super().__init__()
                    self.estimator = estimator
                def forward(self, x):
                    return self.estimator(x)
            
            estimator_wrapper = EstimatorWrapper(self.estimator)
            estimator_wrapper.to('cpu').eval()
            
            with torch.no_grad():
                estimator_traced = torch.jit.trace(estimator_wrapper, estimator_dummy, strict=False)
            
            if isinstance(estimator_traced, torch.jit.ScriptModule):
                estimator_traced.save(estimator_path)
                print(f"Exported estimator to: {estimator_path}")
            else:
                torch.save(estimator_wrapper.state_dict(), estimator_path)
                print(f"Saved estimator state dict to: {estimator_path}")
                
        except Exception as e:
            print(f"Separate module export failed: {e}")
    
    
