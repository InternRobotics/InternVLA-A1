#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import importlib

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.qwenvla.configuration_qwenvla import QwenVLAConfig
from lerobot.policies.InternVLA_A1_3B.configuration_qwena1 import QwenA1Config
from lerobot.policies.internvla.configuration_internvla import InternVLAConfig
from lerobot.policies.InternVLA_A1_2B.configuration_a1 import InternVLAA1Config
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pretrained import PreTrainedPolicy


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """
    Retrieves a policy class by its registered name.

    This function uses dynamic imports to avoid loading all policy classes into memory
    at once, improving startup time and reducing dependencies.

    Args:
        name: The name of the policy. Supported names are "tdmpc", "diffusion", "act",
              "vqbet", "pi0", "pi05", "sac", "reward_classifier", "smolvla".

    Returns:
        The policy class corresponding to the given name.

    Raises:
        NotImplementedError: If the policy name is not recognized.
    """
    if name == "qwenvla":
        from lerobot.policies.qwenvla.modeling_qwenvla import QwenVLAPolicy

        return QwenVLAPolicy
    elif name == "qwena1":
        from lerobot.policies.InternVLA_A1_3B.modeling_qwena1 import QwenA1Policy

        return QwenA1Policy
    elif name == "internvla":
        from lerobot.policies.internvla.modeling_internvla import InternVLAPolicy

        return InternVLAPolicy
    
    elif name == "internvla_a1":
        from lerobot.policies.InternVLA_A1_2B.modeling_a1 import InternVLAA1Policy

        return InternVLAA1Policy

    elif name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy

    else:
        try:
            return _get_policy_cls_from_policy_name(name=name)
        except Exception as e:
            raise ValueError(f"Policy type '{name}' is not available.") from e


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """
    Instantiates a policy configuration object based on the policy type.

    This factory function simplifies the creation of policy configuration objects by
    mapping a string identifier to the corresponding config class.

    Args:
        policy_type: The type of the policy. Supported types include "tdmpc",
                     "diffusion", "act", "vqbet", "pi0", "pi05", "sac", "smolvla",
                     "reward_classifier".
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        An instance of a `PreTrainedConfig` subclass.

    Raises:
        ValueError: If the `policy_type` is not recognized.
    """
    if policy_type == "qwenvla":
        return QwenVLAConfig(**kwargs)
    elif policy_type == "qwena1":
        return QwenA1Config(**kwargs)
    elif policy_type == "internvla":
        return InternVLAConfig(**kwargs)
    elif policy_type == "internvla_a1":
        return InternVLAA1Config(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    else:
        try:
            config_cls = PreTrainedConfig.get_choice_class(policy_type)
            return config_cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Policy type '{policy_type}' is not available.") from e


def make_policy(
    cfg: PreTrainedConfig,
) -> PreTrainedPolicy:
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created. If `cfg.pretrained_path` is
             set, the policy will be loaded with weights from that path.
    Returns:
        An instantiated and device-placed policy model.

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided.
        NotImplementedError: If attempting to use an unsupported policy-backend
                             combination (e.g., VQBeT with 'mps').
    """
    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    return policy


def _get_policy_cls_from_policy_name(name: str) -> type[PreTrainedConfig]:
    """Get policy class from its registered name using dynamic imports.

    This is used as a helper function to import policies from 3rd party lerobot plugins.

    Args:
        name: The name of the policy.
    Returns:
        The policy class corresponding to the given name.
    """
    if name not in PreTrainedConfig.get_known_choices():
        raise ValueError(
            f"Unknown policy name '{name}'. Available policies: {PreTrainedConfig.get_known_choices()}"
        )

    config_cls = PreTrainedConfig.get_choice_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix("Config")  # e.g., DiffusionConfig -> Diffusion
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention."
            f"Make sure it ends with 'Config'!"
        )
    cls_name = model_name + "Policy"  # e.g., DiffusionConfig -> DiffusionPolicy
    module_path = config_cls.__module__.replace(
        "configuration_", "modeling_"
    )  # e.g., configuration_diffusion -> modeling_diffusion

    module = importlib.import_module(module_path)
    policy_cls = getattr(module, cls_name)
    return policy_cls
