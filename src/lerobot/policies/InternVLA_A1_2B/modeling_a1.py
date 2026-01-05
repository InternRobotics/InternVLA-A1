#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import builtins
import logging
import os
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from einops import rearrange
from typing_extensions import Unpack

from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.internvl import InternVLForConditionalGeneration

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.InternVLA_A1_2B.cosmos_tokenizer.image_lib import ImageTokenizer
from lerobot.policies.InternVLA_A1_2B.configuration_a1 import InternVLAA1Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.utils import format_big_number
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
    HF_HOME, 
)


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, und_expert, gen_expert, act_expert
):
    models = [und_expert.language_model, gen_expert.model, act_expert.model]
    query_states = []
    key_states = []
    value_states = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states= layer.input_layernorm(hidden_states)  # noqa: PLW2901
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        hidden_states = hidden_states.to(dtype=torch.bfloat16)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = und_expert.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = und_expert.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_qwen2.eager_attention_forward(
        und_expert.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = und_expert.language_model.layers[layer_idx].self_attn.head_dim
    num_attention_heads = und_expert.language_model.layers[layer_idx].self_attn.config.num_attention_heads
    att_output = att_output.reshape(batch_size, -1, 1 * num_attention_heads * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = out_emb + hidden_states
        after_first_residual = out_emb.clone()
        out_emb = layer.post_attention_layernorm(out_emb)
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = out_emb + after_first_residual
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class QwenConfig:  
    """Configuration for Qwen model variants."""

    def __init__(self, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, num_key_value_heads):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads


def get_qwen_config(variant: str) -> QwenConfig:  
    """Returns config for specified gemma variant."""
    num_hidden_layers = int(variant.split('_')[-1][:-1])  # pattern: qwen3_vl_28l or qwen3_xxl
    if variant.startswith("internvl"):
        return QwenConfig(
            hidden_size=896,
            intermediate_size=4864, 
            num_attention_heads=14,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
        )
    elif variant.startswith("qwen2"):
        return QwenConfig(
            hidden_size=896,
            intermediate_size=4864, 
            num_attention_heads=14,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class InternVL3WithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """InternVL3 model with action expert for InternVLA."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["internvl"]()
        vlm_config_hf._vocab_size = 151674  # noqa: SLF001
        vlm_config_hf.text_config.vocab_size = 151674
        vlm_config_hf.text_config.hidden_size = vlm_config.hidden_size
        vlm_config_hf.text_config.intermediate_size = vlm_config.intermediate_size
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_attention_heads
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.num_hidden_layers
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_key_value_heads

        vlm_config_hf.vision_config.hidden_size = 1024              
        vlm_config_hf.vision_config.intermediate_size = 4096        
        vlm_config_hf.vision_config.num_attention_heads = 16        
        vlm_config_hf.vision_config.num_hidden_layers = 24          
        vlm_config_hf.vision_config.moe_intermediate_size = 768
        vlm_config_hf.vision_config.shared_expert_intermediate_size = 3072
        vlm_config_hf.vision_config.drop_path_rate = 0.1
        vlm_config_hf.vision_config.norm_type = "layer_norm"
        vlm_config_hf.vision_config.qk_normalization = False
        vlm_config_hf.vision_config.qkv_bias = True

        # self.internvl = InternVLForConditionalGeneration(config=vlm_config_hf)
        self.und_expert = InternVLForConditionalGeneration.from_pretrained(
            os.path.join(HF_HOME, "hub", "OpenGVLab/InternVL3-1B-pt"), 
            config=vlm_config_hf, 
            ignore_mismatched_sizes=True
        )

        gen_expert_config_hf = CONFIG_MAPPING["qwen2"]()
        gen_expert_config_hf.hidden_size=action_expert_config.hidden_size
        gen_expert_config_hf.intermediate_size=action_expert_config.intermediate_size
        gen_expert_config_hf.num_attention_heads=action_expert_config.num_attention_heads
        gen_expert_config_hf.num_hidden_layers=action_expert_config.num_hidden_layers
        gen_expert_config_hf.num_key_value_heads=action_expert_config.num_key_value_heads
        
        # self.qwen2_gen_expert = Qwen2ForCausalLM(config=gen_expert_config_hf)
        # self.qwen2_gen_expert.model.embed_tokens = None
        # self.qwen2_gen_expert.lm_head = None
        self.gen_expert = Qwen2ForCausalLM(config=gen_expert_config_hf)
        self.gen_expert.model.embed_tokens = None
        self.gen_expert.lm_head = None

        action_expert_config_hf = CONFIG_MAPPING["qwen2"]()
        action_expert_config_hf.hidden_size=action_expert_config.hidden_size
        action_expert_config_hf.intermediate_size=action_expert_config.intermediate_size
        action_expert_config_hf.num_attention_heads=action_expert_config.num_attention_heads
        action_expert_config_hf.num_hidden_layers=action_expert_config.num_hidden_layers
        action_expert_config_hf.num_key_value_heads=action_expert_config.num_key_value_heads
        
        # self.qwen2_expert = Qwen2ForCausalLM(config=action_expert_config_hf)
        # self.qwen2_expert.model.embed_tokens = None
        # self.qwen2_expert.lm_head = None
        self.act_expert = Qwen2ForCausalLM(config=action_expert_config_hf)
        self.act_expert.model.embed_tokens = None
        self.act_expert.lm_head = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
            params_to_keep_float32 = [
                # "vision_tower.vision_model.embeddings.patch_embedding.weight",
                # "vision_tower.vision_model.embeddings.patch_embedding.bias",
                # "vision_tower.vision_model.embeddings.position_embedding.weight",
                "input_layernorm",
                "post_attention_layernorm",
                "model.norm",
            ]
            for name, param in self.named_parameters():
                if any(selector in name for selector in params_to_keep_float32):
                    param.data = param.data.to(dtype=torch.float32)
            return
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")



    def embed_image(self, image: torch.Tensor):
        return self.und_expert.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.und_expert.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
    ):
        if inputs_embeds[1] is None and inputs_embeds[2] is None:
            und_output = self.und_expert.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = und_output.past_key_values
            und_output = und_output.last_hidden_state
            gen_output = None
            act_output = None # suffix_output
        elif inputs_embeds[0] is None and inputs_embeds[2] is None:
            gen_output = self.gen_expert.model.forward(
                inputs_embeds=inputs_embeds[1], # (1, 96, 896)
                attention_mask=attention_mask, # (1, 1, 96, 912)
                position_ids=position_ids, # (1, 96)
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = gen_output.past_key_values
            und_output = None
            gen_output = gen_output.last_hidden_state
            act_output = None
        elif inputs_embeds[0] is None and inputs_embeds[1] is None:
            act_output = self.act_expert.model.forward(
                inputs_embeds=inputs_embeds[2], # (1, 51, 896)
                attention_mask=attention_mask, # 
                position_ids=position_ids, # (1, 51)
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = None
            und_output = None
            gen_output = None
            act_output = act_output.last_hidden_state
        else:
            models = [self.und_expert.language_model, self.gen_expert.model, self.act_expert.model]
            num_layers = self.und_expert.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.act_expert.model, "gradient_checkpointing")
                and self.gen_expert.model.gradient_checkpointing
                and self.act_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        und_expert=self.und_expert,
                        gen_expert=self.gen_expert, 
                        act_expert=self.act_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        und_expert=self.und_expert,
                        gen_expert=self.gen_expert, 
                        act_expert=self.act_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb = models[i].norm(hidden_states)
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds)

            past_key_values = None
            und_output = outputs_embeds[0]
            gen_output = outputs_embeds[1]
            act_output = outputs_embeds[2]

        return [und_output, gen_output, act_output], past_key_values


class InternVLAA1(nn.Module):  

    def __init__(self, config: InternVLAA1Config):
        super().__init__()
        self.config = config

        vlm_config = get_qwen_config(config.internvl_variant)
        action_expert_config = get_qwen_config(config.qwen2_variant)

        self.internvl_with_expert = InternVL3WithExpertModel(
            vlm_config,
            action_expert_config,
            precision=config.dtype,
        )

        self.cosmos_tokenizer = ImageTokenizer(
            checkpoint_enc=f"{HF_HOME}/hub/Cosmos-Tokenizer-CI8x8/encoder.jit", 
            checkpoint_dec=f"{HF_HOME}/hub/Cosmos-Tokenizer-CI8x8/decoder.jit", 
        )

        vae_dim = 16
        gen_proj_dim = vlm_config.hidden_size
        ds = self.config.scale_factor
        self.gen_in_proj = nn.Conv2d(in_channels=vae_dim, out_channels=gen_proj_dim, kernel_size=1, stride=1, padding=0)
        self.downsample_conv = nn.Conv2d(in_channels=gen_proj_dim, out_channels=gen_proj_dim, kernel_size=ds, stride=ds, padding=0)
        self.upsample_conv = nn.ConvTranspose2d(in_channels=gen_proj_dim, out_channels=gen_proj_dim, kernel_size=ds, stride=ds, padding=0, output_padding=0) 
        self.gen_out_proj = nn.Linear(gen_proj_dim, vae_dim)
        self.gen_out_layer_norm = nn.LayerNorm(gen_proj_dim)

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.hidden_size)
        self.action_out_proj = nn.Linear(action_expert_config.hidden_size, config.max_action_dim)

        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.hidden_size, action_expert_config.hidden_size)
        self.action_time_mlp_out = nn.Linear(action_expert_config.hidden_size, action_expert_config.hidden_size)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(config.imagenet_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(config.imagenet_std).view(1, 3, 1, 1)
        )

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

        self.set_requires_grad()
    
    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.internvl_with_expert.und_expert.vision_tower.eval()
            for params in self.internvl_with_expert.und_expert.vision_tower.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.internvl_with_expert.und_expert.eval()
            for params in self.internvl_with_expert.und_expert.parameters():
                params.requires_grad = False
        
        if self.config.train_vlm_only:
            self.internvl_with_expert.act_expert.eval()
            for params in self.internvl_with_expert.act_expert.parameters():
                params.requires_grad = False
        
        self.cosmos_tokenizer.eval()
        for params in self.cosmos_tokenizer.parameters():
            params.requires_grad = False
    
    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.internvl_with_expert.und_expert.vision_tower.eval()

        if self.config.train_expert_only:
            self.internvl_with_expert.und_expert.eval()
        
        if self.config.train_vlm_only:
            self.internvl_with_expert.act_expert.eval()
        
        self.cosmos_tokenizer.eval()
        return self

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.internvl_with_expert.und_expert.language_model.gradient_checkpointing = True
        self.internvl_with_expert.und_expert.vision_tower.gradient_checkpointing = True
        self.internvl_with_expert.gen_expert.gradient_checkpointing = True
        self.internvl_with_expert.act_expert.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for InternVLA model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.internvl_with_expert.und_expert.language_model.gradient_checkpointing = False
        self.internvl_with_expert.und_expert.vision_tower.gradient_checkpointing = False
        self.internvl_with_expert.gen_expert.gradient_checkpointing = False
        self.internvl_with_expert.act_expert.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for InternVLA model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_und(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for i in range(3):
            img, img_mask = images[:, i, 1], img_masks[:, i]

            def image_embed_func(img):
                img = (img - self.imagenet_mean) / self.imagenet_std
                return self.internvl_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.internvl_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1).to(torch.bool)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks
    
    def prepare_gen_features(self, images):
        shape = images.shape[:-3]
        c, h, w = images.shape[-3:]
        images = images.reshape(-1, c, h, w)
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = images * 2 - 1  # [-1, 1]
        features = self.cosmos_tokenizer.encode(images)
        c, h, w = features.shape[-3:]
        features = features.view(*shape, c, h, w)
        return features
    
    def embed_gen(self, images, img_masks):
        device = images[0].device
        features = self.prepare_gen_features(images)

        B, N_view, T = features.shape[:3]
        features = rearrange(features, 'b n t c h w -> (b n t) c h w')
        if self.gen_in_proj.weight.dtype == torch.float32:
            features = features.to(torch.float32)

        features = self.gen_in_proj(features) # dtype: torch.float32
        features = self.downsample_conv(features) # dtype: torch.float32
        features = rearrange(features, '(b n t) c h w -> b n t c h w', b=B, n=N_view, t=T)
        self.gen_feat_shape = features.shape

        B, N_view, T, _, H, W = features.shape
        embs = rearrange(features, 'b n t c h w -> b (n t h w) c', b=B, n=N_view, t=T)

        pad_masks = torch.zeros((B, N_view, T, H, W), dtype=torch.bool, device=device)
        pad_masks[img_masks] = True
        pad_masks = rearrange(pad_masks, 'b n t h w -> b (n t h w)', b=B, n=N_view, t=T)

        att_masks = [1] + [0] * (embs.shape[1] - 1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(B, len(att_masks))
        return embs, pad_masks, att_masks

    def embed_act(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_proj_func(state):
            return self.state_proj(state) # dtype: torch.float32

        state_emb = self._apply_checkpoint(state_proj_func, state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions) # dtype: torch.float32

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb) # torch.float32
            x = F.silu(x)
            return self.action_time_mlp_out(x) # torch.float32

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks
    
    def pred_gen_features(self, features):
        b, n, t, c, h, w =self.gen_feat_shape
        features = rearrange(features, 'b (n t h w) c -> b n t c h w', b=b, n=n, t=t, h=h, w=w)
        features = features.mean(2)  # b n c h w, average across temporal dimension
        features = rearrange(features, 'b n c h w -> (b n) c h w')

        upsampled_features = self.upsample_conv(features) # dtype: torch.float32
        h_upsampled, w_upsampled = upsampled_features.shape[-2:]
        upsampled_features = upsampled_features.permute(0, 2, 3, 1)
        upsampled_features = upsampled_features.reshape(b * n, -1, c)
        pred_features = self.gen_out_proj(self.gen_out_layer_norm(upsampled_features)) # dtype: torch.float32
        pred_features = pred_features.view(b, n, h_upsampled, w_upsampled, pred_features.shape[-1])
        pred_features = pred_features.permute(0, 1, 4, 2, 3)
        return pred_features

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        und_embs, und_pad_masks, und_att_masks = self.embed_und(
            images, img_masks, lang_tokens, lang_masks
        )
        gen_embs, gen_pad_masks, gen_att_masks = self.embed_gen(
            images[:, :, :2], img_masks,  # remove the future observation
        )
        act_embs, act_pad_masks, act_att_masks = self.embed_act(state, x_t, time)

        if (
            self.internvl_with_expert.und_expert.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            und_embs = und_embs.to(dtype=torch.bfloat16)
            gen_embs = gen_embs.to(dtype=torch.bfloat16)
            act_embs = act_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([und_pad_masks, gen_pad_masks, act_pad_masks], dim=1)
        att_masks = torch.cat([und_att_masks, gen_att_masks, act_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(und_embs, gen_embs, act_embs, att_2d_masks_4d, position_ids):
            (_, gen_out, act_out), _ = self.internvl_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[und_embs, gen_embs, act_embs],
                use_cache=False,
            )
            return gen_out, act_out

        gen_out, act_out = self._apply_checkpoint(
            forward_func, und_embs, gen_embs, act_embs, att_2d_masks_4d, position_ids
        )

        def gen_out_func(gen_out):
            return self.pred_gen_features(gen_out)
        
        pred_gen_features = self._apply_checkpoint(gen_out_func, gen_out.to(dtype=torch.float32))

        future_embs = self.prepare_gen_features(images[:, :, 2])
        loss_gen = F.mse_loss(pred_gen_features[img_masks], future_embs.to(dtype=torch.float32)[img_masks])

        act_out = act_out[:, -self.config.chunk_size :]
        act_out = act_out.to(dtype=torch.float32)

        def action_out_proj_func(act_out):
            return self.action_out_proj(act_out)

        v_t = self._apply_checkpoint(action_out_proj_func, act_out)

        loss_action = F.mse_loss(u_t, v_t, reduction="none")

        return loss_action, loss_gen

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        num_steps=None,
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)


        und_expert_embs, und_expert_pad_masks, und_expert_att_masks = self.embed_und(
            images, img_masks, lang_tokens, lang_masks
        )
        und_expert_att_2d_masks = make_att_2d_masks(und_expert_pad_masks, und_expert_att_masks)
        und_expert_position_ids = torch.cumsum(und_expert_pad_masks, dim=1) - 1

        und_expert_att_2d_masks_4d = self._prepare_attention_masks_4d(und_expert_att_2d_masks)

        self.internvl_with_expert.und_expert.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        if (
            self.internvl_with_expert.und_expert.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            und_expert_embs = und_expert_embs.to(dtype=torch.bfloat16)

        _, past_key_values = self.internvl_with_expert.forward(
            attention_mask=und_expert_att_2d_masks_4d,
            position_ids=und_expert_position_ids,
            past_key_values=None,
            inputs_embeds=[und_expert_embs, None, None],
        )

        gen_expert_embs, gen_expert_pad_masks, gen_expert_att_masks = self.embed_gen(
            images[:, :, :2], img_masks,  # remove the future observation
        )
        gen_att_2d_masks = make_att_2d_masks(gen_expert_pad_masks, gen_expert_att_masks)
        gen_expert_token_num = gen_expert_pad_masks.shape[1]
        und_gen_att_2d_masks = torch.cat([
            und_expert_pad_masks[:, None, :].repeat(1, gen_expert_token_num, 1),
            gen_att_2d_masks,
        ], dim=2)

        position_ids = torch.sum(und_expert_pad_masks, dim=-1)[:, None] + torch.cumsum(gen_expert_pad_masks, dim=1) - 1
        # position_ids = torch.cumsum(pad_masks, dim=1) - 1
        und_gen_2d_masks_4d = self._prepare_attention_masks_4d(und_gen_att_2d_masks)
       
        if (
            self.internvl_with_expert.gen_expert.model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            gen_expert_embs = gen_expert_embs.to(dtype=torch.bfloat16)

        self.internvl_with_expert.gen_expert.config._attn_implementation = "eager"  # noqa: SLF001

        (_, gen_out, _), past_key_values = self.internvl_with_expert.forward(
            attention_mask=und_gen_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, gen_expert_embs, None],
        )
        # if decode_image:
        #     recon_images = self.decode_images(gen_out, n_view=image_grid_thw.shape[1])
        # else:
        #     recon_images = None

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                torch.cat([und_expert_pad_masks, gen_expert_pad_masks], dim=1),
                past_key_values,
                x_t.to(dtype),
                expanded_time.to(dtype),
            )
            x_t = x_t + dt * v_t
            time += dt

        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_act(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.internvl_with_expert.act_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        if past_key_values.layers[0].keys.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        outputs_embeds, _ = self.internvl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
        )

        suffix_out = outputs_embeds[2]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out) # dtype: torch.float32


class InternVLAA1Policy(PreTrainedPolicy):
    """InternVLA-A1 Policy for LeRobot."""

    config_class = InternVLAA1Config
    name = "internvla_a1"

    def __init__(
        self,
        config: InternVLAA1Config,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = InternVLAA1(config)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()
    
    def __str__(self) -> str:
        lines = []

        # ---- basic info ----
        lines.append("=" * 60)
        lines.append(f"Policy: {self.__class__.__name__}")
        lines.append("")

        # ---- parameter counts ----
        num_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_trainable_params = sum(p.numel() for p in self.parameters())

        num_und = sum(p.numel() for p in self.model.internvl_with_expert.und_expert.parameters())
        num_gen = sum(p.numel() for p in self.model.internvl_with_expert.gen_expert.parameters())
        num_act = sum(p.numel() for p in self.model.internvl_with_expert.act_expert.parameters())

        lines.append("Parameter statistics:")
        lines.append(f"  - Total params        : {num_total_params} ({format_big_number(num_total_params)})")
        lines.append(f"  - Trainable params    : {num_trainable_params} ({format_big_number(num_trainable_params)})")
        lines.append(f"  - Und params          : {num_und} ({format_big_number(num_und)})")
        lines.append(f"  - Gen params          : {num_gen} ({format_big_number(num_gen)})")
        lines.append(f"  - Act params          : {num_act} ({format_big_number(num_act)})")

        lines.append("=" * 60)

        return "\n".join(lines)
    
    def to(self, *args, **kwargs):
        if args[0] == torch.bfloat16:
            super().to(*args, **kwargs)
            self.model.internvl_with_expert.to_bfloat16_for_selected_params("bfloat16")
            self.model.action_out_proj.to(torch.float32)
            self.model.gen_in_proj.to(torch.float32)
            self.model.downsample_conv.to(torch.float32)
            self.model.upsample_conv.to(torch.float32)
            self.model.gen_out_proj.to(torch.float32)
            self.model.gen_out_layer_norm.to(torch.float32)
            self.model.action_in_proj.to(torch.float32)
            self.model.action_out_proj.to(torch.float32)
            self.model.state_proj.to(torch.float32)
            self.model.action_time_mlp_in.to(torch.float32)
            self.model.action_time_mlp_out.to(torch.float32)
            self.model.imagenet_mean = self.model.imagenet_mean.to(torch.float32)
            self.model.imagenet_std = self.model.imagenet_std.to(torch.float32)
        elif args[0] == torch.float32:
            super().to(*args, **kwargs)
            self.model.cosmos_tokenizer.to(torch.bfloat16)
        else:
            super().to(*args, **kwargs)
        return self

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        InternVL expects images in [B, C, H, W] format and normalized with ImageNet Statistics.
        """
        images = []
        img_masks = []

        for img_idx in range(3):
            img = batch[f"{OBS_IMAGES}.image{img_idx}"]
            mask = batch[f"{OBS_IMAGES}.image{img_idx}_mask"]

            images.append(img)
            img_masks.append(mask)
        
        images = torch.stack(images, dim=1)  # B, N_view, T, C, H, W
        img_masks = torch.stack(img_masks, dim=1)

        return images, img_masks
    
    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)

        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        # Compute loss
        losses_action, loss_gen = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses_action = losses_action[:, :, :original_action_dim]
        loss_action = losses_action.mean()

        loss = loss_action + self.config.lambda_gen * loss_gen

        loss_dict = {
            "loss": loss.item(),
            "loss_action": loss_action.item(), 
            "loss_gen": loss_gen.item(), 
        }
        
        losses_action = losses_action.mean(dim=[0, 1]).detach().cpu().numpy().tolist()
        loss_dict.update({
            f"loss_action_dim{i}": losses_action[i] for i in range(original_action_dim)
        })

        return loss, loss_dict


if __name__ == "__main__":
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, ACTION
    from pprint import pp
    torch.manual_seed(0)
    device = torch.device("cuda")

    cfg = InternVLAA1Config()
    cfg.internvl_variant="internvl_24l"
    cfg.qwen2_variant="qwen2_24l"
    cfg.freeze_vision_encoder=True
    dtype = torch.float32 if cfg.dtype == 'float32' else torch.bfloat16

    model = InternVLAA1Policy(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"Trainable parameters: {trainable_params:,}  ({trainable_params / 1e9:.2f}B)")
    print(f"Und Expert params: {sum(p.numel() for p in model.model.internvl_with_expert.und_expert.parameters()) / 1e9:.2f}B")
    print(f"Gen Expert params: {sum(p.numel() for p in model.model.internvl_with_expert.gen_expert.parameters()) / 1e9:.2f}B")
    print(f"Act Expert params: {sum(p.numel() for p in model.model.internvl_with_expert.act_expert.parameters()) / 1e9:.2f}B")

    inputs = {
        OBS_STATE: torch.rand((1, 32), dtype=dtype).cuda(), 
        ACTION: torch.rand((1, 50, 32), dtype=dtype).cuda(), 
        f"{OBS_IMAGES}.image0": torch.rand((1, 3, 3, 448, 448), dtype=dtype).cuda(),  # [B, T, C, H, W]
        f"{OBS_IMAGES}.image1": torch.rand((1, 3, 3, 448, 448), dtype=dtype).cuda(),  # T = [past, now, future]
        f"{OBS_IMAGES}.image2": torch.rand((1, 3, 3, 448, 448), dtype=dtype).cuda(), 
        f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(), 
        f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(), 
        f"{OBS_IMAGES}.image2_mask": torch.tensor([True]).cuda(), 
        "task": ['Put the pen from the table into the pen holder.'], 
        "observation.language.tokens": torch.randint(0, 7777, (1, 48)).cuda(), 
        "observation.language.attention_mask": torch.ones((1, 48)).cuda(), 
    }

    loss, loss_dict = model.forward(inputs)
    pp(loss)
    pp(loss_dict)
