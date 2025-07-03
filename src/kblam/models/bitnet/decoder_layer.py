# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
# Adapted for KBLaM from the Llama and Phi-3 implementations.
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

import torch
from torch import nn
from typing import Optional, Tuple

from transformers.models.bitnet import configuration_bitnet, modeling_bitnet
from transformers.utils import logging

from .attention import KBLaMBitNetAttention
from .mlp import KBLaMBitNetMLP
from ..kblam_config import KBLaMConfig

logger = logging.get_logger(__name__)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        # Only log initialization, not every forward
        print(f"[LayerScale] Initialized gamma with value: {init_values}")

    def forward(self, x):
        return x * self.gamma

    def get_gamma_mean(self):
        return self.gamma.data.mean().item()


class KBLaMBitNetDecoderLayer(nn.Module):
    """
    Single decoder layer for BitNet, with self-attention and MLP blocks.
    Integrates KBLaM attention for knowledge base retrieval if configured.
    """
    def __init__(self, config: configuration_bitnet.BitNetConfig, layer_idx: int, use_layerscale: bool = False, layerscale_init_value: float = 1e-5):
        """
        Initialize the decoder layer.
        Args:
            config: BitNetConfig with layer parameters.
            layer_idx: Index of this decoder layer.
            use_layerscale: Whether to use LayerScale.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = KBLaMBitNetAttention(config=config, layer_idx=layer_idx)
        self.mlp = KBLaMBitNetMLP(config)
        self.input_layernorm = modeling_bitnet.BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = modeling_bitnet.BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_layerscale = use_layerscale
        if self.use_layerscale:
            self.attn_layerscale = LayerScale(config.hidden_size, init_values=layerscale_init_value)
            self.mlp_layerscale = LayerScale(config.hidden_size, init_values=layerscale_init_value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        kb_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kb_config: Optional[KBLaMConfig] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass for a single decoder layer.
        Applies self-attention, MLP, and layer normalization, with optional KB integration.
        Returns hidden states and (optionally) attention weights and cache.
        """
        if kb_config is not None and self.self_attn.layer_idx is not None and self.self_attn.layer_idx % kb_config.kb_layer_frequency == 0:
            logger.debug(f"KB-ATTN: DecoderLayer {self.self_attn.layer_idx} received kb_config.")

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            kb_kvs=kb_kvs,
            kb_config=kb_config,
            position_embeddings=position_embeddings,
            save_attention_weights=save_attention_weights,
            attention_save_loc=attention_save_loc,
            attention_file_base_name=attention_file_base_name,
        )
        if self.use_layerscale:
            hidden_states = self.attn_layerscale(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.use_layerscale:
            hidden_states = self.mlp_layerscale(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
