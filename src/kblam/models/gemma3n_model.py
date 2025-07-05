# coding=utf-8
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
"""A KBLAM model based on the Gemma-3N architecture from Google.

This module adapts the Gemma-3N model to integrate with a Knowledge
Base (KB) by modifying the attention mechanism and relevant model components to
process and incorporate KB-derived information during generation. It leverages
the underlying Gemma-2 and Gemma-3N implementations from the `transformers`
library.
"""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nConfig,
    Gemma3nPreTrainedModel,
    Gemma3nTextModel,
    Gemma3nTextDecoderLayer,
    Gemma3nTextAttention,
    Gemma3nTextMLP,
    Gemma3nRMSNorm,
)
from transformers.generation.utils import GenerationMixin
from transformers.utils import logging
from kblam.models.kblam_config import KBLaMConfig

logger = logging.get_logger(__name__)

# This file is heavily adapted from the original Gemma-3N implementation in the
# `transformers` library. The main changes are:
# 1.  Introduction of KblamGemma3nAttention, KblamGemma3nDecoderLayer, and
#     KblamGemma3nTextModel to inject Knowledge Base (KB) information.
# 2.  Modification of the KblamGemma3nForConditionalGeneration class to
#     properly initialize and manage the KBLaM-specific configuration.

class KblamGemma3nAttention(Gemma3nTextAttention):
    """
    Custom attention mechanism for Gemma-3N that integrates KB information.
    Extend this class to add KBLaM logic as needed.
    """
    def __init__(self, config, layer_idx: int):
        text_config = config.text_config if hasattr(config, 'text_config') else config
        super().__init__(text_config, layer_idx)
        # Project KB embeddings to match attention key/value dimension
        self.kb_proj = nn.Linear(text_config.hidden_size, self.head_dim, bias=False)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_value=None, kb_embeds=None, **kwargs):
        # Standard attention projections
        query_states = self.q_proj(hidden_states).view(hidden_states.size(0), -1, self.num_attention_heads, self.head_dim)
        query_states = self.q_norm(query_states)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = self._apply_rope(query_states, cos, sin)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_states.size(0), -1, self.num_key_value_heads, self.head_dim)
        key_states = self.k_norm(key_states)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            key_states = self._apply_rope(key_states, cos, sin)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_states.size(0), -1, self.num_key_value_heads, self.head_dim)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        # KBLaM: Fuse KB embeddings as extra keys/values if provided
        if kb_embeds is not None:
            # Project KB embeddings to key/value space
            kb_keys = self.kb_proj(kb_embeds).unsqueeze(1)  # (batch, 1, head_dim)
            kb_values = self.kb_proj(kb_embeds).unsqueeze(1)
            # Repeat for all heads
            kb_keys = kb_keys.expand(-1, self.num_key_value_heads, -1)
            kb_values = kb_values.expand(-1, self.num_key_value_heads, -1)
            # Concatenate to sequence
            key_states = torch.cat([key_states, kb_keys], dim=2)
            value_states = torch.cat([value_states, kb_values], dim=2)

        # Standard attention computation
        attn_output, attn_weights = self._attention_forward(query_states, key_states, value_states, attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(hidden_states.size(0), -1, self.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _apply_rope(self, x, cos, sin):
        # Apply rotary position embedding (same as Gemma3nTextAttention)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _attention_forward(self, query, key, value, attention_mask):
        # Standard scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class KblamGemma3nDecoderLayer(Gemma3nTextDecoderLayer):
    """Custom decoder layer for Gemma-3N that uses the KblamGemma3nAttention."""
    def __init__(self, config, layer_idx: int):
        text_config = config.text_config if hasattr(config, 'text_config') else config
        super().__init__(text_config, layer_idx)
        self.self_attn = KblamGemma3nAttention(config, layer_idx)
        self.mlp = Gemma3nTextMLP(text_config, layer_idx)
        self.input_layernorm = Gemma3nRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3nRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_value=None, kb_embeds=None, **kwargs):
        # Layer norm
        normed_hidden = self.input_layernorm(hidden_states)
        # KBLaM: Pass kb_embeds to attention
        attn_output, attn_weights = self.self_attn(normed_hidden, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_value=past_key_value, kb_embeds=kb_embeds, **kwargs)
        attn_output = self.post_attention_layernorm(attn_output)
        # Residual connection
        hidden_states = hidden_states + attn_output
        # MLP
        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_output
        return hidden_states, attn_weights


class KblamGemma3nTextModel(Gemma3nTextModel):
    """The text-processing component of the KBLAM Gemma-3N model."""
    def __init__(self, config):
        super().__init__(config.text_config)
        # Replace layers with KBLaM versions
        self.layers = nn.ModuleList(
            [KblamGemma3nDecoderLayer(config, layer_idx) for layer_idx in range(config.text_config.num_hidden_layers)]
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, kb_embeds=None, **kwargs):
        # Standard embedding
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        # Position embeddings (assume precomputed for simplicity)
        position_embeddings = None
        if hasattr(self, 'rotary_emb'):
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # Pass through all layers, injecting KB embeddings
        all_attn_weights = []
        for layer in self.layers:
            hidden_states, attn_weights = layer(hidden_states, position_embeddings, attention_mask=attention_mask, past_key_value=past_key_values, kb_embeds=kb_embeds, **kwargs)
            all_attn_weights.append(attn_weights)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=all_attn_weights,
        )


class KblamGemma3nForConditionalGeneration(Gemma3nPreTrainedModel, GenerationMixin):
    """
    The main KBLAM model for conditional generation using the Gemma-3N architecture.
    It integrates a text model (based on Gemma-2) with vision and audio models.
    """
    _auto_class = "AutoModelForCausalLM"
    config_class = Gemma3nConfig

    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        self.text_model = KblamGemma3nTextModel(config)
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
        # Add KBLaM specific attributes to the config if they don't exist.
        if not hasattr(self.config, "kb_layer_frequency"):
            logger.info("KBLaM config attributes not found. Initializing from KBLaMConfig.")
            kblam_config = KBLaMConfig()
            for key, value in kblam_config.to_dict().items():
                if not hasattr(self.config, key):
                    setattr(self.config, key, value)
                if not hasattr(self.config.text_config, key):
                    setattr(self.config.text_config, key, value)

    def get_input_embeddings(self):
        return self.text_model.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list = None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        labels: torch.LongTensor = None,
        kb_embeds: torch.FloatTensor = None,
        **kwargs,
    ):
        """
        The forward pass of the model with KBLaM support.
        """
        if position_ids is None and input_ids is not None:
            device = input_ids.device
            seq_length = input_ids.shape[1]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)

        output_attentions = output_attentions if output_attentions is not None else self.config.text_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.text_config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.text_config.use_return_dict

        outputs: BaseModelOutputWithPast = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            kb_embeds=kb_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
