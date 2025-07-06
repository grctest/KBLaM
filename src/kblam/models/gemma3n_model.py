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
    Follows official Hugging Face patterns: strict config typing, attribute propagation, and subclassing.
    """
    def __init__(self, config, layer_idx: int):
        # Always use the correct config type
        if hasattr(config, 'text_config'):
            text_config = config.text_config
        else:
            text_config = config
        assert hasattr(text_config, "num_attention_heads"), f"Config missing num_attention_heads: {text_config}"
        assert hasattr(text_config, "head_dim"), f"Config missing head_dim: {text_config}"
        super().__init__(text_config, layer_idx)
        # KBLaM extension: add KB projection for concat+proj fusion
        self.kb_proj = nn.Linear(text_config.hidden_size, text_config.head_dim, bias=False)
        # Optional: separable query head
        self.separable_query = getattr(text_config, 'kblam_separable_query', False)
        if self.separable_query:
            self.kb_query_proj = nn.Linear(text_config.hidden_size, text_config.head_dim, bias=False)
        # Optional: length scaling
        self.length_scaling = getattr(text_config, 'kblam_length_scaling', False)
        self.kb_layer_frequency = getattr(text_config, 'kb_layer_frequency', 1)
        self.kb_fusion_method = getattr(text_config, 'kb_fusion_method', 'concat_proj')
        self.kb_layers = set(getattr(text_config, 'kb_layers', []))
        self.layer_idx = layer_idx

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_value=None, kb_embeds=None, **kwargs):
        # Determine if this layer should inject KB (config-driven)
        inject_kb = False
        if len(self.kb_layers) > 0:
            inject_kb = self.layer_idx in self.kb_layers
        else:
            inject_kb = (self.layer_idx % self.kb_layer_frequency == 0)

        # Optionally, fuse KB embeddings as extra keys/values if provided and enabled
        if kb_embeds is not None and inject_kb:
            if self.kb_fusion_method == 'concat_proj':
                # Project KB embeddings to key/value space
                kb_keys = self.kb_proj(kb_embeds).unsqueeze(1)
                kb_values = self.kb_proj(kb_embeds).unsqueeze(1)
                # Expand to match heads if needed (assume GQA/MQA)
                kb_keys = kb_keys.expand(-1, self.num_key_value_heads, -1)
                kb_values = kb_values.expand(-1, self.num_key_value_heads, -1)
                # Save for later use in the parent's forward
                kwargs['extra_kb_keys'] = kb_keys
                kwargs['extra_kb_values'] = kb_values
            elif self.kb_fusion_method == 'separable_query' and self.separable_query:
                # Use a separate query head for KB
                kb_queries = self.kb_query_proj(kb_embeds).unsqueeze(1)
                kwargs['extra_kb_queries'] = kb_queries
            # Add other fusion methods as needed
            # Optional: length scaling
            if self.length_scaling:
                scale = (hidden_states.size(1) + kb_embeds.size(1)) ** 0.5 / (hidden_states.size(1) ** 0.5)
                hidden_states = hidden_states * scale
        return super().forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            **kwargs
        )


class KblamGemma3nDecoderLayer(Gemma3nTextDecoderLayer):
    """
    Custom decoder layer for KBLaM, following official Gemma3nTextDecoderLayer structure.
    Only overrides attention to use KblamGemma3nAttention.
    """
    def __init__(self, config: Gemma3nConfig, layer_idx: int):
        # Use the correct config type for parent
        text_config = config.text_config if hasattr(config, 'text_config') else config
        super().__init__(text_config, layer_idx)
        # Replace self_attn with KBLaM version
        self.self_attn = KblamGemma3nAttention(config, layer_idx)
        # Save config for forward
        self.kb_layer_frequency = getattr(text_config, 'kb_layer_frequency', 1)
        self.kb_layers = set(getattr(text_config, 'kb_layers', []))
        self.layer_idx = layer_idx

    def forward(self, *args, kb_embeds=None, **kwargs):
        # Pass kb_embeds to attention if this layer is selected for KB fusion
        inject_kb = False
        if len(self.kb_layers) > 0:
            inject_kb = self.layer_idx in self.kb_layers
        else:
            inject_kb = (self.layer_idx % self.kb_layer_frequency == 0)
        if inject_kb:
            kwargs['kb_embeds'] = kb_embeds
        return super().forward(*args, **kwargs)


class KblamGemma3nTextModel(Gemma3nTextModel):
    """
    The text-processing component of the KBLAM Gemma-3N model.
    Subclasses the official Gemma3nTextModel, but replaces decoder layers with KBLaM versions.
    """
    def __init__(self, config: Gemma3nConfig):
        text_config = config.text_config if hasattr(config, 'text_config') else config
        # Call parent constructor (builds layers, but we will replace them)
        super().__init__(text_config)
        self.config = config
        # Replace layers with KBLaM versions
        self.layers = nn.ModuleList([
            KblamGemma3nDecoderLayer(config, i) for i in range(text_config.num_hidden_layers)
        ])

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, kb_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        # Pass kb_embeds to all layers (each layer decides if it uses it)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            kb_embeds=kb_embeds,
            **kwargs
        )


class KblamGemma3nForConditionalGeneration(Gemma3nPreTrainedModel, GenerationMixin):
    """
    The main KBLAM model for conditional generation using the Gemma-3N architecture.
    Subclasses the official model, but uses KblamGemma3nTextModel for text.
    """
    _auto_class = "AutoModelForCausalLM"
    config_class = Gemma3nConfig

    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        # Ensure critical fields are present at the top level for Hugging Face compatibility
        if hasattr(config, 'text_config'):
            # Add all critical fields needed by Hugging Face at the top level
            for attr in ["num_hidden_layers", "hidden_size", "vocab_size", "sliding_window", "max_position_embeddings", "initializer_range", "use_cache", "model_type"]:
                if hasattr(config.text_config, attr):
                    setattr(config, attr, getattr(config.text_config, attr))
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
