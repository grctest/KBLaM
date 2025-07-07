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
import copy
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
    Gemma3nTextRotaryEmbedding,
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
    def __init__(self, config: KBLaMConfig):
        text_config = getattr(config, "text_config", None)
        if text_config:
            super().__init__(text_config)
        else:
            # This case might be for loading from a KBLaM-saved checkpoint
            # where text_config isn't nested. We'll assume config is compatible.
            super().__init__(config)
        self.config = config

        # KBLaM-specific attributes
        self.kb_layer_frequency = getattr(config, 'kb_layer_frequency', 1)

        # The following logic is adapted from the original Gemma3nTextModel.__init__
        # to ensure all necessary attributes are correctly initialized on this subclass.
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size, text_config.padding_idx)
        
        # Replicating the RoPE creation from the reference implementation
        self.rotary_emb = Gemma3nTextRotaryEmbedding(config=text_config)
        
        # Create a deepcopy of the config for the local RoPE, as per the reference
        local_config = copy.deepcopy(text_config)
        local_config.rope_theta = getattr(text_config, 'rope_local_base_freq', 10000.0)
        if hasattr(local_config, 'rope_scaling'):
            local_config.rope_scaling['rope_type'] = 'default'
        else:
            local_config.rope_scaling = {'rope_type': 'default'}
        self.rotary_emb_local = Gemma3nTextRotaryEmbedding(config=local_config)

        self.per_layer_input_embeddings = nn.Embedding(text_config.num_hidden_layers, text_config.hidden_size)
        
        # Replace layers with KBLaM versions
        self.layers = nn.ModuleList([
            KblamGemma3nDecoderLayer(config, i) for i in range(text_config.num_hidden_layers)
        ])
        self.norm = Gemma3nRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, kb_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        # This forward pass is now simplified as the main logic is in KblamGemma3nForConditionalGeneration
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
            # kb_embeds is passed up to the layer that needs it
            **kwargs
        )


class KblamGemma3nForConditionalGeneration(Gemma3nPreTrainedModel, GenerationMixin):
    """
    The main KBLAM model for Gemma-3N, designed for conditional generation tasks.
    This class integrates the KB-aware text model with a language modeling head.
    It also overrides the `forward` and `prepare_inputs_for_generation` methods
    to handle KB-related inputs and logic.
    """
    def __init__(self, config: KBLaMConfig):
        super().__init__(config)
        self.config = config
        self.model = KblamGemma3nTextModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        kb_kvs: list[torch.FloatTensor] | None = None,
        kb_config: dict | None = None,
    ):
        """
        The forward pass for the KBLAM Gemma-3N model.

        This method is a complete override of the base model's forward pass to
        manually control the flow of information through the decoder layers,
        allowing for the injection of knowledge base embeddings (`kb_kvs`) at
        specified intervals.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence token in the position embeddings.
            past_key_values (`list(torch.FloatTensor)`, *optional*):
                Contains pre-computed hidden-states (key and value pairs) of the attention blocks.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, use embeddings directly instead of `input_ids`.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.
            kb_kvs (`list(torch.FloatTensor)`, *optional*):
                A list or tuple of key-value embeddings from the knowledge base.
                Expected to be of shape (batch_size, num_kb_heads, kb_head_dim).
            kb_config (`dict`, *optional*):
                A dictionary containing configuration for KB injection, such as
                `kb_layer_frequency`.

        Returns:
            `CausalLMOutputWithPast` or `tuple`: A standard causal LM output object.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # The core logic from the original `Gemma3nTextModel.forward` is replicated here
        # to allow for manual layer-by-layer processing with KB injection.
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = 0
            if past_key_values is not None:
                past_seen_tokens = past_key_values[0][0].shape[2]
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device
            ).unsqueeze(0)

        hidden_states = inputs_embeds
        
        # Get KB configuration
        kb_layer_frequency = getattr(self.config, 'kb_layer_frequency', 1)
        if kb_config and 'kb_layer_frequency' in kb_config:
            kb_layer_frequency = kb_config['kb_layer_frequency']

        # Prepare for decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Manually iterate through each decoder layer
        for idx, decoder_layer in enumerate(self.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Determine if KB should be injected at this layer
            inject_kb = (idx + 1) % kb_layer_frequency == 0
            kb_embeds_for_layer = None
            if inject_kb and kb_kvs is not None:
                # Unpack the tuple of (keys, values)
                kb_keys, kb_values = kb_kvs
                # For now, we pass the keys. This might need adjustment based on fusion strategy.
                kb_embeds_for_layer = kb_keys

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # The following arguments are required by the Gemma3nTextDecoderLayer.forward method.
            # We must construct them manually.
            layer_input_embed = self.model.per_layer_input_embeddings(torch.tensor(idx, device=hidden_states.device))
            
            # Generate RoPE embeddings for the current hidden states
            position_embeddings_global = self.model.rotary_emb(hidden_states, position_ids)
            position_embeddings_local = self.model.rotary_emb_local(hidden_states, position_ids)
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                per_layer_input_embedding=layer_input_embed,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                kb_embeds=kb_embeds_for_layer, # Pass KB embeds to the layer
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.model.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # Compute final logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return tuple(v for v in [logits, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Prepares inputs for generation, adding `kb_kvs` and `kb_config` to the
        model keyword arguments.
        """
        model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)

        # Add KB-related arguments if they are present in the generation call
        if "kb_kvs" in kwargs:
            model_inputs["kb_kvs"] = kwargs["kb_kvs"]
        if "kb_config" in kwargs:
            model_inputs["kb_config"] = kwargs["kb_config"]

        return model_inputs
