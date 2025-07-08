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

        # The input dimension for the KB projection should match the KB embeddings,
        # which is defined in the top-level KBLaMConfig, not the text_config.
        kb_embed_dim = getattr(config, "kb_embed_dim", text_config.hidden_size)

        # KBLaM extension: add KB projection for concat+proj fusion
        self.kb_proj = nn.Linear(kb_embed_dim, text_config.head_dim, bias=False)
        # Optional: separable query head
        self.separable_query = getattr(text_config, 'kblam_separable_query', False)
        if self.separable_query:
            self.kb_query_proj = nn.Linear(kb_embed_dim, text_config.head_dim, bias=False)
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
            inject_kb = (self.layer_idx + 1) % self.kb_layer_frequency == 0

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

    def forward(self, hidden_states, *args, **kwargs):
        # Check for KB data in kwargs and pass to attention if this layer is selected
        kb_kvs = kwargs.get("kb_kvs")

        inject_kb = False
        if len(self.kb_layers) > 0:
            inject_kb = self.layer_idx in self.kb_layers
        else:
            # Corrected logic to match original implementation's 1-based layer check
            inject_kb = (self.layer_idx + 1) % self.kb_layer_frequency == 0

        if inject_kb and kb_kvs is not None:
            # The attention layer expects `kb_embeds`. We derive it from `kb_kvs`.
            # The original logic used the first element (keys).
            kwargs["kb_embeds"] = kb_kvs[0]

        # The base class forward will be called from KblamGemma3nForConditionalGeneration,
        # which will handle the `use_altup` flag.
        return super().forward(hidden_states, *args, **kwargs)


class KblamGemma3nTextModel(Gemma3nTextModel):
    """
    The text-processing component of the KBLAM Gemma-3N model.
    Subclasses the official Gemma3nTextModel, but replaces decoder layers with KBLaM versions.
    """
    def __init__(self, config: KBLaMConfig):
        text_config = getattr(config, "text_config", None)
        if text_config:
            super().__init__(text_config)
            self.padding_idx = text_config.pad_token_id
            self.vocab_size = text_config.vocab_size
            self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size, self.padding_idx)
        else:
            # This case might be for loading from a KBLaM-saved checkpoint
            # where text_config isn't nested. We'll assume config is compatible.
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Do not overwrite self.config; the parent class sets it correctly.
        # self.config = config

        # KBLaM-specific attributes
        self.kb_layer_frequency = getattr(config, 'kb_layer_frequency', 1)

        # The following logic is adapted from the original Gemma3nTextModel.__init__
        # to ensure all necessary attributes are correctly initialized on this subclass.
        # Replicating the RoPE creation from the reference implementation
        self.rotary_emb = Gemma3nTextRotaryEmbedding(config=text_config)
        
        # Create a deepcopy of the config for the local RoPE, as per the reference
        local_config = copy.deepcopy(text_config)
        local_config.rope_theta = getattr(text_config, 'rope_local_base_freq', 10000.0)
        if not hasattr(local_config, 'rope_scaling') or local_config.rope_scaling is None:
            local_config.rope_scaling = {}
        local_config.rope_scaling['rope_type'] = 'default'
        self.rotary_emb_local = Gemma3nTextRotaryEmbedding(config=local_config)

        self.per_layer_input_embeddings = nn.Embedding(text_config.num_hidden_layers, text_config.hidden_size)
        
        # Replace layers with KBLaM versions, passing the top-level config
        # so that the attention layer can access kb_embed_dim.
        self.layers = nn.ModuleList([
            KblamGemma3nDecoderLayer(config, i) for i in range(text_config.num_hidden_layers)
        ])
        self.norm = Gemma3nRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, use_altup=None, **kwargs):
        # This forward pass is now simplified as the main logic is in KblamGemma3nForConditionalGeneration
        # We explicitly pass use_altup to the parent.
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
            use_altup=use_altup,
            # kb_embeds is passed up to the layer that needs it via kwargs
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

        # To ensure the correct kb_embed_dim is always available, we explicitly
        # add it to the text_config before initializing the model.
        if hasattr(config, "text_config") and hasattr(config, "kb_embed_dim"):
            config.text_config.kb_embed_dim = config.kb_embed_dim

        self.model = KblamGemma3nTextModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        # Make vocab_size accessible from the top-level config for trainer compatibility
        self.config.vocab_size = config.text_config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if 'kb_proj' not in name and 'kb_query_proj' not in name:
                param.requires_grad = False

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
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, 'output_attentions', False)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else getattr(self.config, 'output_hidden_states', False)
        )
        use_cache = use_cache if use_cache is not None else getattr(self.config, 'use_cache', False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, 'use_return_dict', True)

        # Determine if altup should be used based on input dimensions.
        # 3D input is text-only, 4D is multi-modal.
        use_altup_flag = None
        if inputs_embeds is not None and inputs_embeds.dim() == 3:
            use_altup_flag = False

        # The core logic from the original `Gemma3nTextModel.forward` is replicated here
        # to allow for manual layer-by-layer processing with KB injection.
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs_embeds is None:
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

        # We now call the model directly and pass the use_altup flag.
        # The manual layer loop is removed to ensure proper subclass behavior.
        transformer_outputs = self.model(
            input_ids=None,  # We pass embeds, so ids should be None
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_altup=use_altup_flag,
            # Pass kb_kvs and kb_config through kwargs for the layers to use
            kb_kvs=kb_kvs,
            kb_config=kb_config,
        )
        hidden_states = transformer_outputs[0]
        next_cache = transformer_outputs[1] if use_cache else None

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
            return tuple(v for v in [logits, next_cache, transformer_outputs.hidden_states, transformer_outputs.attentions] if v is not None)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
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
