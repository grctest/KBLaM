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
Base (KB) by modifying the attention mechanism to process and incorporate
KB-derived information during generation. It follows the established KBLaM
project pattern of composition and surgical modification.
"""

import torch
from torch import nn
import math
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nConfig,
    Gemma3nPreTrainedModel,
    Gemma3nTextModel,
    Gemma3nTextAttention,
)
from transformers.generation.utils import GenerationMixin
from transformers.utils import logging
from kblam.models.kblam_config import KBLaMConfig
from typing import Optional, Tuple

logger = logging.get_logger(__name__)

class KblamGemma3nAttention(Gemma3nTextAttention):
    """
    Custom attention mechanism for Gemma-3N that integrates KB information,
    following the architectural pattern of other KBLaM models.
    """
    def __init__(self, original_attention, kblam_config: KBLaMConfig):
        # This is a bit of a hack, as super().__init__ will call _init_rope, which we don't want yet.
        # We'll re-initialize with the correct config from the original attention module.
        super().__init__(original_attention.config, original_attention.layer_idx)
        
        # Copy all weights and attributes from the original attention module
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj

        # --- FIX START ---
        # Explicitly set all necessary attributes from the config, as they are needed for layer initialization and forward pass.
        self.hidden_size = original_attention.config.hidden_size
        self.num_heads = original_attention.config.num_attention_heads
        self.head_dim = original_attention.config.head_dim
        self.num_key_value_heads = original_attention.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # --- FIX END ---
        
        # KBLaM-specific attributes and layers
        self.kblam_config = kblam_config
        
        # The input dimension for the KB projection is the hidden size of the model
        kb_embed_dim = self.hidden_size
        
        # This projection is for the separate query head used to score KB entries
        self.q_proj_new = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: torch.Tensor,
        past_key_value,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_altup: bool = False,
        kb_kvs: Optional[tuple] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_embeddings is not None:
            # Gemma3n passes a tuple for position_embeddings
            global_pos_emb, local_pos_emb = position_embeddings
            # --- PATCH START ---
            # Handle both (cos, sin) tuple and single tensor cases for global_pos_emb
            if isinstance(global_pos_emb, (tuple, list)) and len(global_pos_emb) == 2:
                cos, sin = global_pos_emb
                query_states, key_states = (
                    query_states * cos + self._rotate_half(query_states) * sin,
                    key_states * cos + self._rotate_half(key_states) * sin,
                )
            else:
                # Fallback: just pass through (or use base class logic if needed)
                pass  # No rotary applied, or already applied upstream
            # --- PATCH END ---
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # KBLaM Injection Logic
        if kb_kvs is not None and self.layer_idx % self.kblam_config.kb_layer_frequency == 0:
            kb_keys, kb_values = kb_kvs
            kb_idx = self.layer_idx // self.kblam_config.kb_layer_frequency
            
            if len(kb_keys.shape) == 2: # Shared KB embeddings across batch
                kb_len = kb_keys.shape[0]
                num_kb_layers = 1 + self.config.num_hidden_layers // self.kblam_config.kb_layer_frequency
                
                kb_keys_layer = kb_keys.reshape(kb_len, num_kb_layers, -1)[:, kb_idx]
                kb_values_layer = kb_values.reshape(kb_len, num_kb_layers, -1)[:, kb_idx]

                kb_keys_layer = kb_keys_layer.view(kb_len, self.num_heads, self.head_dim).transpose(0, 1)
                kb_values_layer = kb_values_layer.view(kb_len, self.num_heads, self.head_dim).transpose(0, 1)

                kb_keys_layer = kb_keys_layer.unsqueeze(0).expand(bsz, -1, -1, -1)
                kb_values_layer = kb_values_layer.unsqueeze(0).expand(bsz, -1, -1, -1)
            else: # Batch-specific KB embeddings
                kb_len = kb_keys.shape[1]
                num_kb_layers = 1 + self.config.num_hidden_layers // self.kblam_config.kb_layer_frequency

                kb_keys_layer = kb_keys.view(bsz, kb_len, num_kb_layers, -1)[:, :, kb_idx]
                kb_values_layer = kb_values.view(bsz, kb_len, num_kb_layers, -1)[:, :, kb_idx]

                kb_keys_layer = kb_keys_layer.view(bsz, kb_len, self.num_heads, self.head_dim).transpose(1, 2)
                kb_values_layer = kb_values_layer.view(bsz, kb_len, self.num_heads, self.head_dim).transpose(1, 2)

            key_states = torch.cat([kb_keys_layer, key_states], dim=2)
            value_states = torch.cat([kb_values_layer, value_states], dim=2)

            # Modify attention mask
            if attention_mask is not None:
                kb_len = kb_keys_layer.shape[2]
                kb_attention_mask = torch.zeros(bsz, 1, q_len, kb_len, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([kb_attention_mask, attention_mask], dim=-1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
    
    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def repeat_kv(hidden_states, n_rep):
        """
        This function is used to copy the key/value hidden states n_rep times
        """
        if n_rep == 1:
            return hidden_states
        return hidden_states.repeat_interleave(n_rep, dim=1)


class KblamGemma3nForConditionalGeneration(Gemma3nPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: KBLaMConfig):
        # Load the base Gemma3nConfig and update it with KBLaM parameters
        base_model_name_or_path = config.base_model_name_or_path if hasattr(config, "base_model_name_or_path") else config._name_or_path
        gemma_config = Gemma3nConfig.from_pretrained(base_model_name_or_path)
        
        # Transfer KBLaM specific attributes to the Gemma3nConfig
        for key, value in config.to_dict().items():
            setattr(gemma_config, key, value)

        # --- PATCH START ---
        # Promote essential nested attributes to the top level for KBLaM compatibility
        if hasattr(gemma_config, "text_config"):
            for attr in ["vocab_size", "hidden_size", "num_hidden_layers"]:
                if hasattr(gemma_config.text_config, attr):
                    setattr(gemma_config, attr, getattr(gemma_config.text_config, attr))
        # --- PATCH END ---

        # Now, initialize the parent class with the fully-featured config
        super().__init__(gemma_config)
        
        # Load the base Gemma3nTextModel
        self.model = Gemma3nTextModel.from_pretrained(base_model_name_or_path, torch_dtype=config.torch_dtype)
        
        # Replace attention layers with our custom KBLaM version
        for layer in self.model.layers:
            original_attention = layer.self_attn
            # Pass the augmented gemma_config to the attention layer
            layer.self_attn = KblamGemma3nAttention(original_attention, gemma_config)

        # Standard LM head
        self.vocab_size = self.model.config.vocab_size
        self.lm_head = nn.Linear(self.model.config.hidden_size, self.vocab_size, bias=False)

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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        kb_kvs: Optional[list[torch.FloatTensor]] = None,
        **kwargs,
    ):
        # --- PATCH START ---
        if return_dict is None:
            return_dict = True
        # --- PATCH END ---
        # The `use_altup` flag is handled internally by the original Gemma3nTextModel.forward
        use_altup_flag = kwargs.get("use_altup", None)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_altup=use_altup_flag,
            kb_kvs=kb_kvs,
            # Pass the entire kblam_config to the attention layers
            kblam_config=self.config
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

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
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        model_inputs = self.model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)
        
        # Add KB-related arguments if they are present in the generation call
        if "kb_kvs" in kwargs:
            model_inputs["kb_kvs"] = kwargs["kb_kvs"]
        
        return model_inputs
