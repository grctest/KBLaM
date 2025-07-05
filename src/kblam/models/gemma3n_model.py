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
"""PyTorch Gemma-3n model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import Gemma3nForConditionalGeneration as HfGemma3nForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Gemma3nConfig, Gemma3nTextConfig
from transformers.generation import GenerationMixin

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from kblam.models.kblam_config import KBLaMConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/gemma-3n-2b"
_CONFIG_FOR_DOC = "Gemma3nConfig"


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Gemma3n
class Gemma3nRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Gemma3nRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding
class Gemma3nRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Gemma3nMLP(nn.Module):
    """
    A standard MLP block for the Gemma-3n model, consisting of a gate-and-up projection
    followed by a down projection.
    """
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        
        # Handle variable intermediate sizes for MatFormer architectures
        if isinstance(config.intermediate_size, (list, tuple)):
            intermediate_size = config.intermediate_size[layer_idx]
        else:
            intermediate_size = config.intermediate_size

        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)


class Gemma3nPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Gemma3nConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma3nDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.text_config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Gemma3nTextModel(Gemma3nPreTrainedModel):
    """
    The text-only model component of the Gemma-3n architecture. It processes token IDs
    and returns the final hidden states. This model is augmented to accept knowledge
    base key-values (`kb_kvs`) to be injected into its attention layers.
    """
    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        self.padding_idx = config.text_config.pad_token_id
        self.vocab_size = config.text_config.vocab_size

        self.embed_tokens = nn.Embedding(config.text_config.vocab_size, config.text_config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Gemma3nDecoderLayer(config.text_config, i) for i in range(config.text_config.num_hidden_layers)]
        )
        self.norm = Gemma3nRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        kb_kvs: Optional[tuple] = None,
        kb_config: Optional[KBLaMConfig] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass for the Gemma3nTextModel.

        Args:
            input_ids (`torch.LongTensor`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.Tensor`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor`, *optional*):
                Indices of positions of each input sequence token in the position embeddings.
            past_key_values (`Cache`, *optional*):
                Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a `BaseModelOutputWithPast` instead of a plain tuple.
            kb_kvs (`tuple`, *optional*):
                A tuple containing the knowledge base keys and values to be injected into the attention layers.
            kb_config (`KBLaMConfig`, *optional*):
                Configuration for the knowledge base augmentation.

        Returns:
            `Union[Tuple, BaseModelOutputWithPast]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.text_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.text_config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.text_config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length, self.layer_idx)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    kb_kvs,
                    kb_config,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    kb_kvs=kb_kvs,
                    kb_config=kb_config,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Gemma3nVisionModel(Gemma3nPreTrainedModel):
    """
    A placeholder vision model for Gemma-3n. In a real implementation, this would be a
    proper vision backbone (e.g., ViT, MobileNetV5). Currently, it's a simple linear
    projection.
    """
    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        # For now, a simple projection. A real implementation would use a vision backbone like MobileNetV5.
        self.vision_tower = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        return self.vision_tower(pixel_values)

class Gemma3nAudioModel(Gemma3nPreTrainedModel):
    """
    A placeholder audio model for Gemma-3n. In a real implementation, this would be a
    proper audio encoder. Currently, it's a simple linear projection.
    """
    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        # For now, a simple projection. A real implementation would use a transformer-based audio encoder.
        self.audio_encoder = nn.Linear(config.audio_config.hidden_size, config.text_config.hidden_size)

    def forward(self, audio_values: torch.FloatTensor) -> torch.FloatTensor:
        return self.audio_encoder(audio_values)


# --- KBLaM-augmented Gemma3nForConditionalGeneration ---
class KblamGemma3nForConditionalGeneration(Gemma3nPreTrainedModel, GenerationMixin):
    """
    Knowledge Base-augmented Gemma-3n model for conditional generation.

    This model integrates text, vision, and audio modalities and injects knowledge
    base (KB) information into the text decoder's attention layers. It serves as the
    main entry point for using the KB-augmented Gemma-3n model.
    """
    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        self.text_model = Gemma3nTextModel(config)
        self.vision_model = Gemma3nVisionModel(config)
        self.audio_model = Gemma3nAudioModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.text_model = decoder

    def get_decoder(self):
        return self.text_model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        kb_kvs: Optional[tuple] = None,
        kb_config: Optional[KBLaMConfig] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for the KblamGemma3nForConditionalGeneration model.

        Args:
            input_ids (`torch.LongTensor`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.Tensor`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor`, *optional*):
                Indices of positions of each input sequence token in the position embeddings.
            past_key_values (`Cache`, *optional*):
                Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            labels (`torch.LongTensor`, *optional*):
                Labels for computing the causal language modeling loss.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a `CausalLMOutputWithPast` instead of a plain tuple.
            pixel_values (`torch.FloatTensor`, *optional*):
                Pixel values for image inputs.
            audio_values (`torch.FloatTensor`, *optional*):
                Audio values for audio inputs.
            kb_kvs (`tuple`, *optional*):
                A tuple containing the knowledge base keys and values to be injected into the attention layers.
            kb_config (`KBLaMConfig`, *optional*):
                Configuration for the knowledge base augmentation.

        Returns:
            `Union[Tuple, CausalLMOutputWithPast]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.text_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.text_config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.text_config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.text_model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_features = self.vision_model(pixel_values)
            image_token_mask = (input_ids == self.config.image_token_id)
            inputs_embeds[image_token_mask] = image_features.to(inputs_embeds.dtype)

        if audio_values is not None:
            audio_features = self.audio_model(audio_values)
            audio_token_mask = (input_ids == self.config.audio_token_id)
            inputs_embeds[audio_token_mask] = audio_features.to(inputs_embeds.dtype)

        outputs = self.text_model(
            input_ids=None, # We are passing embeddings directly
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            kb_kvs=kb_kvs,
            kb_config=kb_config,
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
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
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
        model_inputs = {"input_ids": input_ids}
        model_inputs.update(kwargs)
        return model_inputs


class KblamGemma3nAttention(nn.Module):
    """
    Multi-headed attention module for Gemma-3n, augmented with Knowledge Base (KB) injection.

    This attention mechanism is the core of the KBLaM framework. On specified layers, it
    concatenates external knowledge base keys and values (`kb_kvs`) to the model's own
    self-attention context. It uses a secondary query projection (`q_proj_new`) to learn
    how to best query this external knowledge.
    """

    def __init__(self, config: Gemma3nTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else -1

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.q_proj_new = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = Gemma3nRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {self.config.rope_scaling['type']}")

    def prune_key_value(self, query, kb_keys, kb_values, topk_size=20):
        """
        Dynamically prunes the knowledge base keys and values based on their attention
        scores with the query. This is used to select the most relevant KB entries
        at inference time.

        Args:
            query (`torch.Tensor`): The query tensor.
            kb_keys (`torch.Tensor`): The knowledge base key tensor.
            kb_values (`torch.Tensor`): The knowledge base value tensor.
            topk_size (`int`): The number of top KB entries to keep.

        Returns:
            A tuple of pruned (kb_keys, kb_values, attn_weights).
        """
        assert (
            query.requires_grad is False
        ), "This function should only be used at test time"
        batch_size, num_heads, kb_len, head_dim = kb_keys.shape
        attn_weights = torch.matmul(query, kb_keys.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )  # Batchsize, num_heads, query_size, key_size
        if topk_size >= kb_len:
            return kb_keys, kb_values, attn_weights
        with torch.autocast(device=query.device, enabled=False):
            top_idx = attn_weights.sum((1, 2)).topk(min(kb_len, topk_size), -1)[1]
            top_idx = top_idx.view(batch_size, -1, topk_size, 1).expand(
                batch_size, num_heads, topk_size, head_dim
            )
            kb_keys = kb_keys.gather(-2, top_idx)
            kb_values = kb_values.gather(-2, top_idx)
        return kb_keys, kb_values, attn_weights[..., :topk_size]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        kb_kvs: Optional[tuple] = None,
        kb_config: Optional[KBLaMConfig] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for the KblamGemma3nAttention.

        This method performs the standard multi-head attention, but with a crucial
        modification: if `kb_kvs` are provided, it injects them into the key and
        value states, dynamically adjusting the attention mask.

        Args:
            hidden_states (`torch.Tensor`): Input to the attention layer.
            attention_mask (`torch.Tensor`, *optional*): Mask to prevent attention to certain positions.
            position_ids (`torch.LongTensor`, *optional*): Position indices for rotary embeddings.
            past_key_value (`Cache`, *optional*): Cached key-value states for faster decoding.
            output_attentions (`bool`): Whether to return attention weights.
            use_cache (`bool`): Whether to use and return the cache.
            kb_kvs (`tuple`, *optional*): Tuple of (kb_keys, kb_values) to inject.
            kb_config (`KBLaMConfig`, *optional*): Configuration for KB injection.
            save_attention_weights (`bool`): (Not implemented) Whether to save attention weights.
            attention_save_loc (`str`, *optional*): (Not implemented) Location to save weights.
            attention_file_base_name (`str`, *optional*): (Not implemented) File name for weights.

        Returns:
            A tuple containing `(attn_output, attn_weights, past_key_value)`.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states_2 = self.q_proj_new(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query_states_2 = query_states_2.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights_2 = None
        kb_layer_frequency = kb_config.kb_layer_frequency if kb_config else -1
        if kb_kvs is not None and kb_layer_frequency > 0 and self.layer_idx % kb_layer_frequency == 0:
            kb_keys, kb_values = kb_kvs
            kb_idx = self.layer_idx // kb_layer_frequency
            dynamic_sparsify = kb_config.dynamic_sparsify
            topk_size = kb_config.top_k_kb

            if len(kb_keys.shape) == 2:  # Not batch dim
                kb_len = kb_keys.shape[0]
                kb_keys = kb_keys.reshape(
                    kb_len,
                    1 + self.config.num_hidden_layers // kb_layer_frequency,
                    -1,
                )[:, kb_idx]
                kb_values = kb_values.reshape(
                    kb_len,
                    1 + self.config.num_hidden_layers // kb_layer_frequency,
                    -1,
                )[:, kb_idx]
                kb_keys = kb_keys.view(
                    kb_len, self.num_heads, self.head_dim
                ).transpose(0, 1)
                kb_values = kb_values.view(
                    kb_len, self.num_heads, self.head_dim
                ).transpose(0, 1)
                kb_keys = kb_keys.unsqueeze(0).expand(
                    bsz, self.num_heads, kb_len, self.head_dim
                )
                kb_values = kb_values.unsqueeze(0).expand(
                    bsz, self.num_heads, kb_len, self.head_dim
                )
                if dynamic_sparsify:
                    kb_keys, kb_values, attn_weights_2 = self.prune_key_value(
                        query_states_2, kb_keys, kb_values, topk_size
                    )
                key_states = torch.cat([kb_keys, key_states], dim=2)
                value_states = torch.cat([kb_values, value_states], dim=2)
            elif len(kb_keys.shape) == 3:  # Has a batch dim
                kb_len = kb_keys.shape[1]
                kb_keys = kb_keys.view(
                    bsz,
                    kb_len,
                    1 + self.config.num_hidden_layers // kb_layer_frequency,
                    -1,
                )[:, :, kb_idx]
                kb_values = kb_values.view(
                    bsz,
                    kb_len,
                    1 + self.config.num_hidden_layers // kb_layer_frequency,
                    -1,
                )[:, :, kb_idx]
                kb_keys = kb_keys.view(
                    bsz, kb_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                kb_values = kb_values.view(
                    bsz, kb_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                if dynamic_sparsify:
                    kb_keys, kb_values, attn_weights_2 = self.prune_key_value(
                        query_states_2, kb_keys, kb_values, topk_size
                    )
                key_states = torch.cat([kb_keys, key_states], dim=2)
                value_states = torch.cat([kb_values, value_states], dim=2)

            kb_len = kb_keys.shape[2]
            kb_atten_mask = attention_mask.new_zeros(bsz, 1, q_len, kb_len)
            padding_mask = torch.all(attention_mask < -1, -1, keepdim=True)
            kb_atten_mask = (
                padding_mask * torch.finfo(attention_mask.dtype).min + (~padding_mask) * kb_atten_mask
            )
            attention_mask = torch.cat([kb_atten_mask, attention_mask], dim=-1)
            kv_seq_len += kb_len

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        sep_query_head = kb_config.sep_query_head if kb_config else False
        if sep_query_head and kb_kvs is not None and kb_layer_frequency > 0 and self.layer_idx % kb_layer_frequency == 0:
            if attn_weights_2 is None:
                attn_weights_2 = torch.matmul(
                    query_states_2, kb_keys.transpose(2, 3)
                ) / math.sqrt(self.head_dim)
            
            attn_weights = attn_weights[:, :, :, kb_len:]
            kb_scale_factor = kb_config.kb_scale_factor
            if kb_scale_factor is not None:
                attn_weights_2 = (
                    attn_weights_2 - torch.log(torch.tensor(kb_len)) + torch.log(torch.tensor(kb_scale_factor))
                )
            attn_weights = torch.cat([attn_weights_2, attn_weights], -1)


        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Gemma3nDecoderLayer(nn.Module):
    """
    A single decoder layer for the Gemma-3n model. It consists of a self-attention
    block (`KblamGemma3nAttention`) and a feed-forward MLP block.
    """
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = KblamGemma3nAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3nMLP(config, layer_idx=layer_idx)
        self.input_layernorm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        kb_kvs: Optional[tuple] = None,
        kb_config: Optional[KBLaMConfig] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass for the Gemma3nDecoderLayer.

        Args:
            hidden_states (`torch.Tensor`): Input to the layer.
            attention_mask (`torch.Tensor`, *optional*): Attention mask.
            position_ids (`torch.LongTensor`, *optional*): Position IDs.
            past_key_value (`Tuple[torch.Tensor]`, *optional*): Cached key-value states.
            output_attentions (`bool`, *optional*): Whether to return attention weights.
            use_cache (`bool`, *optional*): Whether to use the cache.
            kb_kvs (`tuple`, *optional*): Knowledge base key-values.
            kb_config (`KBLaMConfig`, *optional*): KBLaM configuration.

        Returns:
            A tuple containing `(hidden_states, self_attn_weights, present_key_value)`.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            kb_kvs=kb_kvs,
            kb_config=kb_config,
        )
        hidden_states = residual + attn_outputs

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
