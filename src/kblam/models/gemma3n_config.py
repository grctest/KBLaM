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
"""Gemma-3n model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Gemma3nTextConfig(PretrainedConfig):
    def __init__(
        self,
        activation_sparsity_pattern=None,
        altup_active_idx=0,
        altup_coef_clip=120.0,
        altup_correct_scale=True,
        altup_lr_multiplier=1.0,
        altup_num_inputs=4,
        attention_bias=False,
        attention_dropout=0.0,
        final_logit_softcapping=30.0,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        hidden_size=2048,
        hidden_size_per_layer_input=256,
        initializer_range=0.02,
        intermediate_size=8192,
        laurel_rank=64,
        layer_types=None,
        max_position_embeddings=32768,
        model_type="gemma3n_text",
        num_attention_heads=8,
        num_hidden_layers=30,
        num_key_value_heads=2,
        num_kv_shared_layers=10,
        query_pre_attn_scalar=256,
        rms_norm_eps=1e-06,
        rope_local_base_freq=10000.0,
        rope_scaling=None,
        rope_theta=1000000.0,
        sliding_window=512,
        use_cache=True,
        vocab_size=262400,
        vocab_size_per_layer_input=262144,
        **kwargs,
    ):
        self.activation_sparsity_pattern = activation_sparsity_pattern
        self.altup_active_idx = altup_active_idx
        self.altup_coef_clip = altup_coef_clip
        self.altup_correct_scale = altup_correct_scale
        self.altup_lr_multiplier = altup_lr_multiplier
        self.altup_num_inputs = altup_num_inputs
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.final_logit_softcapping = final_logit_softcapping
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.hidden_size = hidden_size
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.laurel_rank = laurel_rank
        self.layer_types = layer_types
        self.max_position_embeddings = max_position_embeddings
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_kv_shared_layers = num_kv_shared_layers
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.rms_norm_eps = rms_norm_eps
        self.rope_local_base_freq = rope_local_base_freq
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        super().__init__(**kwargs)

class Gemma3nVisionConfig(PretrainedConfig):
    def __init__(
        self,
        architecture="mobilenetv5_300m_enc",
        do_pooling=True,
        hidden_size=2048,
        initializer_range=0.02,
        label_names=None,
        model_type="gemma3n_vision",
        num_classes=2,
        rms_norm_eps=1e-06,
        vocab_offset=262144,
        vocab_size=128,
        **kwargs,
    ):
        self.architecture = architecture
        self.do_pooling = do_pooling
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.label_names = label_names if label_names is not None else ["LABEL_0", "LABEL_1"]
        self.model_type = model_type
        self.num_classes = num_classes
        self.rms_norm_eps = rms_norm_eps
        self.vocab_offset = vocab_offset
        self.vocab_size = vocab_size
        super().__init__(**kwargs)

class Gemma3nAudioConfig(PretrainedConfig):
    def __init__(
        self,
        conf_attention_chunk_size=12,
        conf_attention_context_left=13,
        conf_attention_context_right=0,
        conf_attention_logit_cap=50.0,
        conf_conv_kernel_size=5,
        conf_num_attention_heads=8,
        conf_num_hidden_layers=12,
        conf_positional_bias_size=256,
        conf_reduction_factor=4,
        conf_residual_weight=0.5,
        gradient_clipping=10000000000.0,
        hidden_size=1536,
        input_feat_size=128,
        model_type="gemma3n_audio",
        rms_norm_eps=1e-06,
        sscp_conv_channel_size=None,
        sscp_conv_eps=0.001,
        sscp_conv_kernel_size=None,
        sscp_conv_stride_size=None,
        vocab_offset=262272,
        vocab_size=128,
        **kwargs,
    ):
        self.conf_attention_chunk_size = conf_attention_chunk_size
        self.conf_attention_context_left = conf_attention_context_left
        self.conf_attention_context_right = conf_attention_context_right
        self.conf_attention_logit_cap = conf_attention_logit_cap
        self.conf_conv_kernel_size = conf_conv_kernel_size
        self.conf_num_attention_heads = conf_num_attention_heads
        self.conf_num_hidden_layers = conf_num_hidden_layers
        self.conf_positional_bias_size = conf_positional_bias_size
        self.conf_reduction_factor = conf_reduction_factor
        self.conf_residual_weight = conf_residual_weight
        self.gradient_clipping = gradient_clipping
        self.hidden_size = hidden_size
        self.input_feat_size = input_feat_size
        self.model_type = model_type
        self.rms_norm_eps = rms_norm_eps
        self.sscp_conv_channel_size = sscp_conv_channel_size if sscp_conv_channel_size is not None else [128, 32]
        self.sscp_conv_eps = sscp_conv_eps
        self.sscp_conv_kernel_size = sscp_conv_kernel_size if sscp_conv_kernel_size is not None else [[3, 3], [3, 3]]
        self.sscp_conv_stride_size = sscp_conv_stride_size if sscp_conv_stride_size is not None else [[2, 2], [2, 2]]
        self.vocab_offset = vocab_offset
        self.vocab_size = vocab_size
        super().__init__(**kwargs)

class Gemma3nConfig(PretrainedConfig):
    model_type = "gemma3n"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        audio_soft_tokens_per_image=188,
        audio_token_id=262273,
        boa_token_id=256000,
        boi_token_id=255999,
        eoa_token_id=262272,
        eoi_token_id=262144,
        image_token_id=262145,
        initializer_range=0.02,
        vision_soft_tokens_per_image=256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the Gemma3nTextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the Gemma3nVisionConfig with default values.")
        
        if audio_config is None:
            audio_config = {}
            logger.info("audio_config is None. Initializing the Gemma3nAudioConfig with default values.")

        self.text_config = Gemma3nTextConfig(**text_config)
        self.vision_config = Gemma3nVisionConfig(**vision_config)
        self.audio_config = Gemma3nAudioConfig(**audio_config)
        
        self.audio_soft_tokens_per_image = audio_soft_tokens_per_image
        self.audio_token_id = audio_token_id
        self.boa_token_id = boa_token_id
        self.boi_token_id = boi_token_id
        self.eoa_token_id = eoa_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.initializer_range = initializer_range
        self.vision_soft_tokens_per_image = vision_soft_tokens_per_image

    @classmethod
    def from_text_vision_audio_configs(
        cls, text_config: Gemma3nTextConfig, vision_config: Gemma3nVisionConfig, audio_config: Gemma3nAudioConfig, **kwargs
    ) -> "Gemma3nConfig":
        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            audio_config=audio_config.to_dict(),
            **kwargs,
        )
