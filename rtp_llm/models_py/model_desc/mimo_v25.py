# rtp_llm/models_py/model_desc/mimo_v25.py
# MiMo V2.5 前向实现（实施手册 Step 7）：
# 与 qwen3 / generic_moe 模板的核心差异是双 fmha 实例 + 按层路由——
# GA（全局注意力）与 SWA（滑窗注意力）两类层的 kv 头数 / 窗口 / rope theta /
# sink bias 不同，各建一个 fmha 实例，逐层按层型选择。
import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_attention_inputs_for_tag,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    MultimodalEmbeddingInjector,
    RMSNorm,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import get_fmha_impl
from rtp_llm.ops import (
    HWKernelConfig,
    HybridAttentionType,
    MoeConfig,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class MiMoV25DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        is_ga: bool = False,
    ):
        super().__init__()
        self.is_ga = is_ga
        mimo_cfg = config.mimo_v25_config
        tp_size = parallelism_config.get_attn_tp_size()
        attn_configs = config.getAttentionConfigs(tp_size)

        def local_kv_heads(global_kv_heads: int) -> int:
            if global_kv_heads % tp_size == 0:
                return global_kv_heads // tp_size
            return global_kv_heads // math.gcd(global_kv_heads, tp_size)

        # 按层覆盖差异参数：kv_head / 窗口 / rope theta / sink。
        # rope_config.dim = 64（partial rope）两类层相同，沿用 Step 1 的解析结果。
        if is_ga:
            attn_configs.kv_head_num = local_kv_heads(mimo_cfg["ga_kv_head_num"])
            attn_configs.sliding_window = 0  # GA 不限窗
            attn_configs.rope_config.base = int(
                config.attn_config.rope_config.base
            )  # 1e7
            attn_configs.add_sink_bias = False
        else:
            attn_configs.kv_head_num = local_kv_heads(mimo_cfg["swa_kv_head_num"])
            attn_configs.sliding_window = mimo_cfg["window_size"]
            attn_configs.rope_config.base = int(mimo_cfg["swa_rope_theta"])
            attn_configs.add_sink_bias = mimo_cfg["add_sink_bias"]
        self.attn_configs = attn_configs

        self.self_attn = CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
            layer_idx,
        )
        # SWA layers carry a per-head sink bias; GA layers do not.
        self.sink_bias: Optional[torch.Tensor] = weights.get(W.attn_sink_bias, None)
        # layer 0 是 dense FFN（inter=16384），其余是 256 专家 MoE（noaux_tc 路由，
        # correction_bias 逻辑由 GenericMoeLayer 内部处理）
        if layer_idx in config.moe_layer_index:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config,
            )
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiMoV25Model(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        types = config.hybrid_attention_config.hybrid_attention_types
        self.is_ga_layer: List[bool] = [
            t != HybridAttentionType.SLIDING_WINDOW for t in types
        ]

        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.layers = nn.ModuleList(
            [
                MiMoV25DecoderLayer(
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    quant_config=quant_config,
                    hw_kernel_config=py_hw_kernel_config,
                    is_ga=self.is_ga_layer[idx],
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        attention_inputs = get_attention_inputs_value(inputs)
        if not isinstance(attention_inputs, Mapping):
            raise RuntimeError("MiMo hybrid attention requires tagged attention inputs")
        layer_by_tag = {
            "ga": self.layers[self.is_ga_layer.index(True)],
            "swa": self.layers[self.is_ga_layer.index(False)],
        }
        implementations = {}
        for tag, layer in layer_by_tag.items():
            group_inputs = select_attention_inputs_for_tag(attention_inputs, tag)
            group_inputs.headwise_config = getattr(
                self.config, "headwise_config", None
            )
            implementations[tag] = get_fmha_impl(
                layer.attn_configs,
                self.weight,
                group_inputs,
                self.fmha_config,
                self.config.quant_config,
                is_cuda_graph,
                self.config.max_seq_len,
                self.parallelism_config,
            )
        return implementations

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embed_tokens(
            input_ids,
            inputs.combo_position_ids,
            inputs.embedding_inputs.combo_tokens_type_ids,
            inputs.embedding_inputs.text_tokens_mask,
        )
        hidden_states = self.multimodal_embedding_injector(
            hidden_states,
            inputs.multimodal_inputs.multimodal_features,
            inputs.multimodal_inputs.mm_features_locs,
        )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            # Set per-layer sink bias: SWA layers have the bias, GA layers get None.
            if decoder_layer.sink_bias is not None and hasattr(impl, "set_sink_bias"):
                impl.set_sink_bias(decoder_layer.sink_bias)
            hidden_states = decoder_layer(
                hidden_states,
                impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states)


__all__ = [
    "MiMoV25Model",
]
