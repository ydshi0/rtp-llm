# rtp_llm/models/mimo_v25.py
import json
import os
from typing import Any, Dict, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import CacheGroupType, HybridAttentionType, KVCacheSpecDesc, KVCacheSpecType


class MiMoV25(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        with open(os.path.join(ckpt_path, "config.json")) as f:
            cj = json.load(f)

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        cls._parse_basic_config(cj, config)
        cls._parse_stop_words(ckpt_path, config)
        cls._parse_rope_config(cj, config)
        cls._parse_normalization_config(cj, config)
        cls._parse_hybrid_attention_config(cj, config)
        cls._parse_swa_config(cj, config)
        cls._parse_moe_config(cj, config)
        cls._parse_vit_config(cj, config)
        return config

    @classmethod
    def _parse_basic_config(cls, cj: Dict[str, Any], config: ModelConfig):
        config.hidden_size = cj["hidden_size"]  # 4096
        config.num_layers = cj["num_hidden_layers"]  # 48
        config.vocab_size = cj["vocab_size"]  # 152576
        config.max_seq_len = cj["max_position_embeddings"]  # 1048576
        config.config_dtype = cj.get("dtype", None)  # "bfloat16"
        config.tie_word_embeddings = cj.get("tie_word_embeddings", False)  # False
        config.special_tokens.eos_token_id = cj["eos_token_id"]  # 151645 <|im_end|>

        config.attn_config.head_num = cj["num_attention_heads"]  # 64
        config.attn_config.size_per_head = cj["head_dim"]  # 192（QK）
        config.attn_config.v_size_per_head = cj["v_head_dim"]
        # kv_head_num 填 SWA 的 8（39 层多数派），GA 的 4 见下方说明
        config.attn_config.kv_head_num = cj["swa_num_key_value_heads"]  # 8

        # GA/SWA 两类层共用上面这一套 head 维度：fused qkv 的行边界、o_proj 输入维、
        # 两个 KV spec 的 stride 全由它们推出。config schema 允许 swa_* 取不同值
        # （configuration_mimo_v2.py:243-248），一旦不同就必须改成逐层维度，
        # 否则形状静默错到对拍阶段才暴露，所以在入口卡死
        assert cj.get("swa_head_dim", cj["head_dim"]) == cj["head_dim"]
        assert cj.get("swa_v_head_dim", cj["v_head_dim"]) == cj["v_head_dim"]
        assert (
            cj.get("swa_num_attention_heads", cj["num_attention_heads"])
            == cj["num_attention_heads"]
        )
        # Step 3 的权重映射按「融合 qkv 单 tensor」切分
        assert cj.get("attention_projection_layout") == "fused_qkv"

    @classmethod
    def _parse_stop_words(cls, ckpt_path: str, config: ModelConfig):
        # generation_config.json 的 eos 往往是个列表（本 ckpt 三项），比 config.json 的
        # 单值多；框架 eos_token_id 只有一个，其余的走 stop_words（参照 qwen_v2.py:350）。
        # 全部从 ckpt 读，不写死 token id
        path = os.path.join(ckpt_path, "generation_config.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            eos = json.load(f).get("eos_token_id")
        if eos is None:
            return
        if isinstance(eos, int):
            eos = [eos]
        config.special_tokens.stop_words_id_list = [
            [t] for t in eos if t != config.special_tokens.eos_token_id
        ]

    @classmethod
    def _parse_rope_config(cls, cj: Dict[str, Any], config: ModelConfig):
        # partial rope：rope 只作用于 head 内前 int(192*0.334)=64 维
        config.attn_config.rope_config.style = 1  # RopeStyle::Base（NEOX）
        config.attn_config.rope_config.base = int(cj["rope_theta"])  # GA: 1e7
        config.partial_rotary_factor = cj["partial_rotary_factor"]  # 0.334
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * config.partial_rotary_factor
        )  # → 64
        assert config.attn_config.rope_config.dim % 2 == 0

    @classmethod
    def _parse_normalization_config(cls, cj: Dict[str, Any], config: ModelConfig):
        config.layernorm_eps = cj["layernorm_epsilon"]  # 1e-5
        config.norm_type = "rmsnorm"
        # MLP 是 gate/up/down 三件套（modeling_mimo_v2.py:131），即 gated silu
        assert cj["hidden_act"] == "silu", cj["hidden_act"]
        config.activation_type = "SiGLU"
        config.qk_norm = False
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True

    @classmethod
    def _parse_hybrid_attention_config(cls, cj: Dict[str, Any], config: ModelConfig):
        config.hybrid_attention_config.enable_hybrid_attention = True
        pattern = cj["hybrid_layer_pattern"]  # 0=GA(full), 1=SWA
        assert len(pattern) == config.num_layers
        types: List[HybridAttentionType] = []
        for v in pattern:
            types.append(
                HybridAttentionType.NONE
                if v == 0  # full attention
                else HybridAttentionType.SLIDING_WINDOW
            )
        config.hybrid_attention_config.hybrid_attention_types = types

    @classmethod
    def _parse_swa_config(cls, cj: Dict[str, Any], config: ModelConfig):
        config.mimo_v25_config = {
            "window_size": int(cj["sliding_window"]),
            "swa_kv_head_num": int(cj["swa_num_key_value_heads"]),
            "swa_rope_theta": float(cj["swa_rope_theta"]),
            "ga_kv_head_num": int(cj["num_key_value_heads"]),
            "add_sink_bias": bool(cj["add_swa_attention_sink_bias"]),
        }
        config.attn_config.sliding_window = config.mimo_v25_config["window_size"]
        assert not cj.get(
            "add_full_attention_sink_bias", False
        ), "GA layers must not use attention sinks"
        config.attn_config.add_sink_bias = config.mimo_v25_config["add_sink_bias"]

    @classmethod
    def _parse_vit_config(cls, cj: Dict[str, Any], config: ModelConfig):
        vision_config = cj.get("vision_config")
        if not vision_config:
            return
        if int(vision_config["out_hidden_size"]) != config.hidden_size:
            raise ValueError("MiMo VIT output size must match the LLM hidden size")
        config.mm_model_config.is_multimodal = True
        config.mm_related_params.config = dict(vision_config)
        config.mm_related_params.config["ckpt_path"] = config.ckpt_path
        config.mm_related_params.config["processor_config"] = cj.get(
            "processor_config", {}
        )
        processor = cj.get("processor_config", {})
        vision_start = processor.get("vision_start_token_id", cj.get("vision_start_token_id"))
        vision_end = processor.get("vision_end_token_id", cj.get("vision_end_token_id"))
        if vision_start is not None and vision_end is not None:
            config.mm_model_config.mm_sep_tokens = [[int(vision_start), int(vision_end)]]
        config.mm_related_params.special_tokens.update(
            {"default_mm_token": "<|vision_start|><|image_pad|><|vision_end|>"}
        )

    @classmethod
    def _post_build_model_config(cls, config: ModelConfig) -> None:
        mimo = config.mimo_v25_config
        config.kv_cache_spec_descs = []
        for attention_type in config.hybrid_attention_config.hybrid_attention_types:
            desc = KVCacheSpecDesc()
            desc.cache_type = KVCacheSpecType.MHA
            desc.tag = "ga" if attention_type != HybridAttentionType.SLIDING_WINDOW else "swa"
            if attention_type == HybridAttentionType.SLIDING_WINDOW:
                desc.group_type = CacheGroupType.SWA
            desc.mha_kv_head_num = (
                mimo["ga_kv_head_num"]
                if attention_type != HybridAttentionType.SLIDING_WINDOW
                else mimo["swa_kv_head_num"]
            )
            desc.mha_k_head_dim = config.attn_config.size_per_head
            desc.mha_v_head_dim = config.attn_config.v_size_per_head
            config.kv_cache_spec_descs.append([desc])

    @classmethod
    def _parse_moe_config(cls, cj: Dict[str, Any], config: ModelConfig):
        config.moe_k = cj["num_experts_per_tok"]  # 8
        config.expert_num = cj["n_routed_experts"]  # 256
        config.moe_inter_size = cj["moe_intermediate_size"]  # 2048
        config.inter_size = cj["intermediate_size"]  # 16384（layer 0 dense 用）
        config.has_moe_norm = cj.get("norm_topk_prob", True)  # True
        # moe_style 1=只有 routed expert，2=shared + routed。本 ckpt n_shared_experts=null；
        # 若变体带 shared expert，Step 3 的 FFN 权重映射要同步加一路，故在此卡死
        assert cj.get("n_shared_experts") is None, cj.get("n_shared_experts")
        config.moe_style = 1
        # 门控打分函数按 ckpt 字符串映射（参照 deepseek_v2.py:677-685）
        config.scoring_func = {"softmax": 0, "sigmoid": 1}[cj["scoring_func"]]
        # noaux_tc = sigmoid + e_score_correction_bias + 分组 topk，Step 3 要加载
        # gate.e_score_correction_bias；换成别的 topk_method 则 Step 7 的门控要重写
        assert cj["topk_method"] == "noaux_tc", cj["topk_method"]
        config.moe_n_group = cj.get("n_group") or 1  # 1
        config.moe_topk_group = cj.get("topk_group") or 1  # 1
        config.routed_scaling_factor = (
            cj.get("routed_scaling_factor") or 1.0
        )  # null→1.0

        # moe_layer_freq[i] 决定第 i 层是 MoE 还是 dense：layer 0 dense，1~47 MoE
        freq = cj["moe_layer_freq"]
        assert len(freq) == config.num_layers
        config.moe_layer_index = [i for i, f in enumerate(freq) if f]

    @staticmethod
    def get_weight_cls():
        from rtp_llm.models.mimo_v25_weight import MiMoV25Weight  # Step 3 创建

        return MiMoV25Weight

    def _create_python_model(self):
        # Step 7：双 fmha 实例 + 按层路由的前向实现（参照 qwen_v3_moe 的接法）
        from rtp_llm.models_py.model_desc.mimo_v25 import MiMoV25Model

        self.py_model = MiMoV25Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=self.model_config.quant_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


register_model("mimo_v25", MiMoV25, ["MiMoV2ForCausalLM"])
