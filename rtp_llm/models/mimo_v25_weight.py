# rtp_llm/models/mimo_v25_weight.py
# MiMo V2.5 权重映射（实施手册 Step 3）。
#
# 与通用路径的三个差异（均有专门处理）：
#   1. ckpt 的 QKV 是单个融合张量 qkv_proj.weight，通用 FP8 量化包装
#      （PerBlockFp8Weight._get_qkv_quant_weight）假设 Q/K/V 为三个独立张量并
#      硬编码 merge_te_qkv，对 MiMo 会解包失败 → MiMoPerBlockFp8Weight 接管。
#   2. o_proj 是 BF16（在 quantization_config.ignored_layers 中，无
#      weight_scale_inv）→ 打上 MiMo 标记后基类量化包装放行，走普通 BF16 路径。
#   3. 每层 kv 头数不同（GA=4 / SWA=8）且 K≠V 头维度（192/128），标准 TP 切分
#      只读全局 load_config → MiMo 专用权重类重写 _split，用每层 AttnConfig。
#
# 非文本权重（visual.* / audio_encoder.* / speech_embeddings.* / model.mtp.*）
# 不在本文件登记，加载器按登记清单过滤 ckpt，天然排除。
import functools
from typing import Any, Dict, List, Union

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig, QuantizationConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
)
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    QS_SUFFIX,
    W_SUFFIX,
    PerBlockFp8Weight,
    W8A8Fp8PerBlockAttnAtomicWeight,
)
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    WeightModule,
)
from rtp_llm.ops import HybridAttentionType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    get_sp_tensor_kv_asym,
    identity,
    stack_,
    stack_moe_w1,
    transpose,
    transpose_pad,
    zeros,
)

# ---------------------------------------------------------------------------
# FP8 scale 行区间（Step 3.2）
# ---------------------------------------------------------------------------
# qkv_proj.weight_scale_inv 的行数不是简单的 rows/128：官方量化时 Q/K/V 三段
# 各自按 4 分片、每分片行数向上取整到 128 边界后拼接（GA: 24×4 + 2×4 + 1×4 = 108；
# SWA: 24×4 + 3×4 + 2×4 = 116）。以下区间为对实际 ckpt 逐块 amax 验证后的结论；
# 注意实施手册 Step 3.2 示例中 GA 的 K 段写作 (96, 102) 是按简单 ceil 推的，有误，
# 实际为 (96, 104)（每分片 192 行权重对应 2 行 scale：128 + 64 半块 padding）。
QKV_QUANT_SHARDS = 4  # ckpt 量化时的分片数（与 TP 部署无关，是 ckpt 布局常量）
QKV_SCALE_ROWS = {
    4: {"q": (0, 96), "k": (96, 104), "v": (104, 108)},  # GA 层（kv=4）
    8: {"q": (0, 96), "k": (96, 108), "v": (108, 116)},  # SWA 层（kv=8）
}

# attention_value_scale：参考实现在写 KV cache 前对 V 乘 0.707，等价于把它
# 折进 v_proj 的输出。FP8 权重不动数据本身，只缩放 V 段的 weight_scale_inv。
ATTENTION_VALUE_SCALE = 0.707


def check_qkv_scale_rows(s: torch.Tensor, kv_heads: int) -> Dict[str, Any]:
    """校验 scale 总行数与分片布局公式一致；ckpt 换版本时先在这里失败，不静默错。"""
    r = QKV_SCALE_ROWS[kv_heads]
    assert s.shape[0] == r["v"][1], (
        f"qkv scale rows {s.shape[0]} != expected {r['v'][1]} (kv_heads={kv_heads}); "
        f"ckpt 的 scale 分段布局可能已变化，需重跑 Step 0.3 的验证脚本"
    )
    return r


def process_mimo_qkv_scale(ts: List[torch.Tensor], kv_heads: int) -> torch.Tensor:
    """qkv scale 加载处理：行数断言 + attention_value_scale 折入 V 段（Step 3.3）。"""
    s = ts[0]
    rows = check_qkv_scale_rows(s, kv_heads)
    s = s.clone()
    vb, ve = rows["v"]
    s[vb:ve] = s[vb:ve] * ATTENTION_VALUE_SCALE
    return s


def _tp_bypass(load_config: LoadConfig) -> bool:
    """与 AtomicWeight._split 的短路条件一致：单卡无需切分。"""
    return (
        load_config.tp_size <= 1
        and load_config.dp_size <= 1
        and load_config.ep_size <= 1
    )


def sp_mimo_qkv_scale(
    s: torch.Tensor, tp: int, tp_rank: int, kv_heads: int
) -> torch.Tensor:
    """qkv scale 的 TP 切分：scale 行本身按 QKV_QUANT_SHARDS 分片生成，
    每 rank 取各段中属于自己的连续分片行，天然对齐（含 K 段的 padding 半块）。"""
    rows = check_qkv_scale_rows(s, kv_heads)
    assert (
        QKV_QUANT_SHARDS % tp == 0
    ), f"MiMo qkv scale 切分要求 tp 整除 {QKV_QUANT_SHARDS}，got tp={tp}"
    shards_per_rank = QKV_QUANT_SHARDS // tp
    parts = []
    for seg in ("q", "k", "v"):
        b, e = rows[seg]
        per_shard = (e - b) // QKV_QUANT_SHARDS
        s0 = b + tp_rank * shards_per_rank * per_shard
        parts.append(s[s0 : s0 + shards_per_rank * per_shard])
    return torch.concat(parts, dim=0).contiguous()


# ---------------------------------------------------------------------------
# MiMo 标记权重类
# ---------------------------------------------------------------------------
# 类属性 is_mimo_v25 是量化包装分发的判别标记（见 model_weight.is_mimo_v25_weight
# 与 PerBlockFp8Weight.support 中的排除逻辑），作用等同 DSV4 的 is_v4_weight。


class MiMoAttnAtomicWeight(AttnAtomicWeight):
    """MiMo qkv 权重（非量化路径，登记态）。量化 ckpt 下会被
    MiMoPerBlockFp8Weight 接管；此处的 _split 覆盖服务于 BF16 ckpt 场景。"""

    is_mimo_v25 = True

    def _split(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        raw = tensor if isinstance(tensor, torch.Tensor) else tensor[self.name]
        if _tp_bypass(load_config):
            return {self.name: raw}
        # transpose 后布局 [hidden, q+k+v]，沿最后一维按段切；
        # 每层 kv 头数取自本权重的 AttnConfig，不用全局 load_config 的单一值
        cfg = self.config
        ts = get_sp_tensor_kv_asym(
            raw,
            head_num=cfg.head_num,
            head_num_kv=cfg.head_num_kv,
            size_per_head=cfg.size_per_head,
            v_size_per_head=cfg.v_size_per_head,
            tp=load_config.tp_size,
            tp_rank=load_config.tp_rank,
        )
        return {self.name: ts.contiguous().clone()}


class MiMoBf16AtomicWeight(AtomicWeight):
    """MiMo 的 BF16 权重（o_proj）：打标记让 FP8 量化包装放行（ckpt 中无
    o_proj.weight_scale_inv），切分沿用 W.gpt_style_tp_strategy 的默认规则。"""

    is_mimo_v25 = True

    def _get_split_func(self):
        return W.gpt_style_tp_strategy[self.name]


class MiMoQkvW8A8KernelWeight(W8A8Fp8PerBlockAttnAtomicWeight):
    """量化路径下的 qkv 权重（FP8 kernel，[out, in] 布局）。"""

    is_mimo_v25 = True

    def _split(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        raw = tensor if isinstance(tensor, torch.Tensor) else tensor[self.name]
        if _tp_bypass(load_config):
            return {self.name: raw}
        cfg = self.config
        # [out, in] → 转成 [in, out] 沿输出维按段切（对齐 sp_head_gemm_a8 的做法）
        t = raw.reshape([raw.shape[0], -1]).T
        ts = get_sp_tensor_kv_asym(
            t,
            head_num=cfg.head_num,
            head_num_kv=cfg.head_num_kv,
            size_per_head=cfg.size_per_head,
            v_size_per_head=cfg.v_size_per_head,
            tp=load_config.tp_size,
            tp_rank=load_config.tp_rank,
        ).T
        return {self.name: ts.contiguous().clone()}


class MiMoQkvW8A8ScaleWeight(W8A8Fp8PerBlockAttnAtomicWeight):
    """量化路径下的 qkv scale（[rows, in/128] 布局），按分片行区间切分。"""

    is_mimo_v25 = True

    def _split(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        raw = tensor if isinstance(tensor, torch.Tensor) else tensor[self.name]
        if _tp_bypass(load_config):
            return {self.name: raw}
        ts = sp_mimo_qkv_scale(
            raw,
            tp=load_config.tp_size,
            tp_rank=load_config.tp_rank,
            kv_heads=self.config.head_num_kv,
        )
        return {self.name: ts.clone()}


# ---------------------------------------------------------------------------
# MiMo 专用 FP8 量化包装（仿 V4PerBlockFp8Weight 先例）
# ---------------------------------------------------------------------------


class MiMoPerBlockFp8Weight(PerBlockFp8Weight):
    """只接管 MiMo 标记的融合 qkv：
    - kernel：单融合张量原样加载（ckpt 即 [q+k+v, hidden]，与通用路径
      merge_te_qkv 的输出布局一致），不做三元解包；
    - scale：单张量加载 + 行数断言 + value_scale 折入 V 段。
    其余（FFN / MoE）权重不打标记，仍由基类 PerBlockFp8Weight 按通用路径处理。"""

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, Fp8BlockWiseQuantConfig
        ):
            return False
        if not getattr(src_weight_info, "is_mimo_v25", False):
            return False
        return src_weight_info.name == W.attn_qkv_w

    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        assert src_weight_info.name == W.attn_qkv_w, src_weight_info.name
        self.group_size = quant_config.group_size()
        kernel, scale = self._get_mimo_fused_qkv_pair(src_weight_info)
        sub_weights = {kernel.name: kernel, scale.name: scale}
        # 绕开基类按名分发的 if/elif 链（其 qkv 分支假设 3 个独立张量）
        CompositeWeight.__init__(
            self, sub_weights, quant_config=quant_config, *args, **kwargs
        )
        self.kernel = kernel
        self.scale = scale

    def _get_mimo_fused_qkv_pair(self, src_weight_info: MiMoAttnAtomicWeight):
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        kv_heads = src_weight_info.config.head_num_kv
        kernel = MiMoQkvW8A8KernelWeight(
            W.attn_qkv_w,
            [CkptWeightInfo(w_name + W_SUFFIX, identity)],
            identity,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = MiMoQkvW8A8ScaleWeight(
            W.attn_qkv_s,
            [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
            functools.partial(process_mimo_qkv_scale, kv_heads=kv_heads),
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return kernel, scale


# ---------------------------------------------------------------------------
# 权重映射主类
# ---------------------------------------------------------------------------


class MiMoV25Weight(ModelDeployWeightInfo):
    def __init__(self, prefix: str = None, **kwargs: Any):
        self.prefix = prefix or ""
        self.model_prefix = "model."
        self.bias = False  # attention_bias=false，MiMo 无 qkv / o bias
        super().__init__(**kwargs)
        # QK=192 / V=128（Step 2 新增的配置字段）
        self._v_size_per_head = self.model_config.attn_config.v_size_per_head

    def _process_meta(self, meta_dicts: Any, weight_keys: List[str]):
        self.transformer_prefix = self.prefix + self.model_prefix

    def _get_weight_info(self) -> ModelWeightInfo:
        return self._get_hf_weight_info()

    def _get_hf_weight_info(self) -> ModelWeightInfo:
        weights = [
            AtomicWeight(
                W.embedding,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "embed_tokens.weight", identity
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(self.prefix + "lm_head.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.transformer_prefix + "norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
        ]
        layer_weights: List[List[WeightModule]] = []
        for layer_id in range(self._num_layers):
            layer_weights.append(self._get_hf_layer_weight_info(layer_id))
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

    def _is_ga_layer(self, layer_id: int) -> bool:
        types = self.model_config.hybrid_attention_config.hybrid_attention_types
        return types[layer_id] != HybridAttentionType.SLIDING_WINDOW

    def _get_hf_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        mimo_cfg = self.model_config.mimo_v25_config
        is_ga = self._is_ga_layer(layer_id)
        # 不能用 self._head_num_kv —— 它恒为 8（Step 1 填的是 SWA 值），GA 层要取 4
        kv_heads = (
            mimo_cfg["ga_kv_head_num"]
            if is_ga
            else mimo_cfg["swa_kv_head_num"]
        )

        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,  # 192，QK
            v_size_per_head=self._v_size_per_head,  # 128
            head_num=self._head_num,  # 64
            head_num_kv=kv_heads,
        )
        weights: List[WeightModule] = [
            AtomicWeight(
                W.pre_ln_gamma,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.input_layernorm.weight"
                    )
                ],
                identity,
            ),
            # qkv 保持融合，只做 transpose（Step 3.1）；FP8 场景由
            # MiMoPerBlockFp8Weight 接管，scale 存 W.attn_qkv_s
            MiMoAttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.self_attn.qkv_proj.weight"
                    )
                ],
                transpose,
                config=attn_config,
            ),
            # o_proj：BF16（ignored_layers，无 scale）、输入维 64×128=8192，
            # 走非量化路径（Step 3.4）
            MiMoBf16AtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.self_attn.o_proj.weight"
                    )
                ],
                transpose,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        self.transformer_prefix
                        + "layers.{i}.post_attention_layernorm.weight"
                    )
                ],
                identity,
            ),
        ]

        if not is_ga:  # 只有 SWA 层有 sink bias，BF16 [64]（Step 0.2）
            weights.append(
                AtomicWeight(
                    W.attn_sink_bias,
                    [
                        CkptWeightInfo(
                            self.transformer_prefix
                            + "layers.{i}.self_attn.attention_sink_bias"
                        )
                    ],
                    identity,
                )
            )

        weights.extend(self._get_ffn_layer_weight_info(layer_id))
        return weights

    def _get_ffn_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        """layer 0 dense（inter=16384），layer 1~47 MoE（256 专家，inter=2048）。
        MoE 结构与 deepseek_v2 同构（sigmoid + e_score_correction_bias +
        moe_layer_index 区分 dense/MoE，无共享专家），映射对齐其实现。"""
        if layer_id in self.moe_layer_index_:
            return self._get_moe_layer_weight_info(layer_id)
        return self._get_dense_ffn_layer_weight_info(layer_id)

    def _get_dense_ffn_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        align_size = self._align_size
        ffn_config = FfnConfig(
            align_size=align_size,
            is_gated_activation=self._is_gated_activation,
            is_moe=False,
        )
        return [
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.gate_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w2,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.down_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=1),
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.up_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            )
        ]

    def _get_moe_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        moe_config = MoeConfig(
            align_size=self._align_size,
            expert_num=self.expert_num_,
        )
        return [
            MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix + "layers.{i}.mlp.gate.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.experts.{expert_id}.down_proj.weight",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w1,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.experts.{expert_id}.up_proj.weight",
                                identity,
                            )
                        ]
                        + [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.experts.{expert_id}.gate_proj.weight",
                                identity,
                            )
                        ],
                        stack_moe_w1,
                        config=moe_config,
                    ),
                ],
                config=moe_config,
            ),
            # noaux_tc 路由：bias 只参与选专家，topk 权重用不含 bias 的 sigmoid 分数
            AtomicWeight(
                W.e_score_correction_b,
                [
                    CkptWeightInfo(
                        self.transformer_prefix
                        + "layers.{i}.mlp.gate.e_score_correction_bias",
                        identity,
                    )
                ],
                identity,
                data_type=torch.float32,
            ),
        ]
