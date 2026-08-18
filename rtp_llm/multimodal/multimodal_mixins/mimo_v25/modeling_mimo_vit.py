import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiMoVisionConfig:
    def __init__(self, **config):
        self.depth = int(config.get("depth", 28))
        self.hidden_size = int(config.get("hidden_size", 1280))
        self.intermediate_size = int(config.get("intermediate_size", 4608))
        self.num_heads = int(config.get("num_heads", 32))
        self.num_key_value_heads = int(config.get("num_key_value_heads", 8))
        self.qk_channels = int(config.get("qk_channels", 64))
        self.kv_channels = int(config.get("kv_channels", self.qk_channels))
        self.in_channels = int(config.get("in_channels", config.get("in_chans", 3)))
        self.patch_size = int(config.get("patch_size", 16))
        self.temporal_patch_size = int(config.get("temporal_patch_size", 2))
        self.spatial_merge_size = int(config.get("spatial_merge_size", 2))
        self.out_hidden_size = int(config["out_hidden_size"])
        self.visual_token_window_size = int(
            config.get("visual_token_window_size", 64)
        )
        pattern = config.get("vit_window_attn_types")
        self.vit_window_attn_types = list(pattern or [-1] * self.depth)
        pattern_full = {
            i
            for i, attention_type in enumerate(self.vit_window_attn_types)
            if attention_type == -1
        }
        configured_full = config.get("fullatt_block_indexes")
        self.fullatt_block_indexes = list(
            pattern_full if configured_full is None else configured_full
        )
        self.use_sink = bool(config.get("use_sink", False))
        self.hidden_act = str(config.get("hidden_act", "silu"))
        self.rms_norm_eps = float(config.get("rms_norm_eps", 1e-6))
        if len(self.vit_window_attn_types) != self.depth:
            raise ValueError("vit_window_attn_types length must equal vision depth")
        if not set(self.vit_window_attn_types).issubset({-1, 0, 1}):
            raise ValueError("vision attention type must be -1, 0, or 1")
        if pattern_full != set(self.fullatt_block_indexes):
            raise ValueError("full attention indexes disagree with attention pattern")
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("vision Q heads must be divisible by KV heads")


class MiMoRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).to(dtype) * self.weight


class MiMoVisionPatchEmbed(nn.Module):
    def __init__(self, config: MiMoVisionConfig):
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=(
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ),
            stride=(
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ),
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(x.to(self.proj.weight.dtype)).flatten(1)


class MiMoVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        positions = torch.arange(
            length, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(positions, self.inv_freq)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    q: torch.Tensor, k: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    cos = cos[:, None, :].float()
    sin = sin[:, None, :].float()
    q_dtype, k_dtype = q.dtype, k.dtype
    q = (q.float() * cos + _rotate_half(q.float()) * sin).to(q_dtype)
    k = (k.float() * cos + _rotate_half(k.float()) * sin).to(k_dtype)
    return q, k


class MiMoVisionEagerAttention(nn.Module):
    """Reference MiMo VIT attention. This class deliberately has no fast backend."""

    # Query block size used by windowed layers. Only bounds peak activation
    # memory: the result is identical to materialising a dense [seq, seq] mask,
    # but the cost is O(seq * (block + 2 * window)) instead of O(seq^2).
    window_chunk_size = 512

    def __init__(self, config: MiMoVisionConfig, use_sink: bool):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.qk_channels
        self.value_dim = config.kv_channels
        self.window_size = config.visual_token_window_size
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        v_size = self.num_kv_heads * self.value_dim
        self.qkv = nn.Linear(config.hidden_size, q_size + k_size + v_size, bias=True)
        self.proj = nn.Linear(self.num_heads * self.value_dim, config.hidden_size, bias=True)
        self.sinks = (
            nn.Parameter(torch.zeros(self.num_heads)) if use_sink else None
        )

    def _logits(self, qh: torch.Tensor, kh: torch.Tensor) -> torch.Tensor:
        return torch.matmul(qh.float(), kh.float().transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )

    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """FP32 softmax, optionally with an attention sink.

        The sink is a per-head logit that enters the softmax denominator only
        and contributes no value, so it is appended as an extra column and
        dropped afterwards without renormalising. Note that vLLM's MiMo vision
        tower loads ``sinks`` but never applies it (its flash_attn path cannot
        express the extra logit), so this branch has no reference to compare
        against and must be confirmed against the checkpoint's own HF modeling
        code during the golden pass.
        """
        if self.sinks is None:
            return torch.softmax(logits, dim=-1)
        sink = self.sinks.float().view(-1, *([1] * (logits.dim() - 1)))
        sink = sink.expand(*logits.shape[:-1], 1)
        return torch.softmax(torch.cat((logits, sink), dim=-1), dim=-1)[..., :-1]

    def _windowed_attention(
        self, qh: torch.Tensor, kh: torch.Tensor, vh: torch.Tensor
    ) -> torch.Tensor:
        seq_len = qh.shape[1]
        window = self.window_size
        outputs = []
        for start in range(0, seq_len, self.window_chunk_size):
            end = min(start + self.window_chunk_size, seq_len)
            # Every query in [start, end) can only reach keys within +-window,
            # so the whole block needs no key outside this slice.
            key_start = max(0, start - window)
            key_end = min(seq_len, end + window)
            logits = self._logits(qh[:, start:end], kh[:, key_start:key_end])
            query_pos = torch.arange(start, end, device=qh.device)
            key_pos = torch.arange(key_start, key_end, device=qh.device)
            local = (query_pos[:, None] - key_pos[None, :]).abs() <= window
            logits = logits.masked_fill(~local[None, :, :], float("-inf"))
            probs = self._softmax(logits)
            outputs.append(torch.matmul(probs, vh[:, key_start:key_end].float()))
        return torch.cat(outputs, dim=1)

    def _segment_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, full_attn: bool
    ) -> torch.Tensor:
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        if full_attn:
            context = torch.matmul(self._softmax(self._logits(qh, kh)), vh.float())
        else:
            context = self._windowed_attention(qh, kh, vh)
        return context.to(vh.dtype).transpose(0, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        full_attn: bool,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        q, k, v = self.qkv(hidden_states).split(
            (q_size, k_size, self.num_kv_heads * self.value_dim), dim=-1
        )
        q = q.reshape(seq_len, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_kv_heads, self.value_dim)
        q, k = _apply_rotary(q, k, position_embeddings)
        outputs = []
        boundaries = cu_seqlens.detach().cpu().tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            outputs.append(self._segment_attention(q[start:end], k[start:end], v[start:end], full_attn))
        output = torch.cat(outputs, dim=0).reshape(seq_len, -1)
        return self.proj(output)


class MiMoVisionMLP(nn.Module):
    def __init__(self, config: MiMoVisionConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        if config.hidden_act == "silu":
            self.act = F.silu
        elif config.hidden_act == "gelu":
            self.act = F.gelu
        else:
            raise ValueError(f"unsupported MiMo VIT activation: {config.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class MiMoVisionBlock(nn.Module):
    def __init__(self, config: MiMoVisionConfig, use_sink: bool):
        super().__init__()
        self.norm1 = MiMoRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = MiMoRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = MiMoVisionEagerAttention(config, use_sink=use_sink)
        self.mlp = MiMoVisionMLP(config)

    def forward(self, x, cu_seqlens, position_embeddings, full_attn):
        x = x + self.attn(self.norm1(x), cu_seqlens, position_embeddings, full_attn)
        return x + self.mlp(self.norm2(x))


class MiMoVisionPatchMerger(nn.Module):
    def __init__(self, config: MiMoVisionConfig):
        super().__init__()
        merged = config.hidden_size * config.spatial_merge_size**2
        self.spatial_merge_unit = config.spatial_merge_size**2
        # MiMo passes an RMSNorm into the merger rather than keeping the
        # LayerNorm default; vLLM and SGLang agree on this.
        self.ln_q = MiMoRMSNorm(config.hidden_size, config.rms_norm_eps)
        # The two references disagree on the projection bias: vLLM's
        # MiMoVisionPatchMerger builds them bias-free, while SGLang reuses
        # Qwen2_5_VisionPatchMerger with bias=True. We follow SGLang because the
        # tower is Qwen2.5-VL-derived (whose checkpoints do carry these biases)
        # and because SGLang's merger was clearly debugged against real weights.
        # If this is wrong the mismatch surfaces at load time, since every local
        # parameter must be present in the checkpoint.
        self.mlp = nn.Sequential(
            nn.Linear(merged, merged),
            nn.GELU(),
            nn.Linear(merged, config.out_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] % self.spatial_merge_unit != 0:
            raise ValueError("vision token count is not divisible by merge unit")
        return self.mlp(
            self.ln_q(x).reshape(x.shape[0] // self.spatial_merge_unit, -1)
        )


class MiMoVisionTransformer(nn.Module):
    def __init__(self, config_dict: Dict):
        super().__init__()
        self.config = MiMoVisionConfig(**config_dict)
        c = self.config
        self.patch_embed = MiMoVisionPatchEmbed(c)
        self.rotary_pos_emb = MiMoVisionRotaryEmbedding(c.qk_channels // 2)
        self.blocks = nn.ModuleList(
            [
                MiMoVisionBlock(
                    c,
                    use_sink=c.use_sink and i not in c.fullatt_block_indexes,
                )
                for i in range(c.depth)
            ]
        )
        self.merger = MiMoVisionPatchMerger(c)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self):
        return self.patch_embed.proj.weight.device

    def _expanded_column_index(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge = self.config.spatial_merge_size
        indexes: List[torch.Tensor] = []
        offset = 0
        for t, h, w in grid_thw.tolist():
            if h % merge or w % merge:
                raise ValueError("vision grid must be divisible by spatial merge size")
            logical = torch.arange(t * (h // merge) * (w // merge)).reshape(
                t, h // merge, w // merge
            )
            logical = logical.transpose(1, 2).reshape(-1) + offset
            patch_offsets = torch.arange(merge * merge)
            indexes.append((logical[:, None] * merge * merge + patch_offsets).reshape(-1))
            offset += t * (h // merge) * (w // merge)
        return torch.cat(indexes).to(self.device)

    def _position_embeddings(self, grid_thw: torch.Tensor):
        merge = self.config.spatial_merge_size
        ids = []
        for t, h, w in grid_thw.tolist():
            if h % merge or w % merge:
                raise ValueError("vision grid must be divisible by spatial merge size")
            h_ids = torch.arange(h).view(h, 1).expand(h, w)
            w_ids = torch.arange(w).view(1, w).expand(h, w)
            h_ids = h_ids.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
            w_ids = w_ids.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
            ids.append(torch.stack((h_ids, w_ids), dim=-1).repeat(t, 1))
        pos_ids = torch.cat(ids).to(self.device)
        freqs = self.rotary_pos_emb(int(grid_thw[:, 1:].max().item()))[pos_ids]
        # [tokens, 2, qk/4] -> [tokens, qk/2] laid out as [h_freqs, w_freqs],
        # then doubled to [h, w, h, w] so that the chunk(2) split inside
        # _rotate_half pairs every frequency with itself. Doubling before the
        # flatten yields [h, h, w, w] and rotates h against w channels.
        emb = freqs.flatten(1)
        emb = torch.cat((emb, emb), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        grid_thw = grid_thw.to(dtype=torch.long, device="cpu")
        x = self.patch_embed(pixel_values.to(self.device, self.dtype))
        row_pos = self._position_embeddings(grid_thw)
        column_index = self._expanded_column_index(grid_thw)
        reverse_column_index = torch.argsort(column_index)
        col_pos = (row_pos[0][column_index], row_pos[1][column_index])
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0)).to(self.device, torch.int32)
        pattern = self.config.vit_window_attn_types
        column_order = False
        for i, block in enumerate(self.blocks):
            if pattern[i] == 1 and not column_order:
                x = x[column_index]
                column_order = True
            elif pattern[i] != 1 and column_order:
                x = x[reverse_column_index]
                column_order = False
            pos = col_pos if pattern[i] == 1 else row_pos
            x = block(
                x,
                cu_seqlens,
                pos,
                full_attn=i in self.config.fullatt_block_indexes,
            )
        if column_order:
            x = x[reverse_column_index]
        return self.merger(x)
