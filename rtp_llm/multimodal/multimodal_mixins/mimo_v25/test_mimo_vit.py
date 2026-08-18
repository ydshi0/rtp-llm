import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rtp_llm.multimodal.multimodal_mixins.mimo_v25.mimo_v25_image_processing import (
    PIXEL_MEAN,
    PIXEL_STD,
    MiMoV25ImageProcessor,
    smart_resize,
)
from rtp_llm.multimodal.multimodal_mixins.mimo_v25.modeling_mimo_vit import (
    MiMoRMSNorm,
    MiMoVisionConfig,
    MiMoVisionEagerAttention,
    MiMoVisionTransformer,
)


def tiny_config(**overrides):
    config = {
        "depth": 3,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_heads": 2,
        "num_key_value_heads": 1,
        "qk_channels": 4,
        "kv_channels": 4,
        "in_chans": 3,
        "patch_size": 2,
        "temporal_patch_size": 1,
        "spatial_merge_size": 2,
        "out_hidden_size": 12,
        "visual_token_window_size": 1,
        "vit_window_attn_types": [0, 1, -1],
        "fullatt_block_indexes": [2],
        "use_sink": True,
    }
    config.update(overrides)
    return config


def model_config(config):
    return MiMoVisionConfig(**config)


def dense_window_reference(
    attention: MiMoVisionEagerAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Unchunked window attention, written out independently of the module."""
    repeat = attention.num_heads // attention.num_kv_heads
    qh = q.transpose(0, 1).float()
    kh = k.repeat_interleave(repeat, dim=1).transpose(0, 1).float()
    vh = v.repeat_interleave(repeat, dim=1).transpose(0, 1).float()
    logits = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(attention.head_dim)
    pos = torch.arange(q.shape[0])
    local = (pos[:, None] - pos[None, :]).abs() <= attention.window_size
    logits = logits.masked_fill(~local[None, :, :], float("-inf"))
    sink = attention.sinks.float().view(-1, 1, 1).expand(-1, q.shape[0], 1)
    probs = torch.softmax(torch.cat((logits, sink), dim=-1), dim=-1)[..., :-1]
    return torch.matmul(probs, vh).to(v.dtype).transpose(0, 1)


class MiMoVisionTest(unittest.TestCase):
    def test_official_in_chans_alias(self):
        model = MiMoVisionTransformer(tiny_config())
        self.assertEqual(model.config.in_channels, 3)

    def test_column_permutation_is_invertible(self):
        model = MiMoVisionTransformer(tiny_config())
        grid = torch.tensor([[1, 4, 6], [1, 2, 4]])
        index = model._expanded_column_index(grid)
        values = torch.arange(index.numel())
        torch.testing.assert_close(values[index][torch.argsort(index)], values)

    def test_rotary_layout_pairs_each_frequency_with_itself(self):
        model = MiMoVisionTransformer(tiny_config())
        cos, sin = model._position_embeddings(torch.tensor([[1, 2, 4]]))
        half = cos.shape[-1] // 2
        # The [h_freqs, w_freqs] block is doubled, so rotate_half's chunk(2)
        # split sees identical halves. A [h, h, w, w] layout would fail here.
        torch.testing.assert_close(cos[:, :half], cos[:, half:])
        torch.testing.assert_close(sin[:, :half], sin[:, half:])
        # Guard against a vacuous pass when every frequency happens to match.
        self.assertFalse(torch.allclose(cos[:, 0], cos[:, 1]))

    def test_sink_only_enters_softmax_denominator(self):
        attention = MiMoVisionEagerAttention(
            model_config(tiny_config(visual_token_window_size=8)), use_sink=True
        )
        attention.sinks.data.fill_(0.0)
        q = torch.zeros(2, 2, 4)
        k = torch.zeros(2, 1, 4)
        # Only the first key carries a value, so a sink implemented as a bias on
        # key 0 would return 1/2 instead of 1/3.
        v = torch.stack((torch.ones(1, 4), torch.zeros(1, 4)))
        output = attention._segment_attention(q, k, v, full_attn=False)
        torch.testing.assert_close(output, torch.full((2, 2, 4), 1.0 / 3.0))

    def test_without_sink_softmax_is_normalized(self):
        attention = MiMoVisionEagerAttention(
            model_config(tiny_config(visual_token_window_size=8)), use_sink=False
        )
        self.assertIsNone(attention.sinks)
        q = torch.zeros(2, 2, 4)
        k = torch.zeros(2, 1, 4)
        v = torch.ones(2, 1, 4)
        output = attention._segment_attention(q, k, v, full_attn=False)
        torch.testing.assert_close(output, torch.ones(2, 2, 4))

    def test_window_chunking_matches_dense_mask(self):
        attention = MiMoVisionEagerAttention(model_config(tiny_config()), use_sink=True)
        attention.sinks.data.copy_(torch.tensor([0.3, -0.7]))
        torch.manual_seed(0)
        seq_len = 9
        q = torch.randn(seq_len, 2, 4)
        k = torch.randn(seq_len, 1, 4)
        v = torch.randn(seq_len, 1, 4)
        expected = dense_window_reference(attention, q, k, v)
        attention.window_chunk_size = 2
        torch.testing.assert_close(
            attention._segment_attention(q, k, v, full_attn=False), expected
        )

    def test_merger_uses_rms_norm(self):
        model = MiMoVisionTransformer(tiny_config())
        self.assertIsInstance(model.merger.ln_q, MiMoRMSNorm)
        # RMSNorm is weight-only; a LayerNorm here would add a bias that the
        # checkpoint does not carry.
        self.assertNotIn("merger.ln_q.bias", model.state_dict())

    def test_sink_parameters_only_on_window_blocks(self):
        model = MiMoVisionTransformer(tiny_config())
        self.assertIsNotNone(model.blocks[0].attn.sinks)
        self.assertIsNotNone(model.blocks[1].attn.sinks)
        self.assertIsNone(model.blocks[2].attn.sinks)

    def test_forward_shape_and_eager_backend(self):
        model = MiMoVisionTransformer(tiny_config())
        self.assertTrue(
            all(
                isinstance(block.attn, MiMoVisionEagerAttention)
                for block in model.blocks
            )
        )
        grid = torch.tensor([[1, 4, 4]])
        patches = torch.randn(16, 3 * 1 * 2 * 2)
        self.assertEqual(model(patches, grid).shape, (4, 12))


class MiMoImageProcessorTest(unittest.TestCase):
    def make_processor(self):
        return MiMoV25ImageProcessor(
            patch_size=2, spatial_merge_size=2, temporal_patch_size=2
        )

    def test_resize_and_normalize_matches_official_pipeline(self):
        processor = self.make_processor()
        array = np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3)
        pixels, grid_thw = processor(Image.fromarray(array))

        resized_height, resized_width = smart_resize(
            6, 10, processor.size_factor, processor.min_pixels, processor.max_pixels
        )
        source = torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0)
        resized = F.interpolate(
            source,
            (resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1)
        std = torch.tensor(PIXEL_STD).view(1, -1, 1, 1)
        expected = (resized - mean) / std

        self.assertEqual(
            grid_thw.tolist(), [[1, resized_height // 2, resized_width // 2]]
        )
        self.assertEqual(pixels.shape[0], (resized_height // 2) * (resized_width // 2))
        # Row 0 is the top-left patch; its trailing dims are (C, temporal, h, w).
        torch.testing.assert_close(
            pixels[0].view(3, 2, 2, 2)[:, 0], expected[0][:, :2, :2]
        )

    def test_default_pixel_bounds_follow_size_factor(self):
        processor = self.make_processor()
        unit = processor.size_factor**2
        self.assertEqual(processor.min_pixels, 4 * unit)
        self.assertEqual(processor.max_pixels, 4096 * unit)

    def test_processor_config_overrides_pixel_bounds(self):
        processor = MiMoV25ImageProcessor.from_vision_config(
            model_config(tiny_config()),
            {"image_min_pixels": 256, "image_max_pixels": 4096},
        )
        self.assertEqual((processor.min_pixels, processor.max_pixels), (256, 4096))

    def test_max_pixels_caps_the_grid(self):
        processor = self.make_processor()
        _, grid_thw = processor(
            Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)),
            max_pixels=16 * processor.size_factor**2,
        )
        grid_t, grid_h, grid_w = grid_thw[0].tolist()
        self.assertEqual(grid_t, 1)
        self.assertLessEqual((grid_h // 2) * (grid_w // 2), 16)


if __name__ == "__main__":
    unittest.main()
