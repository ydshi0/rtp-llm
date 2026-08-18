from typing import Dict, Iterable, List

import torch
from PIL import Image

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalDeployWeightInfo,
    BaseMultiModalMixin,
    BaseVitWeights,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
    get_bytes_io_from_url,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType, VitParameters
from rtp_llm.utils.model_weight import CkptWeightInfo, identity, sp_id

from .mimo_v25_image_processing import MiMoV25ImageProcessor
from .modeling_mimo_vit import MiMoVisionTransformer


class MiMoV25VitWeights(BaseVitWeights):
    # transformers >= 4.52 saves the vision tower under "model.visual.", while
    # earlier MiMo exports use a bare "visual.".
    ckpt_prefix_candidates = ("visual.", "model.visual.")

    def _set_weight_prefix(self):
        self._ckpt_prefix = "visual."
        self._ft_prefix = "self.mm_part.visual."

    def resolve_ckpt_prefix(self, tensor_names: Iterable[str]) -> str:
        available = set(tensor_names)
        best_prefix, best_hits = None, -1
        for prefix in self.ckpt_prefix_candidates:
            hits = sum(prefix + name in available for name in self.weight_names)
            if hits > best_hits:
                best_prefix, best_hits = prefix, hits
        if best_hits <= 0:
            raise ValueError(
                "No MiMo VIT weights found in the checkpoint under any of "
                f"{self.ckpt_prefix_candidates}; first expected tensor was "
                f"{self.ckpt_prefix_candidates[0] + self.weight_names[0]!r}"
            )
        if best_hits != len(self.weight_names):
            missing = [
                name
                for name in self.weight_names
                if best_prefix + name not in available
            ]
            raise ValueError(
                f"MiMo VIT checkpoint prefix {best_prefix!r} is missing "
                f"{len(missing)} tensor(s), e.g. {missing[:5]}"
            )
        self._ckpt_prefix = best_prefix
        return best_prefix


class MiMoV25WeightInfo(BaseMultiModalDeployWeightInfo):
    def get_weight_info(self):
        weights = []
        for name in self.vit_weights.weight_names:
            weights.append(
                CustomAtomicWeight(
                    name,
                    [CkptWeightInfo(self.vit_weights.ckpt_prefix + name, identity)],
                    identity,
                    split_func=sp_id,
                )
            )
        return ModelWeightInfo(layer_weights=[], weights=weights)


class MiMoV25ImageEmbedding(MultiModalEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters):
        self.mm_related_params = mm_related_params
        config: Dict = mm_related_params.config
        self.visual = MiMoVisionTransformer(config)
        self.image_processor = MiMoV25ImageProcessor.from_vision_config(
            self.visual.config, config.get("processor_config")
        )

    @property
    def _data_type(self):
        return self.visual.dtype

    @property
    def _device(self):
        return self.visual.device

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor,
        **kwargs,
    ):
        if len(mm_inputs) != 1:
            raise ValueError("MiMo VIT preprocess expects one media item")
        mm_input = mm_inputs[0]
        if mm_input.mm_type != MMUrlType.IMAGE:
            raise ValueError("MiMo V2.5 currently supports image input only")
        data = get_bytes_io_from_url(mm_input.url, vit_config.download_headers)
        image = Image.open(data).convert("RGB")
        preprocess = mm_input.mm_preprocess_config
        width, height = image.size
        nominal_size = (
            height if preprocess.height == -1 else preprocess.height,
            width if preprocess.width == -1 else preprocess.width,
        )
        min_pixels = (
            processor.min_pixels
            if preprocess.min_pixels == -1
            else preprocess.min_pixels
        )
        max_pixels = (
            processor.max_pixels
            if preprocess.max_pixels == -1
            else preprocess.max_pixels
        )
        return processor(
            image,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            nominal_size=nominal_size,
        )

    def get_preprocess_params(self):
        return {"processor": self.image_processor}

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        pixel_values, grid_thw = data
        embeddings = self.visual(
            pixel_values.to(self._device, self._data_type), grid_thw
        )
        return embeddings.contiguous(), None


class MiMoV25Mixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = MiMoV25ImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = MiMoV25VitWeights(
            {"visual": self.mm_part.visual}
        )

    def _prepare_vit_weights(self, database) -> None:
        self.mm_related_params.vit_weights.resolve_ckpt_prefix(
            database.get_pretrain_tensor_names()
        )

    @staticmethod
    def get_multimodal_mixin_weight_info():
        return MiMoV25WeightInfo

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return MiMoV25ImageEmbedding(mm_related_params).visual


register_multimodal_mixin(["mimo_v25"], MiMoV25Mixin)
