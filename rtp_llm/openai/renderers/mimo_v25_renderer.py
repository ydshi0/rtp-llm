import copy

from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ContentPartTypeEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.openai.renderers.qwen_renderer import QwenRenderer
from rtp_llm.utils.base_model_datatypes import MMUrlType


class MiMoV25Renderer(QwenRenderer):
    def _render_multimodal(self, request: ChatCompletionRequest) -> PromptWithMMInput:
        messages = []
        urls = []
        types = []
        for message in request.messages:
            data = {"role": message.role.value, "content": message.content}
            if isinstance(message.content, list):
                parts = []
                for part in message.content:
                    if part.type == ContentPartTypeEnum.text:
                        parts.append({"type": "text", "text": part.text})
                    elif part.type == ContentPartTypeEnum.image_url:
                        if part.image_url is None:
                            raise ValueError("image_url content is missing its URL")
                        urls.append(part.image_url.url)
                        types.append(MMUrlType.IMAGE)
                        parts.append({"type": "image", "image": part.image_url.url})
                    elif part.type == ContentPartTypeEnum.video_url:
                        raise ValueError("MiMo V2.5 video input is not supported yet")
                data["content"] = parts
            messages.append(data)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return PromptWithMMInput(prompt=prompt, urls=urls, mm_types=types)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        if not any(isinstance(m.content, list) for m in request.messages):
            return super().render_chat(request)
        mm_input = self._render_multimodal(copy.deepcopy(request))
        return RenderedInputs(
            input_ids=self.tokenizer.encode(mm_input.prompt),
            input_urls=mm_input.urls,
            input_urls_type=mm_input.mm_types,
            rendered_prompt=mm_input.prompt,
        )


register_renderer("mimo_v25", MiMoV25Renderer)
