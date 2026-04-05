# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
LlamaIndex ``CustomLLM`` wrapper around the PyTorch ROCm backend.

This allows HuggingFace models running on AMD ROCm to be used with
LlamaIndex RAG pipelines in exactly the same way as ``TrtLlmAPI``
(NVIDIA) and ``OrtGenaiAPI`` (AMD ONNX Runtime).
"""

import gc
import time
import uuid
from typing import Any, Optional, Sequence

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    LLMMetadata,
)
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

from ChatRTX.inference.pytorch_rocm.rocm_llm import RocmLlm


class RocmAPI(CustomLLM):
    """LlamaIndex Custom LLM backed by PyTorch ROCm."""

    generate_kwargs: dict = Field(default_factory=dict)
    model_kwargs: dict = Field(default_factory=dict)

    _model: Any = PrivateAttr()
    _model_path: Any = PrivateAttr()
    _context_window: int = PrivateAttr()
    _max_new_tokens: int = PrivateAttr()

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.1,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        self._model = RocmLlm(
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
        )
        self._model_path = model_path
        self._context_window = context_window
        self._max_new_tokens = max_new_tokens

        super().__init__(
            model_path=model_path,
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            callback_manager=callback_manager,
            generate_kwargs=kwargs.get("generate_kwargs", {}),
            model_kwargs=kwargs.get("model_kwargs", {}),
            messages_to_prompt=None,
            completion_to_prompt=None,
            verbose=False,
        )

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = "\n".join(m.content for m in messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        prompt = "\n".join(m.content for m in messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        output_txt = self._model.complete(prompt)
        return CompletionResponse(text=output_txt, raw=self._make_raw(output_txt))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        response_iter = self._model.stream_complete(prompt)

        def gen() -> CompletionResponseGen:
            text = ""
            for delta in response_iter:
                text += delta
                yield CompletionResponse(delta=delta, text=text, raw=self._make_raw(delta))
        return gen()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload_llm(self) -> None:
        if self._model is not None:
            self._model.unload_llm()
            del self._model
            self._model = None
        gc.collect()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self._context_window,
            num_output=self._max_new_tokens,
            model_name=self._model_path,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_raw(text: str) -> dict:
        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "text_completion",
            "created": int(time.time()),
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        }
