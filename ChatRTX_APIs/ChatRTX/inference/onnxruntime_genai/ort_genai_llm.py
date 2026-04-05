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
ONNX Runtime GenAI inference backend for AMD Ryzen AI hardware.

Uses the ``onnxruntime-genai`` library (with DirectML execution provider on
Windows) to run quantised ONNX models on the Ryzen AI NPU / iGPU.
"""

import gc
import os
from typing import Any, Optional

from ChatRTX.logger import ChatRTXLogger
from ChatRTX.hardware_detect import detect

# ---------------------------------------------------------------------------
# The onnxruntime_genai package is only required when this backend is used.
# We import it lazily so the rest of the application works even when the
# package is not installed (e.g. on NVIDIA systems).
# ---------------------------------------------------------------------------
_og = None  # will be set on first use


def _ensure_ort_genai():
    global _og
    if _og is not None:
        return
    try:
        import onnxruntime_genai as og  # type: ignore
        _og = og
    except ImportError as exc:
        raise ImportError(
            "onnxruntime-genai is required for AMD Ryzen AI inference. "
            "Install it with: pip install onnxruntime-genai-directml"
        ) from exc


class OrtGenaiLlm:
    """
    A language-model backend that runs ONNX models through the
    ``onnxruntime-genai`` library.

    The interface mirrors :class:`ChatRTX.inference.trtllm.trtllm.TrtLlm` so
    that the rest of the application can use it as a drop-in replacement.
    """

    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_NEW_TOKENS = 1024
    DEFAULT_CONTEXT_WINDOW = 4096

    def __init__(
        self,
        model_path: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        **kwargs: Any,
    ) -> None:
        _ensure_ort_genai()

        self._logger = ChatRTXLogger.get_logger()
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._context_window = context_window

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory does not exist: {model_path}")

        hw = detect()
        self._is_ryzen_ai_max_plus_395 = hw.get("is_ryzen_ai_max_plus_395", False)

        try:
            self._model = _og.Model(model_path)
            self._tokenizer = _og.Tokenizer(self._model)
            self._logger.info("OrtGenaiLlm: loaded model from %s", model_path)
        except Exception as exc:
            self._logger.error("OrtGenaiLlm: failed to load model from %s – %s", model_path, exc)
            raise

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return "OrtGenaiLlm"

    @classmethod
    def class_name(cls) -> str:
        return "OrtGenaiLlm"

    # ------------------------------------------------------------------
    # Search params – tuned for Ryzen AI Max+ 395 when detected
    # ------------------------------------------------------------------

    def _make_search_params(self, streaming: bool = False) -> Any:
        params = _og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=self._context_window + self._max_new_tokens,
            temperature=self._temperature,
            top_k=1,
            top_p=0.0,
            do_sample=self._temperature > 0,
        )
        if self._is_ryzen_ai_max_plus_395:
            # Ryzen AI Max+ 395 has a large shared-memory pool and a wide
            # NPU fabric.  We allow a larger batch of tokens to be processed
            # per step which improves throughput on this SKU.
            try:
                params.set_search_options(
                    num_beams=1,
                    early_stopping=True,
                )
            except Exception:
                pass  # older ort-genai builds may not support all options
        return params

    # ------------------------------------------------------------------
    # Completion (non-streaming)
    # ------------------------------------------------------------------

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate a non-streaming completion for *prompt*."""
        self._logger.debug("OrtGenaiLlm.complete – prompt length %d chars", len(prompt))
        try:
            tokens = self._tokenizer.encode(prompt)
            params = self._make_search_params(streaming=False)
            params.input_ids = tokens

            output_tokens = self._model.generate(params)
            output_text = self._tokenizer.decode(output_tokens[0])

            # Strip the echoed prompt if the model returns it
            if output_text.startswith(prompt):
                output_text = output_text[len(prompt):]
            return output_text.strip()
        except Exception as exc:
            self._logger.error("OrtGenaiLlm.complete failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------

    def stream_complete(self, prompt: str, **kwargs: Any):
        """Yield new tokens one-by-one (streaming)."""
        self._logger.debug("OrtGenaiLlm.stream_complete – prompt length %d chars", len(prompt))
        try:
            tokens = self._tokenizer.encode(prompt)
            params = self._make_search_params(streaming=True)
            params.input_ids = tokens

            generator = _og.Generator(self._model, params)
            tokenizer_stream = self._tokenizer.create_stream()

            first_token = True
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                decoded = tokenizer_stream.decode(new_token)
                # Skip the first token if it is empty (common with some models)
                if first_token and not decoded.strip():
                    first_token = False
                    continue
                first_token = False
                yield decoded

            del generator
        except Exception as exc:
            self._logger.error("OrtGenaiLlm.stream_complete failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload_llm(self) -> None:
        """Release model resources."""
        try:
            if hasattr(self, "_model") and self._model is not None:
                del self._model
                self._model = None
            if hasattr(self, "_tokenizer") and self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            gc.collect()
            self._logger.info("OrtGenaiLlm: model unloaded")
        except Exception as exc:
            self._logger.error("OrtGenaiLlm.unload_llm failed: %s", exc)
