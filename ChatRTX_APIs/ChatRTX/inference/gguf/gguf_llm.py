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
GGUF model inference backend for ChatRTX.

Uses the ``llama-cpp-python`` library to run GGUF-format models.  Automatically
selects the best acceleration layer based on detected hardware:

- **NVIDIA GPUs**: offloads all layers to the GPU via CUDA.
- **AMD Ryzen AI / Radeon**: offloads layers via Vulkan (when available) or
  runs on CPU with optimized thread settings.
- **CPU fallback**: runs entirely on CPU.
"""

import gc
import glob
import os
from typing import Any, Optional

from ChatRTX.logger import ChatRTXLogger
from ChatRTX.hardware_detect import detect

# ---------------------------------------------------------------------------
# llama-cpp-python is imported lazily so the rest of the application works
# even when the package is not installed.
# ---------------------------------------------------------------------------
_Llama = None  # will be set on first use


def _ensure_llama_cpp():
    global _Llama
    if _Llama is not None:
        return
    try:
        from llama_cpp import Llama  # type: ignore
        _Llama = Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required for GGUF model inference. "
            "Install it with: pip install llama-cpp-python"
        ) from exc


def _find_gguf_file(model_path: str) -> str:
    """Return the path to the first ``.gguf`` file under *model_path*.

    If *model_path* itself is a ``.gguf`` file, return it directly.
    Otherwise search recursively.
    """
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        return model_path
    candidates = glob.glob(os.path.join(model_path, "**", "*.gguf"), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No .gguf file found in {model_path}.  "
            "Ensure the GGUF model file has been downloaded."
        )
    # Prefer the largest file (likely the main model rather than a shard)
    candidates.sort(key=os.path.getsize, reverse=True)
    return candidates[0]


def _compute_gpu_layers(hw: dict) -> int:
    """Determine how many layers to offload to the GPU.

    Returns -1 (all layers) for NVIDIA CUDA, a conservative value for AMD
    Vulkan, or 0 for CPU-only.
    """
    vendor = hw.get("vendor", "cpu")
    if vendor == "nvidia":
        # Offload all layers to CUDA
        return -1
    if vendor == "amd":
        # AMD Ryzen AI Max+ 395 has a large shared-memory pool – offload
        # more layers.  Other AMD hardware gets a conservative default.
        if hw.get("is_ryzen_ai_max_plus_395", False):
            return -1
        return 32  # conservative default for generic AMD / Vulkan
    return 0  # CPU-only


class GgufLlm:
    """
    A language-model backend that runs ``.gguf`` files through
    ``llama-cpp-python``.

    The public interface mirrors :class:`ChatRTX.inference.trtllm.trtllm.TrtLlm`
    so that the rest of the application can use it as a drop-in replacement.
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
        n_gpu_layers: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _ensure_llama_cpp()

        self._logger = ChatRTXLogger.get_logger()
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._context_window = context_window

        gguf_file = _find_gguf_file(model_path)
        self._logger.info("GgufLlm: using GGUF file %s", gguf_file)

        hw = detect()
        if n_gpu_layers is None:
            n_gpu_layers = _compute_gpu_layers(hw)

        self._is_ryzen_ai_max_plus_395 = hw.get("is_ryzen_ai_max_plus_395", False)

        # Determine optimal thread count
        n_threads = os.cpu_count() or 4
        if self._is_ryzen_ai_max_plus_395:
            # Ryzen AI Max+ 395 benefits from more threads on CPU-bound ops
            n_threads = min(n_threads, 16)

        try:
            self._model = _Llama(
                model_path=gguf_file,
                n_ctx=context_window,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                verbose=False,
            )
            self._logger.info(
                "GgufLlm: loaded model (n_gpu_layers=%s, n_threads=%d, vendor=%s)",
                n_gpu_layers, n_threads, hw.get("vendor", "unknown"),
            )
        except Exception as exc:
            self._logger.error("GgufLlm: failed to load model from %s – %s", gguf_file, exc)
            raise

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return "GgufLlm"

    @classmethod
    def class_name(cls) -> str:
        return "GgufLlm"

    # ------------------------------------------------------------------
    # Completion (non-streaming)
    # ------------------------------------------------------------------

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate a non-streaming completion for *prompt*."""
        self._logger.debug("GgufLlm.complete – prompt length %d chars", len(prompt))
        try:
            result = self._model(
                prompt,
                max_tokens=self._max_new_tokens,
                temperature=self._temperature,
                top_k=1,
                top_p=0.0,
                echo=False,
            )
            text = result["choices"][0]["text"]
            return text.strip()
        except Exception as exc:
            self._logger.error("GgufLlm.complete failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------

    def stream_complete(self, prompt: str, **kwargs: Any):
        """Yield new tokens one-by-one (streaming)."""
        self._logger.debug("GgufLlm.stream_complete – prompt length %d chars", len(prompt))
        try:
            stream = self._model(
                prompt,
                max_tokens=self._max_new_tokens,
                temperature=self._temperature,
                top_k=1,
                top_p=0.0,
                echo=False,
                stream=True,
            )
            for chunk in stream:
                token_text = chunk["choices"][0]["text"]
                if token_text:
                    yield token_text
        except Exception as exc:
            self._logger.error("GgufLlm.stream_complete failed: %s", exc)
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
            gc.collect()
            self._logger.info("GgufLlm: model unloaded")
        except Exception as exc:
            self._logger.error("GgufLlm.unload_llm failed: %s", exc)
