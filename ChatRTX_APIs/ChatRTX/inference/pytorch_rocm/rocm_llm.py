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
PyTorch ROCm inference backend for ChatRTX.

Uses PyTorch compiled with ROCm / HIP support (``torch+rocm``) to run
HuggingFace ``transformers`` models on AMD Radeon / Ryzen AI hardware.

On Windows with ROCm 7.2.1 and an AMD Ryzen AI Max+ 395 the backend
applies SKU-specific optimisations:

- ``torch.float16`` precision by default (the Ryzen AI Max+ 395 integrated
  GPU has good fp16 throughput).
- Memory-pool tuning via ``PYTORCH_HIP_ALLOC_CONF`` to reduce allocation
  overhead on the shared-memory APU.
- Thread-count capping for the CPU-side tokeniser / sampling work.
"""

import gc
import os
from typing import Any, Optional

from ChatRTX.logger import ChatRTXLogger
from ChatRTX.hardware_detect import detect

# ---------------------------------------------------------------------------
# transformers + torch are imported lazily so the rest of the application
# works even when they are not installed (e.g. on pure NVIDIA setups).
# ---------------------------------------------------------------------------
_pipeline = None      # transformers.pipeline
_torch = None         # torch module
_AutoTokenizer = None
_AutoModelForCausalLM = None


def _ensure_pytorch_rocm():
    """Lazily import PyTorch and HuggingFace transformers."""
    global _pipeline, _torch, _AutoTokenizer, _AutoModelForCausalLM
    if _torch is not None:
        return
    try:
        import torch  # type: ignore
        _torch = torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch with ROCm support is required for the pytorch_rocm backend. "
            "Install it from https://pytorch.org/ (ROCm build) or with: "
            "pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"
        ) from exc

    if not _torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch ROCm backend requires a working ROCm / HIP installation "
            "with GPU access.  torch.cuda.is_available() returned False."
        )

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore
        _AutoTokenizer = AutoTokenizer
        _AutoModelForCausalLM = AutoModelForCausalLM
        _pipeline = pipeline
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for the pytorch_rocm backend. "
            "Install it with: pip install transformers>=4.48.0"
        ) from exc


def _apply_ryzen_ai_max_plus_395_env() -> None:
    """Set environment variables that improve PyTorch-ROCm performance on
    the Ryzen AI Max+ 395 APU.

    The APU shares system memory between CPU and GPU, so the default HIP
    memory allocator settings (tuned for discrete GPUs with dedicated VRAM)
    are sub-optimal.  We configure the expandable-segments allocator with a
    larger initial chunk to reduce repeated allocation calls.
    """
    os.environ.setdefault(
        "PYTORCH_HIP_ALLOC_CONF",
        "expandable_segments:True",
    )
    # Limit the fraction of system memory PyTorch may claim (shared memory APU)
    os.environ.setdefault("PYTORCH_ROCM_MEMORY_FRACTION", "0.5")


class RocmLlm:
    """
    A language-model backend that runs HuggingFace models through PyTorch
    with ROCm / HIP GPU acceleration.

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
        **kwargs: Any,
    ) -> None:
        _ensure_pytorch_rocm()

        self._logger = ChatRTXLogger.get_logger()
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._context_window = context_window

        hw = detect()
        self._is_ryzen_ai_max_plus_395 = hw.get("is_ryzen_ai_max_plus_395", False)

        if self._is_ryzen_ai_max_plus_395:
            _apply_ryzen_ai_max_plus_395_env()

        # Determine dtype – fp16 for Ryzen AI Max+ 395 (good fp16 throughput
        # on the integrated RDNA 3.5 iGPU), otherwise use the model default.
        dtype = _torch.float16 if self._is_ryzen_ai_max_plus_395 else "auto"

        try:
            self._tokenizer = _AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True,
            )
            self._model = _AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self._logger.info(
                "RocmLlm: loaded model from %s (dtype=%s, Max+ 395=%s)",
                model_path, dtype, self._is_ryzen_ai_max_plus_395,
            )
        except Exception as exc:
            self._logger.error("RocmLlm: failed to load model from %s – %s", model_path, exc)
            raise

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return "RocmLlm"

    @classmethod
    def class_name(cls) -> str:
        return "RocmLlm"

    # ------------------------------------------------------------------
    # Completion (non-streaming)
    # ------------------------------------------------------------------

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate a non-streaming completion for *prompt*."""
        self._logger.debug("RocmLlm.complete – prompt length %d chars", len(prompt))
        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._context_window,
            ).to(self._model.device)

            with _torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    temperature=max(self._temperature, 1e-7),
                    do_sample=self._temperature > 0,
                    top_k=1,
                    top_p=1.0,
                )

            # Decode only the newly generated tokens (strip the prompt)
            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            return text.strip()
        except Exception as exc:
            self._logger.error("RocmLlm.complete failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------

    def stream_complete(self, prompt: str, **kwargs: Any):
        """Yield new tokens one-by-one (streaming).

        Uses the ``TextIteratorStreamer`` from HuggingFace transformers to
        stream tokens as they are generated.
        """
        self._logger.debug("RocmLlm.stream_complete – prompt length %d chars", len(prompt))
        try:
            from transformers import TextIteratorStreamer  # type: ignore
            import threading

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._context_window,
            ).to(self._model.device)

            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True,
            )
            generation_kwargs = {
                **inputs,
                "max_new_tokens": self._max_new_tokens,
                "temperature": max(self._temperature, 1e-7),
                "do_sample": self._temperature > 0,
                "top_k": 1,
                "top_p": 1.0,
                "streamer": streamer,
            }

            thread = threading.Thread(
                target=self._model.generate, kwargs=generation_kwargs,
            )
            thread.start()

            for token_text in streamer:
                if token_text:
                    yield token_text

            thread.join()
        except Exception as exc:
            self._logger.error("RocmLlm.stream_complete failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload_llm(self) -> None:
        """Release model resources and free GPU memory."""
        try:
            if hasattr(self, "_model") and self._model is not None:
                del self._model
                self._model = None
            if hasattr(self, "_tokenizer") and self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            gc.collect()
            if _torch is not None and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            self._logger.info("RocmLlm: model unloaded")
        except Exception as exc:
            self._logger.error("RocmLlm.unload_llm failed: %s", exc)
