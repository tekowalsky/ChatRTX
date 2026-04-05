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
Central hardware detection module for ChatRTX.

Detects whether the system has AMD Ryzen AI or NVIDIA hardware and exposes
a consistent API used by all subsystems.  Also checks for ROCm availability
on AMD platforms so that PyTorch can be used with GPU acceleration.
"""

import logging
import os
import platform
import subprocess
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal cached state
# ---------------------------------------------------------------------------
_detected: Optional[Dict] = None


def _run_cmd(cmd: list) -> Optional[str]:
    """Run a command and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _detect_nvidia_gpu() -> Optional[Dict]:
    """Try to detect an NVIDIA GPU via nvidia-smi."""
    output = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if output is None:
        return None
    parts = output.split(",")
    if len(parts) < 2:
        return None
    name = parts[0].strip()
    try:
        vram_mb = int(float(parts[1].strip()))
    except (ValueError, IndexError):
        vram_mb = 0
    return {"name": name, "vram_mb": vram_mb}


def _detect_amd_ryzen_ai() -> Optional[Dict]:
    """
    Detect AMD Ryzen AI hardware.

    On Windows the integrated NPU / iGPU is exposed through the Windows
    Management Instrumentation (WMI) interface.  We first attempt a WMI
    query; if that is unavailable we fall back to a platform-level heuristic
    based on the CPU brand string (Ryzen AI processors embed 'Ryzen AI' or
    'Ryzen' with an NPU indicator in their marketing name).
    """
    system = platform.system()

    # ---- Windows WMI approach ------------------------------------------------
    if system == "Windows":
        # Check for AMD GPU via WMIC (works for iGPU exposed by Ryzen AI)
        wmic_out = _run_cmd(["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv"])
        if wmic_out:
            for line in wmic_out.splitlines():
                if "AMD" in line.upper() or "RADEON" in line.upper():
                    cols = [c.strip() for c in line.split(",")]
                    # CSV format: Node,AdapterRAM,Name
                    adapter_ram = 0
                    name = ""
                    for col in cols:
                        if col.isdigit():
                            adapter_ram = int(col)
                        elif "AMD" in col.upper() or "RADEON" in col.upper():
                            name = col
                    if name:
                        return {"name": name, "vram_mb": adapter_ram // (1024 * 1024) if adapter_ram > 1024 else adapter_ram}

    # ---- Linux lspci approach ------------------------------------------------
    if system == "Linux":
        lspci_out = _run_cmd(["lspci"])
        if lspci_out:
            for line in lspci_out.splitlines():
                lower = line.lower()
                if "amd" in lower and ("display" in lower or "vga" in lower or "3d" in lower):
                    return {"name": line.split(":")[-1].strip(), "vram_mb": 0}

    # ---- CPU brand-string fallback -------------------------------------------
    try:
        import cpuinfo  # type: ignore
        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw", "")
    except Exception:
        brand = platform.processor()

    if brand and "Ryzen" in brand:
        return {"name": brand, "vram_mb": 0}

    return None


def _detect_rocm() -> bool:
    """Return True when a usable ROCm installation is detected.

    Checks for the ``hipconfig`` CLI (part of ROCm) and, as a fallback,
    verifies that PyTorch was compiled with ROCm / HIP support.
    """
    # Quick check: hipconfig binary available?
    if _run_cmd(["hipconfig", "--version"]) is not None:
        return True

    # Environment variable hint (set by the ROCm installer on Windows)
    if os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH"):
        return True

    # Last resort: ask PyTorch (lazy – only imported when needed)
    try:
        import torch  # type: ignore
        return hasattr(torch, "hip") or (
            hasattr(torch.version, "hip") and torch.version.hip is not None
        )
    except Exception:
        pass

    return False


# Ryzen AI Max+ 395 identifiers (substring match on detected GPU / CPU name)
_RYZEN_AI_MAX_PLUS_395_KEYWORDS = ["Ryzen AI Max+ 395", "Ryzen AI MAX+ 395"]


def detect() -> dict:
    """
    Detect the accelerator present on this system.

    Returns a dict with:
        vendor        – "amd" | "nvidia" | "cpu"
        gpu_name      – human-readable name
        vram_mb       – VRAM in MiB (0 when unknown / shared memory)
        is_ryzen_ai   – True when AMD Ryzen AI silicon is detected
        is_ryzen_ai_max_plus_395 – True for the specific Ryzen AI Max+ 395 SKU
        has_rocm      – True when a usable ROCm / HIP stack is present
        preferred_backend – "onnxrt" | "TRTLLM" | "pytorch" | "pytorch_rocm"
    """
    global _detected
    if _detected is not None:
        return _detected

    # Try AMD first so that on a system with *both* (unlikely but possible)
    # we prefer the Ryzen AI path as requested by the problem statement.
    amd = _detect_amd_ryzen_ai()
    if amd is not None:
        name = amd["name"]
        is_max_plus_395 = any(kw in name for kw in _RYZEN_AI_MAX_PLUS_395_KEYWORDS)
        rocm_available = _detect_rocm()
        # When ROCm is available, prefer the pytorch_rocm backend for GPU-
        # accelerated inference via PyTorch; otherwise fall back to onnxrt.
        preferred = "pytorch_rocm" if rocm_available else "onnxrt"
        _detected = {
            "vendor": "amd",
            "gpu_name": name,
            "vram_mb": amd["vram_mb"],
            "is_ryzen_ai": True,
            "is_ryzen_ai_max_plus_395": is_max_plus_395,
            "has_rocm": rocm_available,
            "preferred_backend": preferred,
        }
        logger.info(
            "Detected AMD Ryzen AI hardware: %s (Max+ 395: %s, ROCm: %s)",
            name, is_max_plus_395, rocm_available,
        )
        return _detected

    nv = _detect_nvidia_gpu()
    if nv is not None:
        _detected = {
            "vendor": "nvidia",
            "gpu_name": nv["name"],
            "vram_mb": nv["vram_mb"],
            "is_ryzen_ai": False,
            "is_ryzen_ai_max_plus_395": False,
            "has_rocm": False,
            "preferred_backend": "TRTLLM",
        }
        logger.info("Detected NVIDIA GPU: %s with %d MiB VRAM", nv["name"], nv["vram_mb"])
        return _detected

    # Fallback – CPU-only
    _detected = {
        "vendor": "cpu",
        "gpu_name": "CPU",
        "vram_mb": 0,
        "is_ryzen_ai": False,
        "is_ryzen_ai_max_plus_395": False,
        "has_rocm": False,
        "preferred_backend": "pytorch",
    }
    logger.warning("No GPU accelerator detected – falling back to CPU")
    return _detected


def get_total_system_memory_gb() -> int:
    """Return total system RAM in GiB (useful for shared-memory APUs like Ryzen AI)."""
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024 ** 3))
    except Exception:
        return 0


def get_available_vram_mb() -> int:
    """
    Return available VRAM in MiB.

    For NVIDIA GPUs this queries pynvml.  For AMD Ryzen AI (shared memory)
    we report half of total system RAM as a conservative estimate of what
    the iGPU / NPU can use.
    """
    hw = detect()
    if hw["vendor"] == "nvidia":
        try:
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
            nvmlInit()
            info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
            return int(info.free / (1024 * 1024))
        except Exception:
            return 0
    elif hw["vendor"] == "amd":
        # Ryzen AI uses shared system memory – report half of total RAM
        total_gb = get_total_system_memory_gb()
        return (total_gb * 1024) // 2
    return 0


def get_torch_device() -> str:
    """
    Return the best PyTorch device string for the detected hardware.

    Returns "cuda" for NVIDIA, "hip" (mapped to torch device ``"cuda"`` when
    using PyTorch-ROCm) for AMD with ROCm, and "cpu" as fallback.
    """
    hw = detect()
    if hw["vendor"] == "nvidia":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    if hw["vendor"] == "amd" and hw.get("has_rocm", False):
        try:
            import torch
            # PyTorch ROCm exposes HIP devices via the torch.cuda API
            if torch.cuda.is_available():
                return "cuda"  # ROCm uses the CUDA device name in PyTorch
        except Exception:
            pass
    # For AMD Ryzen AI without ROCm, PyTorch operations (embeddings, CLIP)
    # run on CPU; the heavy LLM inference is handled by ONNX Runtime GenAI
    # with DirectML.
    return "cpu"
