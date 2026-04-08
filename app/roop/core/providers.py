from pathlib import Path
from typing import List

import torch

import roop.config.globals
from roop.memory import get_available_vram_gb
from roop.core.resources import suggest_execution_threads


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace("ExecutionProvider", "").lower() for execution_provider in execution_providers]


def _build_cuda_execution_provider():
    torch.cuda.set_device(roop.config.globals.cuda_device_id)
    return (
        "CUDAExecutionProvider",
        {
            "device_id": roop.config.globals.cuda_device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "1",
        },
    )


def _supports_tensorrt_fp16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(roop.config.globals.cuda_device_id)
        return major >= 7
    except Exception:
        return False


def _resolve_trt_workspace_size() -> int:
    available_vram_gb = get_available_vram_gb()
    if available_vram_gb is None:
        workspace_gb = 2.0
    else:
        workspace_gb = max(1.0, min(6.0, available_vram_gb * 0.5))
    return int(workspace_gb * (1024 ** 3))


def _build_tensorrt_execution_provider():
    trt_cache = str(Path(__file__).resolve().parents[2] / "models" / "trt_cache")
    Path(trt_cache).mkdir(parents=True, exist_ok=True)
    return (
        "TensorrtExecutionProvider",
        {
            "device_id": roop.config.globals.cuda_device_id,
            "trt_fp16_enable": _supports_tensorrt_fp16(),
            "trt_max_workspace_size": _resolve_trt_workspace_size(),
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_cache,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": trt_cache,
        },
    )


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    import onnxruntime

    available_providers = onnxruntime.get_available_providers()
    encoded_available_providers = encode_execution_providers(available_providers)
    requested_providers = [str(provider).lower() for provider in execution_providers]
    list_providers = [
        provider
        for provider, encoded_execution_provider in zip(available_providers, encoded_available_providers)
        if any(execution_provider in encoded_execution_provider for execution_provider in requested_providers)
    ]

    try:
        if "tensorrt" in requested_providers and "TensorrtExecutionProvider" in available_providers:
            prioritized_providers = [_build_tensorrt_execution_provider()]
            if "CUDAExecutionProvider" in available_providers:
                prioritized_providers.append(_build_cuda_execution_provider())
            if "CPUExecutionProvider" in available_providers:
                prioritized_providers.append("CPUExecutionProvider")
            return prioritized_providers

        normalized_providers = []
        for provider in list_providers:
            if provider == "CUDAExecutionProvider":
                normalized_providers.append(_build_cuda_execution_provider())
            elif provider == "TensorrtExecutionProvider":
                normalized_providers.append(_build_tensorrt_execution_provider())
            else:
                normalized_providers.append(provider)
        if normalized_providers and "CPUExecutionProvider" in available_providers:
            if not any(
                provider == "CPUExecutionProvider" or (isinstance(provider, tuple) and provider[0] == "CPUExecutionProvider")
                for provider in normalized_providers
            ):
                normalized_providers.append("CPUExecutionProvider")
        if normalized_providers:
            return normalized_providers
    except Exception:
        pass

    return list_providers


def suggest_execution_providers() -> List[str]:
    import onnxruntime

    return encode_execution_providers(onnxruntime.get_available_providers())


__all__ = [
    "_build_cuda_execution_provider",
    "_build_tensorrt_execution_provider",
    "decode_execution_providers",
    "encode_execution_providers",
    "suggest_execution_providers",
    "suggest_execution_threads",
]
