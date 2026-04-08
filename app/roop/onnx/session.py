from typing import Iterable

import onnxruntime

import roop.config.globals

from roop.onnx.runtime import get_execution_providers_for_processor


_GPU_PROVIDER_NAMES = {
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "TensorrtExecutionProvider",
}


def _provider_name(provider) -> str:
    if isinstance(provider, tuple):
        return str(provider[0])
    return str(provider)


def providers_use_gpu(providers: Iterable | None = None) -> bool:
    provider_list = list(providers or getattr(roop.config.globals, "execution_providers", []) or [])
    return any(_provider_name(provider) in _GPU_PROVIDER_NAMES for provider in provider_list)


def create_session_options(providers: Iterable | None = None) -> onnxruntime.SessionOptions:
    session_options = onnxruntime.SessionOptions()
    try:
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        pass
    try:
        session_options.log_severity_level = 3
    except Exception:
        pass

    if providers_use_gpu(providers):
        session_options.enable_cpu_mem_arena = False
    else:
        execution_threads = getattr(roop.config.globals, "execution_threads", None)
        if execution_threads:
            try:
                session_options.intra_op_num_threads = max(1, int(execution_threads))
                session_options.inter_op_num_threads = 1
            except (TypeError, ValueError):
                pass
    return session_options


def create_inference_session(model_path: str, processor_name: str | None = None, providers=None):
    resolved_providers = list(
        providers
        if providers is not None
        else (
            get_execution_providers_for_processor(processor_name)
            if processor_name is not None
            else (getattr(roop.config.globals, "execution_providers", []) or [])
        )
    )
    return onnxruntime.InferenceSession(
        model_path,
        create_session_options(resolved_providers),
        providers=resolved_providers,
    )


__all__ = ["create_inference_session", "create_session_options", "providers_use_gpu"]
