from types import SimpleNamespace

import roop.config.globals
from roop.core.providers import decode_execution_providers


def test_decode_execution_providers_tensorrt_includes_cuda_and_cpu_fallback(monkeypatch):
    import onnxruntime
    import torch

    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device_id: None)
    monkeypatch.setattr(roop.config.globals, "cuda_device_id", 0, raising=False)

    providers = decode_execution_providers(["tensorrt"])

    assert providers[0][0] == "TensorrtExecutionProvider"
    assert providers[1][0] == "CUDAExecutionProvider"
    assert providers[-1] == "CPUExecutionProvider"
    assert providers[1][1]["cudnn_conv_use_max_workspace"] == "1"
    assert providers[1][1]["do_copy_in_default_stream"] == "1"


def test_decode_execution_providers_tensorrt_sets_cache_and_timing_options(monkeypatch, tmp_path):
    import onnxruntime
    import roop.core.providers
    import torch

    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device_id: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device_id: (8, 9))
    monkeypatch.setattr(roop.core.providers, "get_available_vram_gb", lambda: 10.0)
    monkeypatch.setattr(roop.config.globals, "cuda_device_id", 0, raising=False)

    providers = decode_execution_providers(["tensorrt"])

    tensorrt_options = providers[0][1]
    assert tensorrt_options["trt_fp16_enable"] is True
    assert tensorrt_options["trt_max_workspace_size"] == 5 * 1024 ** 3
    assert tensorrt_options["trt_engine_cache_enable"] is True
    assert tensorrt_options["trt_timing_cache_enable"] is True
    assert tensorrt_options["trt_engine_cache_path"] == tensorrt_options["trt_timing_cache_path"]
    assert "trt_cache" in tensorrt_options["trt_engine_cache_path"]
