import roop.config.globals

from roop.onnx.session import create_session_options, providers_use_gpu


def test_providers_use_gpu_detects_tuple_and_string_providers():
    assert providers_use_gpu([("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]) is True
    assert providers_use_gpu(["CPUExecutionProvider"]) is False


def test_create_session_options_disables_cpu_arena_for_gpu():
    session_options = create_session_options(["CUDAExecutionProvider"])

    assert session_options.enable_cpu_mem_arena is False


def test_create_session_options_uses_execution_threads_for_cpu(monkeypatch):
    monkeypatch.setattr(roop.config.globals, "execution_threads", 6, raising=False)

    session_options = create_session_options(["CPUExecutionProvider"])

    assert session_options.intra_op_num_threads == 6
    assert session_options.inter_op_num_threads == 1
