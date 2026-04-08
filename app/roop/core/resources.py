import platform

import torch

import roop.config.globals


def suggest_max_memory() -> int:
    if platform.system().lower() == "darwin":
        return 4
    return 16


def suggest_execution_threads() -> int:
    if "DmlExecutionProvider" in roop.config.globals.execution_providers:
        return 1
    if "ROCMExecutionProvider" in roop.config.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    if roop.config.globals.max_memory:
        memory = roop.config.globals.max_memory * 1024 ** 3
        if platform.system().lower() == "darwin":
            memory = roop.config.globals.max_memory * 1024 ** 6
        if platform.system().lower() == "windows":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource

            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources(process_manager=None) -> None:
    import gc
    from roop.face import release_face_analyser

    release_face_analyser()
    if process_manager is not None:
        process_manager.release_resources()

    gc.collect()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                with torch.cuda.device(roop.config.globals.cuda_device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        except Exception:
            pass


__all__ = [
    "limit_resources",
    "release_resources",
    "suggest_execution_threads",
    "suggest_max_memory",
]
