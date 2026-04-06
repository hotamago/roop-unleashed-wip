import psutil
import torch
from typing import Optional

import roop.globals


OS_RAM_RESERVE_GB = 2.0
GPU_VRAM_RESERVE_GB = 0.75


def _bytes_to_gb(num_bytes: float) -> float:
    return max(0.0, num_bytes / (1024 ** 3))


def _round_gb(value: float) -> float:
    return round(max(0.0, value), 2)


def provider_uses_gpu() -> bool:
    if not roop.globals.execution_providers:
        return False
    provider = str(roop.globals.execution_providers[0])
    return any(name in provider for name in ("CUDAExecutionProvider", "ROCMExecutionProvider", "TensorrtExecutionProvider"))


def get_available_ram_gb() -> float:
    return _bytes_to_gb(psutil.virtual_memory().available)


def get_available_vram_gb() -> Optional[float]:
    if not provider_uses_gpu():
        return None
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(roop.globals.cuda_device_id)
        return _bytes_to_gb(free_bytes)
    except Exception:
        return None


def resolve_memory_plan(width: int = 0, height: int = 0) -> dict:
    cfg = roop.globals.CFG
    available_ram = get_available_ram_gb()
    available_vram = get_available_vram_gb()

    if cfg.memory_mode == "manual":
        ram_budget = cfg.max_ram_gb if cfg.max_ram_gb and cfg.max_ram_gb > 0 else max(available_ram - OS_RAM_RESERVE_GB, 0.5)
        vram_budget = cfg.max_vram_gb if cfg.max_vram_gb and cfg.max_vram_gb > 0 else available_vram
    else:
        ram_budget = max(available_ram - OS_RAM_RESERVE_GB, 0.5)
        if available_vram is not None:
            vram_budget = max(available_vram - GPU_VRAM_RESERVE_GB, 0.25)
        else:
            vram_budget = None

    if available_vram is None:
        vram_budget = None
    elif vram_budget is not None:
        vram_budget = min(vram_budget, available_vram)

    frame_bytes = max(width * height * 3, 1)
    ram_budget_bytes = max(ram_budget, 0.25) * (1024 ** 3)
    approx_frame_cost = max(frame_bytes * 6, 1)
    chunk_size = int(ram_budget_bytes / approx_frame_cost)
    chunk_size = max(8, min(chunk_size, 240))

    face_pixels = max(roop.globals.subsample_size, 128) ** 2
    swap_batch = 1
    mask_batch = 1
    enhance_batch = 1
    if vram_budget is not None and vram_budget > 0:
        vram_budget_bytes = vram_budget * (1024 ** 3)
        swap_batch = max(1, min(16, int(vram_budget_bytes / max(face_pixels * 6, 1))))
        mask_batch = max(1, min(32, int(vram_budget_bytes / max(face_pixels * 3, 1))))
        enhance_batch = max(1, min(8, int(vram_budget_bytes / max((512 * 512) * 10, 1))))

    prefetch_frames = max(4, min(chunk_size // 2, 48))
    plan = {
        "mode": cfg.memory_mode,
        "available_ram_gb": _round_gb(available_ram),
        "available_vram_gb": None if available_vram is None else _round_gb(available_vram),
        "ram_budget_gb": _round_gb(ram_budget),
        "vram_budget_gb": None if vram_budget is None else _round_gb(vram_budget),
        "chunk_size": chunk_size,
        "prefetch_frames": prefetch_frames,
        "swap_batch_size": swap_batch,
        "mask_batch_size": mask_batch,
        "enhance_batch_size": enhance_batch,
    }
    roop.globals.active_memory_plan = plan
    roop.globals.runtime_memory_status = describe_memory_plan(plan)
    return plan


def describe_memory_plan(plan: Optional[dict] = None) -> str:
    if plan is None:
        plan = roop.globals.active_memory_plan
    if not plan:
        return "Memory budget: not computed yet"
    ram = f"{plan['ram_budget_gb']:.2f} GB RAM"
    if plan["vram_budget_gb"] is None:
        vram = "CPU / no VRAM cap"
    else:
        vram = f"{plan['vram_budget_gb']:.2f} GB VRAM"
    return (
        f"Memory budget: {plan['mode']} | {ram} | {vram} | "
        f"chunk={plan['chunk_size']} | prefetch={plan['prefetch_frames']} | "
        f"swap batch={plan['swap_batch_size']} | mask batch={plan['mask_batch_size']} | "
        f"enhance batch={plan['enhance_batch_size']}"
    )
