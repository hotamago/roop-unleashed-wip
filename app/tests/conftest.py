from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


@pytest.fixture(autouse=True)
def reset_runtime_globals(monkeypatch):
    import roop.globals

    cfg = SimpleNamespace(
        memory_mode="smart",
        max_ram_gb=0,
        max_vram_gb=0,
        detect_pack_frame_count=256,
        staged_chunk_size=0,
    )
    monkeypatch.setattr(roop.globals, "CFG", cfg, raising=False)
    monkeypatch.setattr(roop.globals, "processing", True, raising=False)
    monkeypatch.setattr(roop.globals, "execution_providers", ["CUDAExecutionProvider", "CPUExecutionProvider"], raising=False)
    monkeypatch.setattr(roop.globals, "runtime_memory_status", "Memory budget: not computed yet", raising=False)
    monkeypatch.setattr(roop.globals, "active_memory_plan", None, raising=False)
    monkeypatch.setattr(roop.globals, "runtime_processing_status", "Idle", raising=False)
    monkeypatch.setattr(roop.globals, "runtime_processing_markdown", "**Process Info**\n- Status: Idle", raising=False)
    monkeypatch.setattr(roop.globals, "runtime_processing_state", {}, raising=False)
    monkeypatch.setattr(roop.globals, "runtime_processing_last_log_at", 0.0, raising=False)
    monkeypatch.setattr(roop.globals, "INPUT_FACESETS", [], raising=False)
    monkeypatch.setattr(roop.globals, "TARGET_FACES", [], raising=False)
