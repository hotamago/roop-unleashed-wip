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
    import ui.globals
    import ui.tabs.faceswap_tab as faceswap_tab

    cfg = SimpleNamespace(
        detect_pack_frame_count=256,
        staged_chunk_size=96,
        prefetch_frames=24,
        swap_batch_size=32,
        mask_batch_size=64,
        enhance_batch_size=8,
        single_batch_workers=1,
    )
    monkeypatch.setattr(roop.globals, "CFG", cfg, raising=False)
    monkeypatch.setattr(roop.globals, "processing", True, raising=False)
    monkeypatch.setattr(roop.globals, "execution_providers", ["CUDAExecutionProvider", "CPUExecutionProvider"], raising=False)
    monkeypatch.setattr(roop.globals, "runtime_memory_status", "Resource tuning: not computed yet", raising=False)
    monkeypatch.setattr(roop.globals, "active_memory_plan", None, raising=False)
    monkeypatch.setattr(roop.globals, "runtime_processing_status", "Idle", raising=False)
    monkeypatch.setattr(roop.globals, "runtime_processing_markdown", "**Process Info**\n- Status: Idle", raising=False)
    monkeypatch.setattr(roop.globals, "runtime_processing_state", {}, raising=False)
    monkeypatch.setattr(roop.globals, "runtime_processing_last_log_at", 0.0, raising=False)
    monkeypatch.setattr(roop.globals, "active_resume_key", None, raising=False)
    monkeypatch.setattr(roop.globals, "active_resume_job_key", None, raising=False)
    monkeypatch.setattr(roop.globals, "INPUT_FACESETS", [], raising=False)
    monkeypatch.setattr(roop.globals, "TARGET_FACES", [], raising=False)
    monkeypatch.setattr(ui.globals, "ui_input_thumbs", [], raising=False)
    monkeypatch.setattr(ui.globals, "ui_input_face_refs", [], raising=False)
    monkeypatch.setattr(ui.globals, "ui_target_thumbs", [], raising=False)
    monkeypatch.setattr(ui.globals, "ui_target_files", [], raising=False)
    monkeypatch.setattr(ui.globals, "ui_target_face_refs", [], raising=False)
    monkeypatch.setattr(ui.globals, "ui_resume_last_path", None, raising=False)
    monkeypatch.setattr(ui.globals, "ui_resume_bound_path", None, raising=False)
    faceswap_tab.list_files_process.clear()
    monkeypatch.setattr(faceswap_tab, "selected_preview_index", 0, raising=False)
    monkeypatch.setattr(faceswap_tab, "SELECTED_INPUT_FACE_INDEX", 0, raising=False)
    monkeypatch.setattr(faceswap_tab, "SELECTED_TARGET_FACE_INDEX", 0, raising=False)
