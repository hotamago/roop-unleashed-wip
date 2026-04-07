from types import SimpleNamespace

import numpy as np

import roop.globals
from roop.ProcessEntry import ProcessEntry
from roop.ProcessOptions import ProcessOptions
from roop.staged_executor import (
    StagedBatchExecutor,
    get_entry_signature,
    normalize_cache_image,
)


def make_options(processors):
    return ProcessOptions(processors, 0.6, 0.8, "all", 0, "", None, 1, 256, False, False)


def test_normalize_cache_image_makes_uint8_contiguous():
    image = np.arange(27, dtype=np.float32).reshape(3, 3, 3)

    normalized = normalize_cache_image(image)

    assert normalized.dtype == np.uint8
    assert normalized.flags["C_CONTIGUOUS"]
    assert normalized.shape == (3, 3, 3)


def test_stage_cache_roundtrip_uses_binary_blob(tmp_path):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}, "mask_xseg": {}, "gfpgan": {}}))
    cache_path = executor.get_stage_cache_path(tmp_path / "swap")
    images = {
        "face_a": np.full((2, 2, 3), 7, dtype=np.uint8),
        "face_b": np.full((2, 2, 3), 9, dtype=np.uint8),
    }

    executor.write_stage_cache_map(cache_path, images)
    loaded = executor.read_stage_cache_map(cache_path)

    assert set(loaded.keys()) == {"face_a", "face_b"}
    assert np.array_equal(loaded["face_a"], images["face_a"])
    assert np.array_equal(loaded["face_b"], images["face_b"])


def test_pipeline_steps_and_detect_pack_ranges_follow_current_config(monkeypatch):
    monkeypatch.setattr(roop.globals.CFG, "detect_pack_frame_count", 32, raising=False)
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}, "mask_xseg": {}, "gfpgan": {}}))
    executor.current_entry = SimpleNamespace(filename="clip.mp4")

    assert executor.get_pipeline_steps() == ["prepare", "detect", "swap", "mask", "enhance", "composite", "mux"]
    assert executor.get_stage_step_info("mask") == (4, 7)
    assert list(executor.iter_detect_pack_ranges(70)) == [(1, 32), (33, 64), (65, 70)]


def test_entry_signature_changes_when_cache_shaping_config_changes(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"test-video")
    entry = ProcessEntry(str(media_path), 0, 10, 30.0)
    options = make_options({"faceswap": {}})

    monkeypatch.setattr(roop.globals.CFG, "detect_pack_frame_count", 256, raising=False)
    monkeypatch.setattr(roop.globals.CFG, "staged_chunk_size", 0, raising=False)
    sig_a = get_entry_signature(entry, options, "File")

    monkeypatch.setattr(roop.globals.CFG, "detect_pack_frame_count", 512, raising=False)
    sig_b = get_entry_signature(entry, options, "File")

    monkeypatch.setattr(roop.globals.CFG, "detect_pack_frame_count", 256, raising=False)
    monkeypatch.setattr(roop.globals.CFG, "staged_chunk_size", 96, raising=False)
    sig_c = get_entry_signature(entry, options, "File")

    assert sig_a != sig_b
    assert sig_a != sig_c
