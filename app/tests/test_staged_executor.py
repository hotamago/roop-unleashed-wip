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


def test_entry_signature_uses_effective_single_batch_workers(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"test-video")
    entry = ProcessEntry(str(media_path), 0, 10, 30.0)
    options = make_options({"faceswap": {}})

    monkeypatch.setattr("roop.staged_executor.resolve_single_batch_workers", lambda configured_workers: (1, configured_workers, "GPU-safe cap"))
    monkeypatch.setattr(roop.globals.CFG, "single_batch_workers", 2, raising=False)
    sig_a = get_entry_signature(entry, options, "File")

    monkeypatch.setattr(roop.globals.CFG, "single_batch_workers", 4, raising=False)
    sig_b = get_entry_signature(entry, options, "File")

    assert sig_a == sig_b


def test_entry_signature_stays_stable_when_transient_target_moves_to_resume_cache(tmp_path):
    temp_target = tmp_path / "temp" / "gradio" / "upload" / "clip.mp4"
    temp_target.parent.mkdir(parents=True, exist_ok=True)
    temp_target.write_bytes(b"same-video")
    cached_target = tmp_path / "resume_cache" / "clip.mp4"
    cached_target.parent.mkdir(parents=True, exist_ok=True)
    cached_target.write_bytes(b"same-video")
    options = make_options({"faceswap": {}})
    entry_a = ProcessEntry(str(temp_target), 0, 10, 30.0)
    entry_b = ProcessEntry(str(cached_target), 0, 10, 30.0)
    entry_a.file_signature = "sha256:same-video"
    entry_b.file_signature = "sha256:same-video"

    sig_a = get_entry_signature(entry_a, options, "File")
    sig_b = get_entry_signature(entry_b, options, "File")

    assert sig_a == sig_b


def test_entry_signature_uses_active_resume_key_instead_of_runtime_face_embeddings(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"same-video")
    options = make_options({"faceswap": {}})
    entry = ProcessEntry(str(media_path), 0, 10, 30.0, file_signature="sha256:same-video")
    monkeypatch.setattr(roop.globals, "active_resume_key", "resume-key-123", raising=False)
    roop.globals.INPUT_FACESETS[:] = [SimpleNamespace(faces=[SimpleNamespace(embedding=np.array([1.0]), mask_offsets=[0])])]
    roop.globals.TARGET_FACES[:] = [SimpleNamespace(embedding=np.array([2.0]))]

    sig_a = get_entry_signature(entry, options, "File")

    roop.globals.INPUT_FACESETS[:] = [SimpleNamespace(faces=[SimpleNamespace(embedding=np.array([999.0]), mask_offsets=[0])])]
    roop.globals.TARGET_FACES[:] = [SimpleNamespace(embedding=np.array([555.0]))]

    sig_b = get_entry_signature(entry, options, "File")

    assert sig_a == sig_b


def test_cleanup_job_dir_preserves_processing_cache_for_resume(tmp_path, monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}}))
    job_dir = tmp_path / "job-cache"
    job_dir.mkdir()
    (job_dir / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(roop.globals, "processing", True, raising=False)

    executor.cleanup_job_dir(job_dir)

    assert job_dir.exists()
    assert (job_dir / "manifest.json").exists()


def test_completed_video_job_skips_full_pipeline_when_output_exists(tmp_path, monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}}))
    source_path = tmp_path / "clip.mp4"
    source_path.write_bytes(b"video")
    output_path = tmp_path / "clip_out.mp4"
    output_path.write_bytes(b"ready")
    entry = ProcessEntry(str(source_path), 0, 10, 30.0)
    entry.finalname = str(output_path)

    class FakeCapture:
        def get(self, prop):
            if prop == 3:
                return 1920
            if prop == 4:
                return 1080
            return 0

        def release(self):
            return None

    memory_plan = {
        "chunk_size": 96,
        "prefetch_frames": 24,
        "swap_batch_size": 32,
        "mask_batch_size": 64,
        "enhance_batch_size": 8,
        "single_batch_workers": 1,
    }
    manifest = {
        "status": "completed",
        "stages": {
            "detect": True,
            "swap": True,
            "mask": True,
            "enhance": True,
            "composite": True,
        },
    }
    job_dir = tmp_path / "job-cache"
    job_dir.mkdir()

    monkeypatch.setattr("roop.staged_executor.cv2.VideoCapture", lambda _: FakeCapture())
    monkeypatch.setattr("roop.staged_executor.resolve_memory_plan", lambda width, height: dict(memory_plan))
    monkeypatch.setattr("roop.staged_executor.describe_memory_plan", lambda plan: "memory-plan")
    monkeypatch.setattr("roop.staged_executor.set_memory_status", lambda status: None)
    monkeypatch.setattr(executor, "prepare_job", lambda current_entry, current_plan: (job_dir, dict(manifest)))
    monkeypatch.setattr(
        executor,
        "ensure_full_detect_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("detect stage should be skipped")),
    )

    executor.process_video_entry_full_frames(entry, 0)

    assert executor.completed_units == 10
