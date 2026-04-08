from types import SimpleNamespace

import numpy as np

import roop.config.globals
from roop.pipeline.entry import ProcessEntry
from roop.pipeline.batch_executor import eNoFaceAction
from roop.pipeline.options import ProcessOptions
from roop.pipeline.staged_executor.executor import (
    StagedBatchExecutor,
    get_entry_job_key,
    get_entry_job_relpath,
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


def test_ensure_enhance_stage_flushes_cache_once_per_chunk(tmp_path, monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}, "gfpgan": {}}))
    chunk_dir = tmp_path / "chunk"
    chunk_meta = {
        "start": 0,
        "end": 2,
        "frames": [
            {"frame_number": 0, "tasks": [{"cache_key": "face_a"}]},
            {"frame_number": 1, "tasks": [{"cache_key": "face_b"}]},
        ],
    }
    chunk_state = {"stages": {"enhance": False}}
    memory_plan = {"enhance_batch_size": 1, "single_batch_workers": 1}
    input_cache = {
        "face_a": np.full((2, 2, 3), 7, dtype=np.uint8),
        "face_b": np.full((2, 2, 3), 9, dtype=np.uint8),
    }
    swap_cache_path = executor.get_stage_cache_path(chunk_dir / "swap")
    enhance_cache_path = executor.get_stage_cache_path(chunk_dir / "enhance")
    write_calls = []

    class FakeProcessMgr:
        def __init__(self, _progress):
            self.processors = [object()]

        def initialize(self, *_args, **_kwargs):
            return None

        def run_enhance_tasks_batch(self, task_batch, current_frames, _processor, _batch_size):
            return {
                task_meta["cache_key"]: normalize_cache_image(current_frame + 1)
                for task_meta, current_frame in zip(task_batch, current_frames)
            }

        def release_resources(self):
            return None

    def fake_read_stage_cache_map(path):
        if path == enhance_cache_path:
            return {}
        if path == swap_cache_path:
            return dict(input_cache)
        raise AssertionError(path)

    def fake_write_stage_cache_map(path, cache_map):
        assert path == enhance_cache_path
        write_calls.append(dict(cache_map))

    monkeypatch.setattr("roop.pipeline.staged_executor.enhance_stage.ProcessMgr", FakeProcessMgr)
    monkeypatch.setattr(executor, "read_stage_cache_map", fake_read_stage_cache_map)
    monkeypatch.setattr(executor, "write_stage_cache_map", fake_write_stage_cache_map)
    monkeypatch.setattr(executor, "update_progress", lambda *args, **kwargs: None)

    executor.ensure_enhance_stage(chunk_dir, chunk_meta, chunk_state, memory_plan)

    assert chunk_state["stages"]["enhance"] is True
    assert len(write_calls) == 1
    assert set(write_calls[0].keys()) == {"face_a", "face_b"}


def test_process_full_mask_batch_falls_back_when_batch_masks_are_broken(tmp_path, monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}, "mask_xseg": {}}))
    task_batch = [
        {"cache_key": "face_a", "aligned_frame": np.full((2, 2, 3), 10, dtype=np.uint8)},
        {"cache_key": "face_b", "aligned_frame": np.full((2, 2, 3), 10, dtype=np.uint8)},
    ]
    original_batch = [task["aligned_frame"] for task in task_batch]
    input_cache = {
        "face_a": np.full((2, 2, 3), 200, dtype=np.uint8),
        "face_b": np.full((2, 2, 3), 200, dtype=np.uint8),
    }
    output_cache = {}

    class FakeBrokenMaskProcessor:
        supports_batch = True

        def Run(self, _image, _keywords):
            return np.zeros((2, 2, 1), dtype=np.float32)

        def RunBatch(self, _images, _keywords, _batch_size):
            return [
                np.ones((2, 1), dtype=np.float32),
                np.ones((2, 1), dtype=np.float32),
            ]

    monkeypatch.setattr(executor, "update_progress", lambda *args, **kwargs: None)
    processor = FakeBrokenMaskProcessor()
    executor.process_full_mask_batch(
        task_batch,
        original_batch,
        input_cache,
        output_cache,
        tmp_path / "mask.bin",
        None,
        processor,
        0,
        2,
        {"mask_batch_size": 8},
        flush_cache=False,
    )

    assert processor.supports_batch is False
    assert np.array_equal(output_cache["face_a"], input_cache["face_a"])
    assert np.array_equal(output_cache["face_b"], input_cache["face_b"])


def test_process_full_mask_batch_uses_single_batch_worker_pool_for_non_batch_processors(tmp_path, monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}, "mask_clip2seg": {}}))
    task_batch = [
        {"cache_key": "face_a", "aligned_frame": np.full((2, 2, 3), 10, dtype=np.uint8)},
        {"cache_key": "face_b", "aligned_frame": np.full((2, 2, 3), 20, dtype=np.uint8)},
    ]
    original_batch = [task["aligned_frame"] for task in task_batch]
    input_cache = {
        "face_a": np.full((2, 2, 3), 200, dtype=np.uint8),
        "face_b": np.full((2, 2, 3), 210, dtype=np.uint8),
    }
    output_cache = {}

    class FakeMaskProcessor:
        supports_batch = False
        supports_parallel_single_batch = True
        batch_size_limit = 1

    class FakeProcessMgr:
        def run_mask_tasks_batch(self, task_batch, current_frames, _processor, _batch_size):
            return {
                task_meta["cache_key"]: normalize_cache_image(current_frame + 1)
                for task_meta, current_frame in zip(task_batch, current_frames)
            }

    monkeypatch.setattr(executor, "update_progress", lambda *args, **kwargs: None)
    executor.process_full_mask_batch(
        task_batch,
        original_batch,
        input_cache,
        output_cache,
        tmp_path / "mask.bin",
        FakeProcessMgr(),
        FakeMaskProcessor(),
        0,
        2,
        {"mask_batch_size": 8},
        flush_cache=False,
    )

    assert np.array_equal(output_cache["face_a"], normalize_cache_image(input_cache["face_a"] + 1))
    assert np.array_equal(output_cache["face_b"], normalize_cache_image(input_cache["face_b"] + 1))


def test_ensure_full_mask_stage_reuses_swap_source_cache_without_decoding_video(tmp_path, monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}, "mask_xseg": {}}))
    entry = ProcessEntry("clip.mp4", 0, 2, 30.0)
    detect_dir = tmp_path / "detect"
    swap_dir = tmp_path / "swap"
    mask_dir = tmp_path / "mask"
    manifest = {}
    stages = {"mask": False}
    memory_plan = {"mask_batch_size": 8}
    pack_data = {
        "start_sequence": 1,
        "end_sequence": 2,
        "frames": [
            {"frame_number": 10, "tasks": [{"cache_key": "face_a"}]},
            {"frame_number": 11, "tasks": [{"cache_key": "face_b"}]},
        ],
    }
    swap_cache_path = executor.get_stage_pack_path(swap_dir, 1, 2)
    swap_source_cache_path = executor.get_stage_pack_path(swap_dir.parent / "swap_source", 1, 2)
    mask_cache_path = executor.get_stage_pack_path(mask_dir, 1, 2)
    write_calls = []

    class FakeMaskProcessor:
        supports_batch = False

    class FakeProcessMgr:
        def __init__(self, _progress):
            self.processors = [FakeMaskProcessor()]

        def initialize(self, *_args, **_kwargs):
            return None

        def run_mask_tasks_batch(self, task_batch, current_frames, _processor, _batch_size):
            return {
                task_meta["cache_key"]: normalize_cache_image(current_frame + 1)
                for task_meta, current_frame in zip(task_batch, current_frames)
            }

        def release_resources(self):
            return None

    def fake_read_stage_cache_map(path):
        if path == swap_cache_path:
            return {
                "face_a": np.full((2, 2, 3), 10, dtype=np.uint8),
                "face_b": np.full((2, 2, 3), 20, dtype=np.uint8),
            }
        if path == swap_source_cache_path:
            return {
                "face_a": np.full((2, 2, 3), 30, dtype=np.uint8),
                "face_b": np.full((2, 2, 3), 40, dtype=np.uint8),
            }
        if path == mask_cache_path:
            return {}
        raise AssertionError(path)

    monkeypatch.setattr("roop.pipeline.staged_executor.mask_stage.ProcessMgr", FakeProcessMgr)
    monkeypatch.setattr("roop.pipeline.staged_executor.mask_stage.iter_video_chunk", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("video decode should be skipped")))
    monkeypatch.setattr(executor, "iter_detect_packs", lambda _detect_dir: [pack_data])
    monkeypatch.setattr(executor, "read_stage_cache_map", fake_read_stage_cache_map)
    monkeypatch.setattr(executor, "write_stage_cache_map", lambda path, cache_map: write_calls.append((path, dict(cache_map))))
    monkeypatch.setattr(executor, "update_progress", lambda *args, **kwargs: None)

    executor.ensure_full_mask_stage(entry, 12, detect_dir, swap_dir, mask_dir, 2, stages, manifest, memory_plan)

    assert stages["mask"] is True
    assert len(write_calls) == 1
    assert write_calls[0][0] == mask_cache_path
    assert set(write_calls[0][1].keys()) == {"face_a", "face_b"}


def test_compose_frame_from_cache_skips_fallback_reprocess_for_original_frame_mode(monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}}))
    frame = np.full((2, 2, 3), 77, dtype=np.uint8)

    class FakeFallbackMgr:
        last_swapped_frame = None
        num_frames_no_face = 0
        options = SimpleNamespace(max_num_reuse_frame=3)

        def process_frame(self, _frame):
            raise AssertionError("fallback processing should not run for USE_ORIGINAL_FRAME")

    monkeypatch.setattr(roop.config.globals, "no_face_action", eNoFaceAction.USE_ORIGINAL_FRAME, raising=False)

    result = executor.compose_frame_from_cache(
        None,
        frame,
        {"tasks": [], "fallback": True},
        {},
        {},
        FakeFallbackMgr(),
    )

    assert result is frame


def test_ensure_full_compose_stage_streams_source_once_across_packs(tmp_path, monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}}))
    entry = ProcessEntry("clip.mp4", 0, 4, 30.0)
    intermediate_video = tmp_path / "composite.mp4"
    stages = {"composite": False}
    manifest = {"frame_count": 4}
    memory_plan = {"prefetch_frames": 2}
    iter_calls = []

    class FakeCapture:
        def get(self, prop):
            if prop == 3:
                return 64
            if prop == 4:
                return 64
            return 0

        def release(self):
            return None

    class FakeWriter:
        def __init__(self, filename, *_args, **_kwargs):
            self.filename = filename

        def write_frame(self, _frame):
            return None

        def close(self):
            with open(self.filename, "wb") as handle:
                handle.write(b"ok")

    class FakeProcessMgr:
        def __init__(self, _progress):
            return None

        def initialize(self, *_args, **_kwargs):
            return None

        def compose_task(self, result, _task_meta, _fake_frame, _enhanced_frame=None):
            return result

        def release_resources(self):
            return None

    def fake_iter_detect_packs(_detect_dir):
        return [
            {
                "start_sequence": 1,
                "end_sequence": 2,
                "frames": [
                    {"frame_number": 0, "tasks": [], "fallback": False},
                    {"frame_number": 1, "tasks": [], "fallback": False},
                ],
            },
            {
                "start_sequence": 3,
                "end_sequence": 4,
                "frames": [
                    {"frame_number": 2, "tasks": [], "fallback": False},
                    {"frame_number": 3, "tasks": [], "fallback": False},
                ],
            },
        ]

    def fake_iter_video_chunk(_video_path, frame_start, frame_end, _prefetch_frames):
        iter_calls.append((frame_start, frame_end))
        for frame_number in range(frame_start, frame_end):
            yield frame_number, np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr("roop.pipeline.staged_executor.compose_stage.open_video_capture", lambda _path: FakeCapture())
    monkeypatch.setattr("roop.pipeline.staged_executor.compose_stage.FFMPEG_VideoWriter", FakeWriter)
    monkeypatch.setattr("roop.pipeline.staged_executor.compose_stage.ProcessMgr", FakeProcessMgr)
    monkeypatch.setattr("roop.pipeline.staged_executor.video_iter.iter_video_chunk", fake_iter_video_chunk)
    monkeypatch.setattr(executor, "iter_detect_packs", fake_iter_detect_packs)
    monkeypatch.setattr(executor, "read_stage_cache_map", lambda _path: {})
    monkeypatch.setattr(executor, "update_progress", lambda *args, **kwargs: None)

    executor.ensure_full_compose_stage(
        entry,
        4,
        30.0,
        tmp_path / "detect",
        tmp_path / "swap",
        tmp_path / "mask",
        tmp_path / "enhance",
        intermediate_video,
        stages,
        manifest,
        memory_plan,
    )

    assert iter_calls == [(0, 4)]


def test_ensure_full_compose_stage_writes_cached_swapped_frames(tmp_path, monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}}))
    entry = ProcessEntry("clip.mp4", 0, 2, 30.0)
    intermediate_video = tmp_path / "composite.mp4"
    stages = {"composite": False}
    manifest = {"frame_count": 2}
    memory_plan = {"prefetch_frames": 2}
    written_frames = []

    original_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    swapped_frame = np.full((2, 2, 3), 255, dtype=np.uint8)

    class FakeCapture:
        def get(self, prop):
            if prop == 3:
                return 2
            if prop == 4:
                return 2
            return 0

        def release(self):
            return None

    class FakeWriter:
        def __init__(self, filename, *_args, **_kwargs):
            self.filename = filename

        def write_frame(self, frame):
            written_frames.append(frame.copy())

        def close(self):
            with open(self.filename, "wb") as handle:
                handle.write(b"ok")

    class FakeProcessMgr:
        def __init__(self, _progress):
            return None

        def initialize(self, *_args, **_kwargs):
            return None

        def compose_task(self, result, _task_meta, fake_frame, _enhanced_frame=None):
            return fake_frame.copy()

        def release_resources(self):
            return None

    def fake_iter_detect_packs(_detect_dir):
        return [
            {
                "start_sequence": 1,
                "end_sequence": 2,
                "frames": [
                    {"frame_number": 0, "tasks": [{"cache_key": "f000001_t000", "target_face": {}, "mask_offsets": [0] * 10}], "fallback": False},
                    {"frame_number": 1, "tasks": [{"cache_key": "f000002_t000", "target_face": {}, "mask_offsets": [0] * 10}], "fallback": False},
                ],
            }
        ]

    def fake_iter_video_chunk(_video_path, frame_start, frame_end, _prefetch_frames):
        for frame_number in range(frame_start, frame_end):
            yield frame_number, original_frame.copy()

    monkeypatch.setattr("roop.pipeline.staged_executor.compose_stage.open_video_capture", lambda _path: FakeCapture())
    monkeypatch.setattr("roop.pipeline.staged_executor.compose_stage.FFMPEG_VideoWriter", FakeWriter)
    monkeypatch.setattr("roop.pipeline.staged_executor.compose_stage.ProcessMgr", FakeProcessMgr)
    monkeypatch.setattr("roop.pipeline.staged_executor.video_iter.iter_video_chunk", fake_iter_video_chunk)
    monkeypatch.setattr(executor, "iter_detect_packs", fake_iter_detect_packs)
    monkeypatch.setattr(
        executor,
        "read_stage_cache_map",
        lambda _path: {
            "f000001_t000": swapped_frame.copy(),
            "f000002_t000": swapped_frame.copy(),
        },
    )
    monkeypatch.setattr(executor, "update_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(roop.config.globals.CFG, "max_threads", 1, raising=False)
    monkeypatch.setattr(roop.config.globals, "no_face_action", eNoFaceAction.USE_ORIGINAL_FRAME, raising=False)

    executor.ensure_full_compose_stage(
        entry,
        2,
        30.0,
        tmp_path / "detect",
        tmp_path / "swap",
        tmp_path / "mask",
        tmp_path / "enhance",
        intermediate_video,
        stages,
        manifest,
        memory_plan,
    )

    assert len(written_frames) == 2
    assert np.array_equal(written_frames[0], swapped_frame)
    assert np.array_equal(written_frames[1], swapped_frame)


def test_pipeline_steps_and_detect_pack_ranges_follow_current_config(monkeypatch):
    monkeypatch.setattr(roop.config.globals.CFG, "detect_pack_frame_count", 32, raising=False)
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}, "mask_xseg": {}, "gfpgan": {}}))
    executor.current_entry = SimpleNamespace(filename="clip.mp4")

    assert executor.get_pipeline_steps() == ["prepare", "detect", "swap", "mask", "enhance", "composite", "mux"]
    assert executor.get_stage_step_info("mask") == (4, 7)
    assert list(executor.iter_detect_pack_ranges(70)) == [(1, 32), (33, 64), (65, 70)]


def test_entry_signature_ignores_cache_shaping_config_changes(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"test-video")
    entry = ProcessEntry(str(media_path), 0, 10, 30.0)
    options = make_options({"faceswap": {}})

    monkeypatch.setattr(roop.config.globals.CFG, "detect_pack_frame_count", 256, raising=False)
    monkeypatch.setattr(roop.config.globals.CFG, "staged_chunk_size", 0, raising=False)
    sig_a = get_entry_signature(entry, options, "File")

    monkeypatch.setattr(roop.config.globals.CFG, "detect_pack_frame_count", 512, raising=False)
    sig_b = get_entry_signature(entry, options, "File")

    monkeypatch.setattr(roop.config.globals.CFG, "detect_pack_frame_count", 256, raising=False)
    monkeypatch.setattr(roop.config.globals.CFG, "staged_chunk_size", 96, raising=False)
    sig_c = get_entry_signature(entry, options, "File")

    assert sig_a == sig_b
    assert sig_a == sig_c


def test_entry_job_relpath_uses_resume_cache_id_and_file_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(roop.config.globals, "active_resume_cache_id", "20260406_8d8d244889be", raising=False)
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"v")
    entry = ProcessEntry(str(media_path), 0, 10, 30.0, file_signature="sha256:abc")
    options = make_options({"faceswap": {}})
    rel = get_entry_job_relpath(entry, options)
    assert rel == "20260406_8d8d244889be/sha256_abc"


def test_prepare_job_uses_nested_jobs_folder_for_resume_cache_id(tmp_path, monkeypatch):
    monkeypatch.setattr(roop.config.globals, "active_resume_cache_id", "sess1", raising=False)
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"v")
    entry = ProcessEntry(str(media_path), 0, 5, 30.0, file_signature="sha256:z")
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}}))
    executor.jobs_root = tmp_path / "jobs"
    job_dir, manifest = executor.prepare_job(entry, {"chunk_size": 8})
    assert job_dir == tmp_path / "jobs" / "sess1" / "sha256_z"
    assert manifest["job_key"] == "sess1/sha256_z"


def test_entry_job_key_unchanged_when_global_mask_batch_changes_with_resume_job_key(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"v")
    entry = ProcessEntry(str(media_path), 0, 10, 30.0, file_signature="sha256:same")
    options = make_options({"faceswap": {}})
    monkeypatch.setattr(roop.config.globals, "active_resume_job_key", "resume-job-locked", raising=False)
    k32 = get_entry_job_key(entry, options)
    monkeypatch.setattr(roop.config.globals.CFG, "mask_batch_size", 128, raising=False)
    k128 = get_entry_job_key(entry, options)
    assert k32 == k128


def test_entry_signature_ignores_global_mask_batch_size(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"same-video")
    entry = ProcessEntry(str(media_path), 0, 10, 30.0, file_signature="sha256:same-video")
    options = make_options({"faceswap": {}})
    monkeypatch.setattr(roop.config.globals, "active_resume_job_key", "resume-job-x", raising=False)
    monkeypatch.setattr(roop.config.globals.CFG, "mask_batch_size", 32, raising=False)
    sig_lo = get_entry_signature(entry, options, "File")
    monkeypatch.setattr(roop.config.globals.CFG, "mask_batch_size", 128, raising=False)
    sig_hi = get_entry_signature(entry, options, "File")
    assert sig_lo == sig_hi


def test_entry_signature_uses_effective_single_batch_workers(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"test-video")
    entry = ProcessEntry(str(media_path), 0, 10, 30.0)
    options = make_options({"faceswap": {}})

    monkeypatch.setattr(roop.config.globals.CFG, "single_batch_workers", 2, raising=False)
    sig_a = get_entry_signature(entry, options, "File")

    monkeypatch.setattr(roop.config.globals.CFG, "single_batch_workers", 4, raising=False)
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


def test_entry_signature_uses_resume_job_key_instead_of_runtime_face_embeddings(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"same-video")
    options = make_options({"faceswap": {}})
    entry = ProcessEntry(str(media_path), 0, 10, 30.0, file_signature="sha256:same-video")
    monkeypatch.setattr(roop.config.globals, "active_resume_job_key", "resume-job-stable", raising=False)
    monkeypatch.setattr(roop.config.globals, "active_resume_key", "resume-key-123", raising=False)
    roop.config.globals.INPUT_FACESETS[:] = [SimpleNamespace(faces=[SimpleNamespace(embedding=np.array([1.0]), mask_offsets=[0])])]
    roop.config.globals.TARGET_FACES[:] = [SimpleNamespace(embedding=np.array([2.0]))]

    sig_a = get_entry_signature(entry, options, "File")

    roop.config.globals.INPUT_FACESETS[:] = [SimpleNamespace(faces=[SimpleNamespace(embedding=np.array([999.0]), mask_offsets=[0])])]
    roop.config.globals.TARGET_FACES[:] = [SimpleNamespace(embedding=np.array([555.0]))]

    sig_b = get_entry_signature(entry, options, "File")

    assert sig_a == sig_b


def test_entry_signature_uses_resume_key_when_no_resume_job_key(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"same-video")
    options = make_options({"faceswap": {}})
    entry = ProcessEntry(str(media_path), 0, 10, 30.0, file_signature="sha256:same-video")
    monkeypatch.setattr(roop.config.globals, "active_resume_job_key", None, raising=False)
    monkeypatch.setattr(roop.config.globals, "active_resume_key", "resume-key-only", raising=False)
    roop.config.globals.INPUT_FACESETS[:] = [SimpleNamespace(faces=[SimpleNamespace(embedding=np.array([1.0]), mask_offsets=[0])])]
    roop.config.globals.TARGET_FACES[:] = [SimpleNamespace(embedding=np.array([2.0]))]

    sig_a = get_entry_signature(entry, options, "File")
    roop.config.globals.INPUT_FACESETS[:] = [SimpleNamespace(faces=[SimpleNamespace(embedding=np.array([999.0]), mask_offsets=[0])])]
    sig_b = get_entry_signature(entry, options, "File")
    assert sig_a == sig_b


def test_entry_job_key_uses_active_resume_job_key_when_available(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"same-video")
    options = make_options({"faceswap": {}})
    entry = ProcessEntry(str(media_path), 0, 10, 30.0, file_signature="sha256:same-video")
    monkeypatch.setattr(roop.config.globals, "active_resume_job_key", "resume-job-123", raising=False)

    job_key_a = get_entry_job_key(entry, options)

    monkeypatch.setattr(roop.config.globals, "active_resume_job_key", "resume-job-123", raising=False)
    monkeypatch.setattr(roop.config.globals, "active_resume_key", "resume-snapshot-b", raising=False)
    monkeypatch.setattr(roop.config.globals.CFG, "detect_pack_frame_count", 1024, raising=False)
    job_key_b = get_entry_job_key(entry, options)

    assert job_key_a == job_key_b


def test_get_compose_worker_count_uses_cfg_max_threads_when_order_is_safe(monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}}))
    monkeypatch.setattr(roop.config.globals.CFG, "max_threads", 4, raising=False)
    monkeypatch.setattr(roop.config.globals, "no_face_action", eNoFaceAction.USE_ORIGINAL_FRAME, raising=False)

    assert executor.get_compose_worker_count() == 4


def test_get_compose_worker_count_falls_back_to_single_thread_for_use_last_swapped(monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}}))
    monkeypatch.setattr(roop.config.globals.CFG, "max_threads", 4, raising=False)
    monkeypatch.setattr(roop.config.globals, "no_face_action", eNoFaceAction.USE_LAST_SWAPPED, raising=False)

    assert executor.get_compose_worker_count() == 1


def test_prepare_job_preserves_cache_when_only_resume_snapshot_key_changes(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"same-video")
    entry = ProcessEntry(str(media_path), 0, 10, 30.0, file_signature="sha256:same-video")
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}}))
    executor.jobs_root = tmp_path / "jobs"
    memory_plan = {"chunk_size": 96}

    monkeypatch.setattr(roop.config.globals, "active_resume_job_key", "resume-job-123", raising=False)
    monkeypatch.setattr(roop.config.globals, "active_resume_key", "resume-snapshot-a", raising=False)
    job_dir_a, _manifest_a = executor.prepare_job(entry, memory_plan)
    stale_marker = job_dir_a / "stale.bin"
    stale_marker.write_bytes(b"old-cache")

    monkeypatch.setattr(roop.config.globals, "active_resume_key", "resume-snapshot-b", raising=False)
    job_dir_b, manifest_b = executor.prepare_job(entry, memory_plan)

    assert job_dir_a == job_dir_b
    assert stale_marker.exists()
    assert manifest_b["status"] == "running"


def test_prepare_job_wipes_cache_when_blend_ratio_changes(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"same-video")
    entry = ProcessEntry(str(media_path), 0, 10, 30.0, file_signature="sha256:same-video")
    memory_plan = {"chunk_size": 96}
    jobs_root = tmp_path / "jobs"

    monkeypatch.setattr(roop.config.globals, "active_resume_job_key", "resume-job-123", raising=False)
    monkeypatch.setattr(roop.config.globals, "active_resume_key", "snap", raising=False)

    executor_a = StagedBatchExecutor("File", None, make_options({"faceswap": {}}))
    executor_a.jobs_root = jobs_root
    job_dir_a, _ = executor_a.prepare_job(entry, memory_plan)
    marker = job_dir_a / "pixel.bin"
    marker.write_bytes(b"x")

    executor_b = StagedBatchExecutor(
        "File",
        None,
        ProcessOptions({"faceswap": {}}, 0.6, 0.5, "all", 0, "", None, 1, 256, False, False),
    )
    executor_b.jobs_root = jobs_root
    job_dir_b, _ = executor_b.prepare_job(entry, memory_plan)

    assert job_dir_a == job_dir_b
    assert not marker.exists()


def test_cleanup_job_dir_preserves_processing_cache_for_resume(tmp_path, monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({"faceswap": {}}))
    job_dir = tmp_path / "job-cache"
    job_dir.mkdir()
    (job_dir / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(roop.config.globals, "processing", True, raising=False)

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

    monkeypatch.setattr("roop.pipeline.staged_executor.executor.cv2.VideoCapture", lambda _: FakeCapture())
    monkeypatch.setattr("roop.pipeline.staged_executor.executor.resolve_memory_plan", lambda width, height: dict(memory_plan))
    monkeypatch.setattr("roop.pipeline.staged_executor.executor.describe_memory_plan", lambda plan: "memory-plan")
    monkeypatch.setattr("roop.pipeline.staged_executor.executor.set_memory_status", lambda status: None)
    monkeypatch.setattr(executor, "prepare_job", lambda current_entry, current_plan: (job_dir, dict(manifest)))
    monkeypatch.setattr(
        executor,
        "ensure_full_detect_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("detect stage should be skipped")),
    )

    executor.process_video_entry_full_frames(entry, 0)

    assert executor.completed_units == 10


def test_process_video_entry_full_frames_uses_direct_encode_fast_path_when_no_processors(tmp_path, monkeypatch):
    executor = StagedBatchExecutor("File", None, make_options({}))
    source_path = tmp_path / "clip.mp4"
    source_path.write_bytes(b"video")
    output_path = tmp_path / "clip_out.mp4"
    entry = ProcessEntry(str(source_path), 5, 21, 30.0)
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

    job_dir = tmp_path / "job-cache"
    job_dir.mkdir()
    manifest = {"status": "running", "stages": {}}
    direct_calls = []

    monkeypatch.setattr("roop.pipeline.staged_executor.executor.cv2.VideoCapture", lambda _path: FakeCapture())
    monkeypatch.setattr("roop.pipeline.staged_executor.executor.resolve_memory_plan", lambda width, height: {"prefetch_frames": 24})
    monkeypatch.setattr("roop.pipeline.staged_executor.executor.describe_memory_plan", lambda plan: "memory-plan")
    monkeypatch.setattr("roop.pipeline.staged_executor.executor.set_memory_status", lambda status: None)
    monkeypatch.setattr(executor, "prepare_job", lambda current_entry, current_plan: (job_dir, dict(manifest)))
    monkeypatch.setattr(
        executor,
        "ensure_direct_video_output",
        lambda current_entry, current_index, frame_count, endframe: direct_calls.append(
            (current_entry.filename, current_index, frame_count, endframe)
        ),
    )
    monkeypatch.setattr(
        executor,
        "ensure_full_detect_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("detect stage should be skipped")),
    )

    executor.process_video_entry_full_frames(entry, 3)

    assert direct_calls == [(str(source_path), 3, 16, 21)]

