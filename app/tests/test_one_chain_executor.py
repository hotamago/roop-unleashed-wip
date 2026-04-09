from pathlib import Path

import numpy as np

import roop.config.globals
from roop.pipeline.entry import ProcessEntry
from roop.pipeline.one_chain_executor import (
    OneChainAllExecutor,
    get_one_chain_frame_key,
    get_one_chain_segment_key,
    get_one_chain_segment_path,
)
from roop.pipeline.options import ProcessOptions
from roop.pipeline.staged_executor.cache import read_json, write_json


def make_options(processors):
    return ProcessOptions(processors, 0.6, 0.8, "all", 0, "", None, 1, 256, False, False)


class FakeCapture:
    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height

    def get(self, prop):
        if prop == 3:
            return self.width
        if prop == 4:
            return self.height
        return 0

    def release(self):
        return None


def test_one_chain_streams_source_frames_into_video_stage_cache(tmp_path, monkeypatch):
    source_path = tmp_path / "clip.mp4"
    source_path.write_bytes(b"video")
    entry = ProcessEntry(str(source_path), 0, 4, 30.0)
    executor = OneChainAllExecutor("File", None, make_options({"faceswap": {}}))
    write_calls = []

    class FakeProcessMgr:
        def __init__(self, _progress):
            return None

        def initialize(self, *_args, **_kwargs):
            return None

        def set_progress_context(self, *_args, **_kwargs):
            return None

        def process_frame(self, frame):
            return frame + 1

        def release_resources(self):
            return None

    class FakeStageCache:
        def __init__(self, **_kwargs):
            return None

        def write(self, path, cache_map):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"segment")
            path.with_suffix(".idx.bin").write_text("{}", encoding="utf-8")
            write_calls.append((path, dict(cache_map)))
            return path

    def fake_iter_video_chunk(_video_path, frame_start, frame_end, _prefetch_frames):
        for frame_number in range(frame_start, frame_end):
            yield frame_number, np.full((4, 4, 3), frame_number, dtype=np.uint8)

    def fail_extract_frames(*_args, **_kwargs):
        raise AssertionError("extract_frames should not be called")

    monkeypatch.setattr("roop.pipeline.one_chain_executor.get_jobs_root", lambda: tmp_path)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.ProcessMgr", FakeProcessMgr)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.VideoStageCache", FakeStageCache)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.iter_video_chunk", fake_iter_video_chunk)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.open_video_capture", lambda _path: FakeCapture())
    monkeypatch.setattr("roop.pipeline.one_chain_executor.set_processing_message", lambda *args, **kwargs: None)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.ffmpeg.extract_frames", fail_extract_frames)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.get_one_chain_chunk_size", lambda: 4)

    _job_dir, manifest_path, manifest, cache_dir, _merged_video = executor._prepare_job(entry)

    assert executor._process_stream_to_cache(entry, 0, 1, cache_dir, manifest, manifest_path, 30.0) is True

    updated_manifest = read_json(manifest_path)
    assert updated_manifest["process_complete"] is True
    assert updated_manifest["completed_frames"] == 4
    assert len(write_calls) == 1
    segment_path, cache_map = write_calls[0]
    assert segment_path == get_one_chain_segment_path(cache_dir, 0, 4)
    assert list(cache_map.keys()) == [get_one_chain_frame_key(i) for i in range(4)]
    assert np.array_equal(cache_map[get_one_chain_frame_key(3)], np.full((4, 4, 3), 4, dtype=np.uint8))


def test_one_chain_stream_resume_skips_completed_segments(tmp_path, monkeypatch):
    source_path = tmp_path / "clip.mp4"
    source_path.write_bytes(b"video")
    entry = ProcessEntry(str(source_path), 0, 4, 30.0)
    executor = OneChainAllExecutor("File", None, make_options({"faceswap": {}}))
    iter_calls = []
    write_calls = []

    class FakeProcessMgr:
        def __init__(self, _progress):
            return None

        def initialize(self, *_args, **_kwargs):
            return None

        def set_progress_context(self, *_args, **_kwargs):
            return None

        def process_frame(self, frame):
            return frame + 10

        def release_resources(self):
            return None

    class FakeStageCache:
        def __init__(self, **_kwargs):
            return None

        def write(self, path, cache_map):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"segment")
            path.with_suffix(".idx.bin").write_text("{}", encoding="utf-8")
            write_calls.append((path, dict(cache_map)))
            return path

    def fake_iter_video_chunk(_video_path, frame_start, frame_end, prefetch_frames):
        iter_calls.append((frame_start, frame_end, prefetch_frames))
        for frame_number in range(frame_start, frame_end):
            yield frame_number, np.full((4, 4, 3), frame_number, dtype=np.uint8)

    monkeypatch.setattr("roop.pipeline.one_chain_executor.get_jobs_root", lambda: tmp_path)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.ProcessMgr", FakeProcessMgr)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.VideoStageCache", FakeStageCache)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.iter_video_chunk", fake_iter_video_chunk)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.open_video_capture", lambda _path: FakeCapture())
    monkeypatch.setattr("roop.pipeline.one_chain_executor.set_processing_message", lambda *args, **kwargs: None)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.get_one_chain_chunk_size", lambda: 2)

    _job_dir, manifest_path, manifest, cache_dir, _merged_video = executor._prepare_job(entry)
    first_key = get_one_chain_segment_key(0, 2)
    first_segment = get_one_chain_segment_path(cache_dir, 0, 2)
    first_segment.parent.mkdir(parents=True, exist_ok=True)
    first_segment.write_bytes(b"done")
    first_segment.with_suffix(".idx.bin").write_text("{}", encoding="utf-8")
    manifest["segments"] = {
        first_key: {"completed": True, "frame_count": 2},
    }
    manifest["completed_frames"] = 2
    manifest["frame_count"] = 4
    write_json(manifest_path, manifest)

    assert executor._process_stream_to_cache(entry, 0, 1, cache_dir, manifest, manifest_path, 30.0) is True

    updated_manifest = read_json(manifest_path)
    assert iter_calls == [(2, 4, 24)]
    assert updated_manifest["completed_frames"] == 4
    assert updated_manifest["process_complete"] is True
    assert updated_manifest["segments"][first_key]["completed"] is True
    assert len(write_calls) == 1
    second_segment, cache_map = write_calls[0]
    assert second_segment == get_one_chain_segment_path(cache_dir, 2, 4)
    assert list(cache_map.keys()) == [get_one_chain_frame_key(2), get_one_chain_frame_key(3)]


def test_one_chain_merge_reuses_completed_segment_videos(tmp_path, monkeypatch):
    source_path = tmp_path / "clip.mp4"
    source_path.write_bytes(b"video")
    entry = ProcessEntry(str(source_path), 0, 4, 30.0)
    executor = OneChainAllExecutor("File", None, make_options({"faceswap": {}}))
    joined = {}

    monkeypatch.setattr("roop.pipeline.one_chain_executor.get_jobs_root", lambda: tmp_path)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.set_processing_message", lambda *args, **kwargs: None)
    monkeypatch.setattr("roop.pipeline.one_chain_executor.get_one_chain_chunk_size", lambda: 2)

    _job_dir, manifest_path, manifest, cache_dir, merged_video = executor._prepare_job(entry)
    for start_frame, end_frame in ((0, 2), (2, 4)):
        segment_path = get_one_chain_segment_path(cache_dir, start_frame, end_frame)
        segment_path.parent.mkdir(parents=True, exist_ok=True)
        segment_path.write_bytes(f"{start_frame}-{end_frame}".encode("utf-8"))
        manifest.setdefault("segments", {})[get_one_chain_segment_key(start_frame, end_frame)] = {
            "completed": True,
            "frame_count": end_frame - start_frame,
        }
    write_json(manifest_path, manifest)

    def fake_join_videos(videos, destination, simple):
        joined["videos"] = list(videos)
        joined["destination"] = destination
        joined["simple"] = simple
        Path(destination).write_bytes(b"merged")

    monkeypatch.setattr("roop.pipeline.one_chain_executor.ffmpeg.join_videos", fake_join_videos)

    assert executor._merge_processed_segments(entry, 0, 1, cache_dir, merged_video, manifest, manifest_path) is True

    updated_manifest = read_json(manifest_path)
    assert updated_manifest["merge_complete"] is True
    assert joined["videos"] == [
        str(get_one_chain_segment_path(cache_dir, 0, 2)),
        str(get_one_chain_segment_path(cache_dir, 2, 4)),
    ]
    assert joined["destination"] == str(merged_video)
    assert joined["simple"] is True
