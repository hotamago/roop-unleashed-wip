import shutil

import numpy as np
import pytest
import roop.config.globals

from roop.pipeline.staged_executor.cache import read_cache_blob, read_stage_cache_map, write_cache_blob
from roop.pipeline.staged_executor.video_cache import VideoStageCache


pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video cache tests")


def _make_image(seed, size=16):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = (seed * 31) % 255
    image[:, :, 1] = (seed * 67) % 255
    image[:, :, 2] = (seed * 97) % 255
    image[::2, ::2, :] = 255 - image[::2, ::2, :]
    return image


def _assert_near_equal(lhs, rhs, atol=8.0):
    assert lhs.shape == rhs.shape
    diff = np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))
    assert float(diff.max()) <= atol
    assert float(diff.mean()) <= (atol / 3.0)


def test_video_stage_cache_roundtrip_and_partial_reads(tmp_path):
    cache = VideoStageCache(crf=0, max_frame_extent=64, fps=12)
    cache_path = tmp_path / "swap" / "cache.bin"
    images = {f"face_{index:03d}": _make_image(index, size=24) for index in range(5)}

    cache.write(cache_path, images)

    video_path, index_path = cache._resolve_paths(cache_path)
    assert video_path.exists()
    assert index_path.exists()

    loaded = cache.read(cache_path)
    assert set(loaded) == set(images)
    for cache_key, original in images.items():
        _assert_near_equal(loaded[cache_key], original)

    partial = cache.read_keys(cache_path, ["face_001", "face_004"])
    assert set(partial) == {"face_001", "face_004"}
    _assert_near_equal(partial["face_001"], images["face_001"])
    _assert_near_equal(partial["face_004"], images["face_004"])

    index_data = cache._read_index(index_path)
    frame_indices = {entry["frame_idx"] for entry in index_data["items"].values()}
    assert frame_indices == {0, 1}


def test_read_stage_cache_map_migrates_legacy_pickle_blob_to_video_cache(tmp_path):
    cache_path = tmp_path / "enhance" / "cache.bin"
    legacy_images = {
        "face_a": _make_image(1),
        "face_b": _make_image(2),
    }
    write_cache_blob(cache_path, {"images": legacy_images})

    loaded = read_stage_cache_map(cache_path)

    assert set(loaded) == {"face_a", "face_b"}
    _assert_near_equal(loaded["face_a"], legacy_images["face_a"])
    _assert_near_equal(loaded["face_b"], legacy_images["face_b"])

    migrated_video = cache_path.with_suffix(".mp4")
    migrated_index = cache_path.with_suffix(".idx.bin")
    assert migrated_video.exists()
    assert migrated_index.exists()
    assert not cache_path.exists()


def test_legacy_pickle_blob_payload_is_still_readable_before_migration(tmp_path):
    cache_path = tmp_path / "mask" / "cache.bin"
    images = {"face_x": _make_image(9)}
    write_cache_blob(cache_path, {"images": images})

    payload = read_cache_blob(cache_path)

    assert set(payload["images"]) == {"face_x"}


def test_video_stage_cache_auto_codec_respects_global_video_encoder(monkeypatch):
    cache = VideoStageCache(codec="auto", crf=0, preset="veryfast")

    monkeypatch.setattr(roop.config.globals, "video_encoder", "libx264", raising=False)

    config = cache._resolve_writer_config()

    assert config["codec"] == "libx264"
    assert config["quality_args"] == ["-crf", "0"]
    assert config["ffmpeg_params"] == ["-preset", "veryfast", "-g", "1", "-bf", "0"]
    assert config["allow_fallback"] is False


def test_video_stage_cache_auto_gpu_fallback_retries_with_cpu(monkeypatch, tmp_path):
    cache = VideoStageCache(codec="auto", crf=0, preset="veryfast")
    cache_path = tmp_path / "swap" / "cache.bin"
    frames = [_make_image(1)]
    attempts = []

    monkeypatch.setattr(roop.config.globals, "video_encoder", "h264_nvenc", raising=False)

    def fake_write(video_path, _frames, writer_config):
        attempts.append(writer_config["codec"])
        if writer_config["codec"] == "h264_nvenc":
            raise OSError("nvenc unavailable")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"ok")

    monkeypatch.setattr(cache, "_write_video_with_config", fake_write)
    monkeypatch.setattr(cache, "_video_is_decodable", lambda _path: True)

    video_path, _index_path = cache._resolve_paths(cache_path)
    cache._write_video(video_path, frames)

    assert attempts == ["h264_nvenc", "libx264"]
    assert video_path.exists()
