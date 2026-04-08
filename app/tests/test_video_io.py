import roop.config.globals


def test_get_video_capture_backend_and_params_prefers_hwaccel_without_device_for_any(monkeypatch):
    try:
        import roop.media.video_io as video_io
    except ImportError as exc:
        raise AssertionError("roop.media.video_io helper should exist") from exc

    monkeypatch.setattr(video_io, "provider_uses_gpu", lambda: True)
    monkeypatch.setattr(roop.config.globals, "cuda_device_id", 0, raising=False)
    monkeypatch.setattr(video_io.cv2, "CAP_FFMPEG", 1900, raising=False)
    monkeypatch.setattr(video_io.cv2, "CAP_PROP_HW_ACCELERATION", 50, raising=False)
    monkeypatch.setattr(video_io.cv2, "VIDEO_ACCELERATION_ANY", 1, raising=False)
    monkeypatch.setattr(video_io.cv2, "CAP_PROP_HW_DEVICE", 51, raising=False)

    api_preference, params = video_io.get_video_capture_backend_and_params()

    assert api_preference == 1900
    assert params == [50, 1]


def test_resolve_video_writer_config_respects_explicit_cpu_codec(monkeypatch):
    try:
        import roop.media.video_io as video_io
    except ImportError as exc:
        raise AssertionError("roop.media.video_io helper should exist") from exc

    monkeypatch.setattr(video_io, "provider_uses_gpu", lambda: True)
    monkeypatch.setattr(video_io, "ffmpeg_supports_encoder", lambda encoder: encoder == "h264_nvenc")
    monkeypatch.setattr(roop.config.globals, "cuda_device_id", 0, raising=False)

    config = video_io.resolve_video_writer_config("libx264", 14)

    assert config["codec"] == "libx264"
    assert config["quality_args"] == ["-crf", "14"]
    assert config["ffmpeg_params"] == []


def test_resolve_video_writer_config_keeps_explicit_gpu_codec(monkeypatch):
    try:
        import roop.media.video_io as video_io
    except ImportError as exc:
        raise AssertionError("roop.media.video_io helper should exist") from exc

    monkeypatch.setattr(video_io, "ffmpeg_supports_encoder", lambda encoder: encoder == "h264_nvenc")
    monkeypatch.setattr(roop.config.globals, "cuda_device_id", 0, raising=False)

    config = video_io.resolve_video_writer_config("h264_nvenc", 14)

    assert config["codec"] == "h264_nvenc"
    assert config["quality_args"] == ["-cq", "14"]
    assert config["ffmpeg_params"] == ["-rc", "vbr", "-b:v", "0", "-preset", "p1", "-gpu", "0"]

