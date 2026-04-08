import shutil
import subprocess

import cv2
import roop.config.globals

from roop.memory import provider_uses_gpu


_FFMPEG_SUPPORT_CACHE = {}
_GPU_VIDEO_ENCODERS = {"h264_nvenc", "hevc_nvenc"}


def _ffmpeg_supports(kind: str, name: str) -> bool:
    cache_key = (kind, name)
    cached = _FFMPEG_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ffmpeg_binary = shutil.which("ffmpeg")
    if ffmpeg_binary is None:
        _FFMPEG_SUPPORT_CACHE[cache_key] = False
        return False

    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-h", f"{kind}={name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        supported = result.returncode == 0
    except Exception:
        supported = False

    _FFMPEG_SUPPORT_CACHE[cache_key] = supported
    return supported


def ffmpeg_supports_encoder(encoder_name: str) -> bool:
    return _ffmpeg_supports("encoder", encoder_name)


def get_video_capture_backend_and_params():
    if not provider_uses_gpu():
        return None, None
    if not all(
        hasattr(cv2, name)
        for name in ("CAP_FFMPEG", "CAP_PROP_HW_ACCELERATION", "VIDEO_ACCELERATION_ANY")
    ):
        return None, None

    params = [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
    return cv2.CAP_FFMPEG, params


def _open_video_capture_with_params(video_path: str, api_preference, params):
    try:
        return cv2.VideoCapture(video_path, api_preference, params)
    except TypeError:
        capture = cv2.VideoCapture()
        try:
            capture.open(video_path, api_preference, params)
        except TypeError:
            return None
        return capture


def open_video_capture(video_path: str):
    api_preference, params = get_video_capture_backend_and_params()
    if api_preference is not None and params is not None:
        capture = _open_video_capture_with_params(video_path, api_preference, params)
        if capture is not None and capture.isOpened():
            return capture
        if capture is not None:
            capture.release()
    return cv2.VideoCapture(video_path)


def resolve_video_writer_config(codec: str, quality: int) -> dict:
    resolved_codec = codec
    quality_args = ["-crf", str(quality)]
    ffmpeg_params = []
    if resolved_codec in _GPU_VIDEO_ENCODERS and ffmpeg_supports_encoder(resolved_codec):
        quality_args = ["-cq", str(quality)]
        ffmpeg_params = [
            "-rc",
            "vbr",
            "-b:v",
            "0",
            "-preset",
            "p1",
            "-gpu",
            str(roop.config.globals.cuda_device_id),
        ]

    return {
        "codec": resolved_codec,
        "quality_args": quality_args,
        "ffmpeg_params": ffmpeg_params,
    }

