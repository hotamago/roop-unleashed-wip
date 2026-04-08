import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np

import roop.config.globals
from roop.utils.cache_paths import get_jobs_root as get_persistent_jobs_root

from .video_cache import VideoStageCache, normalize_cache_image


PIPELINE_VERSION = 9
DETECT_PACK_FRAME_COUNT = 256
_STAGE_CACHE = VideoStageCache()


def get_jobs_root():
    return get_persistent_jobs_root()


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_dumps(data):
    return json.dumps(make_json_safe(data), sort_keys=True, separators=(",", ":"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(data), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_cache_blob(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))


def read_cache_blob(path: Path):
    return pickle.loads(path.read_bytes())


def write_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    success, encoded = cv2.imencode(ext, image)
    if success:
        encoded.tofile(str(path))


def chunked(items, size):
    step = max(1, size)
    for start in range(0, len(items), step):
        yield items[start:start + step]


def hash_numpy(value):
    if value is None:
        return None
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def hash_facesets(face_sets):
    payload = []
    for face_set in face_sets:
        faces = []
        for face in face_set.faces:
            faces.append(
                {
                    "embedding": hash_numpy(getattr(face, "embedding", None)),
                    "mask_offsets": list(getattr(face, "mask_offsets", [])),
                }
            )
        payload.append(faces)
    return hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()


def hash_target_faces(target_faces):
    payload = []
    for face in target_faces:
        payload.append(hash_numpy(getattr(face, "embedding", None)))
    return hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()


def get_entry_file_identity(entry):
    stat = os.stat(entry.filename)
    file_signature = getattr(entry, "file_signature", None)
    if file_signature:
        return {
            "signature": file_signature,
            "size": stat.st_size,
            "start": entry.startframe,
            "end": entry.endframe,
        }
    return {
        "path": os.path.abspath(entry.filename),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "start": entry.startframe,
        "end": entry.endframe,
    }


def get_staged_cache_options_snapshot(options):
    return {
        "processors": sorted(options.processors.keys()),
        "face_distance_threshold": options.face_distance_threshold,
        "blend_ratio": options.blend_ratio,
        "swap_mode": options.swap_mode,
        "selected_index": options.selected_index,
        "masking_text": options.masking_text,
        "num_swap_steps": options.num_swap_steps,
        "subsample_size": options.subsample_size,
        "restore_original_mouth": options.restore_original_mouth,
        "show_face_masking": bool(getattr(options, "show_face_masking", False)),
    }


def sanitize_job_path_segment(name: str, max_len: int = 120) -> str:
    if not name or not str(name).strip():
        return "unknown"
    token = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name).strip()).strip("._")
    return (token[:max_len] or "unknown").lower()


def _legacy_hashed_job_folder(entry, options) -> str:
    active_resume_job_key = getattr(roop.config.globals, "active_resume_job_key", None)
    job_identity = {
        "pipeline_version": PIPELINE_VERSION,
        "file": get_entry_file_identity(entry),
    }
    if active_resume_job_key:
        job_identity["resume_job_key"] = active_resume_job_key
    else:
        job_identity["inputs"] = hash_facesets(roop.config.globals.INPUT_FACESETS)
        job_identity["targets"] = hash_target_faces(roop.config.globals.TARGET_FACES)
        job_identity["selected_index"] = getattr(options, "selected_index", 0)
    return hashlib.sha256(json_dumps(job_identity).encode("utf-8")).hexdigest()


def get_entry_job_relpath(entry, options) -> str:
    session = getattr(roop.config.globals, "active_resume_cache_id", None)
    if session and str(session).strip():
        session_seg = sanitize_job_path_segment(str(session).strip())
        sig = getattr(entry, "file_signature", None)
        if isinstance(sig, str) and sig.strip():
            file_seg = sanitize_job_path_segment(sig.replace(":", "_"))
        else:
            fp = os.path.abspath(entry.filename)
            st = os.stat(entry.filename)
            digest = hashlib.sha256(f"{fp}|{st.st_size}|{entry.startframe}|{entry.endframe}".encode("utf-8")).hexdigest()[:24]
            file_seg = f"f_{digest}"
        return f"{session_seg}/{file_seg}"
    return _legacy_hashed_job_folder(entry, options)


def get_entry_job_key(entry, options):
    return get_entry_job_relpath(entry, options)


def get_staged_cache_manifest_signature(entry, options, output_method) -> str:
    blob = {
        "pipeline_version": PIPELINE_VERSION,
        "output_method": output_method,
        "options": get_staged_cache_options_snapshot(options),
        "frame_range": [entry.startframe, entry.endframe],
    }
    fs = getattr(entry, "file_signature", None)
    if isinstance(fs, str) and fs.strip():
        blob["file_content"] = fs.strip()
    else:
        st = os.stat(entry.filename)
        blob["file_anchor"] = f"{os.path.abspath(entry.filename)}|{st.st_size}"
    return hashlib.sha256(json_dumps(blob).encode("utf-8")).hexdigest()


def get_entry_signature(entry, options, output_method):
    return get_staged_cache_manifest_signature(entry, options, output_method)


def list_stage_images(stage_dir: Path, image_format: str):
    if not stage_dir.exists():
        return []
    return sorted(stage_dir.glob(f"*.{image_format}"))


def merge_stage_defaults(stage_state, defaults):
    merged = dict(defaults)
    if isinstance(stage_state, dict):
        merged.update(stage_state)
    return merged


def get_stage_cache_path(stage_dir_or_executor, stage_dir: Path | None = None):
    if stage_dir is None:
        stage_dir = stage_dir_or_executor
    return stage_dir / "cache.bin"


def get_stage_pack_path(stage_dir_or_executor, stage_dir: Path | None = None, start_seq: int | None = None, end_seq: int | None = None):
    if end_seq is None:
        stage_dir, start_seq, end_seq = stage_dir_or_executor, stage_dir, start_seq
    return stage_dir / "packs" / f"{start_seq:06d}_{end_seq:06d}.bin"


def _has_video_stage_cache(cache_path: Path) -> bool:
    video_path, index_path = _STAGE_CACHE._resolve_paths(cache_path)
    return video_path.exists() and index_path.exists()


def read_stage_cache_map(cache_path_or_executor, cache_path: Path | None = None):
    if cache_path is None:
        cache_path = cache_path_or_executor
    cache_path = Path(cache_path)
    if _has_video_stage_cache(cache_path):
        return _STAGE_CACHE.read(cache_path)
    if not cache_path.exists():
        return {}
    payload = read_cache_blob(cache_path)
    images = payload.get("images", {})
    if not isinstance(images, dict):
        return {}
    normalized = {str(key): normalize_cache_image(value) for key, value in images.items()}
    if normalized:
        try:
            _STAGE_CACHE.write(cache_path, normalized)
            cache_path.unlink(missing_ok=True)
        except Exception:
            pass
    return normalized


def read_stage_cache_keys(cache_path_or_executor, cache_path: Path | None = None, keys=None):
    if keys is None:
        cache_path, keys = cache_path_or_executor, cache_path
    cache_path = Path(cache_path)
    requested_keys = [str(key) for key in (keys or [])]
    if not requested_keys:
        return {}
    if _has_video_stage_cache(cache_path):
        return _STAGE_CACHE.read_keys(cache_path, requested_keys)
    legacy_cache = read_stage_cache_map(cache_path)
    return {
        cache_key: legacy_cache[cache_key]
        for cache_key in requested_keys
        if cache_key in legacy_cache
    }


def list_stage_cache_keys(cache_path_or_executor, cache_path: Path | None = None):
    if cache_path is None:
        cache_path = cache_path_or_executor
    cache_path = Path(cache_path)
    if _has_video_stage_cache(cache_path):
        return _STAGE_CACHE.list_keys(cache_path)
    if not cache_path.exists():
        return []
    payload = read_cache_blob(cache_path)
    images = payload.get("images", {})
    if not isinstance(images, dict):
        return []
    return sorted(str(key) for key in images)


def write_stage_cache_map(cache_path_or_executor, cache_path: Path | None = None, cache_map=None):
    if cache_map is None:
        cache_path, cache_map = cache_path_or_executor, cache_path
    serializable = {str(key): normalize_cache_image(value) for key, value in cache_map.items()}
    _STAGE_CACHE.write(cache_path, serializable)
    Path(cache_path).unlink(missing_ok=True)


def count_stage_cache_entries(cache_path_or_executor, cache_path: Path | None = None):
    if cache_path is None:
        cache_path = cache_path_or_executor
    cache_path = Path(cache_path)
    if _has_video_stage_cache(cache_path):
        return _STAGE_CACHE.count(cache_path)
    if not cache_path.exists():
        return 0
    payload = read_cache_blob(cache_path)
    images = payload.get("images", {})
    if not isinstance(images, dict):
        return 0
    return len(images)


def prepare_job(executor, entry, memory_plan):
    job_key = get_entry_job_key(entry, executor.options)
    cache_signature = get_entry_signature(entry, executor.options, executor.output_method)
    parts = [p for p in str(job_key).replace("\\", "/").split("/") if p]
    job_dir = executor.jobs_root.joinpath(*parts) if parts else executor.jobs_root / "unknown"
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = job_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("pipeline_version") != PIPELINE_VERSION or manifest.get("cache_signature") != cache_signature:
            shutil.rmtree(job_dir, ignore_errors=True)
            job_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "job_key": job_key,
                "cache_signature": cache_signature,
                "pipeline_version": PIPELINE_VERSION,
                "status": "running",
                "memory_plan": memory_plan,
                "chunks": {},
                "stages": {},
            }
            write_json(manifest_path, manifest)
    else:
        manifest = {
            "job_key": job_key,
            "cache_signature": cache_signature,
            "pipeline_version": PIPELINE_VERSION,
            "status": "running",
            "memory_plan": memory_plan,
            "chunks": {},
            "stages": {},
        }
        write_json(manifest_path, manifest)
    manifest["job_key"] = job_key
    manifest["cache_signature"] = cache_signature
    manifest["memory_plan"] = memory_plan
    manifest["status"] = "running"
    return job_dir, manifest


def cleanup_job_dir(_executor, _job_dir):
    return


__all__ = [
    "DETECT_PACK_FRAME_COUNT",
    "PIPELINE_VERSION",
    "chunked",
    "cleanup_job_dir",
    "count_stage_cache_entries",
    "get_entry_file_identity",
    "get_entry_job_key",
    "get_entry_job_relpath",
    "get_entry_signature",
    "get_jobs_root",
    "get_stage_cache_path",
    "get_stage_pack_path",
    "get_staged_cache_manifest_signature",
    "get_staged_cache_options_snapshot",
    "hash_facesets",
    "hash_numpy",
    "hash_target_faces",
    "json_dumps",
    "list_stage_images",
    "make_json_safe",
    "merge_stage_defaults",
    "normalize_cache_image",
    "prepare_job",
    "read_cache_blob",
    "read_json",
    "read_stage_cache_keys",
    "read_stage_cache_map",
    "list_stage_cache_keys",
    "sanitize_job_path_segment",
    "write_cache_blob",
    "write_image",
    "write_json",
    "write_stage_cache_map",
]
