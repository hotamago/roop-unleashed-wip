import copy
import hashlib
import os
import json
import shutil
import time
import gradio as gr
import numpy as np
import roop.utils as util
import roop.config.globals
import ui.globals
from roop.face_swap_models import (
    get_face_swap_model_key,
    get_face_swap_upscale_choices,
    get_face_swap_upscale_hint,
    normalize_face_swap_upscale,
    parse_face_swap_upscale_size,
)
from roop.utils.cache_paths import get_gradio_temp_root
from roop.face import extract_face_images, create_blank_image, use_face_analysis_modules
from roop.media.capturer import get_video_frame, get_video_frame_total, get_image_frame
from roop.pipeline.entry import ProcessEntry
from roop.pipeline.options import ProcessOptions
from roop.pipeline.faceset import FaceSet
from roop.progress.status import get_processing_status_markdown, set_memory_status, set_processing_message, start_processing_status

last_image = None


def build_upscale_dropdown_update(model_name=None, selected_upscale=None):
    current_model = get_face_swap_model_key(
        model_name or getattr(roop.config.globals.CFG, "face_swap_model", None)
    )
    normalized_upscale = normalize_face_swap_upscale(
        selected_upscale if selected_upscale is not None else getattr(roop.config.globals.CFG, "subsample_upscale", "256px"),
        current_model,
    )
    return gr.Dropdown(
        choices=get_face_swap_upscale_choices(current_model),
        value=normalized_upscale,
        info=get_face_swap_upscale_hint(current_model),
        interactive=True,
    )


def normalize_upscale_for_current_model(selected_upscale):
    current_model = get_face_swap_model_key(getattr(roop.config.globals.CFG, "face_swap_model", None))
    normalized_upscale = normalize_face_swap_upscale(selected_upscale, current_model)
    roop.config.globals.CFG.subsample_upscale = normalized_upscale
    roop.config.globals.subsample_size = parse_face_swap_upscale_size(normalized_upscale, current_model)
    return build_upscale_dropdown_update(current_model, normalized_upscale)


def sync_resume_processing_cache_id(resume_path) -> None:
    """processing_cache/jobs/<stem>/... uses the resume JSON basename stem (stable human-visible id)."""
    if not resume_path:
        roop.config.globals.active_resume_cache_id = None
        return
    base = os.path.basename(os.path.normpath(str(resume_path)))
    stem, _ext = os.path.splitext(base)
    roop.config.globals.active_resume_cache_id = stem if stem else None


SELECTED_INPUT_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0

input_faces = None
target_faces = None
previewimage = None

selected_preview_index = 0

is_processing = False            

list_files_process : list[ProcessEntry] = []
no_face_choices = ["Use untouched original frame","Retry rotated", "Skip Frame", "Skip Frame if no similar face", "Use last swapped"]
swap_choices = ["First found", "All input faces", "All female", "All male", "All faces", "Selected face"]

current_video_fps = 50

manual_masking = False
RESUME_CACHE_VERSION = 1
FILE_SIGNATURE_CACHE = {}

# Settings keys that only affect VRAM/throughput (memory_plan), not staged pipeline outputs.
# They must not change resume filenames, resume_key, or processing_cache invalidation.
RESUME_SETTINGS_PERF_KEYS = frozenset(
    {
        "mask_batch_size",
        "swap_batch_size",
        "enhance_batch_size",
        "prefetch_frames",
        "staged_chunk_size",
        "detect_pack_frame_count",
        "single_batch_workers",
        "max_threads",
        "video_quality",
    }
)

RESUME_IDENTITY_ROOT_KEYS = frozenset({"version", "sources", "targets", "selection", "settings"})

RESUME_SETTINGS_FLOAT_KEYS = frozenset({"face_distance", "blend_ratio"})


def _normalize_embedding_for_resume_signature(face_ref: dict) -> None:
    emb = face_ref.get("face_embedding")
    if not isinstance(emb, list) or not emb:
        return
    face_ref["face_embedding"] = [float(f"{float(v):.6g}") for v in emb]


def _normalize_numeric_fields_for_resume_identity(canonical: dict) -> None:
    """Make disk vs runtime JSON compare equal (int/float quirks, fps precision)."""
    for source_ref in canonical.get("sources") or []:
        mo = source_ref.get("mask_offsets")
        if isinstance(mo, list) and mo:
            source_ref["mask_offsets"] = [float(f"{float(x):.6g}") for x in mo]
    targets = canonical.get("targets") or {}
    for target_entry in targets.get("files") or []:
        if not isinstance(target_entry, dict):
            continue
        if "fps" in target_entry and target_entry["fps"] is not None:
            target_entry["fps"] = float(f"{float(target_entry['fps']):.6g}")
        for ik in ("startframe", "endframe"):
            if ik in target_entry and target_entry[ik] is not None:
                target_entry[ik] = int(target_entry[ik])
    spi = targets.get("selected_preview_index")
    if spi is not None:
        targets["selected_preview_index"] = int(spi)
    sel = canonical.get("selection") or {}
    if isinstance(sel, dict):
        if "input_face_index" in sel and sel["input_face_index"] is not None:
            sel["input_face_index"] = int(sel["input_face_index"])
        if "target_face_index" in sel and sel["target_face_index"] is not None:
            sel["target_face_index"] = int(sel["target_face_index"])
    settings = canonical.get("settings")
    if isinstance(settings, dict):
        for fk in RESUME_SETTINGS_FLOAT_KEYS:
            if fk in settings and isinstance(settings[fk], (int, float)):
                settings[fk] = float(f"{float(settings[fk]):.6g}")


def default_mask_offsets():
    return [0, 0, 0, 0, 20.0, 10.0, 1.0, 1.0, 1.0, 1.0]


def get_resume_cache_root():
    resume_root = util.resolve_relative_path('../resume_cache')
    os.makedirs(resume_root, exist_ok=True)
    return resume_root


def list_resume_cache_files():
    resume_root = get_resume_cache_root()
    files = [
        os.path.join(resume_root, filename)
        for filename in os.listdir(resume_root)
        if filename.lower().endswith('.json') and os.path.isfile(os.path.join(resume_root, filename))
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def get_resume_status_markdown(active_path=None, detail=None):
    files = list_resume_cache_files()[:5]
    lines = [
        "**Resume Cache**",
        f"- Folder: `{get_resume_cache_root()}`",
    ]
    if active_path:
        lines.append(f"- Active file: `{active_path}`")
    if detail:
        lines.append(f"- Status: {detail}")
    if files:
        lines.append(f"- Recent files: {', '.join(os.path.basename(path) for path in files)}")
    else:
        lines.append("- Recent files: none")
    return "\n".join(lines)


def normalize_resume_path(raw_path):
    if raw_path is None:
        return None
    clean_path = raw_path.strip().strip('"').strip("'")
    if len(clean_path) < 1:
        return None
    if not os.path.isabs(clean_path):
        cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), clean_path))
        cache_candidate = os.path.abspath(os.path.join(get_resume_cache_root(), clean_path))
        clean_path = cwd_candidate if os.path.exists(cwd_candidate) else cache_candidate
    return os.path.normpath(clean_path)


def normalize_source_path(raw_path, warn=True):
    if raw_path is None:
        return None
    clean_path = raw_path.strip().strip('"').strip("'")
    if len(clean_path) < 1:
        return None
    if not os.path.isabs(clean_path):
        clean_path = os.path.abspath(os.path.join(os.getcwd(), clean_path))
    clean_path = os.path.normpath(clean_path)
    if not os.path.exists(clean_path):
        if warn:
            gr.Warning(f"Source path not found: {clean_path}")
        return None
    if os.path.isdir(clean_path):
        if warn:
            gr.Warning(f"Directories are not supported for sources: {clean_path}")
        return None
    if not (util.has_image_extension(clean_path) or clean_path.lower().endswith('fsz')):
        if warn:
            gr.Warning(f"Unsupported source file: {clean_path}")
        return None
    return clean_path


def safe_filename_token(name):
    token = ''.join(char if char.isalnum() or char in ('-', '_') else '_' for char in name)
    token = token.strip('_')
    return token[:80] or "resume"


def hash_file_contents(path):
    if path is None or not os.path.isfile(path):
        return None
    stat = os.stat(path)
    cache_key = (os.path.normcase(os.path.normpath(os.path.abspath(path))), stat.st_size, int(stat.st_mtime_ns))
    cached = FILE_SIGNATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    signature = f"sha256:{digest.hexdigest()}"
    FILE_SIGNATURE_CACHE[cache_key] = signature
    return signature


def should_use_stable_file_signature(path):
    if not path:
        return False
    candidate_roots = [os.environ.get("GRADIO_TEMP_DIR")]
    try:
        candidate_roots.append(str(get_gradio_temp_root()))
    except Exception:
        pass
    try:
        candidate_roots.append(get_resume_cache_root())
    except Exception:
        pass
    return any(root and is_path_within_root(path, root) for root in candidate_roots)


def get_stable_file_signature(path):
    if path is None:
        return None
    normalized_path = os.path.normpath(os.path.abspath(path))
    if not should_use_stable_file_signature(normalized_path):
        return None
    return hash_file_contents(normalized_path)


def get_source_file_signature(source_ref):
    signature = source_ref.get("file_signature")
    if signature:
        return signature
    source_path = normalize_source_path(source_ref.get("path"), warn=False)
    if source_path is None:
        source_path = normalize_source_path(source_ref.get("resume_cached_path"), warn=False)
    return hash_file_contents(source_path)


def get_target_file_signature(target_ref, path_key):
    signature = target_ref.get("file_signature")
    if signature:
        return signature
    target_path = normalize_target_path(target_ref.get(path_key), warn=False)
    if target_path is None:
        target_path = normalize_target_path(target_ref.get("resume_cached_path"), warn=False)
    return get_stable_file_signature(target_path)


def serialize_process_entries(entries):
    payload = []
    for entry in entries:
        entry_payload = {
            "filename": os.path.abspath(entry.filename),
            "startframe": int(entry.startframe),
            "endframe": int(entry.endframe),
            "fps": float(entry.fps or 0),
            "display_name": getattr(entry, "display_name", os.path.basename(entry.filename)),
        }
        file_signature = getattr(entry, "file_signature", None) or get_stable_file_signature(entry.filename)
        if file_signature:
            entry.file_signature = file_signature
            entry_payload["file_signature"] = file_signature
        payload.append(entry_payload)
    return payload


def snapshot_input_face_refs():
    refs = []
    if len(ui.globals.ui_input_face_refs) < len(roop.config.globals.INPUT_FACESETS):
        raise ValueError("Current source inputs are missing resume metadata. Re-add the source files once before running.")
    for index, face_set in enumerate(roop.config.globals.INPUT_FACESETS):
        base_ref = dict(ui.globals.ui_input_face_refs[index])
        if not base_ref.get("path"):
            raise ValueError("A source input is missing its source path. Re-add the source files before running.")
        mask_offsets = default_mask_offsets()
        if getattr(face_set, "faces", None):
            mask_offsets = list(getattr(face_set.faces[0], "mask_offsets", mask_offsets))
        base_ref["path"] = os.path.abspath(base_ref["path"])
        base_ref["mask_offsets"] = list(mask_offsets)
        file_signature = get_source_file_signature(base_ref)
        if file_signature:
            base_ref["file_signature"] = file_signature
        refs.append(base_ref)
    return refs


def snapshot_target_face_refs():
    refs = []
    if len(roop.config.globals.TARGET_FACES) > 0 and len(ui.globals.ui_target_face_refs) < len(roop.config.globals.TARGET_FACES):
        raise ValueError("Current selected target faces are missing resume metadata. Re-add them from preview before running.")
    for index in range(min(len(ui.globals.ui_target_face_refs), len(roop.config.globals.TARGET_FACES))):
        base_ref = dict(ui.globals.ui_target_face_refs[index])
        if not base_ref.get("path"):
            continue
        base_ref["path"] = os.path.abspath(base_ref["path"])
        base_ref["frame_number"] = int(base_ref.get("frame_number", 0) or 0)
        base_ref["face_index"] = int(base_ref.get("face_index", 0) or 0)
        face_embedding = get_face_embedding_vector(roop.config.globals.TARGET_FACES[index])
        if face_embedding is not None:
            base_ref["face_embedding"] = face_embedding.tolist()
        file_signature = get_target_file_signature(base_ref, "path")
        if file_signature:
            base_ref["file_signature"] = file_signature
        refs.append(base_ref)
    return refs


def build_resume_payload(settings):
    if len(list_files_process) < 1:
        raise ValueError("No target files are queued yet, so there is nothing to resume.")
    payload = {
        "version": RESUME_CACHE_VERSION,
        "created_at": int(time.time()),
        "sources": snapshot_input_face_refs(),
        "targets": {
            "files": serialize_process_entries(list_files_process),
            "selected_faces": snapshot_target_face_refs(),
            "selected_preview_index": int(selected_preview_index),
        },
        "selection": {
            "input_face_index": int(SELECTED_INPUT_FACE_INDEX),
            "target_face_index": int(SELECTED_TARGET_FACE_INDEX),
        },
        "settings": settings,
    }
    return payload


def get_resume_payload_identity(payload):
    canonical = copy.deepcopy(payload)
    canonical.pop("created_at", None)
    canonical.pop("resume_key", None)
    canonical.pop("resume_job_key", None)
    canonical.pop("__path__", None)
    for k in list(canonical.keys()):
        if k not in RESUME_IDENTITY_ROOT_KEYS:
            canonical.pop(k, None)
    for source_ref in canonical.get("sources") or []:
        source_ref["path"] = get_source_file_signature(source_ref) or source_ref.get("path")
        source_ref.pop("resume_cached_path", None)
        source_ref.pop("file_signature", None)
    targets = canonical.get("targets") or {}
    for target_entry in targets.get("files") or []:
        target_entry["filename"] = get_target_file_signature(target_entry, "filename") or target_entry.get("filename")
        target_entry.pop("resume_cached_path", None)
        target_entry.pop("file_signature", None)
        target_entry.pop("display_name", None)
    for target_face_ref in targets.get("selected_faces") or []:
        target_face_ref["path"] = get_target_file_signature(target_face_ref, "path") or target_face_ref.get("path")
        target_face_ref.pop("resume_cached_path", None)
        target_face_ref.pop("file_signature", None)
        if "frame_number" in target_face_ref and target_face_ref["frame_number"] is not None:
            target_face_ref["frame_number"] = int(target_face_ref["frame_number"])
        if "face_index" in target_face_ref and target_face_ref["face_index"] is not None:
            target_face_ref["face_index"] = int(target_face_ref["face_index"])
        _normalize_embedding_for_resume_signature(target_face_ref)
    settings = canonical.get("settings")
    if isinstance(settings, dict):
        for k in RESUME_SETTINGS_PERF_KEYS:
            settings.pop(k, None)
    _normalize_numeric_fields_for_resume_identity(canonical)
    return canonical


def get_resume_job_identity(payload):
    canonical = get_resume_payload_identity(payload)
    canonical.pop("settings", None)
    return canonical


def get_resume_payload_signature(payload):
    canonical = get_resume_payload_identity(payload)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_resume_job_signature(payload):
    canonical = get_resume_job_identity(payload)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def pick_resume_job_key_for_write(prior_disk: dict | None, payload: dict) -> str:
    """
    Keep the same resume_job_key JSON field when job identity (sources/targets/selection) is unchanged.
    (Job folders are keyed by resume JSON stem + target file id - see active_resume_cache_id.)
    """
    if isinstance(prior_disk, dict):
        old_jk = prior_disk.get("resume_job_key")
        if isinstance(old_jk, str) and old_jk:
            try:
                if get_resume_job_identity(prior_disk) == get_resume_job_identity(payload):
                    return old_jk
            except Exception:
                pass
    return get_resume_job_signature(payload)


def get_resume_payload_path(payload):
    target_entries = (payload.get("targets") or {}).get("files") or []
    first_target_entry = target_entries[0] if target_entries else {}
    first_target = first_target_entry.get("display_name") or os.path.basename(first_target_entry.get("filename") or "faceswap")
    resume_key = get_resume_payload_signature(payload)
    filename = f"{safe_filename_token(os.path.splitext(first_target)[0])}_{resume_key[:12]}.json"
    return os.path.join(get_resume_cache_root(), filename), resume_key


def resolve_equivalent_resume_path(payload, desired_resume_path):
    active_resume_path = normalize_resume_path(ui.globals.ui_resume_last_path)
    if active_resume_path is None or not os.path.isfile(active_resume_path):
        return desired_resume_path
    if os.path.normcase(os.path.normpath(active_resume_path)) == os.path.normcase(os.path.normpath(desired_resume_path)):
        return desired_resume_path
    try:
        active_payload = read_resume_payload(active_resume_path)
    except Exception:
        return desired_resume_path
    if get_resume_payload_signature(active_payload) == get_resume_payload_signature(payload):
        return active_resume_path
    # Same faces/files/selection + same semantic settings, but hash drifted (legacy JSON, float noise, extra keys).
    # Keep writing to the loaded resume path so resume_cache does not spawn a duplicate file.
    if (
        get_resume_job_signature(active_payload) == get_resume_job_signature(payload)
        and get_resume_payload_identity(active_payload).get("settings")
        == get_resume_payload_identity(payload).get("settings")
    ):
        return active_resume_path
    return desired_resume_path


def get_resume_source_assets_root(resume_path):
    return os.path.join(os.path.splitext(resume_path)[0] + "_assets", "sources")


def get_resume_target_assets_root(resume_path):
    return os.path.join(os.path.splitext(resume_path)[0] + "_assets", "targets")


def is_path_within_root(path, root):
    if not path or not root:
        return False
    try:
        normalized_path = os.path.normcase(os.path.normpath(os.path.abspath(str(path))))
        normalized_root = os.path.normcase(os.path.normpath(os.path.abspath(str(root))))
        return os.path.commonpath([normalized_path, normalized_root]) == normalized_root
    except ValueError:
        return False


def should_snapshot_target_path(target_path):
    if not target_path:
        return False
    candidate_roots = []
    gradio_temp_dir = os.environ.get("GRADIO_TEMP_DIR")
    if gradio_temp_dir:
        candidate_roots.append(gradio_temp_dir)
    try:
        candidate_roots.append(str(get_gradio_temp_root()))
    except Exception:
        pass
    return any(is_path_within_root(target_path, root) for root in candidate_roots)


def reuse_existing_asset_path(asset_path, assets_root):
    if asset_path is None:
        return None
    normalized_asset_path = os.path.normpath(os.path.abspath(asset_path))
    if not os.path.isfile(normalized_asset_path):
        return None
    if not is_path_within_root(normalized_asset_path, assets_root):
        return None
    return normalized_asset_path


def snapshot_resume_source_files(payload, resume_path):
    source_refs = payload.get("sources") or []
    if len(source_refs) < 1:
        return payload
    assets_root = get_resume_source_assets_root(resume_path)
    os.makedirs(assets_root, exist_ok=True)
    copied_paths = {}
    for index, source_ref in enumerate(source_refs):
        source_path = normalize_source_path(source_ref.get("path"), warn=False)
        if source_path is None:
            source_ref.pop("resume_cached_path", None)
            continue
        cache_key = os.path.normcase(os.path.normpath(source_path))
        cached_path = copied_paths.get(cache_key)
        if cached_path is None:
            cached_path = reuse_existing_asset_path(source_ref.get("resume_cached_path"), assets_root)
            if cached_path is None:
                cached_path = reuse_existing_asset_path(source_path, assets_root)
            if cached_path is None:
                stem, ext = os.path.splitext(os.path.basename(source_path))
                safe_name = safe_filename_token(stem)
                filename = f"{index:03d}_{safe_name}{ext.lower()}"
                cached_path = os.path.join(assets_root, filename)
                if os.path.normcase(os.path.normpath(source_path)) != os.path.normcase(os.path.normpath(cached_path)):
                    shutil.copy2(source_path, cached_path)
            copied_paths[cache_key] = cached_path
        source_ref["resume_cached_path"] = cached_path
    return payload


def snapshot_resume_target_files(payload, resume_path):
    targets = payload.get("targets") or {}
    target_refs = []
    for entry in targets.get("files") or []:
        target_refs.append((entry, "filename"))
    for face_ref in targets.get("selected_faces") or []:
        target_refs.append((face_ref, "path"))
    if len(target_refs) < 1:
        return payload
    assets_root = get_resume_target_assets_root(resume_path)
    copied_paths = {}
    for target_ref, path_key in target_refs:
        target_path = normalize_target_path(target_ref.get(path_key), warn=False)
        if target_path is None or not should_snapshot_target_path(target_path):
            target_ref.pop("resume_cached_path", None)
            continue
        cache_key = os.path.normcase(os.path.normpath(target_path))
        cached_path = copied_paths.get(cache_key)
        if cached_path is None:
            cached_path = reuse_existing_asset_path(target_ref.get("resume_cached_path"), assets_root)
            if cached_path is None:
                cached_path = reuse_existing_asset_path(target_path, assets_root)
            if cached_path is None:
                stem, ext = os.path.splitext(os.path.basename(target_path))
                safe_name = safe_filename_token(stem)
                filename = f"{len(copied_paths):03d}_{safe_name}{ext.lower()}"
                cached_path = os.path.join(assets_root, filename)
                os.makedirs(assets_root, exist_ok=True)
                if os.path.normcase(os.path.normpath(target_path)) != os.path.normcase(os.path.normpath(cached_path)):
                    shutil.copy2(target_path, cached_path)
            copied_paths[cache_key] = cached_path
        target_ref["resume_cached_path"] = cached_path
    return payload


def write_resume_payload_with_result(payload):
    payload = copy.deepcopy(payload)
    desired_resume_path, resume_key = get_resume_payload_path(payload)
    resume_path = None
    bound = normalize_resume_path(getattr(ui.globals, "ui_resume_bound_path", None))
    if bound:
        if os.path.isfile(bound):
            resume_path = bound
        else:
            ui.globals.ui_resume_bound_path = None
    if resume_path is None:
        sticky = normalize_resume_path(ui.globals.ui_resume_last_path)
        if sticky and os.path.isfile(sticky):
            resume_path = sticky
        else:
            resume_path = resolve_equivalent_resume_path(payload, desired_resume_path)
    prior_disk = None
    if os.path.isfile(resume_path):
        try:
            with open(resume_path, "r", encoding="utf-8") as handle:
                prior_disk = json.load(handle)
        except Exception:
            prior_disk = None
    payload["resume_key"] = resume_key
    payload["resume_job_key"] = pick_resume_job_key_for_write(prior_disk, payload)
    payload = snapshot_resume_source_files(payload, resume_path)
    payload = snapshot_resume_target_files(payload, resume_path)
    reused_existing = os.path.isfile(resume_path)
    with open(resume_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    ui.globals.ui_resume_last_path = resume_path
    sync_resume_processing_cache_id(resume_path)
    return resume_path, reused_existing, resume_key


def write_resume_payload(payload):
    resume_path, _, _ = write_resume_payload_with_result(payload)
    return resume_path


def get_resume_settings_from_payload(payload):
    settings = dict(payload.get("settings") or {})
    face_swap_model = get_face_swap_model_key(
        settings.get("face_swap_model", getattr(roop.config.globals.CFG, "face_swap_model", None))
    )
    return {
        "output_method": settings.get("output_method", roop.config.globals.CFG.output_method),
        "enhancer": settings.get("enhancer", roop.config.globals.CFG.selected_enhancer),
        "face_swap_model": face_swap_model,
        "detection": settings.get("detection", roop.config.globals.CFG.face_detection_mode),
        "keep_frames": bool(settings.get("keep_frames", roop.config.globals.CFG.keep_frames)),
        "wait_after_extraction": bool(settings.get("wait_after_extraction", roop.config.globals.CFG.wait_after_extraction)),
        "skip_audio": bool(settings.get("skip_audio", roop.config.globals.CFG.skip_audio)),
        "face_distance": float(settings.get("face_distance", roop.config.globals.CFG.max_face_distance)),
        "blend_ratio": float(settings.get("blend_ratio", roop.config.globals.CFG.blend_ratio)),
        "selected_mask_engine": settings.get("selected_mask_engine", roop.config.globals.CFG.mask_engine),
        "clip_text": settings.get("clip_text", roop.config.globals.CFG.mask_clip_text),
        "processing_method": settings.get("processing_method", roop.config.globals.CFG.video_swapping_method),
        "no_face_action": settings.get("no_face_action", roop.config.globals.CFG.no_face_action),
        "vr_mode": bool(settings.get("vr_mode", roop.config.globals.CFG.vr_mode)),
        "autorotate": bool(settings.get("autorotate", roop.config.globals.CFG.autorotate_faces)),
        "restore_original_mouth": bool(settings.get("restore_original_mouth", roop.config.globals.CFG.restore_original_mouth)),
        "num_swap_steps": int(settings.get("num_swap_steps", roop.config.globals.CFG.num_swap_steps)),
        "upsample": normalize_face_swap_upscale(
            settings.get("upsample", roop.config.globals.CFG.subsample_upscale),
            face_swap_model,
        ),
    }


def read_resume_payload(resume_path):
    resolved_path = normalize_resume_path(resume_path)
    if resolved_path is None or not os.path.isfile(resolved_path):
        raise ValueError("Resume file not found.")
    with open(resolved_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("version", 0) or 0) != RESUME_CACHE_VERSION:
        raise ValueError("Unsupported resume file version.")
    payload["__path__"] = resolved_path
    return payload


def clear_input_face_state():
    global SELECTED_INPUT_FACE_INDEX
    ui.globals.ui_input_thumbs.clear()
    ui.globals.ui_input_face_refs.clear()
    roop.config.globals.INPUT_FACESETS.clear()
    SELECTED_INPUT_FACE_INDEX = 0


def clear_target_face_state():
    global SELECTED_TARGET_FACE_INDEX
    ui.globals.ui_target_thumbs.clear()
    ui.globals.ui_target_face_refs.clear()
    roop.config.globals.TARGET_FACES.clear()
    SELECTED_TARGET_FACE_INDEX = 0


def extract_face_images_for_swap(source_path, video_info):
    with use_face_analysis_modules(["recognition"]):
        return extract_face_images(source_path, video_info)


def append_faceset_source(source_path, source_ref):
    unzipfolder = os.path.join(os.environ["TEMP"], 'faceset')
    if os.path.isdir(unzipfolder):
        for filename in os.listdir(unzipfolder):
            os.remove(os.path.join(unzipfolder, filename))
    else:
        os.makedirs(unzipfolder)
    util.mkdir_with_umask(unzipfolder)
    util.unzip(source_path, unzipfolder)
    is_first = True
    face_set = FaceSet()
    with use_face_analysis_modules(["recognition"]):
        for filename in os.listdir(unzipfolder):
            if not filename.endswith(".png"):
                continue
            full_path = os.path.join(unzipfolder, filename)
            selection_faces_data = extract_face_images(full_path, (False, 0))
            for face_data in selection_faces_data:
                face = face_data[0]
                face.mask_offsets = list(default_mask_offsets())
                face_set.faces.append(face)
                if is_first:
                    ui.globals.ui_input_thumbs.append(util.convert_to_gradio(face_data[1]))
                    is_first = False
                face_set.ref_images.append(get_image_frame(full_path))
    if len(face_set.faces) < 1:
        raise ValueError(f"No faces found in faceset: {source_path}")
    if len(face_set.faces) > 1:
        face_set.AverageEmbeddings()
    face_set.faces[0].mask_offsets = list(source_ref.get("mask_offsets", default_mask_offsets()))
    roop.config.globals.INPUT_FACESETS.append(face_set)
    ui.globals.ui_input_face_refs.append(dict(source_ref))


def append_image_source(source_path, source_ref):
    selection_faces_data = extract_face_images_for_swap(source_path, (False, 0))
    face_index = int(source_ref.get("face_index", 0) or 0)
    if face_index < 0 or face_index >= len(selection_faces_data):
        raise ValueError(f"Source face index {face_index} is out of range for {source_path}")
    face_set = FaceSet()
    face = selection_faces_data[face_index][0]
    face.mask_offsets = list(source_ref.get("mask_offsets", default_mask_offsets()))
    face_set.faces.append(face)
    roop.config.globals.INPUT_FACESETS.append(face_set)
    ui.globals.ui_input_thumbs.append(util.convert_to_gradio(selection_faces_data[face_index][1]))
    ui.globals.ui_input_face_refs.append(dict(source_ref))


def restore_input_faces_from_resume(source_refs):
    clear_input_face_state()
    for source_ref in source_refs:
        source_path = normalize_source_path(source_ref.get("path"), warn=False)
        if source_path is None:
            source_path = normalize_source_path(source_ref.get("resume_cached_path"), warn=False)
        if source_path is None:
            original_path = source_ref.get("path")
            if original_path:
                normalize_source_path(original_path, warn=True)
            cached_path = source_ref.get("resume_cached_path")
            if cached_path:
                gr.Warning(f"Resume source snapshot not found: {os.path.normpath(cached_path)}")
            continue
        if source_path is None:
            continue
        restored_ref = dict(source_ref)
        restored_ref["path"] = source_path
        if restored_ref.get("type") == "faceset":
            append_faceset_source(source_path, restored_ref)
        else:
            append_image_source(source_path, restored_ref)
    if len(roop.config.globals.INPUT_FACESETS) < 1:
        raise ValueError("The resume file did not restore any source faces.")


def append_target_face_from_resume(face_ref):
    target_path = resolve_resume_target_path(face_ref, "path")
    if target_path is None:
        return
    frame_number = int(face_ref.get("frame_number", 0) or 0)
    is_video = util.is_video(target_path) or target_path.lower().endswith('gif')
    if is_video and frame_number <= 0:
        frame_number = 1
    faces_data = extract_face_images_for_swap(target_path, (is_video, frame_number))
    face_index = select_target_face_index_from_resume(faces_data, face_ref)
    if face_index < 0 or face_index >= len(faces_data):
        raise ValueError(f"Target face index {face_index} is out of range for {target_path}")
    roop.config.globals.TARGET_FACES.append(faces_data[face_index][0])
    ui.globals.ui_target_thumbs.append(util.convert_to_gradio(faces_data[face_index][1]))
    restored_ref = dict(face_ref)
    restored_ref["path"] = target_path
    restored_ref["frame_number"] = frame_number
    restored_ref["face_index"] = face_index
    ui.globals.ui_target_face_refs.append(restored_ref)


def get_face_embedding_vector(face):
    if face is None:
        return None
    embedding = getattr(face, "embedding", None)
    if embedding is None:
        try:
            embedding = face["embedding"]
        except Exception:
            embedding = None
    if embedding is None:
        embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size < 1 or not np.isfinite(vector).all():
        return None
    return vector


def select_target_face_index_from_resume(faces_data, face_ref):
    stored_embedding = face_ref.get("face_embedding")
    if stored_embedding is not None:
        stored_vector = np.asarray(stored_embedding, dtype=np.float32).reshape(-1)
        stored_norm = float(np.linalg.norm(stored_vector))
        best_index = None
        best_distance = None
        if stored_vector.size > 0 and stored_norm > 0:
            for index, (face, _thumb) in enumerate(faces_data):
                candidate_vector = get_face_embedding_vector(face)
                if candidate_vector is None or candidate_vector.shape != stored_vector.shape:
                    continue
                candidate_norm = float(np.linalg.norm(candidate_vector))
                if candidate_norm <= 0:
                    continue
                similarity = float(np.dot(candidate_vector, stored_vector) / (candidate_norm * stored_norm))
                distance = 1.0 - similarity
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index is not None:
                return best_index
    return int(face_ref.get("face_index", 0) or 0)


def restore_target_faces_from_resume(face_refs):
    clear_target_face_state()
    for face_ref in face_refs:
        append_target_face_from_resume(face_ref)


def restore_process_entries(process_entries):
    global list_files_process

    list_files_process.clear()
    target_paths = []
    for entry_data in process_entries:
        filename = resolve_resume_target_path(entry_data, "filename")
        if filename is None:
            continue
        entry = ProcessEntry(
            filename,
            int(entry_data.get("startframe", 0) or 0),
            int(entry_data.get("endframe", 0) or 0),
            float(entry_data.get("fps", 0) or 0),
            file_signature=entry_data.get("file_signature") or get_stable_file_signature(filename),
            display_name=entry_data.get("display_name") or os.path.basename(entry_data.get("filename") or filename),
        )
        list_files_process.append(entry)
        target_paths.append(filename)
    ui.globals.ui_target_files = list(target_paths)
    if len(list_files_process) < 1:
        raise ValueError("The resume file did not restore any target files.")
    return target_paths


def get_selected_input_mask_values(face_index=None):
    if face_index is None:
        face_index = SELECTED_INPUT_FACE_INDEX
    offsets = list(default_mask_offsets())
    if 0 <= face_index < len(roop.config.globals.INPUT_FACESETS):
        face = roop.config.globals.INPUT_FACESETS[face_index].faces[0]
        offsets = list(getattr(face, "mask_offsets", offsets))
    while len(offsets) < 10:
        offsets.append(1.0)
    return tuple(offsets[:10])


def load_resume_into_runtime(resume_path):
    global selected_preview_index, SELECTED_INPUT_FACE_INDEX, SELECTED_TARGET_FACE_INDEX

    payload = read_resume_payload(resume_path)
    roop.config.globals.active_resume_key = get_resume_payload_signature(payload)
    # Must match JSON + existing jobs folder; recomputing can drift on float/path noise.
    roop.config.globals.active_resume_job_key = payload.get("resume_job_key") or get_resume_job_signature(payload)
    restore_input_faces_from_resume(payload.get("sources") or [])
    target_paths = restore_process_entries((payload.get("targets") or {}).get("files") or [])
    restore_target_faces_from_resume((payload.get("targets") or {}).get("selected_faces") or [])

    settings = get_resume_settings_from_payload(payload)
    roop.config.globals.CFG.face_swap_model = settings["face_swap_model"]
    roop.config.globals.CFG.subsample_upscale = settings["upsample"]
    roop.config.globals.subsample_size = parse_face_swap_upscale_size(
        settings["upsample"],
        settings["face_swap_model"],
    )
    selected_preview_index = int((payload.get("targets") or {}).get("selected_preview_index", 0) or 0)
    selected_preview_index = max(0, min(selected_preview_index, len(target_paths) - 1))
    SELECTED_INPUT_FACE_INDEX = int((payload.get("selection") or {}).get("input_face_index", 0) or 0)
    SELECTED_TARGET_FACE_INDEX = int((payload.get("selection") or {}).get("target_face_index", 0) or 0)
    SELECTED_INPUT_FACE_INDEX = max(0, min(SELECTED_INPUT_FACE_INDEX, len(roop.config.globals.INPUT_FACESETS) - 1))
    if len(roop.config.globals.TARGET_FACES) > 0:
        SELECTED_TARGET_FACE_INDEX = max(0, min(SELECTED_TARGET_FACE_INDEX, len(roop.config.globals.TARGET_FACES) - 1))
    else:
        SELECTED_TARGET_FACE_INDEX = 0
    roop.config.globals.target_path = target_paths[selected_preview_index] if target_paths else None
    slider, frame_text = on_destfiles_selected(None)
    ui.globals.ui_resume_last_path = payload["__path__"]
    ui.globals.ui_resume_bound_path = payload["__path__"]
    sync_resume_processing_cache_id(payload["__path__"])
    summary = f"Loaded {len(roop.config.globals.INPUT_FACESETS)} source face(s), {len(target_paths)} target file(s), {len(roop.config.globals.TARGET_FACES)} selected target face(s)"
    return {
        "payload": payload,
        "settings": settings,
        "target_paths": target_paths,
        "slider": slider,
        "frame_text": frame_text,
        "resume_status": get_resume_status_markdown(payload["__path__"], summary),
    }


def extract_target_path(item):
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if hasattr(item, "name"):
        return item.name
    return str(item)


def normalize_target_path(raw_path: str, warn=True):
    if raw_path is None:
        return None
    clean_path = raw_path.strip().strip('"').strip("'")
    if len(clean_path) < 1:
        return None
    if not os.path.isabs(clean_path):
        clean_path = os.path.abspath(os.path.join(os.getcwd(), clean_path))
    clean_path = os.path.normpath(clean_path)
    if not os.path.exists(clean_path):
        if warn:
            gr.Warning(f"Target path not found: {clean_path}")
        return None
    if os.path.isdir(clean_path):
        if warn:
            gr.Warning(f"Directories are not supported yet: {clean_path}")
        return None
    if not (util.is_image(clean_path) or util.is_video(clean_path) or clean_path.lower().endswith('gif')):
        if warn:
            gr.Warning(f"Unsupported target file: {clean_path}")
        return None
    return clean_path


def resolve_resume_target_path(target_ref, path_key):
    target_path = normalize_target_path(target_ref.get(path_key), warn=False)
    if target_path is None:
        target_path = normalize_target_path(target_ref.get("resume_cached_path"), warn=False)
    if target_path is not None:
        return target_path
    original_path = target_ref.get(path_key)
    if original_path:
        normalize_target_path(original_path, warn=True)
    cached_path = target_ref.get("resume_cached_path")
    if cached_path:
        gr.Warning(f"Resume target snapshot not found: {os.path.normpath(cached_path)}")
    return None


def list_target_paths(files):
    if files is None:
        return []
    paths = []
    seen = set()
    for item in files:
        path = extract_target_path(item)
        norm_path = normalize_target_path(path)
        if norm_path is None or norm_path in seen:
            continue
        seen.add(norm_path)
        paths.append(norm_path)
    return paths


def merge_target_paths(existing_files, new_lines: str):
    merged = list_target_paths(existing_files)
    seen = set(merged)
    for line in (new_lines or "").splitlines():
        norm_path = normalize_target_path(line)
        if norm_path is None or norm_path in seen:
            continue
        seen.add(norm_path)
        merged.append(norm_path)
    ui.globals.ui_target_files = list(merged)
    return merged


def rebuild_process_entries(destfiles):
    global list_files_process

    list_files_process.clear()
    paths = list_target_paths(destfiles)
    ui.globals.ui_target_files = list(paths)
    for path in paths:
        list_files_process.append(
            ProcessEntry(
                path,
                0,
                0,
                0,
                file_signature=get_stable_file_signature(path),
                display_name=os.path.basename(path),
            )
        )
    return paths


def faceswap_tab():
    global no_face_choices, previewimage

    with gr.Tab("Face Swap"):
        with gr.Row(variant='panel'):
            bt_srcfiles = gr.Files(label='Source Images or Facesets', file_count="multiple", file_types=["image", ".fsz"], elem_id='filelist', height=233)
            bt_destfiles = gr.Files(label='Target File(s)', file_count="multiple", file_types=["image", "video"], elem_id='filelist', height=233)
        with gr.Row(variant='panel'):
            target_path_input = gr.Textbox(
                label="Target File(s) Path",
                lines=4,
                placeholder="One file path per line. Relative paths resolve from ./app",
            )
            bt_add_target_paths = gr.Button("Add Target Paths", variant='secondary')
        with gr.Row(variant='panel'):
            with gr.Column(scale=3):
                resume_file_input = gr.Textbox(
                    label="Resume File Path",
                    lines=1,
                    placeholder="Paste a JSON file from ./app/resume_cache and click Resume From File",
                )
            with gr.Column(scale=1):
                bt_resume_from_file = gr.Button("Resume From File", variant='secondary')
                bt_open_resume_cache = gr.Button("Open Resume Cache", size='sm')
        with gr.Row(variant='panel'):
            with gr.Column(scale=2):
                with gr.Row():
                    input_faces = gr.Gallery(label="Input faces gallery", allow_preview=False, preview=False, height=None, columns=2, object_fit="contain", interactive=False)
                    target_faces = gr.Gallery(label="Target faces gallery", allow_preview=False, preview=False, height=None, columns=2, object_fit="contain", interactive=False)
                with gr.Row():
                    bt_move_left_input = gr.Button("Move left", size='sm')
                    bt_move_right_input = gr.Button("Move right", size='sm')
                    bt_move_left_target = gr.Button("Move left", size='sm')
                    bt_move_right_target = gr.Button("Move right", size='sm')
                with gr.Row():
                    bt_remove_selected_input_face = gr.Button("Remove selected", size='sm')
                    bt_clear_input_faces = gr.Button("Clear all", variant='stop', size='sm')
                    bt_remove_selected_target_face = gr.Button("Remove selected", size='sm')

                with gr.Row():
                    with gr.Column():
                        chk_showmaskoffsets = gr.Checkbox(
                            label="Show mask overlay in preview",
                            value=roop.config.globals.CFG.show_mask_offsets,
                            interactive=True,
                        )
                        chk_restoreoriginalmouth = gr.Checkbox(
                            label="Restore original mouth area",
                            value=roop.config.globals.CFG.restore_original_mouth,
                            interactive=True,
                        )
                        mask_top = gr.Slider(
                            0, 2.0, value=roop.config.globals.CFG.mask_top,
                            label="Offset Face Top", step=0.01, interactive=True,
                        )
                        mask_bottom = gr.Slider(
                            0, 2.0, value=roop.config.globals.CFG.mask_bottom,
                            label="Offset Face Bottom", step=0.01, interactive=True,
                        )
                        mask_left = gr.Slider(
                            0, 2.0, value=roop.config.globals.CFG.mask_left,
                            label="Offset Face Left", step=0.01, interactive=True,
                        )
                        mask_right = gr.Slider(
                            0, 2.0, value=roop.config.globals.CFG.mask_right,
                            label="Offset Face Right", step=0.01, interactive=True,
                        )
                        face_mask_blend = gr.Slider(
                            0, 200, value=roop.config.globals.CFG.face_mask_blend,
                            label="Face Mask Edge Blend", step=1, interactive=True,
                        )
                    with gr.Column():
                        mouth_top_scale = gr.Slider(
                            0, 2.0, value=roop.config.globals.CFG.mouth_top_scale,
                            label="Mouth Mask Top", step=0.01, interactive=True,
                        )
                        mouth_bottom_scale = gr.Slider(
                            0, 2.0, value=roop.config.globals.CFG.mouth_bottom_scale,
                            label="Mouth Mask Bottom", step=0.01, interactive=True,
                        )
                        mouth_left_scale = gr.Slider(
                            0, 2.0, value=roop.config.globals.CFG.mouth_left_scale,
                            label="Mouth Mask Left", step=0.01, interactive=True,
                        )
                        mouth_right_scale = gr.Slider(
                            0, 2.0, value=roop.config.globals.CFG.mouth_right_scale,
                            label="Mouth Mask Right", step=0.01, interactive=True,
                        )
                        mouth_mask_blend = gr.Slider(
                            0, 200, value=roop.config.globals.CFG.mouth_mask_blend,
                            label="Mouth Mask Edge Blend", step=1, interactive=True,
                        )
                        bt_toggle_masking = gr.Button(
                            "Toggle manual masking", variant="secondary", size="sm"
                        )
                        selected_mask_engine = gr.Dropdown(
                            ["None", "Clip2Seg", "DFL XSeg"],
                            value=roop.config.globals.CFG.mask_engine,
                            label="Face masking engine",
                        )
                        clip_text = gr.Textbox(
                            label="List of objects to mask and restore back on fake face",
                            value=roop.config.globals.CFG.mask_clip_text,
                            interactive=roop.config.globals.CFG.mask_engine == "Clip2Seg",
                        )
                        bt_preview_mask = gr.Button(
                            "Show Mask Preview", variant="secondary"
                        )

            with gr.Column(scale=2):
                previewimage = gr.Image(label="Preview Image", height=576, interactive=False, visible=True, format=get_gradio_output_format())
                maskimage = gr.ImageEditor(label="Manual mask Image", sources=["clipboard"], transforms="", type="numpy",
                                             brush=gr.Brush(color_mode="fixed", colors=["rgba(255, 255, 255, 1"]), interactive=True, visible=False)
                with gr.Row(variant='panel'):
                    fake_preview = gr.Checkbox(label="Face swap frames", value=False)
                    bt_refresh_preview = gr.Button("Refresh", variant='secondary', size='sm')
                    bt_use_face_from_preview = gr.Button("Use Face from this Frame", variant='primary', size='sm')
                with gr.Row():
                    preview_frame_num = gr.Slider(1, 1, value=1, label="Frame Number", info='0:00:00', step=1.0, interactive=True)
                with gr.Row():
                    text_frame_clip = gr.Markdown('Processing frame range [0 - 0]')
                    set_frame_start = gr.Button("Set as Start", size='sm')
                    set_frame_end = gr.Button("Set as End", size='sm')
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                selected_face_detection = gr.Dropdown(swap_choices, value=roop.config.globals.CFG.face_detection_mode, label="Specify face selection for swapping")
            with gr.Column(scale=1):
                num_swap_steps = gr.Slider(1, 5, value=roop.config.globals.CFG.num_swap_steps, step=1.0, label="Number of swapping steps", info="More steps may increase likeness")
            with gr.Column(scale=2):
                ui.globals.ui_selected_enhancer = gr.Dropdown(["None", "Codeformer", "DMDNet", "GFPGAN", "GPEN", "Restoreformer++"], value=roop.config.globals.CFG.selected_enhancer, label="Select post-processing")

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                max_face_distance = gr.Slider(0.01, 1.0, value=roop.config.globals.CFG.max_face_distance, label="Max Face Similarity Threshold", info="0.0 = identical 1.0 = no similarity", elem_id='max_face_distance', interactive=True)
            with gr.Column(scale=1):
                ui.globals.ui_upscale = gr.Dropdown(
                    get_face_swap_upscale_choices(getattr(roop.config.globals.CFG, "face_swap_model", None)),
                    value=normalize_face_swap_upscale(
                        roop.config.globals.CFG.subsample_upscale,
                        getattr(roop.config.globals.CFG, "face_swap_model", None),
                    ),
                    label="Subsample upscale to",
                    info=get_face_swap_upscale_hint(getattr(roop.config.globals.CFG, "face_swap_model", None)),
                    interactive=True,
                )
            with gr.Column(scale=2):
                ui.globals.ui_blend_ratio = gr.Slider(0.0, 1.0, value=roop.config.globals.CFG.blend_ratio, label="Original/Enhanced image blend ratio", info="Only used with active post-processing")

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                video_swapping_method = gr.Dropdown(["Smart staged processing", "Legacy extract frames"], value=roop.config.globals.CFG.video_swapping_method, label="Select video processing method", interactive=True)
                no_face_action = gr.Dropdown(choices=no_face_choices, value=roop.config.globals.CFG.no_face_action, label="Action on no face detected", interactive=True)
                vr_mode = gr.Checkbox(label="VR Mode", value=roop.config.globals.CFG.vr_mode)
            with gr.Column(scale=1):
                with gr.Group():
                    autorotate = gr.Checkbox(label="Auto rotate horizontal Faces", value=roop.config.globals.CFG.autorotate_faces)
                    roop.config.globals.skip_audio = gr.Checkbox(label="Skip audio", value=roop.config.globals.CFG.skip_audio)
                    roop.config.globals.keep_frames = gr.Checkbox(label="Keep Frames (relevant only when extracting frames)", value=roop.config.globals.CFG.keep_frames)
                    roop.config.globals.wait_after_extraction = gr.Checkbox(label="Wait for user key press before creating video ", value=roop.config.globals.CFG.wait_after_extraction)

        with gr.Row(variant='panel'):
            with gr.Column():
                bt_start = gr.Button("Start", variant='primary')
            with gr.Column():
                bt_stop = gr.Button("Stop", variant='secondary', interactive=False)
                gr.Button("Open Output Folder", size='sm').click(fn=lambda: util.open_folder(roop.config.globals.output_path))
            with gr.Column(scale=2):
                output_method = gr.Dropdown(["File","Virtual Camera", "Both"], value=roop.config.globals.CFG.output_method, label="Select Output Method", interactive=True)
        with gr.Row(variant='panel'):
            processing_info = gr.Markdown(get_processing_status_markdown())
            processing_timer = gr.Timer(1.0, active=True)
        with gr.Row(variant='panel'):
            resume_status = gr.Markdown(get_resume_status_markdown())

    # Store saveable component refs in ui.globals for cross-tab access (Save/Load session)
    ui.globals.ui_selected_face_detection = selected_face_detection
    ui.globals.ui_num_swap_steps = num_swap_steps
    ui.globals.ui_max_face_distance = max_face_distance
    ui.globals.ui_video_swapping_method = video_swapping_method
    ui.globals.ui_no_face_action = no_face_action
    ui.globals.ui_vr_mode = vr_mode
    ui.globals.ui_autorotate = autorotate
    ui.globals.ui_skip_audio = roop.config.globals.skip_audio
    ui.globals.ui_keep_frames = roop.config.globals.keep_frames
    ui.globals.ui_wait_after_extraction = roop.config.globals.wait_after_extraction
    ui.globals.ui_output_method = output_method
    ui.globals.ui_selected_mask_engine = selected_mask_engine
    ui.globals.ui_clip_text = clip_text
    ui.globals.ui_chk_showmaskoffsets = chk_showmaskoffsets
    ui.globals.ui_chk_restoreoriginalmouth = chk_restoreoriginalmouth
    ui.globals.ui_mask_top = mask_top
    ui.globals.ui_mask_bottom = mask_bottom
    ui.globals.ui_mask_left = mask_left
    ui.globals.ui_mask_right = mask_right
    ui.globals.ui_face_mask_blend = face_mask_blend
    ui.globals.ui_mouth_mask_blend = mouth_mask_blend
    ui.globals.ui_mouth_top_scale = mouth_top_scale
    ui.globals.ui_mouth_bottom_scale = mouth_bottom_scale
    ui.globals.ui_mouth_left_scale = mouth_left_scale
    ui.globals.ui_mouth_right_scale = mouth_right_scale
    ui.globals.ui_processing_info = processing_info
    ui.globals.ui_resume_status = resume_status

    previewinputs = [preview_frame_num, bt_destfiles, fake_preview, ui.globals.ui_selected_enhancer, selected_face_detection,
                        max_face_distance, ui.globals.ui_blend_ratio, selected_mask_engine, clip_text, no_face_action, vr_mode, autorotate, maskimage, chk_showmaskoffsets, chk_restoreoriginalmouth, num_swap_steps, ui.globals.ui_upscale]
    previewoutputs = [previewimage, maskimage, preview_frame_num] 
    input_faces.select(on_select_input_face, None, None).success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)
    
    bt_move_left_input.click(fn=move_selected_input, inputs=[bt_move_left_input], outputs=[input_faces])
    bt_move_right_input.click(fn=move_selected_input, inputs=[bt_move_right_input], outputs=[input_faces])
    bt_move_left_target.click(fn=move_selected_target, inputs=[bt_move_left_target], outputs=[target_faces])
    bt_move_right_target.click(fn=move_selected_target, inputs=[bt_move_right_target], outputs=[target_faces])

    bt_remove_selected_input_face.click(fn=remove_selected_input_face, outputs=[input_faces])
    bt_srcfiles.upload(fn=on_srcfile_changed, show_progress='full', inputs=bt_srcfiles, outputs=[input_faces, bt_srcfiles])

    mask_top.release(fn=on_mask_top_changed, inputs=[mask_top], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mask_bottom.release(fn=on_mask_bottom_changed, inputs=[mask_bottom], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mask_left.release(fn=on_mask_left_changed, inputs=[mask_left], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mask_right.release(fn=on_mask_right_changed, inputs=[mask_right], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    face_mask_blend.release(fn=on_face_mask_blend_changed, inputs=[face_mask_blend], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mouth_mask_blend.release(fn=on_mouth_mask_blend_changed, inputs=[mouth_mask_blend], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mouth_top_scale.release(fn=on_mouth_top_scale_changed, inputs=[mouth_top_scale], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mouth_bottom_scale.release(fn=on_mouth_bottom_scale_changed, inputs=[mouth_bottom_scale], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mouth_left_scale.release(fn=on_mouth_left_scale_changed, inputs=[mouth_left_scale], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mouth_right_scale.release(fn=on_mouth_right_scale_changed, inputs=[mouth_right_scale], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    chk_showmaskoffsets.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    chk_restoreoriginalmouth.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    selected_mask_engine.change(fn=on_mask_engine_changed, inputs=[selected_mask_engine], outputs=[clip_text], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')

    target_faces.select(on_select_target_face, None, None)
    bt_remove_selected_target_face.click(fn=remove_selected_target_face, outputs=[target_faces])

    bt_destfiles.change(fn=on_destfiles_changed, inputs=[bt_destfiles], outputs=[preview_frame_num, text_frame_clip], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    bt_destfiles.select(fn=on_destfiles_selected, outputs=[preview_frame_num, text_frame_clip], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    bt_add_target_paths.click(
        fn=on_add_target_paths,
        inputs=[target_path_input, bt_destfiles],
        outputs=[bt_destfiles, target_path_input, preview_frame_num, text_frame_clip],
        show_progress='hidden'
    ).success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    bt_destfiles.clear(fn=on_clear_destfiles, outputs=[target_faces, preview_frame_num, text_frame_clip, target_path_input])
    bt_clear_input_faces.click(fn=on_clear_input_faces, outputs=[input_faces])

    bt_preview_mask.click(fn=on_preview_mask, inputs=[preview_frame_num, bt_destfiles, clip_text, selected_mask_engine], outputs=[previewimage]) 

    start_event = bt_start.click(fn=start_swap,
        inputs=[output_method, ui.globals.ui_selected_enhancer, selected_face_detection, roop.config.globals.keep_frames, roop.config.globals.wait_after_extraction,
                    roop.config.globals.skip_audio, max_face_distance, ui.globals.ui_blend_ratio, selected_mask_engine, clip_text,video_swapping_method, no_face_action, vr_mode, autorotate, chk_restoreoriginalmouth, num_swap_steps, ui.globals.ui_upscale, maskimage],
        outputs=[bt_start, bt_stop, processing_info, resume_status], show_progress='full')

    resume_event = bt_resume_from_file.click(
        fn=resume_swap_from_file,
        inputs=[resume_file_input],
        outputs=[bt_start, bt_stop, processing_info, resume_status],
        show_progress='full'
    )
    bt_open_resume_cache.click(fn=lambda: util.open_folder(get_resume_cache_root()), outputs=[], queue=False)

    bt_stop.click(fn=stop_swap, cancels=[start_event, resume_event], outputs=[bt_start, bt_stop, processing_info, resume_status], queue=False)
    processing_timer.tick(fn=get_runtime_processing_info, outputs=[processing_info], show_progress='hidden', queue=False)

    bt_refresh_preview.click(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)            
    bt_toggle_masking.click(fn=on_toggle_masking, inputs=[previewimage, maskimage], outputs=[previewimage, maskimage])            
    fake_preview.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)
    ui.globals.ui_upscale.change(
        fn=normalize_upscale_for_current_model,
        inputs=[ui.globals.ui_upscale],
        outputs=[ui.globals.ui_upscale],
        show_progress='hidden',
    ).success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    preview_frame_num.release(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden', )
    bt_use_face_from_preview.click(fn=on_use_face_from_selected, show_progress='full', inputs=[bt_destfiles, preview_frame_num], outputs=[target_faces, selected_face_detection])
    set_frame_start.click(fn=on_set_frame, inputs=[set_frame_start, preview_frame_num], outputs=[text_frame_clip])
    set_frame_end.click(fn=on_set_frame, inputs=[set_frame_end, preview_frame_num], outputs=[text_frame_clip])

    return bt_destfiles


def on_mask_top_changed(mask_offset):
    set_mask_offset(0, mask_offset)

def on_mask_bottom_changed(mask_offset):
    set_mask_offset(1, mask_offset)

def on_mask_left_changed(mask_offset):
    set_mask_offset(2, mask_offset)

def on_mask_right_changed(mask_offset):
    set_mask_offset(3, mask_offset)

def on_face_mask_blend_changed(value):
    set_mask_offset(4, value)

def on_mouth_mask_blend_changed(value):
    set_mask_offset(5, value)

def on_mouth_top_scale_changed(value):
    set_mask_offset(6, value)

def on_mouth_bottom_scale_changed(value):
    set_mask_offset(7, value)

def on_mouth_left_scale_changed(value):
    set_mask_offset(8, value)

def on_mouth_right_scale_changed(value):
    set_mask_offset(9, value)


def set_mask_offset(index, mask_offset):
    global SELECTED_INPUT_FACE_INDEX

    if len(roop.config.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        offs = roop.config.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets
        while len(offs) < 10:
            offs.append(1.0)
        offs[index] = mask_offset
        roop.config.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets = offs
        if len(ui.globals.ui_input_face_refs) > SELECTED_INPUT_FACE_INDEX:
            ui.globals.ui_input_face_refs[SELECTED_INPUT_FACE_INDEX]["mask_offsets"] = list(offs)

def on_mask_engine_changed(mask_engine):
    if mask_engine == "Clip2Seg":
        return gr.Textbox(interactive=True)
    return gr.Textbox(interactive=False)


def on_add_target_paths(path_text, current_files):
    merged = merge_target_paths(current_files, path_text)
    slider, frame_text = on_destfiles_changed(merged)
    return merged, "", slider, frame_text



def on_srcfile_changed(srcfiles, progress=gr.Progress()):
    global input_faces, last_image

    if srcfiles is None or len(srcfiles) < 1:
        return ui.globals.ui_input_thumbs, None

    for f in srcfiles:    
        source_path = normalize_source_path(f.name)
        if source_path is None:
            continue
        if source_path.lower().endswith('fsz'):
            progress(0, desc="Retrieving faces from Faceset File")      
            append_faceset_source(source_path, {
                "type": "faceset",
                "path": source_path,
                "mask_offsets": list(default_mask_offsets()),
            })
                                        
        elif util.has_image_extension(source_path):
            progress(0, desc="Retrieving faces from image")      
            roop.config.globals.source_path = source_path
            selection_faces_data = extract_face_images_for_swap(roop.config.globals.source_path,  (False, 0))
            progress(0.5, desc="Retrieving faces from image")
            for face_index, face_data in enumerate(selection_faces_data):
                face_set = FaceSet()
                face = face_data[0]
                face.mask_offsets = list(default_mask_offsets())
                face_set.faces.append(face)
                image = util.convert_to_gradio(face_data[1])
                ui.globals.ui_input_thumbs.append(image)
                roop.config.globals.INPUT_FACESETS.append(face_set)
                ui.globals.ui_input_face_refs.append({
                    "type": "image_face",
                    "path": source_path,
                    "face_index": face_index,
                    "mask_offsets": list(default_mask_offsets()),
                })
                
    progress(1.0)
    if len(ui.globals.ui_input_thumbs) >= 6:
        gr.Warning(
            "You have more than 6 input faces. Consider using the Face Management tab "
            "to consolidate multiple images of the same source into a single faceset file."
        )
    return ui.globals.ui_input_thumbs, None


def ensure_loaded_face_embeddings():
    global SELECTED_INPUT_FACE_INDEX, SELECTED_TARGET_FACE_INDEX

    reload_inputs = any(
        getattr(face_set, "faces", None) and get_face_embedding_vector(face_set.faces[0]) is None
        for face_set in roop.config.globals.INPUT_FACESETS
    )
    reload_targets = any(
        get_face_embedding_vector(face) is None
        for face in roop.config.globals.TARGET_FACES
    )
    if not reload_inputs and not reload_targets:
        return False

    input_refs = [dict(face_ref) for face_ref in ui.globals.ui_input_face_refs]
    target_refs = [dict(face_ref) for face_ref in ui.globals.ui_target_face_refs]
    selected_input_index = SELECTED_INPUT_FACE_INDEX
    selected_target_index = SELECTED_TARGET_FACE_INDEX

    if reload_inputs and input_refs:
        restore_input_faces_from_resume(input_refs)
        if roop.config.globals.INPUT_FACESETS:
            SELECTED_INPUT_FACE_INDEX = max(0, min(selected_input_index, len(roop.config.globals.INPUT_FACESETS) - 1))

    if reload_targets and target_refs:
        clear_target_face_state()
        for face_ref in target_refs:
            append_target_face_from_resume(face_ref)
        if roop.config.globals.TARGET_FACES:
            SELECTED_TARGET_FACE_INDEX = max(0, min(selected_target_index, len(roop.config.globals.TARGET_FACES) - 1))

    return True


def on_select_input_face(evt: gr.SelectData):
    global SELECTED_INPUT_FACE_INDEX

    SELECTED_INPUT_FACE_INDEX = evt.index


def remove_selected_input_face():
    global SELECTED_INPUT_FACE_INDEX

    if len(roop.config.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        f = roop.config.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
        del f
    if len(ui.globals.ui_input_face_refs) > SELECTED_INPUT_FACE_INDEX:
        f = ui.globals.ui_input_face_refs.pop(SELECTED_INPUT_FACE_INDEX)
        del f
    if len(ui.globals.ui_input_thumbs) > SELECTED_INPUT_FACE_INDEX:
        f = ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
        del f

    return ui.globals.ui_input_thumbs

def move_selected_input(button_text):
    global SELECTED_INPUT_FACE_INDEX

    if button_text == "Move left":
        if SELECTED_INPUT_FACE_INDEX <= 0:
            return ui.globals.ui_input_thumbs
        offset = -1
    else:
        if len(ui.globals.ui_input_thumbs) <= SELECTED_INPUT_FACE_INDEX:
            return ui.globals.ui_input_thumbs
        offset = 1
    
    f = roop.config.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
    roop.config.globals.INPUT_FACESETS.insert(SELECTED_INPUT_FACE_INDEX + offset, f)
    if len(ui.globals.ui_input_face_refs) > SELECTED_INPUT_FACE_INDEX:
        f = ui.globals.ui_input_face_refs.pop(SELECTED_INPUT_FACE_INDEX)
        ui.globals.ui_input_face_refs.insert(SELECTED_INPUT_FACE_INDEX + offset, f)
    f = ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
    ui.globals.ui_input_thumbs.insert(SELECTED_INPUT_FACE_INDEX + offset, f)
    return ui.globals.ui_input_thumbs
        

def move_selected_target(button_text):
    global SELECTED_TARGET_FACE_INDEX

    if button_text == "Move left":
        if SELECTED_TARGET_FACE_INDEX <= 0:
            return ui.globals.ui_target_thumbs
        offset = -1
    else:
        if len(ui.globals.ui_target_thumbs) <= SELECTED_TARGET_FACE_INDEX:
            return ui.globals.ui_target_thumbs
        offset = 1
    
    f = roop.config.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
    roop.config.globals.TARGET_FACES.insert(SELECTED_TARGET_FACE_INDEX + offset, f)
    if len(ui.globals.ui_target_face_refs) > SELECTED_TARGET_FACE_INDEX:
        f = ui.globals.ui_target_face_refs.pop(SELECTED_TARGET_FACE_INDEX)
        ui.globals.ui_target_face_refs.insert(SELECTED_TARGET_FACE_INDEX + offset, f)
    f = ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
    ui.globals.ui_target_thumbs.insert(SELECTED_TARGET_FACE_INDEX + offset, f)
    return ui.globals.ui_target_thumbs




def on_select_target_face(evt: gr.SelectData):
    global SELECTED_TARGET_FACE_INDEX

    SELECTED_TARGET_FACE_INDEX = evt.index

def remove_selected_target_face():
    if len(ui.globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = roop.config.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    if len(ui.globals.ui_target_face_refs) > SELECTED_TARGET_FACE_INDEX:
        f = ui.globals.ui_target_face_refs.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    if len(ui.globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    return ui.globals.ui_target_thumbs


def on_use_face_from_selected(files, frame_num):
    target_paths = list_target_paths(files)
    if selected_preview_index >= len(target_paths):
        return ui.globals.ui_target_thumbs, gr.Dropdown(visible=True)
    roop.config.globals.target_path = target_paths[selected_preview_index]
    faces_data = []

    if util.is_image(roop.config.globals.target_path) and not roop.config.globals.target_path.lower().endswith(('gif')):
        faces_data = extract_face_images_for_swap(roop.config.globals.target_path, (False, 0))
    elif util.is_video(roop.config.globals.target_path) or roop.config.globals.target_path.lower().endswith(('gif')):
        faces_data = extract_face_images_for_swap(roop.config.globals.target_path, (True, frame_num))
    else:
        gr.Info('Unknown image/video type!')
        roop.config.globals.target_path = None
        return ui.globals.ui_target_thumbs, gr.Dropdown(visible=True)

    if len(faces_data) == 0:
        gr.Info('No faces detected!')
        roop.config.globals.target_path = None
        return ui.globals.ui_target_thumbs, gr.Dropdown(visible=True)

    for face_index, face_data in enumerate(faces_data):
        roop.config.globals.TARGET_FACES.append(face_data[0])
        ui.globals.ui_target_thumbs.append(util.convert_to_gradio(face_data[1]))
        ui.globals.ui_target_face_refs.append({
            "path": roop.config.globals.target_path,
            "frame_number": int(frame_num) if (util.is_video(roop.config.globals.target_path) or roop.config.globals.target_path.lower().endswith(('gif'))) else 0,
            "face_index": face_index,
        })

    return ui.globals.ui_target_thumbs, gr.Dropdown(value='Selected face')


def on_preview_frame_changed(frame_num, files, fake_preview, enhancer, detection, face_distance, blend_ratio,
                              selected_mask_engine, clip_text, no_face_action, vr_mode, auto_rotate, maskimage, show_face_area, restore_original_mouth, num_steps, upsample):
    global SELECTED_INPUT_FACE_INDEX, manual_masking, current_video_fps

    from roop.core.app import live_swap, get_processing_plugins

    manual_masking = False
    mask_offsets = [0,0,0,0,20.0,10.0,1.0,1.0,1.0,1.0]
    if len(roop.config.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        if not hasattr(roop.config.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0], 'mask_offsets'):
            roop.config.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets = list(mask_offsets)
        mask_offsets = roop.config.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets
        while len(mask_offsets) < 10:
            mask_offsets.append(1.0)

    timeinfo = '0:00:00'
    target_paths = list_target_paths(files)
    if len(target_paths) < 1 or selected_preview_index >= len(target_paths) or frame_num is None:
        return None,None, gr.Slider(info=timeinfo)

    filename = target_paths[selected_preview_index]
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num)
        if current_video_fps == 0:
            current_video_fps = 1
        secs = (frame_num - 1) / current_video_fps
        minutes = secs / 60
        secs = secs % 60
        hours = minutes / 60
        minutes = minutes % 60
        milliseconds = (secs - int(secs)) * 1000
        timeinfo = f"{int(hours):0>2}:{int(minutes):0>2}:{int(secs):0>2}.{int(milliseconds):0>3}"  
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return None, None, gr.Slider(info=timeinfo)

    if not fake_preview or len(roop.config.globals.INPUT_FACESETS) < 1:
        return gr.Image(value=util.convert_to_gradio(current_frame), visible=True), gr.ImageEditor(visible=False), gr.Slider(info=timeinfo)

    roop.config.globals.face_swap_mode = translate_swap_mode(detection)
    roop.config.globals.selected_enhancer = enhancer
    roop.config.globals.distance_threshold = face_distance
    roop.config.globals.blend_ratio = blend_ratio
    roop.config.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.config.globals.vr_mode = vr_mode
    roop.config.globals.autorotate_faces = auto_rotate
    selected_face_swap_model = get_face_swap_model_key(getattr(roop.config.globals.CFG, "face_swap_model", None))
    normalized_upscale = normalize_face_swap_upscale(upsample, selected_face_swap_model)
    roop.config.globals.CFG.subsample_upscale = normalized_upscale
    roop.config.globals.subsample_size = parse_face_swap_upscale_size(
        normalized_upscale,
        selected_face_swap_model,
    )


    mask_engine = map_mask_engine(selected_mask_engine, clip_text)

    roop.config.globals.execution_threads = roop.config.globals.CFG.max_threads
    ensure_loaded_face_embeddings()
    face_index = SELECTED_INPUT_FACE_INDEX
    if len(roop.config.globals.INPUT_FACESETS) <= face_index:
        face_index = 0
   
    options = ProcessOptions(get_processing_plugins(mask_engine), roop.config.globals.distance_threshold, roop.config.globals.blend_ratio,
                              roop.config.globals.face_swap_mode, face_index, clip_text, maskimage, num_steps, roop.config.globals.subsample_size, show_face_area, restore_original_mouth,
                              face_swap_model=selected_face_swap_model)

    current_frame = live_swap(current_frame, options)
    if current_frame is None:
        return gr.Image(visible=True), None, gr.Slider(info=timeinfo)
    return gr.Image(value=util.convert_to_gradio(current_frame), visible=True), gr.ImageEditor(visible=False), gr.Slider(info=timeinfo)

def map_mask_engine(selected_mask_engine, clip_text):
    if selected_mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
          mask_engine = None
    elif selected_mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    else:
        mask_engine = None
    return mask_engine


def on_toggle_masking(previewimage, mask):
    global manual_masking

    manual_masking = not manual_masking
    if manual_masking:
        layers = mask["layers"]
        if len(layers) == 1:
            layers = [create_blank_image(previewimage.shape[1],previewimage.shape[0])]
        return gr.Image(visible=False), gr.ImageEditor(value={"background": previewimage, "layers": layers, "composite": None}, visible=True)
    return gr.Image(visible=True), gr.ImageEditor(visible=False)

def gen_processing_text(start, end):
    return f'Processing frame range [{start} - {end}]'


def get_runtime_processing_info():
    return get_processing_status_markdown()

def on_set_frame(sender:str, frame_num):
    global selected_preview_index, list_files_process
    
    if len(list_files_process) < 1:
        return gen_processing_text(0, 0)
    idx = selected_preview_index
    if list_files_process[idx].endframe == 0:
        return gen_processing_text(0,0)
    
    start = list_files_process[idx].startframe
    end = list_files_process[idx].endframe
    if sender.lower().endswith('start'):
        list_files_process[idx].startframe = min(frame_num, end)
    else:
        list_files_process[idx].endframe = max(frame_num, start)
    
    return gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe)


def on_preview_mask(frame_num, files, clip_text, mask_engine):
    from roop.core.app import live_swap, get_processing_plugins
    global is_processing

    target_paths = list_target_paths(files)
    if is_processing or len(target_paths) < 1 or selected_preview_index >= len(target_paths) or clip_text is None or frame_num is None:
        return None
        
    filename = target_paths[selected_preview_index]
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num
                                        )
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None or mask_engine is None:
        return None
    ensure_loaded_face_embeddings()
    if mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
          mask_engine = None
    elif mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    options = ProcessOptions(get_processing_plugins(mask_engine), roop.config.globals.distance_threshold, roop.config.globals.blend_ratio,
                              "all", 0, clip_text, None, 0, 128, False, False, True)

    current_frame = live_swap(current_frame, options)
    return util.convert_to_gradio(current_frame)


def on_clear_input_faces():
    clear_input_face_state()
    return ui.globals.ui_input_thumbs

def on_clear_destfiles():
    global selected_preview_index
    clear_target_face_state()
    ui.globals.ui_target_files = []
    ui.globals.ui_resume_bound_path = None
    roop.config.globals.active_resume_cache_id = None
    list_files_process.clear()
    selected_preview_index = 0
    return ui.globals.ui_target_thumbs, gr.Slider(value=1, maximum=1, info='0:00:00'), '', ''


def index_of_no_face_action(dropdown_text):
    global no_face_choices

    return no_face_choices.index(dropdown_text) 

def translate_swap_mode(dropdown_text):
    if dropdown_text == "Selected face":
        return "selected"
    elif dropdown_text == "First found":
        return "first"
    elif dropdown_text == "All input faces":
        return "all_input"
    elif dropdown_text == "All female":
        return "all_female"
    elif dropdown_text == "All male":
        return "all_male"
    
    return "all"


def create_resume_settings_snapshot(output_method, enhancer, detection, keep_frames, wait_after_extraction, skip_audio, face_distance, blend_ratio,
                                    selected_mask_engine, clip_text, processing_method, no_face_action, vr_mode, autorotate,
                                    restore_original_mouth, num_swap_steps, upsample):
    face_swap_model = get_face_swap_model_key(getattr(roop.config.globals.CFG, "face_swap_model", None))
    normalized_upscale = normalize_face_swap_upscale(upsample, face_swap_model)
    return {
        "output_method": output_method,
        "enhancer": enhancer,
        "face_swap_model": face_swap_model,
        "detection": detection,
        "keep_frames": bool(keep_frames),
        "wait_after_extraction": bool(wait_after_extraction),
        "skip_audio": bool(skip_audio),
        "face_distance": float(face_distance),
        "blend_ratio": float(blend_ratio),
        "selected_mask_engine": selected_mask_engine,
        "clip_text": clip_text,
        "processing_method": processing_method,
        "no_face_action": no_face_action,
        "vr_mode": bool(vr_mode),
        "autorotate": bool(autorotate),
        "restore_original_mouth": bool(restore_original_mouth),
        "num_swap_steps": int(num_swap_steps),
        "upsample": normalized_upscale,
    }


def save_resume_snapshot_for_run(output_method, enhancer, detection, keep_frames, wait_after_extraction, skip_audio, face_distance, blend_ratio,
                                 selected_mask_engine, clip_text, processing_method, no_face_action, vr_mode, autorotate,
                                 restore_original_mouth, num_swap_steps, upsample):
    settings = create_resume_settings_snapshot(
        output_method, enhancer, detection, keep_frames, wait_after_extraction, skip_audio, face_distance, blend_ratio,
        selected_mask_engine, clip_text, processing_method, no_face_action, vr_mode, autorotate,
        restore_original_mouth, num_swap_steps, upsample
    )
    payload = build_resume_payload(settings)
    resume_job_key = get_resume_job_signature(payload)
    resume_path, reused_existing, resume_key = write_resume_payload_with_result(payload)
    try:
        verified_payload = read_resume_payload(resume_path)
        roop.config.globals.active_resume_key = get_resume_payload_signature(verified_payload)
        roop.config.globals.active_resume_job_key = verified_payload.get("resume_job_key") or get_resume_job_signature(verified_payload)
    except Exception:
        roop.config.globals.active_resume_key = resume_key
        roop.config.globals.active_resume_job_key = resume_job_key
    status_detail = "Reused existing resume config for this run" if reused_existing else "Saved resume config for this run"
    return resume_path, get_resume_status_markdown(resume_path, status_detail), reused_existing


def start_swap( output_method, enhancer, detection, keep_frames, wait_after_extraction, skip_audio, face_distance, blend_ratio,
                selected_mask_engine, clip_text, processing_method, no_face_action, vr_mode, autorotate, restore_original_mouth, num_swap_steps, upsample, imagemask, progress=gr.Progress()):
    from ui.main import prepare_environment
    from roop.core.app import batch_process_regular
    from roop.memory.planner import describe_memory_plan, resolve_memory_plan
    global is_processing, list_files_process

    resume_status = get_resume_status_markdown(ui.globals.ui_resume_last_path)
    if list_files_process is None or len(list_files_process) <= 0:
        yield gr.Button(variant="primary", interactive=True), gr.Button(variant="secondary", interactive=False), get_processing_status_markdown(), resume_status
        return
    
    if roop.config.globals.CFG.clear_output:
        shutil.rmtree(roop.config.globals.output_path, ignore_errors=True)

    if not util.is_installed("ffmpeg"):
        msg = "ffmpeg is not installed! No video processing possible."
        gr.Warning(msg)

    prepare_environment()

    roop.config.globals.selected_enhancer = enhancer
    roop.config.globals.target_path = None
    roop.config.globals.distance_threshold = face_distance
    roop.config.globals.blend_ratio = blend_ratio
    roop.config.globals.keep_frames = keep_frames
    roop.config.globals.wait_after_extraction = wait_after_extraction
    roop.config.globals.skip_audio = skip_audio
    roop.config.globals.face_swap_mode = translate_swap_mode(detection)
    roop.config.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.config.globals.vr_mode = vr_mode
    roop.config.globals.autorotate_faces = autorotate
    selected_face_swap_model = get_face_swap_model_key(getattr(roop.config.globals.CFG, "face_swap_model", None))
    normalized_upscale = normalize_face_swap_upscale(upsample, selected_face_swap_model)
    roop.config.globals.CFG.subsample_upscale = normalized_upscale
    roop.config.globals.subsample_size = parse_face_swap_upscale_size(
        normalized_upscale,
        selected_face_swap_model,
    )
    mask_engine = map_mask_engine(selected_mask_engine, clip_text)

    if roop.config.globals.face_swap_mode == 'selected':
        if len(roop.config.globals.TARGET_FACES) < 1:
            gr.Error('No Target Face selected!')
            yield gr.Button(variant="primary", interactive=True), gr.Button(variant="secondary", interactive=False), get_processing_status_markdown(), resume_status
            return

    ensure_loaded_face_embeddings()

    is_processing = True
    start_processing_status("Preparing faceswap job", total_files=len(list_files_process))
    try:
        resume_path, resume_status, reused_resume = save_resume_snapshot_for_run(
            output_method, enhancer, detection, keep_frames, wait_after_extraction, skip_audio, face_distance, blend_ratio,
            selected_mask_engine, clip_text, processing_method, no_face_action, vr_mode, autorotate,
            restore_original_mouth, num_swap_steps, upsample
        )
        if reused_resume:
            gr.Info(f"Resume config reused: {resume_path}")
        else:
            gr.Info(f"Resume config saved: {resume_path}")
    except Exception as exc:
        roop.config.globals.active_resume_key = None
        roop.config.globals.active_resume_job_key = None
        roop.config.globals.active_resume_cache_id = None
        resume_status = get_resume_status_markdown(ui.globals.ui_resume_last_path, f"Resume config was not saved: {exc}")
        gr.Warning(f"Resume config was not saved for this run: {exc}")
    yield gr.Button(variant="secondary", interactive=False), gr.Button(variant="primary", interactive=True), get_processing_status_markdown(), resume_status
    roop.config.globals.execution_threads = roop.config.globals.CFG.max_threads
    roop.config.globals.video_encoder = roop.config.globals.CFG.output_video_codec
    roop.config.globals.video_quality = roop.config.globals.CFG.video_quality
    roop.config.globals.max_memory = None
    roop.config.globals.max_vram = None
    set_memory_status(describe_memory_plan(resolve_memory_plan()))
    set_processing_message("Preparing faceswap job", stage="prepare", detail="Resolving staged resource tuning and worker plan", force_log=True)
    gr.Info(roop.config.globals.runtime_memory_status)

    batch_process_regular(output_method, list_files_process, mask_engine, clip_text, processing_method, imagemask, restore_original_mouth, num_swap_steps, progress, SELECTED_INPUT_FACE_INDEX)
    is_processing = False
    yield gr.Button(variant="primary", interactive=True), gr.Button(variant="secondary", interactive=False), get_processing_status_markdown(), resume_status


def resume_swap_from_file(resume_path, progress=gr.Progress()):
    try:
        resume_state = load_resume_into_runtime(resume_path)
        settings = resume_state["settings"]
        gr.Info(f"Loaded resume config: {resume_state['payload']['__path__']}")
    except Exception as exc:
        status = get_resume_status_markdown(normalize_resume_path(resume_path), str(exc))
        gr.Warning(str(exc))
        yield gr.Button(variant="primary", interactive=True), gr.Button(variant="secondary", interactive=False), get_processing_status_markdown(), status
        return

    for update in start_swap(
        settings["output_method"],
        settings["enhancer"],
        settings["detection"],
        settings["keep_frames"],
        settings["wait_after_extraction"],
        settings["skip_audio"],
        settings["face_distance"],
        settings["blend_ratio"],
        settings["selected_mask_engine"],
        settings["clip_text"],
        settings["processing_method"],
        settings["no_face_action"],
        settings["vr_mode"],
        settings["autorotate"],
        settings["restore_original_mouth"],
        settings["num_swap_steps"],
        settings["upsample"],
        None,
        progress,
    ):
        yield update


def stop_swap():
    roop.config.globals.processing = False
    set_processing_message('Stopping... waiting for workers to finish', status='stopping', detail='The current stage will stop as soon as workers drain', force_log=True)
    gr.Info('Aborting processing - please wait for the remaining threads to be stopped')
    return gr.Button(variant="primary", interactive=True), gr.Button(variant="secondary", interactive=False), get_processing_status_markdown(), get_resume_status_markdown(ui.globals.ui_resume_last_path)


def on_destfiles_changed(destfiles):
    global selected_preview_index, list_files_process, current_video_fps

    target_paths = rebuild_process_entries(destfiles)
    if len(target_paths) < 1:
        return gr.Slider(value=1, maximum=1, info='0:00:00'), ''

    selected_preview_index = 0
    idx = selected_preview_index    
    
    filename = list_files_process[idx].filename
    
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
        if total_frames is None or total_frames < 1:
            total_frames = 1
            gr.Warning(f"Corrupted video {filename}, can't detect number of frames!")
        else:
            current_video_fps = util.detect_fps(filename)
    else:
        total_frames = 1
    list_files_process[idx].endframe = total_frames
    if total_frames > 1:
        return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe)
    return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), ''


def on_destfiles_selected(evt: gr.SelectData):
    global selected_preview_index, list_files_process, current_video_fps

    if len(list_files_process) < 1:
        return gr.Slider(value=1, maximum=1, info='0:00:00'), gen_processing_text(0, 0)
    if evt is not None:
        selected_preview_index = evt.index
    selected_preview_index = min(selected_preview_index, len(list_files_process) - 1)
    idx = selected_preview_index
    filename = list_files_process[idx].filename
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
        current_video_fps = util.detect_fps(filename)
        if list_files_process[idx].endframe == 0:
            list_files_process[idx].endframe = total_frames
    else:
        total_frames = 1

    if total_frames > 1:
        return gr.Slider(value=list_files_process[idx].startframe, maximum=total_frames, info='0:00:00'), gen_processing_text(list_files_process[idx].startframe, list_files_process[idx].endframe)
    return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), gen_processing_text(0, 0)


def get_gradio_output_format():
    if roop.config.globals.CFG.output_image_format == "jpg":
        return "jpeg"
    return roop.config.globals.CFG.output_image_format

