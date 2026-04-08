import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import roop.config.globals
import roop.utils as util
from roop.config.settings import Settings
from roop.core.app import get_processing_plugins
from roop.core.providers import decode_execution_providers
from roop.memory import describe_memory_plan, resolve_memory_plan
from roop.pipeline.entry import ProcessEntry
from roop.pipeline.options import ProcessOptions
from roop.pipeline.staged_executor.cache import (
    get_stage_pack_path,
    read_cache_blob,
    read_stage_cache_keys,
    sanitize_job_path_segment,
    write_cache_blob,
    write_stage_cache_map,
)
from roop.pipeline.staged_executor.executor import StagedBatchExecutor
from roop.utils.cache_paths import get_jobs_root
from ui.main import prepare_environment
from ui.tabs.faceswap_tab import (
    get_resume_settings_from_payload,
    index_of_no_face_action,
    load_resume_into_runtime,
    map_mask_engine,
    read_resume_payload,
    translate_swap_mode,
)


def configure_runtime(app_root: Path):
    roop.config.globals.CFG = Settings(str(app_root / "config.yaml"))
    prepare_environment()
    roop.config.globals.execution_providers = decode_execution_providers([roop.config.globals.CFG.provider])
    roop.config.globals.execution_threads = roop.config.globals.CFG.max_threads
    roop.config.globals.video_encoder = roop.config.globals.CFG.output_video_codec
    roop.config.globals.video_quality = roop.config.globals.CFG.video_quality


def apply_resume_settings(settings):
    roop.config.globals.selected_enhancer = settings["enhancer"]
    roop.config.globals.distance_threshold = settings["face_distance"]
    roop.config.globals.blend_ratio = settings["blend_ratio"]
    roop.config.globals.keep_frames = settings["keep_frames"]
    roop.config.globals.wait_after_extraction = settings["wait_after_extraction"]
    roop.config.globals.skip_audio = settings["skip_audio"]
    roop.config.globals.face_swap_mode = translate_swap_mode(settings["detection"])
    roop.config.globals.no_face_action = index_of_no_face_action(settings["no_face_action"])
    roop.config.globals.vr_mode = settings["vr_mode"]
    roop.config.globals.autorotate_faces = settings["autorotate"]


def make_options(processors, settings, selected_index=0):
    return ProcessOptions(
        processors,
        roop.config.globals.distance_threshold,
        roop.config.globals.blend_ratio,
        roop.config.globals.face_swap_mode,
        selected_index,
        settings["clip_text"],
        None,
        settings["num_swap_steps"],
        roop.config.globals.subsample_size,
        False,
        settings["restore_original_mouth"],
        face_swap_model=settings["face_swap_model"],
    )


def _resolve_file_segment(entry: ProcessEntry) -> str:
    file_signature = entry.file_signature or entry.filename
    return sanitize_job_path_segment(str(file_signature).replace(":", "_"))


def find_existing_job_dir(entry: ProcessEntry) -> Path:
    jobs_root = get_jobs_root()
    file_seg = _resolve_file_segment(entry)
    preferred_session = sanitize_job_path_segment(str(roop.config.globals.active_resume_cache_id or "adhoc"))
    candidates = []
    preferred = jobs_root / preferred_session / file_seg
    if preferred.exists():
        candidates.append(preferred)
    for session_dir in sorted(jobs_root.iterdir()):
        candidate = session_dir / file_seg
        if candidate.exists() and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if list((candidate / "detect" / "plans").glob("*.bin")):
            return candidate
    raise FileNotFoundError(f"No staged cache with detect packs found for {file_seg}")


def build_subset_stage_inputs(job_dir: Path, frame_window: int, workspace: Path):
    detect_pack_paths = sorted((job_dir / "detect" / "plans").glob("*.bin"))
    if not detect_pack_paths:
        raise FileNotFoundError(f"No detect pack cache found in {job_dir}")
    best_candidate = find_dense_window(job_dir, frame_window)
    if best_candidate is None:
        raise ValueError("No frame window with cached face tasks was found.")

    selected_frames = best_candidate["pack_frames"][
        best_candidate["start_index"]:best_candidate["start_index"] + best_candidate["window_size"]
    ]
    task_keys = [task["cache_key"] for frame_meta in selected_frames for task in frame_meta.get("tasks", [])]
    subset_pack = {
        "start_sequence": int(selected_frames[0]["sequence"]),
        "end_sequence": int(selected_frames[-1]["sequence"]),
        "frames": selected_frames,
    }
    detect_dir = workspace / "detect"
    swap_dir = workspace / "swap"
    detect_dir.mkdir(parents=True, exist_ok=True)
    swap_dir.mkdir(parents=True, exist_ok=True)
    write_cache_blob(get_stage_pack_path(detect_dir, subset_pack["start_sequence"], subset_pack["end_sequence"]), subset_pack)

    source_swap_cache_path = job_dir / "swap" / "packs" / best_candidate["source_pack_path"].name
    subset_swap_cache = read_stage_cache_keys(source_swap_cache_path, task_keys)
    if not subset_swap_cache:
        raise ValueError("Selected window has detect tasks but no swap cache entries.")
    write_stage_cache_map(get_stage_pack_path(swap_dir, subset_pack["start_sequence"], subset_pack["end_sequence"]), subset_swap_cache)
    return subset_pack, task_keys, detect_dir, swap_dir


def find_dense_window(job_dir: Path, frame_window: int):
    detect_pack_paths = sorted((job_dir / "detect" / "plans").glob("*.bin"))
    if not detect_pack_paths:
        raise FileNotFoundError(f"No detect pack cache found in {job_dir}")
    best_candidate = None
    for source_pack_path in detect_pack_paths:
        source_pack = read_cache_blob(source_pack_path)
        pack_frames = list(source_pack.get("frames", []))
        if not pack_frames:
            continue
        task_counts = [len(frame_meta.get("tasks", [])) for frame_meta in pack_frames]
        window_size = min(max(1, frame_window), len(pack_frames))
        window_tasks = sum(task_counts[:window_size])
        best_pack_window = (window_tasks, 0, window_size)
        for start_index in range(1, len(pack_frames) - window_size + 1):
            window_tasks += task_counts[start_index + window_size - 1] - task_counts[start_index - 1]
            if window_tasks > best_pack_window[0]:
                best_pack_window = (window_tasks, start_index, window_size)
        if best_pack_window[0] <= 0:
            continue
        if best_candidate is None or best_pack_window[0] > best_candidate["task_total"]:
            best_candidate = {
                "task_total": best_pack_window[0],
                "start_index": best_pack_window[1],
                "window_size": best_pack_window[2],
                "pack_frames": pack_frames,
                "source_pack_path": source_pack_path,
            }

    return best_candidate


def benchmark_mask_stage(entry, settings, memory_plan, workspace, detect_dir, swap_dir, subset_pack):
    processors = get_processing_plugins(map_mask_engine(settings["selected_mask_engine"], settings["clip_text"]))
    executor = StagedBatchExecutor("File", None, make_options(processors, settings))
    mask_dir = workspace / "mask"
    manifest = {"frame_count": len(subset_pack["frames"])}
    stages = {"mask": False}
    task_count = sum(len(frame_meta.get("tasks", [])) for frame_meta in subset_pack["frames"])

    roop.config.globals.processing = True
    started_at = time.perf_counter()
    executor.ensure_full_mask_stage(entry, entry.endframe, detect_dir, swap_dir, mask_dir, task_count, stages, manifest, memory_plan)
    elapsed = time.perf_counter() - started_at
    return elapsed, task_count, mask_dir


def benchmark_enhance_stage(entry, settings, memory_plan, workspace, detect_dir, swap_dir, mask_dir, subset_pack):
    processors = get_processing_plugins(map_mask_engine(settings["selected_mask_engine"], settings["clip_text"]))
    executor = StagedBatchExecutor("File", None, make_options(processors, settings))
    enhance_dir = workspace / "enhance"
    manifest = {"frame_count": len(subset_pack["frames"])}
    stages = {"enhance": False}
    task_count = sum(len(frame_meta.get("tasks", [])) for frame_meta in subset_pack["frames"])

    roop.config.globals.processing = True
    started_at = time.perf_counter()
    executor.ensure_full_enhance_stage(detect_dir, swap_dir, mask_dir, enhance_dir, task_count, stages, manifest, memory_plan)
    elapsed = time.perf_counter() - started_at
    return elapsed, task_count, enhance_dir


def benchmark_direct_encode(entry, memory_plan, output_dir):
    options = ProcessOptions({}, 0.6, 0.8, "all", 0, "", None, 1, roop.config.globals.subsample_size, False, False)
    executor = StagedBatchExecutor("File", None, options)
    executor.jobs_root = output_dir / "jobs_direct"
    short_entry = ProcessEntry(entry.filename, entry.startframe, entry.endframe, entry.fps, file_signature=entry.file_signature, display_name=entry.display_name)
    short_entry.finalname = str(output_dir / "direct_encode.mp4")

    roop.config.globals.processing = True
    started_at = time.perf_counter()
    executor.process_video_entry_full_frames(short_entry, 0)
    elapsed = time.perf_counter() - started_at
    frame_count = max(short_entry.endframe - short_entry.startframe, 1)
    return elapsed, frame_count


def benchmark_full_flow(entry, options):
    import cv2

    cap = cv2.VideoCapture(entry.filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    memory_plan = resolve_memory_plan(width, height)
    executor = StagedBatchExecutor("File", None, options)
    executor.jobs_root = Path(tempfile.mkdtemp(prefix="fullflow_jobs_", dir=os.environ["TEMP"]))
    stage_stats = {}
    original_update = executor.update_progress

    def patched_update(stage, *args, **kwargs):
        now = time.perf_counter()
        stats = stage_stats.setdefault(
            stage,
            {"first": now, "last": now, "max_step": 0, "unit": kwargs.get("step_unit")},
        )
        stats["last"] = now
        stats["unit"] = kwargs.get("step_unit", stats.get("unit"))
        step_completed = kwargs.get("step_completed")
        if isinstance(step_completed, int) and step_completed > stats["max_step"]:
            stats["max_step"] = step_completed
        return original_update(stage, *args, **kwargs)

    executor.update_progress = patched_update
    try:
        roop.config.globals.processing = True
        started = time.perf_counter()
        executor.process_video_entry_full_frames(entry, 0)
        total_elapsed = time.perf_counter() - started
    finally:
        shutil.rmtree(executor.jobs_root, ignore_errors=True)

    results = {"total_elapsed": total_elapsed, "memory_plan": memory_plan}
    for stage in ("detect", "swap", "mask", "enhance", "composite", "mux"):
        stats = stage_stats.get(stage)
        if not stats:
            continue
        elapsed = max(stats["last"] - stats["first"], 0.0)
        completed = stats["max_step"]
        results[stage] = {
            "elapsed": elapsed,
            "completed": completed,
            "unit": stats.get("unit"),
            "rate": (completed / elapsed) if elapsed > 0 and completed > 0 else 0.0,
        }
    return results


def format_rate(units, elapsed, suffix):
    if elapsed <= 0:
        return "n/a"
    return f"{units / elapsed:.2f} {suffix}/s"


def main():
    parser = argparse.ArgumentParser(description="Short staged benchmark using a real resume file.")
    parser.add_argument("--resume", required=True, help="Path to a resume JSON file.")
    parser.add_argument("--frames", type=int, default=64, help="Number of frames to benchmark from the start of the resume clip.")
    parser.add_argument(
        "--window",
        choices=("dense", "start"),
        default="dense",
        help="Choose the densest cached frame window or start from the resume clip beginning.",
    )
    parser.add_argument(
        "--mode",
        choices=("full-flow", "hot-path"),
        default="full-flow",
        help="Benchmark full staged flow or only the isolated pack-cache hot path.",
    )
    args = parser.parse_args()

    app_root = APP_ROOT
    os.chdir(app_root)
    configure_runtime(app_root)

    payload = read_resume_payload(args.resume)
    settings = get_resume_settings_from_payload(payload)
    apply_resume_settings(settings)
    load_resume_into_runtime(args.resume)

    target_info = (payload.get("targets") or {}).get("files") or []
    if not target_info:
        raise ValueError("Resume payload does not contain target files.")

    target = target_info[0]
    entry = ProcessEntry(
        target.get("resume_cached_path") or target["filename"],
        int(target.get("startframe", 0) or 0),
        int(target.get("startframe", 0) or 0) + int(args.frames),
        float(target.get("fps", 0) or 0) or util.detect_fps(target.get("resume_cached_path") or target["filename"]),
        file_signature=target.get("file_signature"),
        display_name=target.get("display_name") or os.path.basename(target.get("filename") or "target"),
    )
    if target.get("endframe"):
        entry.endframe = min(int(target.get("endframe")), entry.endframe)
    entry.finalname = str(app_root / "output" / "benchmark_staged_short.mp4")

    if args.window == "dense":
        job_dir = find_existing_job_dir(entry)
        dense_window = find_dense_window(job_dir, args.frames)
        if dense_window is not None:
            dense_frames = dense_window["pack_frames"][
                dense_window["start_index"]:dense_window["start_index"] + dense_window["window_size"]
            ]
            entry.startframe = int(dense_frames[0]["frame_number"])
            entry.endframe = int(dense_frames[-1]["frame_number"]) + 1

    import cv2

    cap = cv2.VideoCapture(entry.filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    memory_plan = resolve_memory_plan(width, height)
    print(f"Memory plan: {describe_memory_plan(memory_plan)}")
    options = make_options(get_processing_plugins(map_mask_engine(settings["selected_mask_engine"], settings["clip_text"])), settings)
    frame_count = max(entry.endframe - entry.startframe, 1)

    if args.mode == "full-flow":
        results = benchmark_full_flow(entry, options)
        print(f"Resume: {Path(args.resume).resolve()}")
        print(f"Window: frames {entry.startframe}..{entry.endframe} ({frame_count} frames)")
        print(f"Mode: full-flow ({args.window} window)")
        print(f"Total: {results['total_elapsed']:.3f}s")
        for stage in ("detect", "swap", "mask", "enhance", "composite"):
            stats = results.get(stage)
            if not stats:
                continue
            print(f"{stage}: {stats['elapsed']:.3f}s | {stats['rate']:.2f} {stats['unit']}/s")
        return

    job_dir = find_existing_job_dir(entry)
    workspace = Path(tempfile.mkdtemp(prefix="staged_short_", dir=os.environ["TEMP"]))
    try:
        subset_pack, task_keys, detect_dir, swap_dir = build_subset_stage_inputs(job_dir, frame_count, workspace)
        entry.startframe = int(subset_pack["frames"][0]["frame_number"])
        entry.endframe = int(subset_pack["frames"][-1]["frame_number"]) + 1
        frame_count = max(entry.endframe - entry.startframe, 1)
        mask_elapsed, mask_tasks, mask_dir = benchmark_mask_stage(entry, settings, memory_plan, workspace, detect_dir, swap_dir, subset_pack)
        enhance_elapsed, enhance_tasks, _enhance_dir = benchmark_enhance_stage(entry, settings, memory_plan, workspace, detect_dir, swap_dir, mask_dir, subset_pack)
        direct_elapsed, direct_frames = benchmark_direct_encode(entry, memory_plan, workspace)

        print(f"Resume: {Path(args.resume).resolve()}")
        print(f"Window: frames {entry.startframe}..{entry.endframe} ({frame_count} frames)")
        print(f"Mode: hot-path")
        print(f"Tasks: {len(task_keys)} faces")
        print(f"Mask stage: {mask_elapsed:.3f}s | {format_rate(mask_tasks, mask_elapsed, 'faces')}")
        print(f"Enhance stage: {enhance_elapsed:.3f}s | {format_rate(enhance_tasks, enhance_elapsed, 'faces')}")
        print(f"Direct encode: {direct_elapsed:.3f}s | {format_rate(direct_frames, direct_elapsed, 'frames')}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
