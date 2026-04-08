from roop.pipeline.batch_executor import ProcessMgr
import roop.config.globals

from .cache import DETECT_PACK_FRAME_COUNT, read_cache_blob, write_cache_blob, write_json
from .video_iter import iter_video_chunk


def get_detect_pack_dir(_executor, detect_dir):
    return detect_dir / "plans"


def get_detect_pack_frame_count(_executor):
    value = getattr(roop.config.globals.CFG, "detect_pack_frame_count", 0) or DETECT_PACK_FRAME_COUNT
    return max(8, int(value))


def get_detect_pack_path(executor, detect_dir, start_seq, end_seq):
    return executor.get_detect_pack_dir(detect_dir) / f"{start_seq:06d}_{end_seq:06d}.bin"


def iter_detect_pack_ranges(executor, frame_count):
    pack_frame_count = executor.get_detect_pack_frame_count()
    for start_seq in range(1, frame_count + 1, pack_frame_count):
        end_seq = min(start_seq + pack_frame_count - 1, frame_count)
        yield start_seq, end_seq


def iter_detect_packs(executor, detect_dir):
    pack_dir = executor.get_detect_pack_dir(detect_dir)
    if not pack_dir.exists():
        return
    for pack_path in sorted(pack_dir.glob("*.bin")):
        yield read_cache_blob(pack_path)


def iter_detect_frame_meta(executor, detect_dir, frame_count=None):
    emitted = 0
    for pack_data in executor.iter_detect_packs(detect_dir):
        for frame_meta in pack_data.get("frames", []):
            yield frame_meta["sequence"], frame_meta
            emitted += 1
            if frame_count is not None and emitted >= frame_count:
                return


def iter_detect_tasks(executor, detect_dir, frame_count=None):
    for _, frame_meta in executor.iter_detect_frame_meta(detect_dir, frame_count):
        for task_meta in frame_meta["tasks"]:
            yield task_meta


def flatten_pack_tasks(_executor, pack_data):
    tasks = []
    for frame_meta in pack_data.get("frames", []):
        for task_meta in frame_meta.get("tasks", []):
            tasks.append((frame_meta, task_meta))
    return tasks


def summarize_detect_cache(executor, detect_dir, frame_count=None):
    cached_frames = 0
    total_tasks = 0
    for _, frame_meta in executor.iter_detect_frame_meta(detect_dir, frame_count):
        cached_frames += 1
        total_tasks += len(frame_meta["tasks"])
    return cached_frames, total_tasks


def iter_full_source_frames_with_meta(executor, entry, endframe, detect_dir, frame_count, memory_plan):
    frame_meta_iter = iter(executor.iter_detect_frame_meta(detect_dir, frame_count))
    next_meta = next(frame_meta_iter, None)
    for frame_number, frame in iter_video_chunk(entry.filename, entry.startframe, endframe, memory_plan["prefetch_frames"]):
        seq_index = (frame_number - entry.startframe) + 1
        while next_meta is not None and next_meta[0] < seq_index:
            next_meta = next(frame_meta_iter, None)
        if next_meta is not None and next_meta[0] == seq_index:
            frame_meta = next_meta[1]
            next_meta = next(frame_meta_iter, None)
        else:
            frame_meta = {
                "frame_number": frame_number,
                "sequence": seq_index,
                "fallback": True,
                "tasks": [],
            }
        yield seq_index, frame_number, frame, frame_meta


def ensure_full_detect_stage(executor, entry, endframe, detect_dir, stages, manifest, memory_plan):
    plans_dir = executor.get_detect_pack_dir(detect_dir)
    plans_dir.mkdir(parents=True, exist_ok=True)
    frame_count = max(endframe - entry.startframe, 1)
    if stages["detect"]:
        cached_frames, task_count = executor.summarize_detect_cache(detect_dir, frame_count)
        if cached_frames >= frame_count:
            manifest["frame_count"] = frame_count
            manifest["task_count"] = task_count
            executor.update_progress(
                "detect",
                detail="Reusing detect cache",
                step_completed=frame_count,
                step_total=frame_count,
                step_unit="frames",
                force_log=True,
            )
            return task_count
    planner = ProcessMgr(None)
    planner.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
    processed_frames = 0
    total_tasks = 0
    try:
        for pack_start, pack_end in executor.iter_detect_pack_ranges(frame_count):
            pack_path = executor.get_detect_pack_path(detect_dir, pack_start, pack_end)
            if pack_path.exists():
                pack_data = read_cache_blob(pack_path)
                pack_frames = pack_data.get("frames", [])
                processed_frames += len(pack_frames)
                total_tasks += sum(len(frame_meta["tasks"]) for frame_meta in pack_frames)
                executor.update_progress(
                    "detect",
                    detail="Reusing packed detect cache",
                    step_completed=processed_frames,
                    step_total=frame_count,
                    step_unit="frames",
                )
                continue

            pack_frames = []
            absolute_start = entry.startframe + pack_start - 1
            absolute_end = entry.startframe + pack_end
            for frame_number, frame in iter_video_chunk(entry.filename, absolute_start, absolute_end, memory_plan["prefetch_frames"]):
                seq_index = (frame_number - entry.startframe) + 1
                frame_plan = planner.build_frame_plan(frame)
                frame_meta = {
                    "frame_number": frame_number,
                    "sequence": seq_index,
                    "fallback": frame_plan["fallback"],
                    "tasks": [],
                }
                for task_index, task in enumerate(frame_plan["tasks"]):
                    cache_key = f"f{seq_index:06d}_t{task_index:03d}"
                    task_meta = {key: value for key, value in task.items() if key != "aligned_frame"}
                    task_meta["cache_key"] = cache_key
                    frame_meta["tasks"].append(task_meta)
                pack_frames.append(frame_meta)
                total_tasks += len(frame_meta["tasks"])
                processed_frames += 1
                executor.update_progress(
                    "detect",
                    detail="Streaming source decode + packed face detection",
                    step_completed=processed_frames,
                    step_total=frame_count,
                    step_unit="frames",
                )
            if pack_frames:
                write_cache_blob(
                    pack_path,
                    {
                        "start_sequence": pack_start,
                        "end_sequence": pack_end,
                        "frames": pack_frames,
                    },
                )
            manifest["frame_count"] = processed_frames
            manifest["task_count"] = total_tasks
            write_json(detect_dir.parent / "manifest.json", manifest)
            if not roop.config.globals.processing:
                break
    finally:
        planner.release_resources()
    stages["detect"] = roop.config.globals.processing and processed_frames >= frame_count
    manifest["frame_count"] = processed_frames
    manifest["task_count"] = total_tasks
    return total_tasks


def ensure_detect_cache(executor, job_dir, detect_dir, image, frame_number):
    detect_dir.mkdir(parents=True, exist_ok=True)
    plan_path = detect_dir / "plan.bin"
    if plan_path.exists():
        executor.update_progress("detect", detail="Reusing cached detect plan", step_completed=1, step_total=1, step_unit="image")
        return read_cache_blob(plan_path)
    planner = ProcessMgr(None)
    planner.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
    try:
        frame_plan = planner.build_frame_plan(image)
        frame_meta = {"frame_number": frame_number, "fallback": frame_plan["fallback"], "tasks": []}
        for task_index, task in enumerate(frame_plan["tasks"]):
            cache_key = f"f{frame_number:08d}_t{task_index:03d}"
            task_meta = {key: value for key, value in task.items() if key != "aligned_frame"}
            task_meta["cache_key"] = cache_key
            frame_meta["tasks"].append(task_meta)
    finally:
        planner.release_resources()
    write_cache_blob(plan_path, frame_meta)
    executor.update_progress("detect", detail="Detection cache created", step_completed=1, step_total=1, step_unit="image")
    return frame_meta


def ensure_chunk_detect(executor, video_path, chunk_dir, chunk_start, chunk_end, memory_plan, chunk_state):
    plan_path = chunk_dir / "plan.bin"
    if plan_path.exists():
        chunk_state["stages"]["detect"] = True
        frame_total = max(chunk_end - chunk_start, 1)
        executor.update_progress("detect", detail="Reusing detect cache", step_completed=frame_total, step_total=frame_total, step_unit="frames")
        return read_cache_blob(plan_path)

    chunk_dir.mkdir(parents=True, exist_ok=True)
    planner = ProcessMgr(None)
    planner.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
    chunk_meta = {"start": chunk_start, "end": chunk_end, "frames": []}
    processed_frames = 0
    total_frames = max(chunk_end - chunk_start, 1)
    try:
        for frame_number, frame in iter_video_chunk(video_path, chunk_start, chunk_end, memory_plan["prefetch_frames"]):
            frame_plan = planner.build_frame_plan(frame)
            frame_meta = {"frame_number": frame_number, "fallback": frame_plan["fallback"], "tasks": []}
            for task_index, task in enumerate(frame_plan["tasks"]):
                cache_key = f"f{frame_number:08d}_t{task_index:03d}"
                task_meta = {key: value for key, value in task.items() if key != "aligned_frame"}
                task_meta["cache_key"] = cache_key
                frame_meta["tasks"].append(task_meta)
            chunk_meta["frames"].append(frame_meta)
            processed_frames += 1
            executor.update_progress("detect", detail="Detecting and aligning faces", step_completed=processed_frames, step_total=total_frames, step_unit="frames")
    finally:
        planner.release_resources()
    write_cache_blob(plan_path, chunk_meta)
    chunk_state["stages"]["detect"] = True
    return chunk_meta


__all__ = [
    "ensure_chunk_detect",
    "ensure_detect_cache",
    "ensure_full_detect_stage",
    "flatten_pack_tasks",
    "get_detect_pack_dir",
    "get_detect_pack_frame_count",
    "get_detect_pack_path",
    "iter_detect_frame_meta",
    "iter_detect_pack_ranges",
    "iter_detect_packs",
    "iter_detect_tasks",
    "iter_full_source_frames_with_meta",
    "summarize_detect_cache",
]
