import roop.config.globals
from roop.pipeline.batch_executor import ProcessMgr

from .chunk_processor import flatten_tasks, iter_chunk_source_frames_with_meta
from .detect_stage import flatten_pack_tasks
from .video_iter import iter_video_chunk
from .cache import normalize_cache_image, write_json


def get_swap_task_batch_size(executor, memory_plan):
    tiles_per_task = max((max(executor.options.subsample_size, 128) // 128) ** 2, 1)
    native_batch_window = max(1, min(64, (max(memory_plan["swap_batch_size"], 1) * 2) // tiles_per_task))
    worker_window = max(memory_plan.get("single_batch_workers", 1), 1) * 2
    return max(native_batch_window, min(64, worker_window))


def ensure_full_swap_stage(executor, entry, endframe, detect_dir, swap_dir, task_count, stages, manifest, memory_plan):
    if not executor.swap_enabled or task_count <= 0:
        stages["swap"] = True
        write_json(swap_dir.parent / "manifest.json", manifest)
        return
    swap_dir.mkdir(parents=True, exist_ok=True)
    if stages["swap"]:
        cached = 0
        for pack_data in executor.iter_detect_packs(detect_dir):
            cached += executor.count_stage_cache_entries(executor.get_stage_pack_path(swap_dir, pack_data["start_sequence"], pack_data["end_sequence"]))
        if cached >= task_count:
            executor.update_progress("swap", detail="Reusing swap cache", step_completed=task_count, step_total=task_count, step_unit="faces", force_log=True)
            return

    swap_mgr = ProcessMgr(None)
    swap_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options(["faceswap"]))
    processor = swap_mgr.processors[0]
    processed_tasks = 0
    task_batch_size = get_swap_task_batch_size(executor, memory_plan)
    try:
        for pack_data in executor.iter_detect_packs(detect_dir):
            pack_tasks = flatten_pack_tasks(executor, pack_data)
            if not pack_tasks:
                continue
            pack_cache_path = executor.get_stage_pack_path(swap_dir, pack_data["start_sequence"], pack_data["end_sequence"])
            pack_cache = executor.read_stage_cache_map(pack_cache_path)
            if len(pack_cache) >= len(pack_tasks):
                processed_tasks += len(pack_tasks)
                executor.update_progress("swap", detail="Reusing packed swap cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                continue

            frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in pack_data["frames"]}
            task_batch = []
            pack_cache_dirty = False
            for frame_number, frame in iter_video_chunk(
                entry.filename,
                pack_data["frames"][0]["frame_number"],
                pack_data["frames"][-1]["frame_number"] + 1,
                memory_plan["prefetch_frames"],
            ):
                frame_meta = frame_lookup.get(frame_number, {"tasks": []})
                for task_meta in frame_meta["tasks"]:
                    if task_meta["cache_key"] in pack_cache:
                        processed_tasks += 1
                        executor.update_progress("swap", detail="Reusing packed swap cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                        continue
                    task = dict(task_meta)
                    task["aligned_frame"] = swap_mgr.rebuild_aligned_frame(frame, task_meta)
                    task_batch.append(task)
                    if len(task_batch) >= task_batch_size:
                        batch_outputs = swap_mgr.run_swap_tasks_batch(task_batch, processor, memory_plan["swap_batch_size"])
                        for cache_key, fake_frame in batch_outputs.items():
                            pack_cache[cache_key] = normalize_cache_image(fake_frame)
                        pack_cache_dirty = True
                        processed_tasks += len(task_batch)
                        executor.update_progress("swap", detail="Streaming source decode + batched face swap", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                        task_batch.clear()
            if task_batch:
                batch_outputs = swap_mgr.run_swap_tasks_batch(task_batch, processor, memory_plan["swap_batch_size"])
                for cache_key, fake_frame in batch_outputs.items():
                    pack_cache[cache_key] = normalize_cache_image(fake_frame)
                pack_cache_dirty = True
                processed_tasks += len(task_batch)
                executor.update_progress("swap", detail="Streaming source decode + batched face swap", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
            if pack_cache_dirty:
                executor.write_stage_cache_map(pack_cache_path, pack_cache)
    finally:
        swap_mgr.release_resources()
    stages["swap"] = True
    write_json(swap_dir.parent / "manifest.json", manifest)


def ensure_swap_stage(executor, chunk_dir, chunk_meta, chunk_state, memory_plan, video_path=None, source_image=None):
    if not executor.swap_enabled:
        chunk_state["stages"]["swap"] = True
        return
    swap_dir = chunk_dir / "swap"
    flat_tasks = flatten_tasks(executor, chunk_meta)
    total_tasks = len(flat_tasks)
    cache_path = executor.get_stage_cache_path(swap_dir)
    swap_cache = executor.read_stage_cache_map(cache_path)
    if chunk_state["stages"]["swap"] or len(swap_cache) >= total_tasks:
        chunk_state["stages"]["swap"] = True
        if total_tasks > 0:
            executor.update_progress("swap", detail="Reusing swap cache", step_completed=total_tasks, step_total=total_tasks, step_unit="faces")
        return
    swap_dir.mkdir(parents=True, exist_ok=True)
    swap_mgr = ProcessMgr(None)
    swap_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options(["faceswap"]))
    processor = swap_mgr.processors[0]
    processed_tasks = 0
    task_batch = []
    task_batch_size = get_swap_task_batch_size(executor, memory_plan)
    cache_dirty = False
    try:
        for _, frame, frame_meta in iter_chunk_source_frames_with_meta(executor, chunk_meta, memory_plan, video_path=video_path, source_image=source_image):
            for task_meta in frame_meta["tasks"]:
                if task_meta["cache_key"] in swap_cache:
                    processed_tasks += 1
                    executor.update_progress("swap", detail="Reusing packed swap cache", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
                    continue
                task = dict(task_meta)
                task["aligned_frame"] = swap_mgr.rebuild_aligned_frame(frame, task_meta)
                task_batch.append(task)
                if len(task_batch) >= task_batch_size:
                    batch_outputs = swap_mgr.run_swap_tasks_batch(task_batch, processor, memory_plan["swap_batch_size"])
                    for cache_key, fake_frame in batch_outputs.items():
                        swap_cache[cache_key] = normalize_cache_image(fake_frame)
                    cache_dirty = True
                    processed_tasks += len(task_batch)
                    executor.update_progress("swap", detail="Streaming source decode + batched face swap", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
                    task_batch.clear()
        if task_batch:
            batch_outputs = swap_mgr.run_swap_tasks_batch(task_batch, processor, memory_plan["swap_batch_size"])
            for cache_key, fake_frame in batch_outputs.items():
                swap_cache[cache_key] = normalize_cache_image(fake_frame)
            cache_dirty = True
            processed_tasks += len(task_batch)
            executor.update_progress("swap", detail="Streaming source decode + batched face swap", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
        if cache_dirty:
            executor.write_stage_cache_map(cache_path, swap_cache)
    finally:
        swap_mgr.release_resources()
    chunk_state["stages"]["swap"] = True


__all__ = ["ensure_full_swap_stage", "ensure_swap_stage", "get_swap_task_batch_size"]
