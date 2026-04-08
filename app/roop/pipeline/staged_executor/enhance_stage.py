import numpy as np

import roop.config.globals
from roop.pipeline.batch_executor import ProcessMgr

from .chunk_processor import flatten_tasks
from .detect_stage import flatten_pack_tasks
from .cache import chunked, normalize_cache_image, write_json


def get_enhance_task_batch_size(executor, memory_plan):
    requested_batch = max(memory_plan["enhance_batch_size"], 1)
    worker_window = max(memory_plan.get("single_batch_workers", 1), 1) * 2
    return max(1, min(64, max(requested_batch, worker_window)))


def process_full_enhance_batch(
    executor,
    task_batch,
    input_cache,
    output_cache,
    output_cache_path,
    enhance_mgr,
    processor,
    processed_tasks,
    task_count,
    memory_plan,
    flush_cache=True,
):
    current_frames = [np.ascontiguousarray(input_cache[task_meta["cache_key"]]) for task_meta in task_batch]
    enhanced_frames = enhance_mgr.run_enhance_tasks_batch(task_batch, current_frames, processor, memory_plan["enhance_batch_size"])
    for task_meta in task_batch:
        output_cache[task_meta["cache_key"]] = normalize_cache_image(enhanced_frames[task_meta["cache_key"]])
    if flush_cache:
        executor.write_stage_cache_map(output_cache_path, output_cache)
    executor.update_progress("enhance", detail="Running enhancement stage", step_completed=processed_tasks + len(task_batch), step_total=task_count, step_unit="faces")


def ensure_full_enhance_stage(executor, detect_dir, swap_dir, mask_dir, enhance_dir, task_count, stages, manifest, memory_plan):
    if executor.enhancer_name is None or task_count <= 0:
        stages["enhance"] = True
        write_json(enhance_dir.parent / "manifest.json", manifest)
        return
    enhance_dir.mkdir(parents=True, exist_ok=True)
    if stages["enhance"]:
        cached = 0
        for pack_data in executor.iter_detect_packs(detect_dir):
            cached += executor.count_stage_cache_entries(executor.get_stage_pack_path(enhance_dir, pack_data["start_sequence"], pack_data["end_sequence"]))
        if cached >= task_count:
            executor.update_progress("enhance", detail="Reusing enhance cache", step_completed=task_count, step_total=task_count, step_unit="faces", force_log=True)
            return

    enhance_mgr = ProcessMgr(None)
    enhance_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([executor.enhancer_name]))
    processor = enhance_mgr.processors[0]
    processed_tasks = 0
    task_batch_size = get_enhance_task_batch_size(executor, memory_plan)
    try:
        for pack_data in executor.iter_detect_packs(detect_dir):
            pack_tasks = flatten_pack_tasks(executor, pack_data)
            if not pack_tasks:
                continue
            input_cache = executor.read_stage_cache_map(executor.get_stage_pack_path(mask_dir if executor.mask_name else swap_dir, pack_data["start_sequence"], pack_data["end_sequence"]))
            pack_cache_path = executor.get_stage_pack_path(enhance_dir, pack_data["start_sequence"], pack_data["end_sequence"])
            pack_cache = executor.read_stage_cache_map(pack_cache_path)
            if len(pack_cache) >= len(pack_tasks):
                processed_tasks += len(pack_tasks)
                executor.update_progress("enhance", detail="Reusing packed enhance cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                continue
            task_batch = []
            pack_cache_dirty = False
            for _, task_meta in pack_tasks:
                if task_meta["cache_key"] in pack_cache:
                    processed_tasks += 1
                    executor.update_progress("enhance", detail="Reusing packed enhance cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                    continue
                task_batch.append(dict(task_meta))
                if len(task_batch) >= task_batch_size:
                    process_full_enhance_batch(
                        executor,
                        task_batch,
                        input_cache,
                        pack_cache,
                        pack_cache_path,
                        enhance_mgr,
                        processor,
                        processed_tasks,
                        task_count,
                        memory_plan,
                        flush_cache=False,
                    )
                    pack_cache_dirty = True
                    processed_tasks += len(task_batch)
                    task_batch.clear()
            if task_batch:
                process_full_enhance_batch(
                    executor,
                    task_batch,
                    input_cache,
                    pack_cache,
                    pack_cache_path,
                    enhance_mgr,
                    processor,
                    processed_tasks,
                    task_count,
                    memory_plan,
                    flush_cache=False,
                )
                pack_cache_dirty = True
                processed_tasks += len(task_batch)
            if pack_cache_dirty:
                executor.write_stage_cache_map(pack_cache_path, pack_cache)
    finally:
        enhance_mgr.release_resources()
    stages["enhance"] = True
    write_json(enhance_dir.parent / "manifest.json", manifest)


def ensure_enhance_stage(executor, chunk_dir, chunk_meta, chunk_state, memory_plan):
    if executor.enhancer_name is None:
        chunk_state["stages"]["enhance"] = True
        return
    enhance_dir = chunk_dir / "enhance"
    flat_tasks = flatten_tasks(executor, chunk_meta)
    total_tasks = len(flat_tasks)
    cache_path = executor.get_stage_cache_path(enhance_dir)
    enhance_cache = executor.read_stage_cache_map(cache_path)
    if chunk_state["stages"]["enhance"] or len(enhance_cache) >= total_tasks:
        chunk_state["stages"]["enhance"] = True
        if total_tasks > 0:
            executor.update_progress("enhance", detail="Reusing enhance cache", step_completed=total_tasks, step_total=total_tasks, step_unit="faces")
        return
    input_cache = executor.read_stage_cache_map(executor.get_stage_cache_path(chunk_dir / ("mask" if executor.mask_name else "swap")))
    enhance_dir.mkdir(parents=True, exist_ok=True)
    enhance_mgr = ProcessMgr(None)
    enhance_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([executor.enhancer_name]))
    processor = enhance_mgr.processors[0]
    processed_tasks = 0
    task_batch_size = get_enhance_task_batch_size(executor, memory_plan)
    cache_dirty = False
    try:
        for batch in chunked(flat_tasks, task_batch_size):
            pending = [(frame_meta, task_meta) for frame_meta, task_meta in batch if task_meta["cache_key"] not in enhance_cache]
            if not pending:
                processed_tasks += len(batch)
                executor.update_progress("enhance", detail="Reusing packed enhance cache", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
                continue
            current_frames = [np.ascontiguousarray(input_cache[task_meta["cache_key"]]) for _, task_meta in pending]
            task_batch = [task_meta for _, task_meta in pending]
            enhanced_frames = enhance_mgr.run_enhance_tasks_batch(task_batch, current_frames, processor, memory_plan["enhance_batch_size"])
            for task_meta in task_batch:
                enhance_cache[task_meta["cache_key"]] = normalize_cache_image(enhanced_frames[task_meta["cache_key"]])
            cache_dirty = True
            processed_tasks += len(batch)
            executor.update_progress("enhance", detail="Running enhancement stage", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
        if cache_dirty:
            executor.write_stage_cache_map(cache_path, enhance_cache)
    finally:
        enhance_mgr.release_resources()
    chunk_state["stages"]["enhance"] = True


__all__ = [
    "ensure_enhance_stage",
    "ensure_full_enhance_stage",
    "get_enhance_task_batch_size",
    "process_full_enhance_batch",
]
