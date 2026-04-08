import cv2
import numpy as np

import roop.config.globals
from roop.pipeline.batch_executor import ProcessMgr

from .chunk_processor import flatten_tasks, iter_chunk_source_frames_with_meta
from .detect_stage import flatten_pack_tasks
from .video_iter import iter_video_chunk
from .cache import normalize_cache_image, write_json


def run_mask_single_outputs(executor, processor, original_batch):
    return [processor.Run(original, executor.options.masking_text) for original in original_batch]


def disable_broken_mask_batch(_executor, processor):
    processor._mask_batch_output_validation_state = "broken"
    if hasattr(processor, "supports_batch"):
        processor.supports_batch = False
    if hasattr(processor, "batch_size_limit"):
        processor.batch_size_limit = 1
    if hasattr(processor, "supports_parallel_single_batch"):
        processor.supports_parallel_single_batch = False


def validate_mask_batch_outputs(executor, processor, original_batch, masks):
    if len(original_batch) <= 1:
        return masks
    if len(masks) != len(original_batch):
        disable_broken_mask_batch(executor, processor)
        return run_mask_single_outputs(executor, processor, original_batch)
    if getattr(processor, "_mask_batch_output_validation_state", None) == "verified":
        return masks
    if getattr(processor, "_mask_batch_output_validation_state", None) == "broken":
        return run_mask_single_outputs(executor, processor, original_batch)

    reference_mask = processor.Run(original_batch[0], executor.options.masking_text)
    batch_mask = masks[0]
    if reference_mask.shape == batch_mask.shape and np.allclose(
        reference_mask.astype(np.float32),
        batch_mask.astype(np.float32),
        atol=1e-3,
    ):
        processor._mask_batch_output_validation_state = "verified"
        return masks

    disable_broken_mask_batch(executor, processor)
    return [
        reference_mask,
        *run_mask_single_outputs(executor, processor, original_batch[1:]),
    ]


def process_full_mask_batch(
    executor,
    task_batch,
    original_batch,
    input_cache_path,
    output_cache,
    output_cache_path,
    mask_mgr,
    processor,
    processed_tasks,
    task_count,
    memory_plan,
    flush_cache=True,
):
    input_cache = executor.read_stage_cache_keys(input_cache_path, [task_meta["cache_key"] for task_meta in task_batch])
    if getattr(processor, "supports_batch", False):
        batch_inputs = [np.ascontiguousarray(original) for original in original_batch]
        masks = processor.RunBatch(batch_inputs, executor.options.masking_text, memory_plan["mask_batch_size"])
        masks = validate_mask_batch_outputs(executor, processor, original_batch, masks)
        for task_meta, original, mask in zip(task_batch, original_batch, masks):
            current_frame = np.ascontiguousarray(input_cache[task_meta["cache_key"]])
            mask = cv2.resize(mask, (current_frame.shape[1], current_frame.shape[0]))
            mask = np.reshape(mask, [mask.shape[0], mask.shape[1], 1])
            result = (1 - mask) * current_frame.astype(np.float32)
            result += mask * np.ascontiguousarray(original).astype(np.float32)
            output_cache[task_meta["cache_key"]] = normalize_cache_image(np.uint8(result))
    else:
        for task_meta, original in zip(task_batch, original_batch):
            task = dict(task_meta)
            task["aligned_frame"] = original
            current_frame = input_cache[task_meta["cache_key"]]
            result = mask_mgr.run_mask_task(task, current_frame, processor)
            output_cache[task_meta["cache_key"]] = normalize_cache_image(result)
    if flush_cache:
        executor.write_stage_cache_map(output_cache_path, output_cache)
    executor.update_progress(
        "mask",
        detail="Streaming source decode + mask stage",
        step_completed=processed_tasks + len(task_batch),
        step_total=task_count,
        step_unit="faces",
    )


def ensure_full_mask_stage(executor, entry, endframe, detect_dir, swap_dir, mask_dir, task_count, stages, manifest, memory_plan):
    if executor.mask_name is None or task_count <= 0:
        stages["mask"] = True
        write_json(mask_dir.parent / "manifest.json", manifest)
        return
    mask_dir.mkdir(parents=True, exist_ok=True)
    if stages["mask"]:
        cached = 0
        for pack_data in executor.iter_detect_packs(detect_dir):
            cached += executor.count_stage_cache_entries(executor.get_stage_pack_path(mask_dir, pack_data["start_sequence"], pack_data["end_sequence"]))
        if cached >= task_count:
            executor.update_progress("mask", detail="Reusing mask cache", step_completed=task_count, step_total=task_count, step_unit="faces", force_log=True)
            return

    mask_mgr = ProcessMgr(None)
    mask_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([executor.mask_name]))
    processor = mask_mgr.processors[0]
    processed_tasks = 0
    task_batch_size = max(1, min(128, memory_plan["mask_batch_size"]))
    try:
        for pack_data in executor.iter_detect_packs(detect_dir):
            pack_tasks = flatten_pack_tasks(executor, pack_data)
            if not pack_tasks:
                continue
            input_cache_path = executor.get_stage_pack_path(swap_dir, pack_data["start_sequence"], pack_data["end_sequence"])
            pack_cache_path = executor.get_stage_pack_path(mask_dir, pack_data["start_sequence"], pack_data["end_sequence"])
            pack_cache = executor.read_stage_cache_map(pack_cache_path)
            if len(pack_cache) >= len(pack_tasks):
                processed_tasks += len(pack_tasks)
                executor.update_progress("mask", detail="Reusing packed mask cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                continue

            frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in pack_data["frames"]}
            task_batch = []
            original_batch = []
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
                        executor.update_progress("mask", detail="Reusing packed mask cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                        continue
                    task_batch.append(dict(task_meta))
                    original_batch.append(mask_mgr.rebuild_aligned_frame(frame, task_meta))
                    if len(task_batch) >= task_batch_size:
                        process_full_mask_batch(
                            executor,
                            task_batch,
                            original_batch,
                            input_cache_path,
                            pack_cache,
                            pack_cache_path,
                            mask_mgr,
                            processor,
                            processed_tasks,
                            task_count,
                            memory_plan,
                            flush_cache=False,
                        )
                        pack_cache_dirty = True
                        processed_tasks += len(task_batch)
                        task_batch.clear()
                        original_batch.clear()
            if task_batch:
                process_full_mask_batch(
                    executor,
                    task_batch,
                    original_batch,
                    input_cache_path,
                    pack_cache,
                    pack_cache_path,
                    mask_mgr,
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
        mask_mgr.release_resources()
    stages["mask"] = True
    write_json(mask_dir.parent / "manifest.json", manifest)


def ensure_mask_stage(executor, chunk_dir, chunk_meta, chunk_state, memory_plan, video_path=None, source_image=None):
    if executor.mask_name is None:
        chunk_state["stages"]["mask"] = True
        return
    mask_dir = chunk_dir / "mask"
    flat_tasks = flatten_tasks(executor, chunk_meta)
    total_tasks = len(flat_tasks)
    cache_path = executor.get_stage_cache_path(mask_dir)
    mask_cache = executor.read_stage_cache_map(cache_path)
    if chunk_state["stages"]["mask"] or len(mask_cache) >= total_tasks:
        chunk_state["stages"]["mask"] = True
        if total_tasks > 0:
            executor.update_progress("mask", detail="Reusing mask cache", step_completed=total_tasks, step_total=total_tasks, step_unit="faces")
        return
    input_cache_path = executor.get_stage_cache_path(chunk_dir / "swap")
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_mgr = ProcessMgr(None)
    mask_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([executor.mask_name]))
    processor = mask_mgr.processors[0]
    processed_tasks = 0
    task_batch = []
    original_batch = []
    task_batch_size = max(1, min(128, memory_plan["mask_batch_size"]))
    cache_dirty = False
    try:
        for _, frame, frame_meta in iter_chunk_source_frames_with_meta(executor, chunk_meta, memory_plan, video_path=video_path, source_image=source_image):
            for task_meta in frame_meta["tasks"]:
                if task_meta["cache_key"] in mask_cache:
                    processed_tasks += 1
                    executor.update_progress("mask", detail="Reusing packed masks", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
                    continue
                task_batch.append(dict(task_meta))
                original_batch.append(mask_mgr.rebuild_aligned_frame(frame, task_meta))
                if len(task_batch) >= task_batch_size:
                    process_full_mask_batch(
                        executor,
                        task_batch,
                        original_batch,
                        input_cache_path,
                        mask_cache,
                        cache_path,
                        mask_mgr,
                        processor,
                        processed_tasks,
                        total_tasks,
                        memory_plan,
                        flush_cache=False,
                    )
                    cache_dirty = True
                    processed_tasks += len(task_batch)
                    task_batch.clear()
                    original_batch.clear()
        if task_batch:
            process_full_mask_batch(
                executor,
                task_batch,
                original_batch,
                input_cache_path,
                mask_cache,
                cache_path,
                mask_mgr,
                processor,
                processed_tasks,
                total_tasks,
                memory_plan,
                flush_cache=False,
            )
            cache_dirty = True
            processed_tasks += len(task_batch)
        if cache_dirty:
            executor.write_stage_cache_map(cache_path, mask_cache)
    finally:
        mask_mgr.release_resources()
    chunk_state["stages"]["mask"] = True


__all__ = [
    "disable_broken_mask_batch",
    "ensure_full_mask_stage",
    "ensure_mask_stage",
    "process_full_mask_batch",
    "run_mask_single_outputs",
    "validate_mask_batch_outputs",
]
