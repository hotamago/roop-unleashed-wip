import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import local

import cv2

import roop.config.globals
import roop.media.ffmpeg_ops as ffmpeg
import roop.utils as util
from roop.media.ffmpeg_writer import FFMPEG_VideoWriter
from roop.media.video_io import open_video_capture, resolve_video_writer_config
from roop.pipeline.batch_executor import ProcessMgr, eNoFaceAction
from .cache import write_json

try:
    from roop.media.stream_writer import StreamWriter
except Exception:
    StreamWriter = None


def get_composite_progress_detail():
    return "Streaming source decode + cached face compositing + video encode"


def get_compose_gpu_batch_size(executor):
    requested_workers = getattr(roop.config.globals.CFG, "max_threads", 1)
    try:
        resolved = int(requested_workers)
    except (TypeError, ValueError):
        resolved = 1
    return max(1, min(max(resolved, 1) * 2, 8))


def get_compose_worker_count(executor):
    requested_workers = getattr(roop.config.globals.CFG, "max_threads", 1)
    try:
        resolved_workers = int(requested_workers)
    except (TypeError, ValueError):
        resolved_workers = 1
    resolved_workers = max(1, min(resolved_workers, 8))
    if roop.config.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
        return 1
    return resolved_workers


def should_use_cached_fallback_fast_path():
    return roop.config.globals.no_face_action in (
        eNoFaceAction.USE_ORIGINAL_FRAME,
        eNoFaceAction.USE_LAST_SWAPPED,
        eNoFaceAction.SKIP_FRAME,
    )


def can_direct_encode_without_processing(executor, entry):
    return (
        executor.output_method == "File"
        and not executor.swap_enabled
        and executor.mask_name is None
        and executor.enhancer_name is None
        and not util.has_extension(entry.filename, ["gif"])
        and bool(getattr(entry, "finalname", None))
    )


def ensure_direct_video_output(executor, entry, index, frame_count, endframe):
    destination = util.replace_template(entry.finalname, index=index)
    Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
    executor.update_progress(
        "composite",
        detail="Direct ffmpeg video encode without processing stages",
        step_completed=0,
        step_total=frame_count,
        step_unit="frames",
        force_log=True,
    )
    ffmpeg.transcode_video_range(
        entry.filename,
        destination,
        entry.startframe,
        endframe,
        entry.fps or util.detect_fps(entry.filename),
        include_audio=not roop.config.globals.skip_audio,
    )
    executor.completed_units += frame_count
    executor.update_progress(
        "mux",
        detail="Completed direct video encode",
        step_completed=frame_count,
        step_total=frame_count,
        step_unit="frames",
        force_log=True,
    )


def prepare_fallback_mgr(executor, fallback_mgr=None, last_result_frame=None):
    if fallback_mgr is None:
        fallback_mgr = executor.get_fallback_mgr()
    if fallback_mgr.last_swapped_frame is None and last_result_frame is not None:
        fallback_mgr.last_swapped_frame = last_result_frame.copy()
        fallback_mgr.num_frames_no_face = 0
    return fallback_mgr


def resolve_cached_fallback_frame(executor, frame, fallback_mgr=None, last_result_frame=None):
    no_face_action = roop.config.globals.no_face_action
    if no_face_action == eNoFaceAction.USE_ORIGINAL_FRAME:
        return frame
    if no_face_action == eNoFaceAction.SKIP_FRAME:
        return None
    fallback_mgr = prepare_fallback_mgr(executor, fallback_mgr, last_result_frame)
    if no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
        max_reuse_frame = max(0, int(getattr(getattr(fallback_mgr, "options", None), "max_num_reuse_frame", 0)))
        if fallback_mgr.last_swapped_frame is not None and fallback_mgr.num_frames_no_face < max_reuse_frame:
            fallback_mgr.num_frames_no_face += 1
            return fallback_mgr.last_swapped_frame.copy()
        return frame
    return fallback_mgr.process_frame(frame)


def compose_frame_from_cache(executor, compose_mgr, frame, frame_meta, input_cache, enhance_cache, fallback_mgr=None, last_result_frame=None):
    if frame_meta["fallback"]:
        return resolve_cached_fallback_frame(executor, frame, fallback_mgr, last_result_frame)

    result = frame
    for task_meta in frame_meta["tasks"]:
        fake_frame = input_cache[task_meta["cache_key"]]
        enhanced_frame = enhance_cache.get(task_meta["cache_key"]) if executor.enhancer_name is not None else None
        result = compose_mgr.compose_task(result, task_meta, fake_frame, enhanced_frame)
    return result


def compose_frames_from_cache_batch(executor, compose_mgr, frame_entries, input_cache, enhance_cache):
    if not frame_entries:
        return []
    if len(frame_entries) == 1 or not compose_mgr.should_use_gpu_compositor():
        return [
            compose_frame_from_cache(executor, compose_mgr, frame, frame_meta, input_cache, enhance_cache)
            for frame, frame_meta in frame_entries
        ]

    states = [
        {
            "result": frame,
            "tasks": list(frame_meta.get("tasks", [])),
        }
        for frame, frame_meta in frame_entries
    ]
    max_tasks = max((len(state["tasks"]) for state in states), default=0)
    for task_index in range(max_tasks):
        batch_items = []
        batch_state_indices = []
        for state_index, state in enumerate(states):
            if task_index >= len(state["tasks"]):
                continue
            task_meta = state["tasks"][task_index]
            cache_key = task_meta["cache_key"]
            batch_items.append(
                {
                    "base_frame": state["result"],
                    "task": task_meta,
                    "fake_frame": input_cache[cache_key],
                    "enhanced_frame": enhance_cache.get(cache_key) if executor.enhancer_name is not None else None,
                }
            )
            batch_state_indices.append(state_index)
        if not batch_items:
            continue
        composed_frames = compose_mgr.compose_tasks_batch(batch_items)
        for state_index, composed_frame in zip(batch_state_indices, composed_frames):
            states[state_index]["result"] = composed_frame
    return [state["result"] for state in states]


def ensure_full_compose_stage(executor, entry, endframe, fps, detect_dir, swap_dir, mask_dir, enhance_dir, intermediate_video, stages, manifest, memory_plan):
    frame_count = manifest.get("frame_count", max(endframe - entry.startframe, 1))
    composite_state = manifest.setdefault("composite_state", {})
    completed_frames = int(composite_state.get("completed_frames", 0) or 0)
    if stages["composite"] and intermediate_video.exists() and completed_frames >= frame_count:
        executor.completed_units += frame_count
        executor.update_progress("composite", detail="Reusing encoded composite video cache", step_completed=frame_count, step_total=frame_count, step_unit="frames", force_log=True)
        return
    if intermediate_video.exists():
        os.remove(str(intermediate_video))
    stages["composite"] = False
    composite_state["completed_frames"] = 0
    composite_state["frame_count"] = frame_count
    write_json(intermediate_video.parent / "manifest.json", manifest)
    compose_mgr = ProcessMgr(None)
    compose_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
    fallback_mgr = None
    last_result_frame = None
    processed_frames = 0
    last_progress_emit_at = 0.0
    progress_emit_frames = 8
    cap = open_video_capture(entry.filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    writer_config = resolve_video_writer_config(roop.config.globals.video_encoder, roop.config.globals.video_quality)
    writer = FFMPEG_VideoWriter(
        str(intermediate_video),
        (width, height),
        fps,
        codec=writer_config["codec"],
        crf=roop.config.globals.video_quality,
        ffmpeg_params=writer_config["ffmpeg_params"],
        quality_args=writer_config["quality_args"],
    )

    def load_pack_state(pack_data):
        if pack_data is None:
            return None, {}, {}, {}
        frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in pack_data["frames"]}
        task_keys = [
            task_meta["cache_key"]
            for frame_meta in pack_data["frames"]
            for task_meta in frame_meta.get("tasks", [])
        ]
        input_cache = executor.read_stage_cache_keys(
            executor.get_stage_pack_path(mask_dir if executor.mask_name else swap_dir, pack_data["start_sequence"], pack_data["end_sequence"]),
            task_keys,
        )
        enhance_cache = (
            executor.read_stage_cache_keys(
                executor.get_stage_pack_path(enhance_dir, pack_data["start_sequence"], pack_data["end_sequence"]),
                task_keys,
            )
            if executor.enhancer_name is not None
            else {}
        )
        return pack_data["frames"][-1]["frame_number"], frame_lookup, input_cache, enhance_cache

    try:
        from .video_iter import iter_video_chunk

        pack_iter = iter(executor.iter_detect_packs(detect_dir))
        current_pack = next(pack_iter, None)
        current_pack_end, frame_lookup, input_cache, enhance_cache = load_pack_state(current_pack)
        compose_worker_count = get_compose_worker_count(executor)
        compose_gpu_batch_size = get_compose_gpu_batch_size(executor)
        use_gpu_frame_batch = bool(getattr(compose_mgr, "should_use_gpu_compositor", lambda: False)()) and compose_gpu_batch_size > 1

        def emit_progress(force=False):
            nonlocal last_progress_emit_at
            if not force:
                if processed_frames <= 0:
                    return
                if processed_frames < frame_count and processed_frames % progress_emit_frames != 0:
                    now = time.time()
                    if (now - last_progress_emit_at) < 0.25:
                        return
            last_progress_emit_at = time.time()
            executor.update_progress("composite", detail=get_composite_progress_detail(), step_completed=processed_frames, step_total=frame_count, step_unit="frames")

        def cache_last_result_frame(result):
            nonlocal last_result_frame, fallback_mgr
            if result is None:
                return
            last_result_frame = result
            if fallback_mgr is not None:
                fallback_mgr.last_swapped_frame = result.copy()
                fallback_mgr.num_frames_no_face = 0

        def write_composed_frames(results):
            nonlocal processed_frames, last_result_frame, fallback_mgr
            valid_results = [result for result in results if result is not None]
            if valid_results:
                write_many = getattr(writer, "write_frames", None)
                if callable(write_many):
                    write_many(valid_results)
                else:
                    for result in valid_results:
                        writer.write_frame(result)
            for result in results:
                executor.completed_units += 1
                processed_frames += 1
                cache_last_result_frame(result)
            composite_state["completed_frames"] = processed_frames
            emit_progress(force=processed_frames >= frame_count)

        def write_composed_frame(result):
            write_composed_frames([result])

        def flush_gpu_batch(batch_entries):
            if not batch_entries:
                return
            results = compose_frames_from_cache_batch(executor, compose_mgr, batch_entries, input_cache, enhance_cache)
            write_composed_frames(results)
            batch_entries.clear()

        if use_gpu_frame_batch:
            batched_frames = []
            for frame_number, frame in iter_video_chunk(entry.filename, entry.startframe, endframe, memory_plan["prefetch_frames"]):
                while current_pack_end is not None and frame_number > current_pack_end:
                    flush_gpu_batch(batched_frames)
                    current_pack = next(pack_iter, None)
                    current_pack_end, frame_lookup, input_cache, enhance_cache = load_pack_state(current_pack)

                frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
                if frame_meta["fallback"]:
                    flush_gpu_batch(batched_frames)
                    if roop.config.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
                        fallback_mgr = prepare_fallback_mgr(executor, fallback_mgr, last_result_frame)
                    elif not should_use_cached_fallback_fast_path():
                        fallback_mgr = executor.get_fallback_mgr()
                    result = compose_frame_from_cache(
                        executor,
                        compose_mgr,
                        frame,
                        frame_meta,
                        input_cache,
                        enhance_cache,
                        fallback_mgr,
                        last_result_frame,
                    )
                    write_composed_frame(result)
                    continue

                batched_frames.append((frame, frame_meta))
                if len(batched_frames) >= compose_gpu_batch_size:
                    flush_gpu_batch(batched_frames)
            flush_gpu_batch(batched_frames)
        elif compose_worker_count == 1:
            for frame_number, frame in iter_video_chunk(entry.filename, entry.startframe, endframe, memory_plan["prefetch_frames"]):
                while current_pack_end is not None and frame_number > current_pack_end:
                    current_pack = next(pack_iter, None)
                    current_pack_end, frame_lookup, input_cache, enhance_cache = load_pack_state(current_pack)

                frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
                if frame_meta["fallback"]:
                    if roop.config.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
                        fallback_mgr = prepare_fallback_mgr(executor, fallback_mgr, last_result_frame)
                    elif not should_use_cached_fallback_fast_path():
                        fallback_mgr = executor.get_fallback_mgr()
                result = compose_frame_from_cache(
                    executor,
                    compose_mgr,
                    frame,
                    frame_meta,
                    input_cache,
                    enhance_cache,
                    fallback_mgr,
                    last_result_frame,
                )
                write_composed_frame(result)
        else:
            worker_state = local()
            worker_managers = []
            pending_futures = {}
            buffered_results = {}
            next_result_index = 0
            max_in_flight = max(compose_worker_count * 2, 1)

            def get_thread_compose_mgr():
                thread_compose_mgr = getattr(worker_state, "compose_mgr", None)
                if thread_compose_mgr is None:
                    thread_compose_mgr = ProcessMgr(None)
                    thread_compose_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
                    worker_state.compose_mgr = thread_compose_mgr
                    worker_managers.append(thread_compose_mgr)
                return thread_compose_mgr

            def compose_non_fallback_frame(frame, frame_meta, source_cache, enhanced_cache):
                thread_compose_mgr = get_thread_compose_mgr()
                return compose_frame_from_cache(executor, thread_compose_mgr, frame, frame_meta, source_cache, enhanced_cache)

            def flush_done_futures(done_futures):
                nonlocal next_result_index
                for done_future in done_futures:
                    frame_index = pending_futures.pop(done_future)
                    buffered_results[frame_index] = done_future.result()
                while next_result_index in buffered_results:
                    result = buffered_results.pop(next_result_index)
                    write_composed_frame(result)
                    next_result_index += 1

            def flush_all_pending():
                while pending_futures:
                    done_futures, _ = wait(set(pending_futures), return_when=FIRST_COMPLETED)
                    flush_done_futures(done_futures)

            with ThreadPoolExecutor(max_workers=compose_worker_count, thread_name_prefix="staged_compose") as pool:
                frame_index = 0
                for frame_number, frame in iter_video_chunk(entry.filename, entry.startframe, endframe, memory_plan["prefetch_frames"]):
                    while current_pack_end is not None and frame_number > current_pack_end:
                        current_pack = next(pack_iter, None)
                        current_pack_end, frame_lookup, input_cache, enhance_cache = load_pack_state(current_pack)

                    frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
                    if frame_meta["fallback"]:
                        flush_all_pending()
                        if roop.config.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
                            fallback_mgr = prepare_fallback_mgr(executor, fallback_mgr, last_result_frame)
                        elif not should_use_cached_fallback_fast_path():
                            fallback_mgr = executor.get_fallback_mgr()
                        result = compose_frame_from_cache(
                            executor,
                            compose_mgr,
                            frame,
                            frame_meta,
                            input_cache,
                            enhance_cache,
                            fallback_mgr,
                            last_result_frame,
                        )
                        write_composed_frame(result)
                        frame_index += 1
                        next_result_index = frame_index
                        continue

                    if len(pending_futures) >= max_in_flight:
                        done_futures, _ = wait(set(pending_futures), return_when=FIRST_COMPLETED)
                        flush_done_futures(done_futures)

                    future = pool.submit(compose_non_fallback_frame, frame, frame_meta, input_cache, enhance_cache)
                    pending_futures[future] = frame_index
                    frame_index += 1

                flush_all_pending()

            for worker_manager in worker_managers:
                worker_manager.release_resources()
        emit_progress(force=True)
    finally:
        compose_mgr.release_resources()
        writer.close()
    completed_successfully = roop.config.globals.processing and processed_frames >= frame_count and intermediate_video.exists()
    if not completed_successfully:
        stages["composite"] = False
        composite_state["completed_frames"] = processed_frames
        if intermediate_video.exists():
            intermediate_video.unlink(missing_ok=True)
    else:
        stages["composite"] = True
        composite_state["completed_frames"] = frame_count
    write_json(intermediate_video.parent / "manifest.json", manifest)


def ensure_full_encode_stage(executor, entry, index, intermediate_video, frame_count, endframe):
    destination = util.replace_template(entry.finalname, index=index)
    Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
    if util.has_extension(entry.filename, ["gif"]):
        executor.update_progress("mux", detail="Creating final GIF", step_completed=frame_count, step_total=frame_count, step_unit="frames", force_log=True)
        ffmpeg.create_gif_from_video(str(intermediate_video), destination)
    elif roop.config.globals.skip_audio:
        if os.path.isfile(destination):
            os.remove(destination)
        shutil.move(str(intermediate_video), destination)
    else:
        executor.update_progress("mux", detail="Restoring source audio", step_completed=frame_count, step_total=frame_count, step_unit="frames", force_log=True)
        ffmpeg.restore_audio(str(intermediate_video), entry.filename, entry.startframe, endframe, destination)
        if intermediate_video.exists() and os.path.isfile(destination):
            os.remove(str(intermediate_video))


def compose_image_from_cache(executor, image, job_dir, frame_meta):
    if frame_meta["fallback"]:
        fallback_mgr = executor.get_fallback_mgr() if not should_use_cached_fallback_fast_path() else None
        return resolve_cached_fallback_frame(executor, image, fallback_mgr)
    compose_mgr = ProcessMgr(None)
    compose_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
    task_keys = [task_meta["cache_key"] for task_meta in frame_meta["tasks"]]
    input_cache = executor.read_stage_cache_keys(executor.get_stage_cache_path(job_dir / ("mask" if executor.mask_name else "swap")), task_keys)
    enhance_cache = executor.read_stage_cache_keys(executor.get_stage_cache_path(job_dir / "enhance"), task_keys) if executor.enhancer_name is not None else {}
    result = image.copy()
    try:
        for task_meta in frame_meta["tasks"]:
            fake_frame = input_cache[task_meta["cache_key"]]
            enhanced_frame = enhance_cache.get(task_meta["cache_key"]) if executor.enhancer_name is not None else None
            result = compose_mgr.compose_task(result, task_meta, fake_frame, enhanced_frame)
    finally:
        compose_mgr.release_resources()
    return result


def compose_chunk(executor, entry, chunk_dir, chunk_meta, chunk_state, memory_plan, chunk_video):
    if chunk_state["stages"]["composite"] and chunk_video.exists():
        return
    compose_mgr = ProcessMgr(None)
    compose_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
    frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in chunk_meta["frames"]}
    task_keys = [task_meta["cache_key"] for frame_meta in chunk_meta["frames"] for task_meta in frame_meta.get("tasks", [])]
    input_cache = executor.read_stage_cache_keys(executor.get_stage_cache_path(chunk_dir / ("mask" if executor.mask_name else "swap")), task_keys)
    enhance_cache = executor.read_stage_cache_keys(executor.get_stage_cache_path(chunk_dir / "enhance"), task_keys) if executor.enhancer_name is not None else {}
    cap = open_video_capture(entry.filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_to_file = executor.output_method != "Virtual Camera"
    output_to_cam = executor.output_method in ("Virtual Camera", "Both") and StreamWriter is not None
    writer_config = resolve_video_writer_config(roop.config.globals.video_encoder, roop.config.globals.video_quality)
    writer = (
        FFMPEG_VideoWriter(
            str(chunk_video),
            (width, height),
            entry.fps or util.detect_fps(entry.filename),
            codec=writer_config["codec"],
            crf=roop.config.globals.video_quality,
            ffmpeg_params=writer_config["ffmpeg_params"],
            quality_args=writer_config["quality_args"],
        )
        if output_to_file
        else None
    )
    stream = StreamWriter((width, height), int(entry.fps or util.detect_fps(entry.filename))) if output_to_cam else None
    fallback_mgr = None
    last_result_frame = None

    try:
        from .video_iter import iter_video_chunk

        processed_frames = 0
        total_frames = max(chunk_meta["end"] - chunk_meta["start"], 1)
        for frame_number, frame in iter_video_chunk(entry.filename, chunk_meta["start"], chunk_meta["end"], memory_plan["prefetch_frames"]):
            frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
            if frame_meta["fallback"]:
                if roop.config.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
                    fallback_mgr = prepare_fallback_mgr(executor, fallback_mgr, last_result_frame)
                elif not should_use_cached_fallback_fast_path():
                    fallback_mgr = executor.get_fallback_mgr()
                result = resolve_cached_fallback_frame(executor, frame, fallback_mgr, last_result_frame)
            else:
                result = frame.copy()
                for task_meta in frame_meta["tasks"]:
                    fake_frame = input_cache[task_meta["cache_key"]]
                    enhanced_frame = enhance_cache.get(task_meta["cache_key"]) if executor.enhancer_name is not None else None
                    result = compose_mgr.compose_task(result, task_meta, fake_frame, enhanced_frame)
                fallback_mgr = executor.fallback_mgr
                if fallback_mgr is not None and result is not None:
                    fallback_mgr.last_swapped_frame = result.copy()
                    fallback_mgr.num_frames_no_face = 0
            if result is not None:
                if writer is not None:
                    writer.write_frame(result)
                if stream is not None:
                    stream.WriteToStream(result)
                last_result_frame = result.copy()
            executor.completed_units += 1
            processed_frames += 1
            executor.update_progress("composite", detail="Compositing output frames", step_completed=processed_frames, step_total=total_frames, step_unit="frames")
    finally:
        compose_mgr.release_resources()
        if writer is not None:
            writer.close()
        if stream is not None:
            stream.Close()
    chunk_state["stages"]["composite"] = True


def should_skip_completed_output(_executor, entry, manifest):
    return manifest.get("status") == "completed" and bool(getattr(entry, "finalname", None)) and os.path.isfile(entry.finalname)


__all__ = [
    "can_direct_encode_without_processing",
    "compose_chunk",
    "compose_frame_from_cache",
    "compose_frames_from_cache_batch",
    "compose_image_from_cache",
    "ensure_direct_video_output",
    "ensure_full_compose_stage",
    "ensure_full_encode_stage",
    "get_composite_progress_detail",
    "get_compose_gpu_batch_size",
    "get_compose_worker_count",
    "should_skip_completed_output",
]
