import os
import shutil
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


def compose_frame_from_cache(executor, compose_mgr, frame, frame_meta, input_cache, enhance_cache, fallback_mgr=None):
    if frame_meta["fallback"]:
        if fallback_mgr is None:
            fallback_mgr = executor.get_fallback_mgr()
        return fallback_mgr.process_frame(frame)

    result = frame.copy()
    for task_meta in frame_meta["tasks"]:
        fake_frame = input_cache[task_meta["cache_key"]]
        enhanced_frame = enhance_cache.get(task_meta["cache_key"]) if executor.enhancer_name is not None else None
        result = compose_mgr.compose_task(result, task_meta, fake_frame, enhanced_frame)
    if fallback_mgr is not None and result is not None:
        fallback_mgr.last_swapped_frame = result.copy()
        fallback_mgr.num_frames_no_face = 0
    return result


def ensure_full_compose_stage(executor, entry, endframe, fps, detect_dir, swap_dir, mask_dir, enhance_dir, intermediate_video, stages, manifest, memory_plan):
    frame_count = manifest.get("frame_count", max(endframe - entry.startframe, 1))
    if stages["composite"] and intermediate_video.exists():
        executor.completed_units += frame_count
        executor.update_progress("composite", detail="Reusing encoded composite video cache", step_completed=frame_count, step_total=frame_count, step_unit="frames", force_log=True)
        return
    if intermediate_video.exists():
        os.remove(str(intermediate_video))
    compose_mgr = ProcessMgr(None)
    compose_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
    fallback_mgr = None
    last_result_frame = None
    processed_frames = 0
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
        input_cache = executor.read_stage_cache_map(
            executor.get_stage_pack_path(mask_dir if executor.mask_name else swap_dir, pack_data["start_sequence"], pack_data["end_sequence"])
        )
        enhance_cache = (
            executor.read_stage_cache_map(executor.get_stage_pack_path(enhance_dir, pack_data["start_sequence"], pack_data["end_sequence"]))
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

        def write_composed_frame(result):
            nonlocal processed_frames, last_result_frame, fallback_mgr
            if result is not None:
                writer.write_frame(result)
                last_result_frame = result.copy()
            if fallback_mgr is not None and result is not None:
                fallback_mgr.last_swapped_frame = result.copy()
                fallback_mgr.num_frames_no_face = 0
            executor.completed_units += 1
            processed_frames += 1
            executor.update_progress("composite", detail="Streaming source decode + direct video encode", step_completed=processed_frames, step_total=frame_count, step_unit="frames")

        if compose_worker_count == 1:
            for frame_number, frame in iter_video_chunk(entry.filename, entry.startframe, endframe, memory_plan["prefetch_frames"]):
                while current_pack_end is not None and frame_number > current_pack_end:
                    current_pack = next(pack_iter, None)
                    current_pack_end, frame_lookup, input_cache, enhance_cache = load_pack_state(current_pack)

                frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
                if frame_meta["fallback"]:
                    fallback_mgr = executor.get_fallback_mgr()
                result = compose_frame_from_cache(executor, compose_mgr, frame, frame_meta, input_cache, enhance_cache, fallback_mgr)
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
                        fallback_mgr = executor.get_fallback_mgr()
                        if fallback_mgr.last_swapped_frame is None and last_result_frame is not None:
                            fallback_mgr.last_swapped_frame = last_result_frame.copy()
                            fallback_mgr.num_frames_no_face = 0
                        result = compose_frame_from_cache(executor, compose_mgr, frame, frame_meta, input_cache, enhance_cache, fallback_mgr)
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
    finally:
        compose_mgr.release_resources()
        writer.close()
    stages["composite"] = intermediate_video.exists()
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
        return executor.get_fallback_mgr().process_frame(image)
    compose_mgr = ProcessMgr(None)
    compose_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
    input_cache = executor.read_stage_cache_map(executor.get_stage_cache_path(job_dir / ("mask" if executor.mask_name else "swap")))
    enhance_cache = executor.read_stage_cache_map(executor.get_stage_cache_path(job_dir / "enhance")) if executor.enhancer_name is not None else {}
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
    input_cache = executor.read_stage_cache_map(executor.get_stage_cache_path(chunk_dir / ("mask" if executor.mask_name else "swap")))
    enhance_cache = executor.read_stage_cache_map(executor.get_stage_cache_path(chunk_dir / "enhance")) if executor.enhancer_name is not None else {}
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

    try:
        from .video_iter import iter_video_chunk

        processed_frames = 0
        total_frames = max(chunk_meta["end"] - chunk_meta["start"], 1)
        for frame_number, frame in iter_video_chunk(entry.filename, chunk_meta["start"], chunk_meta["end"], memory_plan["prefetch_frames"]):
            frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
            if frame_meta["fallback"]:
                fallback_mgr = executor.get_fallback_mgr()
                result = fallback_mgr.process_frame(frame)
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
    "compose_chunk",
    "compose_frame_from_cache",
    "compose_image_from_cache",
    "ensure_full_compose_stage",
    "ensure_full_encode_stage",
    "get_compose_worker_count",
    "should_skip_completed_output",
]
