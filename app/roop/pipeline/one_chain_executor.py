import hashlib
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import local

import roop.config.globals
import roop.media.ffmpeg_ops as ffmpeg
import roop.utils as util
from roop.media.capturer import get_video_frame_total
from roop.media.video_io import open_video_capture
from roop.memory import provider_uses_gpu, resolve_single_batch_workers
from roop.pipeline.batch_executor import ProcessMgr
from roop.pipeline.staged_executor.cache import (
    AsyncWritePipeline,
    get_entry_file_identity,
    get_entry_job_relpath,
    get_staged_cache_options_snapshot,
    json_dumps,
    read_json,
    write_json,
)
from roop.pipeline.staged_executor.video_cache import VideoStageCache
from roop.pipeline.staged_executor.video_iter import iter_video_chunk
from roop.progress.status import get_processing_status_line, publish_processing_progress, set_processing_message, update_rate_window
from roop.utils.cache_paths import get_jobs_root


ONE_CHAIN_PIPELINE_VERSION = 1
ONE_CHAIN_MERGE_PROFILE = "concat_reencode_v3"


def get_one_chain_job_dir(entry, options, output_method):
    return get_jobs_root() / get_entry_job_relpath(entry, options) / "one_chain_all"


def get_one_chain_manifest_signature(entry, options, output_method):
    blob = {
        "pipeline_version": ONE_CHAIN_PIPELINE_VERSION,
        "method": "one_chain_all",
        "output_method": output_method,
        "options": get_staged_cache_options_snapshot(options),
        "file": get_entry_file_identity(entry),
    }
    return hashlib.sha256(json_dumps(blob).encode("utf-8")).hexdigest()


def get_one_chain_chunk_size():
    for candidate in (
        getattr(getattr(roop.config.globals, "CFG", None), "staged_chunk_size", None),
        getattr(getattr(roop.config.globals, "CFG", None), "detect_pack_frame_count", None),
    ):
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 128


def iter_one_chain_ranges(start_frame, end_frame, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for chunk_start in range(int(start_frame), int(end_frame), chunk_size):
        yield chunk_start, min(int(end_frame), chunk_start + chunk_size)


def get_one_chain_segment_key(start_frame, end_frame):
    return f"{int(start_frame):06d}_{int(end_frame) - 1:06d}"


def get_one_chain_segment_path(cache_dir, start_frame, end_frame):
    return Path(cache_dir) / f"{get_one_chain_segment_key(start_frame, end_frame)}.mp4"


def get_one_chain_frame_key(frame_number):
    return f"f{int(frame_number):08d}"


def get_one_chain_worker_count():
    try:
        return max(1, int(getattr(roop.config.globals, "execution_threads", 1) or 1))
    except (TypeError, ValueError):
        return 1


def get_one_chain_prefetch_frames():
    try:
        return max(1, int(getattr(getattr(roop.config.globals, "CFG", None), "prefetch_frames", 32) or 32))
    except (TypeError, ValueError):
        return 32


def _append_worker_reason(reason: str | None, extra_reason: str | None) -> str | None:
    if not extra_reason:
        return reason
    if not reason:
        return extra_reason
    return f"{reason}; {extra_reason}"


def resolve_one_chain_worker_config(options=None):
    requested_threads = get_one_chain_worker_count()
    effective_workers, _requested_workers, reason = resolve_single_batch_workers(requested_threads)
    processor_keys = set((getattr(options, "processors", None) or {}).keys())
    if provider_uses_gpu() and "faceswap" in processor_keys and effective_workers > 2:
        effective_workers = 2
        reason = _append_worker_reason(reason, "one-chain GPU face pipeline cap 2")
    prefetch_frames = get_one_chain_prefetch_frames()
    max_in_flight = max(prefetch_frames, requested_threads * 2, effective_workers)
    return {
        "requested_threads": requested_threads,
        "effective_workers": effective_workers,
        "reason": reason,
        "prefetch_frames": prefetch_frames,
        "max_in_flight": max_in_flight,
    }


class OneChainAllExecutor:
    def __init__(self, output_method, progress, options):
        self.output_method = output_method
        self.progress = progress
        self.options = options
        self.process_mgr = None
        self.completed_units = 0
        self.total_units = 0
        self.total_files = 0
        self.current_entry = None
        self.current_file_index = None
        self.current_chunk_index = None
        self.current_total_chunks = None
        self.current_stage = None
        self.stage_started_at = None
        self.rate_phase = None

    def _ensure_process_mgr(self):
        if self.process_mgr is None:
            self.process_mgr = ProcessMgr(self.progress)
            self.process_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, self.options)
        return self.process_mgr

    def release_resources(self):
        if self.process_mgr is not None:
            self.process_mgr.release_resources()
            self.process_mgr = None

    def count_total_units(self, files):
        total = 0
        for entry in files:
            if util.has_image_extension(entry.filename):
                total += 1
            else:
                endframe = entry.endframe or get_video_frame_total(entry.filename)
                total += max(endframe - entry.startframe, 1)
        self.total_units = max(total, 1)

    def get_pipeline_steps(self):
        if self.current_entry is not None and util.has_image_extension(self.current_entry.filename):
            return ["prepare", "one_chain"]
        steps = ["prepare", "one_chain", "encode"]
        if self.output_method != "Virtual Camera":
            steps.append("mux")
        return steps

    def get_stage_step_info(self, stage):
        steps = self.get_pipeline_steps()
        stage_alias = {
            "resume": "prepare",
            "images": "one_chain",
        }
        stage_key = stage_alias.get(stage, stage)
        if stage_key not in steps:
            if stage == "prepare":
                return 1, len(steps)
            return None, len(steps)
        return steps.index(stage_key) + 1, len(steps)

    def update_progress(
        self,
        stage,
        detail=None,
        step_completed=None,
        step_total=None,
        step_unit="items",
        force_log=False,
        rate_completed=None,
        rate_total=None,
        rate_unit=None,
        rate_label="processing",
        rate_enabled=None,
    ):
        if stage != self.current_stage or step_completed in (None, 0):
            self.current_stage = stage
            self.stage_started_at = time.time()
            self.rate_phase = None
            self._rate_samples = None
        stage_elapsed = None
        stage_rate = None
        stage_eta = None
        if self.stage_started_at is not None:
            stage_elapsed = max(time.time() - self.stage_started_at, 0.0)
        if rate_enabled is None:
            rate_enabled = step_completed is not None and step_completed > 0
        if rate_enabled:
            rate_phase = rate_label or "processing"
            if self.rate_phase != rate_phase:
                self.rate_phase = rate_phase
                self._rate_samples = None
            rate_completed_value = rate_completed if rate_completed is not None else step_completed
            rate_total_value = rate_total if rate_total is not None else step_total
            stage_rate = update_rate_window(self, rate_completed_value)
            if stage_rate is not None and rate_total_value is not None and rate_total_value > 0:
                stage_eta = max(rate_total_value - rate_completed_value, 0) / stage_rate
        else:
            self.rate_phase = None
            self._rate_samples = None
        target_name = self.current_entry.filename if self.current_entry is not None else None
        current_step, total_steps = self.get_stage_step_info(stage)
        publish_processing_progress(
            stage=stage,
            completed=self.completed_units,
            total=self.total_units,
            unit="units",
            target_name=target_name,
            file_index=self.current_file_index,
            total_files=self.total_files,
            chunk_index=self.current_chunk_index,
            total_chunks=self.current_total_chunks,
            current_step=current_step,
            total_steps=total_steps,
            step_completed=step_completed,
            step_total=step_total,
            step_unit=step_unit,
            rate=stage_rate,
            rate_label=rate_label if stage_rate is not None else None,
            rate_unit=rate_unit or step_unit,
            elapsed=stage_elapsed,
            eta=stage_eta,
            detail=detail,
            memory_status=roop.config.globals.runtime_memory_status,
            force_log=force_log,
        )
        if self.progress is not None:
            self.progress(
                (self.completed_units, self.total_units),
                desc=get_processing_status_line(),
                total=self.total_units,
                unit="units",
            )

    def _prepare_job(self, entry):
        job_dir = get_one_chain_job_dir(entry, self.options, self.output_method)
        manifest_path = job_dir / "manifest.json"
        cache_dir = job_dir / "processed_cache"
        merged_video = job_dir / "merged.mp4"
        signature = get_one_chain_manifest_signature(entry, self.options, self.output_method)

        if manifest_path.exists():
            try:
                manifest = read_json(manifest_path)
            except Exception:
                manifest = {}
        else:
            manifest = {}

        merge_profile_changed = manifest.get("merge_profile") != ONE_CHAIN_MERGE_PROFILE

        if manifest.get("signature") != signature:
            shutil.rmtree(job_dir, ignore_errors=True)
            manifest = {}
            merge_profile_changed = False

        job_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(job_dir / "source_frames", ignore_errors=True)
        shutil.rmtree(job_dir / "processed_frames", ignore_errors=True)
        if merge_profile_changed:
            merged_video.unlink(missing_ok=True)

        manifest.update(
            {
                "signature": signature,
                "status": manifest.get("status", "pending"),
                "process_complete": bool(manifest.get("process_complete", False)),
                "merge_complete": False if merge_profile_changed else bool(manifest.get("merge_complete", False)),
                "final_complete": False if merge_profile_changed else bool(manifest.get("final_complete", False)),
                "frame_count": int(manifest.get("frame_count", 0) or 0),
                "completed_frames": int(manifest.get("completed_frames", 0) or 0),
                "segments": dict(manifest.get("segments") or {}),
                "merge_profile": ONE_CHAIN_MERGE_PROFILE,
            }
        )
        write_json(manifest_path, manifest)
        return job_dir, manifest_path, manifest, cache_dir, merged_video

    def _process_images(self, imagefiles):
        if not imagefiles:
            return
        process_mgr = self._ensure_process_mgr()
        process_mgr.set_progress_context("one_chain_images", f"{len(imagefiles)} image(s)", unit="images")
        if imagefiles:
            self.update_progress("images", detail=f"Processing {len(imagefiles)} image(s) through one-chain-all", step_completed=0, step_total=len(imagefiles), step_unit="images", force_log=True)
        process_mgr.run_batch([entry.filename for entry in imagefiles], [entry.finalname for entry in imagefiles], roop.config.globals.execution_threads)
        processed_images = len(imagefiles)
        self.completed_units += processed_images
        self.update_progress("images", detail="Completed one-chain-all image processing", step_completed=processed_images, step_total=processed_images, step_unit="images", force_log=True)

    def _process_stream_to_cache(self, entry, index, total_files, cache_dir, manifest, manifest_path, fps):
        process_mgr = None
        worker_config = resolve_one_chain_worker_config(self.options)
        worker_count = worker_config["effective_workers"]
        requested_threads = worker_config["requested_threads"]
        prefetch_frames = worker_config["prefetch_frames"]
        max_in_flight = worker_config["max_in_flight"]
        if worker_count <= 1:
            process_mgr = self._ensure_process_mgr()
            process_mgr.set_progress_context("one_chain", entry.filename, index + 1, total_files, unit="frames")

        capture = open_video_capture(entry.filename)
        width = int(capture.get(3) or 0)
        height = int(capture.get(4) or 0)
        capture.release()

        chunk_ranges = list(iter_one_chain_ranges(entry.startframe, entry.endframe, get_one_chain_chunk_size()))
        expected_segment_keys = [get_one_chain_segment_key(start_frame, end_frame) for start_frame, end_frame in chunk_ranges]
        segment_states = manifest.setdefault("segments", {})
        for segment_path in Path(cache_dir).glob("*.mp4"):
            if segment_path.stem not in expected_segment_keys:
                segment_path.unlink(missing_ok=True)
                segment_path.with_suffix(".idx.bin").unlink(missing_ok=True)

        completed_frames = 0
        for start_frame, end_frame in chunk_ranges:
            segment_key = get_one_chain_segment_key(start_frame, end_frame)
            segment_path = get_one_chain_segment_path(cache_dir, start_frame, end_frame)
            segment_state = segment_states.get(segment_key) or {}
            segment_frame_count = max(0, end_frame - start_frame)
            if segment_path.exists() and segment_state.get("completed"):
                segment_state["frame_count"] = segment_frame_count
                completed_frames += segment_frame_count
            else:
                segment_path.unlink(missing_ok=True)
                segment_path.with_suffix(".idx.bin").unlink(missing_ok=True)
                segment_state = {"completed": False, "frame_count": segment_frame_count}
            segment_states[segment_key] = segment_state

        manifest["completed_frames"] = completed_frames
        manifest["frame_count"] = max(0, entry.endframe - entry.startframe)
        write_json(manifest_path, manifest)
        entry_progress_base = self.completed_units
        self.current_total_chunks = len(chunk_ranges)
        self.completed_units = entry_progress_base + completed_frames

        if completed_frames >= manifest["frame_count"] and all(
            (segment_states.get(get_one_chain_segment_key(start_frame, end_frame)) or {}).get("completed")
            for start_frame, end_frame in chunk_ranges
        ):
            manifest["process_complete"] = True
            write_json(manifest_path, manifest)
            self.update_progress("resume", detail="Reusing completed one-chain packed cache", step_completed=completed_frames, step_total=manifest["frame_count"], step_unit="frames", rate_enabled=False, force_log=True)
            return True

        set_processing_message(
            "Running one-chain-all frame processing",
            stage="one_chain",
            target_name=entry.filename,
            file_index=index + 1,
            total_files=total_files,
            current_step=1,
            total_steps=3,
            detail="Streaming source decode + full-chain processing into packed video cache",
            force_log=True,
        )
        manifest["status"] = "running"
        worker_detail = f"Streaming source decode + full-chain processing into packed video cache | GPU workers: {worker_count}/{requested_threads}"
        if worker_config["reason"]:
            worker_detail = f"{worker_detail} ({worker_config['reason']})"
        self.update_progress("one_chain", detail=worker_detail, step_completed=completed_frames, step_total=manifest["frame_count"], step_unit="frames", rate_enabled=False, force_log=True)
        stage_cache = VideoStageCache(
            codec="auto",
            crf=0,
            preset="veryfast",
            max_frame_extent=max(width, height),
            fps=max(1, round(fps)),
        )
        processed_frames = completed_frames
        initial_completed_frames = completed_frames
        computed_frames = 0
        committed_frames = completed_frames
        worker_pool = None
        worker_state = local()
        worker_managers = []
        cache_writer = AsyncWritePipeline("one_chain_cache_write")

        def get_thread_process_mgr():
            thread_process_mgr = getattr(worker_state, "process_mgr", None)
            if thread_process_mgr is None:
                thread_process_mgr = ProcessMgr(None)
                thread_process_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, self.options)
                thread_process_mgr.set_progress_context("one_chain", entry.filename, index + 1, total_files, unit="frames")
                worker_state.process_mgr = thread_process_mgr
                worker_managers.append(thread_process_mgr)
            return thread_process_mgr

        def process_one_frame(frame_number, frame):
            thread_process_mgr = get_thread_process_mgr()
            result = thread_process_mgr.process_frame(frame)
            if result is None:
                result = frame
            return frame_number, result

        if worker_count > 1:
            worker_pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="one_chain")

        def process_segment_frames(segment_start, segment_end):
            if worker_pool is None:
                frame_cache = {}
                frames_seen = 0
                for frame_number, frame in iter_video_chunk(entry.filename, segment_start, segment_end, prefetch_frames):
                    result = process_mgr.process_frame(frame)
                    if result is None:
                        result = frame
                    frame_cache[get_one_chain_frame_key(frame_number)] = result
                    frames_seen += 1
                    yield frames_seen, frame_cache
                return

            pending_futures = {}
            buffered_results = {}
            frame_cache = {}
            frames_seen = 0
            next_result_frame = segment_start

            def flush_done_futures(done_futures):
                nonlocal frames_seen, next_result_frame
                for done_future in done_futures:
                    future_frame_number = pending_futures.pop(done_future)
                    buffered_results[future_frame_number] = done_future.result()[1]
                while next_result_frame in buffered_results:
                    frame_cache[get_one_chain_frame_key(next_result_frame)] = buffered_results.pop(next_result_frame)
                    frames_seen += 1
                    next_result_frame += 1
                    yield frames_seen, frame_cache

            for frame_number, frame in iter_video_chunk(entry.filename, segment_start, segment_end, prefetch_frames):
                while len(pending_futures) >= max_in_flight:
                    done_futures, _ = wait(set(pending_futures), return_when=FIRST_COMPLETED)
                    for progress_update in flush_done_futures(done_futures):
                        yield progress_update
                pending_futures[worker_pool.submit(process_one_frame, frame_number, frame)] = frame_number
            while pending_futures:
                done_futures, _ = wait(set(pending_futures), return_when=FIRST_COMPLETED)
                for progress_update in flush_done_futures(done_futures):
                    yield progress_update

        try:
            for chunk_index, (start_frame, end_frame) in enumerate(chunk_ranges, start=1):
                segment_key = get_one_chain_segment_key(start_frame, end_frame)
                segment_path = get_one_chain_segment_path(cache_dir, start_frame, end_frame)
                segment_state = segment_states.setdefault(segment_key, {})
                segment_frame_count = max(0, end_frame - start_frame)
                self.current_chunk_index = chunk_index
                if segment_path.exists() and segment_state.get("completed"):
                    self.completed_units = entry_progress_base + processed_frames
                    self.update_progress("resume", detail=f"Reusing one-chain cache segment {segment_key}", step_completed=processed_frames, step_total=manifest["frame_count"], step_unit="frames", rate_enabled=False, force_log=True)
                    continue

                frame_cache = {}
                frames_seen = 0
                checkpoint_completed_frames = processed_frames
                for frames_seen, frame_cache in process_segment_frames(start_frame, end_frame):
                    processed_frames += 1
                    computed_frames += 1
                    self.completed_units = entry_progress_base + processed_frames
                    self.update_progress("one_chain", detail="Streaming source decode + full-chain processing into packed video cache", step_completed=processed_frames, step_total=manifest["frame_count"], step_unit="frames", rate_completed=computed_frames, rate_total=max(manifest["frame_count"] - initial_completed_frames, computed_frames))

                if not roop.config.globals.processing or frames_seen < segment_frame_count:
                    manifest["status"] = "interrupted"
                    manifest["process_complete"] = False
                    processed_frames = checkpoint_completed_frames
                    self.completed_units = entry_progress_base + checkpoint_completed_frames
                    manifest["completed_frames"] = committed_frames
                    segment_state["completed"] = False
                    segment_path.unlink(missing_ok=True)
                    segment_path.with_suffix(".idx.bin").unlink(missing_ok=True)
                    write_json(manifest_path, manifest)
                    self.update_progress("one_chain", detail=f"Interrupted while processing one-chain cache segment {segment_key}", step_completed=checkpoint_completed_frames, step_total=manifest["frame_count"], step_unit="frames", rate_enabled=False, force_log=True)
                    return False

                def finalize_segment_write(
                    current_segment_state=segment_state,
                    current_segment_frame_count=segment_frame_count,
                ):
                    nonlocal committed_frames
                    current_segment_state["completed"] = True
                    current_segment_state["frame_count"] = current_segment_frame_count
                    committed_frames += current_segment_frame_count
                    manifest["completed_frames"] = committed_frames
                    write_json(manifest_path, manifest)

                cache_writer.submit(stage_cache.write, segment_path, frame_cache, on_complete=finalize_segment_write)
                self.completed_units = entry_progress_base + processed_frames
                self.update_progress("one_chain", detail=f"Completed one-chain cache segment {segment_key}", step_completed=processed_frames, step_total=manifest["frame_count"], step_unit="frames", rate_completed=computed_frames, rate_total=max(manifest["frame_count"] - initial_completed_frames, computed_frames), force_log=True)

            cache_writer.close()
            cache_writer = None
            manifest["process_complete"] = True
            manifest["completed_frames"] = committed_frames
            write_json(manifest_path, manifest)
            self.completed_units = entry_progress_base + processed_frames
            self.update_progress("one_chain", detail="Completed one-chain packed cache generation", step_completed=processed_frames, step_total=manifest["frame_count"], step_unit="frames", rate_completed=computed_frames, rate_total=max(manifest["frame_count"] - initial_completed_frames, computed_frames), force_log=True)
            return True
        finally:
            if cache_writer is not None:
                cache_writer.close()
            if worker_pool is not None:
                worker_pool.shutdown(wait=True)
            for worker_manager in worker_managers:
                worker_manager.release_resources()

    def _merge_processed_segments(self, entry, index, total_files, cache_dir, merged_video, manifest, manifest_path):
        if manifest.get("merge_complete") and merged_video.exists():
            total_segments = max(len(list(iter_one_chain_ranges(entry.startframe, entry.endframe, get_one_chain_chunk_size()))), 1)
            self.update_progress("encode", detail="Reusing merged one-chain video cache", step_completed=total_segments, step_total=total_segments, step_unit="segments", rate_enabled=False, force_log=True)
            return True

        merged_video.unlink(missing_ok=True)
        segment_paths = []
        for start_frame, end_frame in iter_one_chain_ranges(entry.startframe, entry.endframe, get_one_chain_chunk_size()):
            segment_key = get_one_chain_segment_key(start_frame, end_frame)
            segment_state = (manifest.get("segments") or {}).get(segment_key) or {}
            segment_path = get_one_chain_segment_path(cache_dir, start_frame, end_frame)
            if segment_state.get("completed") and segment_path.exists():
                segment_paths.append(str(segment_path))
        if not segment_paths:
            manifest["merge_complete"] = False
            manifest["status"] = "interrupted"
            write_json(manifest_path, manifest)
            return False

        set_processing_message(
            "Merging processed frames",
            stage="encode",
            target_name=entry.filename,
            file_index=index + 1,
            total_files=total_files,
            current_step=2,
            total_steps=3,
            detail="Joining packed processed-cache segments and compressing the final merged video",
            force_log=True,
        )
        total_segments = max(len(segment_paths), 1)
        self.update_progress("encode", detail="Joining packed processed-cache segments and compressing the final merged video", step_completed=0, step_total=total_segments, step_unit="segments", rate_enabled=False, force_log=True)
        ffmpeg.join_videos(segment_paths, str(merged_video), True, reencode=True)
        merged_ok = merged_video.exists()
        manifest["merge_complete"] = merged_ok
        manifest["status"] = "interrupted" if not merged_ok else manifest.get("status", "running")
        write_json(manifest_path, manifest)
        self.update_progress("encode", detail="Merged one-chain processed cache segments into compressed final video", step_completed=total_segments if merged_ok else 0, step_total=total_segments, step_unit="segments", rate_enabled=False, force_log=True)
        return merged_ok

    def _finalize_output(self, entry, index, total_files, merged_video, manifest, manifest_path):
        destination = util.replace_template(entry.finalname, index=index)
        Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
        if manifest.get("final_complete") and os.path.isfile(destination):
            self.update_progress("mux", detail="Reusing finalized one-chain output", step_completed=1, step_total=1, step_unit="output", rate_enabled=False, force_log=True)
            return True

        set_processing_message(
            "Finalizing output",
            stage="mux",
            target_name=entry.filename,
            file_index=index + 1,
            total_files=total_files,
            current_step=3,
            total_steps=3,
            detail="Muxing encoded video with the final container/output",
            force_log=True,
        )
        self.update_progress("mux", detail="Muxing encoded video with the final container/output", step_completed=0, step_total=1, step_unit="output", rate_enabled=False, force_log=True)
        if util.has_extension(entry.filename, ["gif"]):
            ffmpeg.create_gif_from_video(str(merged_video), destination)
        elif roop.config.globals.skip_audio:
            if os.path.isfile(destination):
                os.remove(destination)
            shutil.copyfile(str(merged_video), destination)
        else:
            ffmpeg.restore_audio(str(merged_video), entry.filename, entry.startframe, entry.endframe, destination)
        manifest["final_complete"] = os.path.isfile(destination)
        manifest["status"] = "completed" if manifest["final_complete"] else "interrupted"
        write_json(manifest_path, manifest)
        self.update_progress("mux", detail="Finished final one-chain output", step_completed=1 if manifest["final_complete"] else 0, step_total=1, step_unit="output", rate_enabled=False, force_log=True)
        return manifest["final_complete"]

    def _process_video_entry(self, entry, index, total_files):
        if entry.endframe == 0:
            entry.endframe = get_video_frame_total(entry.filename)
        fps = entry.fps if entry.fps > 0 else util.detect_fps(entry.filename)
        _job_dir, manifest_path, manifest, cache_dir, merged_video = self._prepare_job(entry)

        destination = util.replace_template(entry.finalname, index=index)
        if manifest.get("final_complete") and os.path.isfile(destination):
            frame_count = max(entry.endframe - entry.startframe, 1)
            self.completed_units += frame_count
            set_processing_message(
                "Reusing one-chain-all output cache",
                stage="mux",
                target_name=entry.filename,
                file_index=index + 1,
                total_files=total_files,
                current_step=3,
                total_steps=3,
                detail="Final output already exists for this job",
                force_log=True,
            )
            self.update_progress("resume", detail="Skipping completed one-chain output", step_completed=frame_count, step_total=frame_count, step_unit="frames", rate_enabled=False, force_log=True)
            return

        if not self._process_stream_to_cache(entry, index, total_files, cache_dir, manifest, manifest_path, fps):
            return
        if not self._merge_processed_segments(entry, index, total_files, cache_dir, merged_video, manifest, manifest_path):
            return
        self._finalize_output(entry, index, total_files, merged_video, manifest, manifest_path)

    def run(self, files):
        imagefiles = []
        videofiles = []
        self.count_total_units(files)
        self.total_files = len(files)
        for original_index, entry in enumerate(files):
            if util.has_image_extension(entry.filename):
                imagefiles.append(entry)
            elif util.is_video(entry.filename) or util.has_extension(entry.filename, ["gif"]):
                videofiles.append((original_index, entry))

        try:
            if files:
                self.update_progress("prepare", detail="Preparing one-chain-all job queue", step_completed=0, step_total=len(files), step_unit="files", force_log=True)
            self._process_images(imagefiles)
            for original_index, entry in videofiles:
                if not roop.config.globals.processing:
                    break
                self.current_entry = entry
                self.current_file_index = original_index + 1
                self.current_chunk_index = None
                self.current_total_chunks = None
                self._process_video_entry(entry, original_index, len(files))
        finally:
            self.release_resources()


__all__ = [
    "ONE_CHAIN_PIPELINE_VERSION",
    "OneChainAllExecutor",
    "get_one_chain_job_dir",
    "get_one_chain_chunk_size",
    "get_one_chain_frame_key",
    "get_one_chain_manifest_signature",
    "get_one_chain_prefetch_frames",
    "get_one_chain_segment_key",
    "get_one_chain_segment_path",
    "get_one_chain_worker_count",
    "iter_one_chain_ranges",
    "resolve_one_chain_worker_config",
]
