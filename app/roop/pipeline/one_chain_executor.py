import hashlib
import os
import shutil
from pathlib import Path

import roop.config.globals
import roop.media.ffmpeg_ops as ffmpeg
import roop.utils as util
from roop.media.capturer import get_video_frame_total
from roop.media.video_io import open_video_capture
from roop.pipeline.batch_executor import ProcessMgr
from roop.pipeline.staged_executor.cache import (
    get_entry_file_identity,
    get_entry_job_relpath,
    get_staged_cache_options_snapshot,
    json_dumps,
    read_json,
    write_json,
)
from roop.pipeline.staged_executor.video_cache import VideoStageCache
from roop.pipeline.staged_executor.video_iter import iter_video_chunk
from roop.progress.status import set_processing_message
from roop.utils.cache_paths import get_jobs_root


ONE_CHAIN_PIPELINE_VERSION = 1


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


class OneChainAllExecutor:
    def __init__(self, output_method, progress, options):
        self.output_method = output_method
        self.progress = progress
        self.options = options
        self.process_mgr = None

    def _ensure_process_mgr(self):
        if self.process_mgr is None:
            self.process_mgr = ProcessMgr(self.progress)
            self.process_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, self.options)
        return self.process_mgr

    def release_resources(self):
        if self.process_mgr is not None:
            self.process_mgr.release_resources()
            self.process_mgr = None

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

        if manifest.get("signature") != signature:
            shutil.rmtree(job_dir, ignore_errors=True)
            manifest = {}

        job_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(job_dir / "source_frames", ignore_errors=True)
        shutil.rmtree(job_dir / "processed_frames", ignore_errors=True)

        manifest.update(
            {
                "signature": signature,
                "status": manifest.get("status", "pending"),
                "process_complete": bool(manifest.get("process_complete", False)),
                "merge_complete": bool(manifest.get("merge_complete", False)),
                "final_complete": bool(manifest.get("final_complete", False)),
                "frame_count": int(manifest.get("frame_count", 0) or 0),
                "completed_frames": int(manifest.get("completed_frames", 0) or 0),
                "segments": dict(manifest.get("segments") or {}),
            }
        )
        write_json(manifest_path, manifest)
        return job_dir, manifest_path, manifest, cache_dir, merged_video

    def _process_images(self, imagefiles):
        if not imagefiles:
            return
        process_mgr = self._ensure_process_mgr()
        process_mgr.set_progress_context("one_chain_images", f"{len(imagefiles)} image(s)", unit="images")
        process_mgr.run_batch([entry.filename for entry in imagefiles], [entry.finalname for entry in imagefiles], roop.config.globals.execution_threads)

    def _process_stream_to_cache(self, entry, index, total_files, cache_dir, manifest, manifest_path, fps):
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

        if completed_frames >= manifest["frame_count"] and all(
            (segment_states.get(get_one_chain_segment_key(start_frame, end_frame)) or {}).get("completed")
            for start_frame, end_frame in chunk_ranges
        ):
            manifest["process_complete"] = True
            write_json(manifest_path, manifest)
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
        stage_cache = VideoStageCache(
            codec="auto",
            crf=0,
            preset="veryfast",
            max_frame_extent=max(width, height),
            fps=max(1, round(fps)),
        )
        processed_frames = completed_frames

        for start_frame, end_frame in chunk_ranges:
            segment_key = get_one_chain_segment_key(start_frame, end_frame)
            segment_path = get_one_chain_segment_path(cache_dir, start_frame, end_frame)
            segment_state = segment_states.setdefault(segment_key, {})
            segment_frame_count = max(0, end_frame - start_frame)
            if segment_path.exists() and segment_state.get("completed"):
                continue

            frame_cache = {}
            frames_seen = 0
            checkpoint_completed_frames = processed_frames
            for frame_number, frame in iter_video_chunk(entry.filename, start_frame, end_frame, getattr(roop.config.globals.CFG, "prefetch_frames", 32)):
                result = process_mgr.process_frame(frame)
                if result is None:
                    result = frame
                frame_cache[get_one_chain_frame_key(frame_number)] = result
                frames_seen += 1
                processed_frames += 1

            if not roop.config.globals.processing or frames_seen < segment_frame_count:
                manifest["status"] = "interrupted"
                manifest["process_complete"] = False
                processed_frames = checkpoint_completed_frames
                manifest["completed_frames"] = checkpoint_completed_frames
                segment_state["completed"] = False
                segment_path.unlink(missing_ok=True)
                segment_path.with_suffix(".idx.bin").unlink(missing_ok=True)
                write_json(manifest_path, manifest)
                return False

            stage_cache.write(segment_path, frame_cache)
            segment_state["completed"] = True
            segment_state["frame_count"] = segment_frame_count
            manifest["completed_frames"] = processed_frames
            write_json(manifest_path, manifest)

        manifest["process_complete"] = True
        write_json(manifest_path, manifest)
        return True

    def _merge_processed_segments(self, entry, index, total_files, cache_dir, merged_video, manifest, manifest_path):
        if manifest.get("merge_complete") and merged_video.exists():
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
            detail="Joining packed processed-cache segments into a single video",
            force_log=True,
        )
        if len(segment_paths) == 1:
            shutil.copyfile(segment_paths[0], str(merged_video))
        else:
            ffmpeg.join_videos(segment_paths, str(merged_video), True)
        merged_ok = merged_video.exists()
        manifest["merge_complete"] = merged_ok
        manifest["status"] = "interrupted" if not merged_ok else manifest.get("status", "running")
        write_json(manifest_path, manifest)
        return merged_ok

    def _finalize_output(self, entry, index, total_files, merged_video, manifest, manifest_path):
        destination = util.replace_template(entry.finalname, index=index)
        Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
        if manifest.get("final_complete") and os.path.isfile(destination):
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
        return manifest["final_complete"]

    def _process_video_entry(self, entry, index, total_files):
        if entry.endframe == 0:
            entry.endframe = get_video_frame_total(entry.filename)
        fps = entry.fps if entry.fps > 0 else util.detect_fps(entry.filename)
        job_dir, manifest_path, manifest, cache_dir, merged_video = self._prepare_job(entry)

        destination = util.replace_template(entry.finalname, index=index)
        if manifest.get("final_complete") and os.path.isfile(destination):
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
            return

        if not self._process_stream_to_cache(entry, index, total_files, cache_dir, manifest, manifest_path, fps):
            return
        if not self._merge_processed_segments(entry, index, total_files, cache_dir, merged_video, manifest, manifest_path):
            return
        self._finalize_output(entry, index, total_files, merged_video, manifest, manifest_path)

    def run(self, files):
        imagefiles = []
        videofiles = []
        for entry in files:
            if util.has_image_extension(entry.filename):
                imagefiles.append(entry)
            elif util.is_video(entry.filename) or util.has_extension(entry.filename, ["gif"]):
                videofiles.append(entry)

        try:
            self._process_images(imagefiles)
            for index, entry in enumerate(videofiles):
                if not roop.config.globals.processing:
                    break
                self._process_video_entry(entry, index, len(videofiles))
        finally:
            self.release_resources()


__all__ = [
    "ONE_CHAIN_PIPELINE_VERSION",
    "OneChainAllExecutor",
    "get_one_chain_job_dir",
    "get_one_chain_chunk_size",
    "get_one_chain_frame_key",
    "get_one_chain_manifest_signature",
    "get_one_chain_segment_key",
    "get_one_chain_segment_path",
    "iter_one_chain_ranges",
]
