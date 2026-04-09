import hashlib
import os
import shutil
from pathlib import Path

import roop.config.globals
import roop.media.ffmpeg_ops as ffmpeg
import roop.utils as util
from roop.media.capturer import get_video_frame_total
from roop.pipeline.batch_executor import ProcessMgr
from roop.pipeline.staged_executor.cache import (
    get_entry_file_identity,
    get_entry_job_relpath,
    get_staged_cache_options_snapshot,
    json_dumps,
    read_json,
    write_json,
)
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


def list_extracted_frames(frame_dir):
    image_format = getattr(roop.config.globals.CFG, "output_image_format", "png")
    return sorted(Path(frame_dir).glob(f"*.{image_format}"))


def build_processed_frame_path(processed_dir, source_frame_path):
    return Path(processed_dir) / Path(source_frame_path).name


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
        extract_dir = job_dir / "source_frames"
        processed_dir = job_dir / "processed_frames"
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
        extract_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        manifest.update(
            {
                "signature": signature,
                "status": manifest.get("status", "pending"),
                "extract_complete": bool(manifest.get("extract_complete", False)),
                "process_complete": bool(manifest.get("process_complete", False)),
                "merge_complete": bool(manifest.get("merge_complete", False)),
                "final_complete": bool(manifest.get("final_complete", False)),
                "frame_count": int(manifest.get("frame_count", 0) or 0),
            }
        )
        write_json(manifest_path, manifest)
        return job_dir, manifest_path, manifest, extract_dir, processed_dir, merged_video

    def _process_images(self, imagefiles):
        if not imagefiles:
            return
        process_mgr = self._ensure_process_mgr()
        process_mgr.set_progress_context("one_chain_images", f"{len(imagefiles)} image(s)", unit="images")
        process_mgr.run_batch([entry.filename for entry in imagefiles], [entry.finalname for entry in imagefiles], roop.config.globals.execution_threads)

    def _extract_frames(self, entry, index, total_files, extract_dir, manifest, manifest_path, fps):
        existing_frames = list_extracted_frames(extract_dir)
        if manifest.get("extract_complete") and existing_frames:
            manifest["frame_count"] = len(existing_frames)
            write_json(manifest_path, manifest)
            return existing_frames

        shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        set_processing_message(
            "Extracting frames",
            stage="extract",
            target_name=entry.filename,
            file_index=index + 1,
            total_files=total_files,
            current_step=1,
            total_steps=4,
            detail="One-chain-all: extracting source frames to cache",
            force_log=True,
        )
        ffmpeg.extract_frames(
            entry.filename,
            entry.startframe,
            entry.endframe,
            fps,
            temp_directory_path=str(extract_dir),
        )
        if not roop.config.globals.processing:
            manifest["status"] = "interrupted"
            manifest["extract_complete"] = False
            write_json(manifest_path, manifest)
            return []
        extracted_frames = list_extracted_frames(extract_dir)
        manifest["extract_complete"] = True
        manifest["frame_count"] = len(extracted_frames)
        write_json(manifest_path, manifest)
        return extracted_frames

    def _process_pending_frames(self, entry, index, total_files, extracted_frames, processed_dir, manifest, manifest_path):
        pending_sources = []
        pending_targets = []
        for extracted_frame in extracted_frames:
            processed_frame = build_processed_frame_path(processed_dir, extracted_frame)
            if not processed_frame.exists():
                pending_sources.append(str(extracted_frame))
                pending_targets.append(str(processed_frame))

        if not pending_sources:
            manifest["process_complete"] = True
            write_json(manifest_path, manifest)
            return True

        process_mgr = self._ensure_process_mgr()
        process_mgr.set_progress_context("one_chain", entry.filename, index + 1, total_files, unit="frames")
        set_processing_message(
            "Running one-chain-all frame processing",
            stage="one_chain",
            target_name=entry.filename,
            file_index=index + 1,
            total_files=total_files,
            current_step=2,
            total_steps=4,
            detail="Processing pending extracted frames through the full chain",
            force_log=True,
        )
        process_mgr.run_batch(pending_sources, pending_targets, roop.config.globals.execution_threads)
        if not roop.config.globals.processing:
            manifest["status"] = "interrupted"
            manifest["process_complete"] = False
            write_json(manifest_path, manifest)
            return False

        remaining = [
            extracted_frame
            for extracted_frame in extracted_frames
            if not build_processed_frame_path(processed_dir, extracted_frame).exists()
        ]
        manifest["process_complete"] = not remaining
        write_json(manifest_path, manifest)
        return not remaining

    def _merge_processed_frames(self, entry, index, total_files, processed_dir, merged_video, manifest, manifest_path, fps):
        if manifest.get("merge_complete") and merged_video.exists():
            return True

        merged_video.unlink(missing_ok=True)
        set_processing_message(
            "Merging processed frames",
            stage="encode",
            target_name=entry.filename,
            file_index=index + 1,
            total_files=total_files,
            current_step=3,
            total_steps=4,
            detail="Encoding processed frame cache into a video",
            force_log=True,
        )
        ffmpeg.create_video(entry.filename, str(merged_video), fps, temp_directory_path=str(processed_dir))
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
            current_step=4,
            total_steps=4,
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
        job_dir, manifest_path, manifest, extract_dir, processed_dir, merged_video = self._prepare_job(entry)

        destination = util.replace_template(entry.finalname, index=index)
        if manifest.get("final_complete") and os.path.isfile(destination):
            set_processing_message(
                "Reusing one-chain-all output cache",
                stage="mux",
                target_name=entry.filename,
                file_index=index + 1,
                total_files=total_files,
                current_step=4,
                total_steps=4,
                detail="Final output already exists for this job",
                force_log=True,
            )
            return

        extracted_frames = self._extract_frames(entry, index, total_files, extract_dir, manifest, manifest_path, fps)
        if not extracted_frames:
            return
        if not self._process_pending_frames(entry, index, total_files, extracted_frames, processed_dir, manifest, manifest_path):
            return
        if not self._merge_processed_frames(entry, index, total_files, processed_dir, merged_video, manifest, manifest_path, fps):
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
    "build_processed_frame_path",
    "get_one_chain_job_dir",
    "get_one_chain_manifest_signature",
    "list_extracted_frames",
]
