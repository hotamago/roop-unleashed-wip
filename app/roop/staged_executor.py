import hashlib
import json
import os
import pickle
import queue
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

import roop.globals
import roop.util_ffmpeg as ffmpeg
import roop.utilities as util
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.capturer import get_image_frame, get_video_frame_total
from roop.ffmpeg_writer import FFMPEG_VideoWriter
from roop.memory import describe_memory_plan, resolve_memory_plan
from roop.progress_status import get_processing_status_line, publish_processing_progress, set_memory_status

try:
    from roop.StreamWriter import StreamWriter
except Exception:
    StreamWriter = None


def get_jobs_root():
    jobs_root = Path(os.environ.get("TEMP", os.getcwd())) / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    return jobs_root


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_dumps(data):
    return json.dumps(make_json_safe(data), sort_keys=True, separators=(",", ":"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(data), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_cache_blob(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))


def read_cache_blob(path: Path):
    return pickle.loads(path.read_bytes())


def normalize_cache_image(image):
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def write_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    success, encoded = cv2.imencode(ext, image)
    if success:
        encoded.tofile(str(path))


def chunked(items, size):
    step = max(1, size)
    for start in range(0, len(items), step):
        yield items[start:start + step]


def hash_numpy(value):
    if value is None:
        return None
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def hash_facesets(face_sets):
    payload = []
    for face_set in face_sets:
        faces = []
        for face in face_set.faces:
            faces.append({
                "embedding": hash_numpy(getattr(face, "embedding", None)),
                "mask_offsets": list(getattr(face, "mask_offsets", [])),
            })
        payload.append(faces)
    return hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()


def hash_target_faces(target_faces):
    payload = []
    for face in target_faces:
        payload.append(hash_numpy(getattr(face, "embedding", None)))
    return hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()


PIPELINE_VERSION = 7
DETECT_PACK_FRAME_COUNT = 256


def get_entry_signature(entry, options, output_method):
    stat = os.stat(entry.filename)
    signature = {
        "pipeline_version": PIPELINE_VERSION,
        "file": {
            "path": os.path.abspath(entry.filename),
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "start": entry.startframe,
            "end": entry.endframe,
        },
        "inputs": hash_facesets(roop.globals.INPUT_FACESETS),
        "targets": hash_target_faces(roop.globals.TARGET_FACES),
        "provider": str(roop.globals.execution_providers),
        "output_method": output_method,
        "memory_mode": roop.globals.CFG.memory_mode,
        "max_ram_gb": roop.globals.CFG.max_ram_gb,
        "max_vram_gb": roop.globals.CFG.max_vram_gb,
        "options": {
            "processors": list(options.processors.keys()),
            "face_distance_threshold": options.face_distance_threshold,
            "blend_ratio": options.blend_ratio,
            "swap_mode": options.swap_mode,
            "selected_index": options.selected_index,
            "masking_text": options.masking_text,
            "num_swap_steps": options.num_swap_steps,
            "subsample_size": options.subsample_size,
            "restore_original_mouth": options.restore_original_mouth,
            "show_face_area_overlay": options.show_face_area_overlay,
        },
    }
    return hashlib.sha256(json_dumps(signature).encode("utf-8")).hexdigest()


def list_stage_images(stage_dir: Path, image_format: str):
    if not stage_dir.exists():
        return []
    return sorted(stage_dir.glob(f"*.{image_format}"))


def merge_stage_defaults(stage_state, defaults):
    merged = dict(defaults)
    if isinstance(stage_state, dict):
        merged.update(stage_state)
    return merged


def iter_video_chunk(video_path, frame_start, frame_end, prefetch_frames):
    q = queue.Queue(maxsize=max(2, prefetch_frames))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    def producer():
        current = frame_start
        while current < frame_end and roop.globals.processing:
            ok, frame = cap.read()
            if not ok:
                break
            q.put((current, frame), block=True)
            current += 1
        q.put(None)

    thread = Thread(target=producer, daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item
    thread.join()
    cap.release()


class StagedBatchExecutor:
    def __init__(self, output_method, progress, options):
        self.output_method = output_method
        self.progress = progress
        self.options = options
        self.jobs_root = get_jobs_root()
        self.completed_units = 0
        self.total_units = 0
        self.fallback_mgr = None
        self.swap_enabled = "faceswap" in options.processors
        self.mask_name = next((name for name in options.processors if name.startswith("mask_")), None)
        self.enhancer_name = next((name for name in options.processors if name not in ("faceswap", self.mask_name)), None)
        self.total_files = 0
        self.current_entry = None
        self.current_file_index = None
        self.current_chunk_index = None
        self.current_total_chunks = None
        self.current_stage = None
        self.stage_started_at = None


    def get_pipeline_steps(self):
        steps = ["prepare", "detect"]
        if self.swap_enabled:
            steps.append("swap")
        if self.mask_name is not None:
            steps.append("mask")
        if self.enhancer_name is not None:
            steps.append("enhance")
        steps.append("composite")
        is_video = self.current_entry is not None and not util.has_image_extension(self.current_entry.filename)
        if is_video and self.output_method != "Virtual Camera":
            steps.append("mux")
        return steps


    def get_stage_step_info(self, stage):
        steps = self.get_pipeline_steps()
        stage_alias = {
            "resume": "prepare",
            "image": "composite",
            "legacy": "prepare",
        }
        stage_key = stage_alias.get(stage, stage)
        if stage_key not in steps:
            if stage == "prepare":
                return 1, len(steps)
            return None, len(steps)
        return steps.index(stage_key) + 1, len(steps)


    def update_progress(self, stage, detail=None, step_completed=None, step_total=None, step_unit="items", force_log=False):
        if stage != self.current_stage or step_completed in (None, 0):
            self.current_stage = stage
            self.stage_started_at = time.time()
        stage_elapsed = None
        stage_rate = None
        stage_eta = None
        if self.stage_started_at is not None:
            stage_elapsed = max(time.time() - self.stage_started_at, 0.0)
        if step_completed is not None and step_completed > 0 and stage_elapsed and stage_elapsed > 0:
            stage_rate = step_completed / stage_elapsed
        if stage_rate is not None and step_total is not None and step_total > 0:
            stage_eta = max(step_total - step_completed, 0) / stage_rate
        target_name = None
        if self.current_entry is not None:
            target_name = self.current_entry.filename
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
            rate_unit=step_unit,
            elapsed=stage_elapsed,
            eta=stage_eta,
            detail=detail,
            memory_status=roop.globals.runtime_memory_status,
            force_log=force_log,
        )
        if self.progress is not None:
            self.progress((self.completed_units, self.total_units), desc=get_processing_status_line(), total=self.total_units, unit='units')


    def build_stage_options(self, processor_names):
        processors = {name: {} for name in processor_names if name}
        return ProcessOptions(
            processors,
            self.options.face_distance_threshold,
            self.options.blend_ratio,
            self.options.swap_mode,
            self.options.selected_index,
            self.options.masking_text,
            self.options.imagemask,
            self.options.num_swap_steps,
            self.options.subsample_size,
            self.options.show_face_area_overlay,
            self.options.restore_original_mouth,
            self.options.show_face_masking,
        )


    def get_fallback_mgr(self):
        if self.fallback_mgr is None:
            self.fallback_mgr = ProcessMgr(None)
            self.fallback_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.options)
        return self.fallback_mgr


    def count_total_units(self, files):
        total = 0
        for entry in files:
            if util.has_image_extension(entry.filename):
                total += 1
            else:
                endframe = entry.endframe or get_video_frame_total(entry.filename)
                total += max(endframe - entry.startframe, 1)
        self.total_units = max(total, 1)


    def run(self, files):
        self.count_total_units(files)
        self.total_files = len(files)
        for index, entry in enumerate(files):
            if not roop.globals.processing:
                break
            self.current_entry = entry
            self.current_file_index = index + 1
            self.current_chunk_index = None
            self.current_total_chunks = None
            if util.has_image_extension(entry.filename):
                self.process_image_entry(entry, index)
            else:
                self.process_video_entry(entry, index)


    def process_image_entry(self, entry, index):
        image = get_image_frame(entry.filename)
        if image is None:
            self.completed_units += 1
            self.update_progress("image", detail=f"Skipping unreadable image {os.path.basename(entry.filename)}", step_completed=1, step_total=1, step_unit="image", force_log=True)
            return
        height, width = image.shape[:2]
        memory_plan = resolve_memory_plan(width, height)
        set_memory_status(describe_memory_plan(memory_plan))
        job_dir, manifest = self.prepare_job(entry, memory_plan)
        self.update_progress("prepare", detail="Preparing image stages", step_completed=0, step_total=1, step_unit="image", force_log=True)
        if manifest.get("status") == "completed" and os.path.isfile(entry.finalname):
            self.completed_units += 1
            self.update_progress("resume", detail=f"Skipping cached image {os.path.basename(entry.filename)}", step_completed=1, step_total=1, step_unit="image", force_log=True)
            return

        detect_dir = job_dir / "detect"
        self.update_progress("detect", detail="Detecting and aligning faces", step_completed=0, step_total=1, step_unit="image")
        task_meta = self.ensure_detect_cache(job_dir, detect_dir, image, 0)
        chunk_meta = {"start": 0, "end": 1, "frames": [task_meta]}
        chunk_state = {
            "stages": {
                "detect": True,
                "swap": False,
                "mask": self.mask_name is None,
                "enhance": self.enhancer_name is None,
                "composite": False,
            }
        }
        self.update_progress("swap", detail="Running face swap", step_completed=0, step_total=max(len(task_meta["tasks"]), 1), step_unit="faces")
        self.ensure_swap_stage(job_dir, chunk_meta, chunk_state, memory_plan, source_image=image)
        self.update_progress("mask", detail="Applying mask stage", step_completed=0, step_total=max(len(task_meta["tasks"]), 1), step_unit="faces")
        self.ensure_mask_stage(job_dir, chunk_meta, chunk_state, memory_plan, source_image=image)
        self.update_progress("enhance", detail="Applying enhancement stage", step_completed=0, step_total=max(len(task_meta["tasks"]), 1), step_unit="faces")
        self.ensure_enhance_stage(job_dir, chunk_meta, chunk_state, memory_plan)
        self.update_progress("composite", detail="Compositing final image", step_completed=0, step_total=1, step_unit="image")
        result = self.compose_image_from_cache(image, job_dir, task_meta)
        if result is not None:
            write_image(Path(entry.finalname), result)
        manifest["status"] = "completed"
        write_json(job_dir / "manifest.json", manifest)
        self.completed_units += 1
        self.update_progress("image", detail=f"Processed {os.path.basename(entry.filename)}", step_completed=1, step_total=1, step_unit="image", force_log=True)
        self.cleanup_job_dir(job_dir)


    def process_video_entry(self, entry, index):
        if self.output_method == "File":
            self.process_video_entry_full_frames(entry, index)
            return
        self.process_video_entry_streaming(entry, index)


    def process_video_entry_streaming(self, entry, index):
        cap = cv2.VideoCapture(entry.filename)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        memory_plan = resolve_memory_plan(width, height)
        set_memory_status(describe_memory_plan(memory_plan))
        job_dir, manifest = self.prepare_job(entry, memory_plan)
        endframe = entry.endframe or get_video_frame_total(entry.filename)
        chunk_size = max(8, memory_plan["chunk_size"])
        chunks = [(start, min(start + chunk_size, endframe)) for start in range(entry.startframe, endframe, chunk_size)]
        self.current_total_chunks = len(chunks)
        self.update_progress("prepare", detail=f"Prepared {len(chunks)} chunk(s) for staged processing", step_completed=0, step_total=max(endframe - entry.startframe, 1), step_unit="frames", force_log=True)

        chunk_outputs = []
        for chunk_index, (chunk_start, chunk_end) in enumerate(chunks):
            if not roop.globals.processing:
                break
            self.current_chunk_index = chunk_index + 1
            chunk_key = f"{chunk_index:04d}_{chunk_start:08d}_{chunk_end:08d}"
            chunk_dir = job_dir / "chunks" / chunk_key
            chunk_video = chunk_dir / f"{chunk_key}.mp4"
            chunk_state = manifest["chunks"].setdefault(chunk_key, {
                "start": chunk_start,
                "end": chunk_end,
                "stages": {
                    "detect": False,
                    "swap": False,
                    "mask": self.mask_name is None,
                    "enhance": self.enhancer_name is None,
                    "composite": False,
                }
            })

            if chunk_state["stages"]["composite"] and chunk_video.exists():
                chunk_outputs.append(str(chunk_video))
                self.completed_units += max(chunk_end - chunk_start, 1)
                self.update_progress("resume", detail=f"Skipping cached chunk {chunk_index + 1}/{len(chunks)}", step_completed=max(chunk_end - chunk_start, 1), step_total=max(chunk_end - chunk_start, 1), step_unit="frames", force_log=True)
                continue

            chunk_meta = self.ensure_chunk_detect(entry.filename, chunk_dir, chunk_start, chunk_end, memory_plan, chunk_state)
            if not roop.globals.processing:
                break
            self.ensure_swap_stage(chunk_dir, chunk_meta, chunk_state, memory_plan, video_path=entry.filename)
            if not roop.globals.processing:
                break
            self.ensure_mask_stage(chunk_dir, chunk_meta, chunk_state, memory_plan, video_path=entry.filename)
            if not roop.globals.processing:
                break
            self.ensure_enhance_stage(chunk_dir, chunk_meta, chunk_state, memory_plan)
            if not roop.globals.processing:
                break
            self.compose_chunk(entry, chunk_dir, chunk_meta, chunk_state, memory_plan, chunk_video)
            if chunk_state["stages"]["composite"] and chunk_video.exists():
                chunk_outputs.append(str(chunk_video))
            write_json(job_dir / "manifest.json", manifest)

        if not roop.globals.processing:
            manifest["status"] = "interrupted"
            write_json(job_dir / "manifest.json", manifest)
            return

        if self.output_method != "Virtual Camera" and chunk_outputs:
            joined_video = job_dir / "joined_video.mp4"
            self.update_progress("mux", detail=f"Joining {len(chunk_outputs)} chunk(s)", step_completed=len(chunk_outputs), step_total=len(chunk_outputs), step_unit="chunks", force_log=True)
            ffmpeg.join_videos(chunk_outputs, str(joined_video), True)
            destination = util.replace_template(entry.finalname, index=index)
            Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
            if util.has_extension(entry.filename, ['gif']):
                ffmpeg.create_gif_from_video(str(joined_video), destination)
            elif roop.globals.skip_audio:
                shutil.move(str(joined_video), destination)
            else:
                ffmpeg.restore_audio(str(joined_video), entry.filename, entry.startframe, endframe, destination)
                if joined_video.exists() and os.path.isfile(destination):
                    os.remove(str(joined_video))
        manifest["status"] = "completed"
        write_json(job_dir / "manifest.json", manifest)
        self.cleanup_job_dir(job_dir)


    def process_video_entry_full_frames(self, entry, index):
        cap = cv2.VideoCapture(entry.filename)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        memory_plan = resolve_memory_plan(width, height)
        set_memory_status(describe_memory_plan(memory_plan))
        job_dir, manifest = self.prepare_job(entry, memory_plan)
        endframe = entry.endframe or get_video_frame_total(entry.filename)
        fps = entry.fps or util.detect_fps(entry.filename)
        frame_count = max(endframe - entry.startframe, 1)
        self.update_progress("prepare", detail="Preparing packed detect cache and stage pipeline", step_completed=0, step_total=frame_count, step_unit="frames", force_log=True)
        stages = merge_stage_defaults(manifest.get("stages"), {
            "detect": False,
            "swap": False,
            "mask": self.mask_name is None,
            "enhance": self.enhancer_name is None,
            "composite": False,
        })
        manifest["stages"] = stages
        manifest["frame_count"] = frame_count
        write_json(job_dir / "manifest.json", manifest)
        self.current_total_chunks = None
        self.current_chunk_index = None

        detect_dir = job_dir / "detect"
        swap_dir = job_dir / "swap"
        mask_dir = job_dir / "mask"
        enhance_dir = job_dir / "enhance"
        intermediate_video = job_dir / "intermediate.mp4"

        task_count = self.ensure_full_detect_stage(entry, endframe, detect_dir, stages, manifest, memory_plan)
        manifest["task_count"] = task_count
        write_json(job_dir / "manifest.json", manifest)
        if not roop.globals.processing:
            manifest["status"] = "interrupted"
            write_json(job_dir / "manifest.json", manifest)
            return

        self.ensure_full_swap_stage(entry, endframe, detect_dir, swap_dir, task_count, stages, manifest, memory_plan)
        if not roop.globals.processing:
            manifest["status"] = "interrupted"
            write_json(job_dir / "manifest.json", manifest)
            return

        self.ensure_full_mask_stage(entry, endframe, detect_dir, swap_dir, mask_dir, task_count, stages, manifest, memory_plan)
        if not roop.globals.processing:
            manifest["status"] = "interrupted"
            write_json(job_dir / "manifest.json", manifest)
            return

        self.ensure_full_enhance_stage(detect_dir, swap_dir, mask_dir, enhance_dir, task_count, stages, manifest, memory_plan)
        if not roop.globals.processing:
            manifest["status"] = "interrupted"
            write_json(job_dir / "manifest.json", manifest)
            return

        self.ensure_full_compose_stage(entry, endframe, fps, detect_dir, swap_dir, mask_dir, enhance_dir, intermediate_video, stages, manifest, memory_plan)
        if not roop.globals.processing:
            manifest["status"] = "interrupted"
            write_json(job_dir / "manifest.json", manifest)
            return

        self.ensure_full_encode_stage(entry, index, intermediate_video, manifest.get("frame_count", frame_count), endframe)
        manifest["status"] = "completed"
        write_json(job_dir / "manifest.json", manifest)
        self.cleanup_job_dir(job_dir)


    def get_detect_pack_dir(self, detect_dir):
        return detect_dir / "plans"


    def get_detect_pack_frame_count(self):
        value = getattr(roop.globals.CFG, "detect_pack_frame_count", 0) or DETECT_PACK_FRAME_COUNT
        return max(8, int(value))


    def get_detect_pack_path(self, detect_dir, start_seq, end_seq):
        return self.get_detect_pack_dir(detect_dir) / f"{start_seq:06d}_{end_seq:06d}.bin"


    def iter_detect_pack_ranges(self, frame_count):
        pack_frame_count = self.get_detect_pack_frame_count()
        for start_seq in range(1, frame_count + 1, pack_frame_count):
            end_seq = min(start_seq + pack_frame_count - 1, frame_count)
            yield start_seq, end_seq


    def iter_detect_packs(self, detect_dir):
        pack_dir = self.get_detect_pack_dir(detect_dir)
        if not pack_dir.exists():
            return
        for pack_path in sorted(pack_dir.glob("*.bin")):
            yield read_cache_blob(pack_path)


    def iter_detect_frame_meta(self, detect_dir, frame_count=None):
        emitted = 0
        for pack_data in self.iter_detect_packs(detect_dir):
            for frame_meta in pack_data.get("frames", []):
                yield frame_meta["sequence"], frame_meta
                emitted += 1
                if frame_count is not None and emitted >= frame_count:
                    return


    def iter_detect_tasks(self, detect_dir, frame_count=None):
        for _, frame_meta in self.iter_detect_frame_meta(detect_dir, frame_count):
            for task_meta in frame_meta["tasks"]:
                yield task_meta


    def flatten_pack_tasks(self, pack_data):
        tasks = []
        for frame_meta in pack_data.get("frames", []):
            for task_meta in frame_meta.get("tasks", []):
                tasks.append((frame_meta, task_meta))
        return tasks


    def summarize_detect_cache(self, detect_dir, frame_count=None):
        cached_frames = 0
        total_tasks = 0
        for _, frame_meta in self.iter_detect_frame_meta(detect_dir, frame_count):
            cached_frames += 1
            total_tasks += len(frame_meta["tasks"])
        return cached_frames, total_tasks


    def iter_full_source_frames_with_meta(self, entry, endframe, detect_dir, frame_count, memory_plan):
        frame_meta_iter = iter(self.iter_detect_frame_meta(detect_dir, frame_count))
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


    def get_stage_cache_path(self, stage_dir: Path):
        return stage_dir / "cache.bin"


    def get_stage_pack_path(self, stage_dir: Path, start_seq: int, end_seq: int):
        return stage_dir / "packs" / f"{start_seq:06d}_{end_seq:06d}.bin"


    def read_stage_cache_map(self, cache_path: Path):
        if not cache_path.exists():
            return {}
        payload = read_cache_blob(cache_path)
        images = payload.get("images", {})
        if not isinstance(images, dict):
            return {}
        return images


    def write_stage_cache_map(self, cache_path: Path, cache_map):
        serializable = {str(key): normalize_cache_image(value) for key, value in cache_map.items()}
        write_cache_blob(cache_path, {"images": serializable})


    def count_stage_cache_entries(self, cache_path: Path):
        if not cache_path.exists():
            return 0
        return len(self.read_stage_cache_map(cache_path))


    def ensure_full_detect_stage(self, entry, endframe, detect_dir, stages, manifest, memory_plan):
        plans_dir = self.get_detect_pack_dir(detect_dir)
        plans_dir.mkdir(parents=True, exist_ok=True)
        frame_count = max(endframe - entry.startframe, 1)
        if stages["detect"]:
            cached_frames, task_count = self.summarize_detect_cache(detect_dir, frame_count)
            if cached_frames >= frame_count:
                manifest["frame_count"] = frame_count
                manifest["task_count"] = task_count
                self.update_progress("detect", detail="Reusing detect cache", step_completed=frame_count, step_total=frame_count, step_unit="frames", force_log=True)
                return task_count
        planner = ProcessMgr(None)
        planner.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([]))
        processed_frames = 0
        total_tasks = 0
        try:
            for pack_start, pack_end in self.iter_detect_pack_ranges(frame_count):
                pack_path = self.get_detect_pack_path(detect_dir, pack_start, pack_end)
                if pack_path.exists():
                    pack_data = read_cache_blob(pack_path)
                    pack_frames = pack_data.get("frames", [])
                    processed_frames += len(pack_frames)
                    total_tasks += sum(len(frame_meta["tasks"]) for frame_meta in pack_frames)
                    self.update_progress("detect", detail="Reusing packed detect cache", step_completed=processed_frames, step_total=frame_count, step_unit="frames")
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
                    self.update_progress("detect", detail="Streaming source decode + packed face detection", step_completed=processed_frames, step_total=frame_count, step_unit="frames")
                if pack_frames:
                    write_cache_blob(pack_path, {
                        "start_sequence": pack_start,
                        "end_sequence": pack_end,
                        "frames": pack_frames,
                    })
                manifest["frame_count"] = processed_frames
                manifest["task_count"] = total_tasks
                write_json(detect_dir.parent / "manifest.json", manifest)
                if not roop.globals.processing:
                    break
        finally:
            planner.release_resources()
        stages["detect"] = roop.globals.processing and processed_frames >= frame_count
        manifest["frame_count"] = processed_frames
        manifest["task_count"] = total_tasks
        return total_tasks


    def get_swap_task_batch_size(self, memory_plan):
        tiles_per_task = max((max(self.options.subsample_size, 128) // 128) ** 2, 1)
        return max(1, min(64, (max(memory_plan["swap_batch_size"], 1) * 2) // tiles_per_task))


    def ensure_full_swap_stage(self, entry, endframe, detect_dir, swap_dir, task_count, stages, manifest, memory_plan):
        if not self.swap_enabled:
            stages["swap"] = True
            write_json(swap_dir.parent / "manifest.json", manifest)
            return
        if task_count <= 0:
            stages["swap"] = True
            write_json(swap_dir.parent / "manifest.json", manifest)
            return
        swap_dir.mkdir(parents=True, exist_ok=True)
        if stages["swap"]:
            cached = 0
            for pack_data in self.iter_detect_packs(detect_dir):
                cached += self.count_stage_cache_entries(self.get_stage_pack_path(swap_dir, pack_data["start_sequence"], pack_data["end_sequence"]))
            if cached >= task_count:
                self.update_progress("swap", detail="Reusing swap cache", step_completed=task_count, step_total=task_count, step_unit="faces", force_log=True)
                return

        swap_mgr = ProcessMgr(None)
        swap_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options(["faceswap"]))
        processor = swap_mgr.processors[0]
        processed_tasks = 0
        task_batch_size = self.get_swap_task_batch_size(memory_plan)
        try:
            for pack_data in self.iter_detect_packs(detect_dir):
                pack_tasks = self.flatten_pack_tasks(pack_data)
                if not pack_tasks:
                    continue
                pack_cache_path = self.get_stage_pack_path(swap_dir, pack_data["start_sequence"], pack_data["end_sequence"])
                pack_cache = self.read_stage_cache_map(pack_cache_path)
                if len(pack_cache) >= len(pack_tasks):
                    processed_tasks += len(pack_tasks)
                    self.update_progress("swap", detail="Reusing packed swap cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                    continue

                frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in pack_data["frames"]}
                task_batch = []
                for frame_number, frame in iter_video_chunk(entry.filename, pack_data["frames"][0]["frame_number"], pack_data["frames"][-1]["frame_number"] + 1, memory_plan["prefetch_frames"]):
                    frame_meta = frame_lookup.get(frame_number, {"tasks": []})
                    for task_meta in frame_meta["tasks"]:
                        if task_meta["cache_key"] in pack_cache:
                            processed_tasks += 1
                            self.update_progress("swap", detail="Reusing packed swap cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                            continue
                        task = dict(task_meta)
                        task["aligned_frame"] = swap_mgr.rebuild_aligned_frame(frame, task_meta)
                        task_batch.append(task)
                        if len(task_batch) >= task_batch_size:
                            batch_outputs = swap_mgr.run_swap_tasks_batch(task_batch, processor, memory_plan["swap_batch_size"])
                            for cache_key, fake_frame in batch_outputs.items():
                                pack_cache[cache_key] = normalize_cache_image(fake_frame)
                            self.write_stage_cache_map(pack_cache_path, pack_cache)
                            processed_tasks += len(task_batch)
                            self.update_progress("swap", detail="Streaming source decode + batched face swap", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                            task_batch.clear()
                if task_batch:
                    batch_outputs = swap_mgr.run_swap_tasks_batch(task_batch, processor, memory_plan["swap_batch_size"])
                    for cache_key, fake_frame in batch_outputs.items():
                        pack_cache[cache_key] = normalize_cache_image(fake_frame)
                    self.write_stage_cache_map(pack_cache_path, pack_cache)
                    processed_tasks += len(task_batch)
                    self.update_progress("swap", detail="Streaming source decode + batched face swap", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
        finally:
            swap_mgr.release_resources()
        stages["swap"] = True
        write_json(swap_dir.parent / "manifest.json", manifest)


    def ensure_full_mask_stage(self, entry, endframe, detect_dir, swap_dir, mask_dir, task_count, stages, manifest, memory_plan):
        if self.mask_name is None or task_count <= 0:
            stages["mask"] = True
            write_json(mask_dir.parent / "manifest.json", manifest)
            return
        mask_dir.mkdir(parents=True, exist_ok=True)
        if stages["mask"]:
            cached = 0
            for pack_data in self.iter_detect_packs(detect_dir):
                cached += self.count_stage_cache_entries(self.get_stage_pack_path(mask_dir, pack_data["start_sequence"], pack_data["end_sequence"]))
            if cached >= task_count:
                self.update_progress("mask", detail="Reusing mask cache", step_completed=task_count, step_total=task_count, step_unit="faces", force_log=True)
                return

        mask_mgr = ProcessMgr(None)
        mask_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([self.mask_name]))
        processor = mask_mgr.processors[0]
        processed_tasks = 0
        task_batch_size = max(1, min(128, memory_plan["mask_batch_size"]))
        try:
            for pack_data in self.iter_detect_packs(detect_dir):
                pack_tasks = self.flatten_pack_tasks(pack_data)
                if not pack_tasks:
                    continue
                input_cache = self.read_stage_cache_map(self.get_stage_pack_path(swap_dir, pack_data["start_sequence"], pack_data["end_sequence"]))
                pack_cache_path = self.get_stage_pack_path(mask_dir, pack_data["start_sequence"], pack_data["end_sequence"])
                pack_cache = self.read_stage_cache_map(pack_cache_path)
                if len(pack_cache) >= len(pack_tasks):
                    processed_tasks += len(pack_tasks)
                    self.update_progress("mask", detail="Reusing packed mask cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                    continue

                frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in pack_data["frames"]}
                task_batch = []
                original_batch = []
                for frame_number, frame in iter_video_chunk(entry.filename, pack_data["frames"][0]["frame_number"], pack_data["frames"][-1]["frame_number"] + 1, memory_plan["prefetch_frames"]):
                    frame_meta = frame_lookup.get(frame_number, {"tasks": []})
                    for task_meta in frame_meta["tasks"]:
                        if task_meta["cache_key"] in pack_cache:
                            processed_tasks += 1
                            self.update_progress("mask", detail="Reusing packed mask cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                            continue
                        task_batch.append(dict(task_meta))
                        original_batch.append(mask_mgr.rebuild_aligned_frame(frame, task_meta))
                        if len(task_batch) >= task_batch_size:
                            self.process_full_mask_batch(task_batch, original_batch, input_cache, pack_cache, pack_cache_path, mask_mgr, processor, processed_tasks, task_count, memory_plan)
                            processed_tasks += len(task_batch)
                            task_batch.clear()
                            original_batch.clear()
                if task_batch:
                    self.process_full_mask_batch(task_batch, original_batch, input_cache, pack_cache, pack_cache_path, mask_mgr, processor, processed_tasks, task_count, memory_plan)
                    processed_tasks += len(task_batch)
        finally:
            mask_mgr.release_resources()
        stages["mask"] = True
        write_json(mask_dir.parent / "manifest.json", manifest)


    def process_full_mask_batch(self, task_batch, original_batch, input_cache, output_cache, output_cache_path, mask_mgr, processor, processed_tasks, task_count, memory_plan):
        if getattr(processor, "supports_batch", False):
            masks = processor.RunBatch(original_batch, self.options.masking_text, memory_plan["mask_batch_size"])
            for task_meta, original, mask in zip(task_batch, original_batch, masks):
                current_frame = input_cache[task_meta["cache_key"]]
                mask = cv2.resize(mask, (current_frame.shape[1], current_frame.shape[0]))
                mask = np.reshape(mask, [mask.shape[0], mask.shape[1], 1])
                result = (1 - mask) * current_frame.astype(np.float32)
                result += mask * original.astype(np.float32)
                output_cache[task_meta["cache_key"]] = normalize_cache_image(np.uint8(result))
        else:
            for task_meta, original in zip(task_batch, original_batch):
                task = dict(task_meta)
                task["aligned_frame"] = original
                current_frame = input_cache[task_meta["cache_key"]]
                result = mask_mgr.run_mask_task(task, current_frame, processor)
                output_cache[task_meta["cache_key"]] = normalize_cache_image(result)
        self.write_stage_cache_map(output_cache_path, output_cache)
        self.update_progress("mask", detail="Streaming source decode + mask stage", step_completed=processed_tasks + len(task_batch), step_total=task_count, step_unit="faces")


    def ensure_full_enhance_stage(self, detect_dir, swap_dir, mask_dir, enhance_dir, task_count, stages, manifest, memory_plan):
        if self.enhancer_name is None or task_count <= 0:
            stages["enhance"] = True
            write_json(enhance_dir.parent / "manifest.json", manifest)
            return
        enhance_dir.mkdir(parents=True, exist_ok=True)
        if stages["enhance"]:
            cached = 0
            for pack_data in self.iter_detect_packs(detect_dir):
                cached += self.count_stage_cache_entries(self.get_stage_pack_path(enhance_dir, pack_data["start_sequence"], pack_data["end_sequence"]))
            if cached >= task_count:
                self.update_progress("enhance", detail="Reusing enhance cache", step_completed=task_count, step_total=task_count, step_unit="faces", force_log=True)
                return

        enhance_mgr = ProcessMgr(None)
        enhance_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([self.enhancer_name]))
        processor = enhance_mgr.processors[0]
        processed_tasks = 0
        task_batch_size = max(1, min(64, memory_plan["enhance_batch_size"]))
        try:
            for pack_data in self.iter_detect_packs(detect_dir):
                pack_tasks = self.flatten_pack_tasks(pack_data)
                if not pack_tasks:
                    continue
                input_cache = self.read_stage_cache_map(self.get_stage_pack_path(mask_dir if self.mask_name else swap_dir, pack_data["start_sequence"], pack_data["end_sequence"]))
                pack_cache_path = self.get_stage_pack_path(enhance_dir, pack_data["start_sequence"], pack_data["end_sequence"])
                pack_cache = self.read_stage_cache_map(pack_cache_path)
                if len(pack_cache) >= len(pack_tasks):
                    processed_tasks += len(pack_tasks)
                    self.update_progress("enhance", detail="Reusing packed enhance cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                    continue
                task_batch = []
                for _, task_meta in pack_tasks:
                    if task_meta["cache_key"] in pack_cache:
                        processed_tasks += 1
                        self.update_progress("enhance", detail="Reusing packed enhance cache", step_completed=processed_tasks, step_total=task_count, step_unit="faces")
                        continue
                    task_batch.append(dict(task_meta))
                    if len(task_batch) >= task_batch_size:
                        self.process_full_enhance_batch(task_batch, input_cache, pack_cache, pack_cache_path, enhance_mgr, processor, processed_tasks, task_count, memory_plan)
                        processed_tasks += len(task_batch)
                        task_batch.clear()
                if task_batch:
                    self.process_full_enhance_batch(task_batch, input_cache, pack_cache, pack_cache_path, enhance_mgr, processor, processed_tasks, task_count, memory_plan)
                    processed_tasks += len(task_batch)
        finally:
            enhance_mgr.release_resources()
        stages["enhance"] = True
        write_json(enhance_dir.parent / "manifest.json", manifest)


    def process_full_enhance_batch(self, task_batch, input_cache, output_cache, output_cache_path, enhance_mgr, processor, processed_tasks, task_count, memory_plan):
        if getattr(processor, "supports_batch", False):
            current_frames = [input_cache[task_meta["cache_key"]] for task_meta in task_batch]
            source_sets = [roop.globals.INPUT_FACESETS[task_meta["input_index"]] for task_meta in task_batch]
            target_faces = [enhance_mgr.deserialize_face(task_meta["target_face"]) for task_meta in task_batch]
            enhanced_frames = processor.RunBatch(source_sets, target_faces, current_frames, memory_plan["enhance_batch_size"])
            for task_meta, enhanced_frame in zip(task_batch, enhanced_frames):
                output_cache[task_meta["cache_key"]] = normalize_cache_image(enhanced_frame)
        else:
            for task_meta in task_batch:
                task = dict(task_meta)
                current_frame = input_cache[task_meta["cache_key"]]
                enhanced_frame, _ = enhance_mgr.run_enhance_task(task, current_frame, processor)
                output_cache[task_meta["cache_key"]] = normalize_cache_image(enhanced_frame)
        self.write_stage_cache_map(output_cache_path, output_cache)
        self.update_progress("enhance", detail="Running enhancement stage", step_completed=processed_tasks + len(task_batch), step_total=task_count, step_unit="faces")


    def ensure_full_compose_stage(self, entry, endframe, fps, detect_dir, swap_dir, mask_dir, enhance_dir, intermediate_video, stages, manifest, memory_plan):
        frame_count = manifest.get("frame_count", max(endframe - entry.startframe, 1))
        if stages["composite"] and intermediate_video.exists():
            self.completed_units += frame_count
            self.update_progress("composite", detail="Reusing encoded composite video cache", step_completed=frame_count, step_total=frame_count, step_unit="frames", force_log=True)
            return
        if intermediate_video.exists():
            os.remove(str(intermediate_video))
        compose_mgr = ProcessMgr(None)
        compose_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([]))
        fallback_mgr = None
        processed_frames = 0
        cap = cv2.VideoCapture(entry.filename)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        writer = FFMPEG_VideoWriter(str(intermediate_video), (width, height), fps, codec=roop.globals.video_encoder, crf=roop.globals.video_quality)
        try:
            for pack_data in self.iter_detect_packs(detect_dir):
                frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in pack_data["frames"]}
                input_cache = self.read_stage_cache_map(self.get_stage_pack_path(mask_dir if self.mask_name else swap_dir, pack_data["start_sequence"], pack_data["end_sequence"]))
                enhance_cache = self.read_stage_cache_map(self.get_stage_pack_path(enhance_dir, pack_data["start_sequence"], pack_data["end_sequence"])) if self.enhancer_name is not None else {}
                for frame_number, frame in iter_video_chunk(entry.filename, pack_data["frames"][0]["frame_number"], pack_data["frames"][-1]["frame_number"] + 1, memory_plan["prefetch_frames"]):
                    frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
                    if frame_meta["fallback"]:
                        fallback_mgr = self.get_fallback_mgr()
                        result = fallback_mgr.process_frame(frame)
                    else:
                        result = frame.copy()
                        for task_meta in frame_meta["tasks"]:
                            fake_frame = input_cache[task_meta["cache_key"]]
                            enhanced_frame = enhance_cache.get(task_meta["cache_key"]) if self.enhancer_name is not None else None
                            result = compose_mgr.compose_task(result, task_meta, fake_frame, enhanced_frame)
                        fallback_mgr = self.fallback_mgr
                        if fallback_mgr is not None and result is not None:
                            fallback_mgr.last_swapped_frame = result.copy()
                            fallback_mgr.num_frames_no_face = 0
                    if result is not None:
                        writer.write_frame(result)
                    self.completed_units += 1
                    processed_frames += 1
                    self.update_progress("composite", detail="Streaming source decode + direct video encode", step_completed=processed_frames, step_total=frame_count, step_unit="frames")
        finally:
            compose_mgr.release_resources()
            writer.close()
        stages["composite"] = intermediate_video.exists()
        write_json(intermediate_video.parent / "manifest.json", manifest)


    def ensure_full_encode_stage(self, entry, index, intermediate_video, frame_count, endframe):
        destination = util.replace_template(entry.finalname, index=index)
        Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
        if util.has_extension(entry.filename, ['gif']):
            self.update_progress("mux", detail="Creating final GIF", step_completed=frame_count, step_total=frame_count, step_unit="frames", force_log=True)
            ffmpeg.create_gif_from_video(str(intermediate_video), destination)
        elif roop.globals.skip_audio:
            if os.path.isfile(destination):
                os.remove(destination)
            shutil.move(str(intermediate_video), destination)
        else:
            self.update_progress("mux", detail="Restoring source audio", step_completed=frame_count, step_total=frame_count, step_unit="frames", force_log=True)
            ffmpeg.restore_audio(str(intermediate_video), entry.filename, entry.startframe, endframe, destination)
            if intermediate_video.exists() and os.path.isfile(destination):
                os.remove(str(intermediate_video))


    def prepare_job(self, entry, memory_plan):
        job_hash = get_entry_signature(entry, self.options, self.output_method)
        job_dir = self.jobs_root / job_hash
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = job_dir / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
        else:
            manifest = {
                "job_hash": job_hash,
                "pipeline_version": PIPELINE_VERSION,
                "status": "running",
                "memory_plan": memory_plan,
                "chunks": {},
                "stages": {},
            }
            write_json(manifest_path, manifest)
        manifest["memory_plan"] = memory_plan
        manifest["status"] = "running"
        return job_dir, manifest


    def ensure_detect_cache(self, job_dir, detect_dir, image, frame_number):
        detect_dir.mkdir(parents=True, exist_ok=True)
        plan_path = detect_dir / "plan.bin"
        if plan_path.exists():
            self.update_progress("detect", detail="Reusing cached detect plan", step_completed=1, step_total=1, step_unit="image")
            return read_cache_blob(plan_path)
        planner = ProcessMgr(None)
        planner.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([]))
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
        self.update_progress("detect", detail="Detection cache created", step_completed=1, step_total=1, step_unit="image")
        return frame_meta


    def ensure_chunk_detect(self, video_path, chunk_dir, chunk_start, chunk_end, memory_plan, chunk_state):
        plan_path = chunk_dir / "plan.bin"
        if plan_path.exists():
            chunk_state["stages"]["detect"] = True
            frame_total = max(chunk_end - chunk_start, 1)
            self.update_progress("detect", detail="Reusing detect cache", step_completed=frame_total, step_total=frame_total, step_unit="frames")
            return read_cache_blob(plan_path)

        chunk_dir.mkdir(parents=True, exist_ok=True)
        planner = ProcessMgr(None)
        planner.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([]))
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
                self.update_progress("detect", detail="Detecting and aligning faces", step_completed=processed_frames, step_total=total_frames, step_unit="frames")
        finally:
            planner.release_resources()
        write_cache_blob(plan_path, chunk_meta)
        chunk_state["stages"]["detect"] = True
        return chunk_meta


    def flatten_tasks(self, chunk_meta):
        tasks = []
        for frame_meta in chunk_meta["frames"]:
            for task_meta in frame_meta["tasks"]:
                tasks.append((frame_meta, task_meta))
        return tasks


    def iter_chunk_source_frames_with_meta(self, chunk_meta, memory_plan, video_path=None, source_image=None):
        if source_image is not None:
            frame_meta = chunk_meta["frames"][0] if chunk_meta["frames"] else {"frame_number": 0, "fallback": True, "tasks": []}
            yield frame_meta.get("frame_number", 0), source_image, frame_meta
            return
        frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in chunk_meta["frames"]}
        for frame_number, frame in iter_video_chunk(video_path, chunk_meta["start"], chunk_meta["end"], memory_plan["prefetch_frames"]):
            frame_meta = frame_lookup.get(frame_number, {"frame_number": frame_number, "fallback": True, "tasks": []})
            yield frame_number, frame, frame_meta


    def ensure_swap_stage(self, chunk_dir, chunk_meta, chunk_state, memory_plan, video_path=None, source_image=None):
        if not self.swap_enabled:
            chunk_state["stages"]["swap"] = True
            return
        swap_dir = chunk_dir / "swap"
        flat_tasks = self.flatten_tasks(chunk_meta)
        total_tasks = len(flat_tasks)
        cache_path = self.get_stage_cache_path(swap_dir)
        swap_cache = self.read_stage_cache_map(cache_path)
        if chunk_state["stages"]["swap"] or len(swap_cache) >= total_tasks:
            chunk_state["stages"]["swap"] = True
            if total_tasks > 0:
                self.update_progress("swap", detail="Reusing swap cache", step_completed=total_tasks, step_total=total_tasks, step_unit="faces")
            return
        swap_dir.mkdir(parents=True, exist_ok=True)
        swap_mgr = ProcessMgr(None)
        swap_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options(["faceswap"]))
        processor = swap_mgr.processors[0]
        processed_tasks = 0
        task_batch = []
        task_batch_size = self.get_swap_task_batch_size(memory_plan)
        try:
            for _, frame, frame_meta in self.iter_chunk_source_frames_with_meta(chunk_meta, memory_plan, video_path=video_path, source_image=source_image):
                for task_meta in frame_meta["tasks"]:
                    if task_meta["cache_key"] in swap_cache:
                        processed_tasks += 1
                        self.update_progress("swap", detail="Reusing packed swap cache", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
                        continue
                    task = dict(task_meta)
                    task["aligned_frame"] = swap_mgr.rebuild_aligned_frame(frame, task_meta)
                    task_batch.append(task)
                    if len(task_batch) >= task_batch_size:
                        batch_outputs = swap_mgr.run_swap_tasks_batch(task_batch, processor, memory_plan["swap_batch_size"])
                        for cache_key, fake_frame in batch_outputs.items():
                            swap_cache[cache_key] = normalize_cache_image(fake_frame)
                        self.write_stage_cache_map(cache_path, swap_cache)
                        processed_tasks += len(task_batch)
                        self.update_progress("swap", detail="Streaming source decode + batched face swap", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
                        task_batch.clear()
            if task_batch:
                batch_outputs = swap_mgr.run_swap_tasks_batch(task_batch, processor, memory_plan["swap_batch_size"])
                for cache_key, fake_frame in batch_outputs.items():
                    swap_cache[cache_key] = normalize_cache_image(fake_frame)
                self.write_stage_cache_map(cache_path, swap_cache)
                processed_tasks += len(task_batch)
                self.update_progress("swap", detail="Streaming source decode + batched face swap", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
        finally:
            swap_mgr.release_resources()
        chunk_state["stages"]["swap"] = True


    def ensure_mask_stage(self, chunk_dir, chunk_meta, chunk_state, memory_plan, video_path=None, source_image=None):
        if self.mask_name is None:
            chunk_state["stages"]["mask"] = True
            return
        mask_dir = chunk_dir / "mask"
        flat_tasks = self.flatten_tasks(chunk_meta)
        total_tasks = len(flat_tasks)
        cache_path = self.get_stage_cache_path(mask_dir)
        mask_cache = self.read_stage_cache_map(cache_path)
        if chunk_state["stages"]["mask"] or len(mask_cache) >= total_tasks:
            chunk_state["stages"]["mask"] = True
            if total_tasks > 0:
                self.update_progress("mask", detail="Reusing mask cache", step_completed=total_tasks, step_total=total_tasks, step_unit="faces")
            return
        input_cache = self.read_stage_cache_map(self.get_stage_cache_path(chunk_dir / "swap"))
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_mgr = ProcessMgr(None)
        mask_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([self.mask_name]))
        processor = mask_mgr.processors[0]
        processed_tasks = 0
        task_batch = []
        original_batch = []
        task_batch_size = max(1, min(128, memory_plan["mask_batch_size"]))
        try:
            for _, frame, frame_meta in self.iter_chunk_source_frames_with_meta(chunk_meta, memory_plan, video_path=video_path, source_image=source_image):
                for task_meta in frame_meta["tasks"]:
                    if task_meta["cache_key"] in mask_cache:
                        processed_tasks += 1
                        self.update_progress("mask", detail="Reusing packed masks", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
                        continue
                    task_batch.append(dict(task_meta))
                    original_batch.append(mask_mgr.rebuild_aligned_frame(frame, task_meta))
                    if len(task_batch) >= task_batch_size:
                        self.process_full_mask_batch(task_batch, original_batch, input_cache, mask_cache, cache_path, mask_mgr, processor, processed_tasks, total_tasks, memory_plan)
                        processed_tasks += len(task_batch)
                        task_batch.clear()
                        original_batch.clear()
            if task_batch:
                self.process_full_mask_batch(task_batch, original_batch, input_cache, mask_cache, cache_path, mask_mgr, processor, processed_tasks, total_tasks, memory_plan)
                processed_tasks += len(task_batch)
        finally:
            mask_mgr.release_resources()
        chunk_state["stages"]["mask"] = True


    def ensure_enhance_stage(self, chunk_dir, chunk_meta, chunk_state, memory_plan):
        if self.enhancer_name is None:
            chunk_state["stages"]["enhance"] = True
            return
        enhance_dir = chunk_dir / "enhance"
        flat_tasks = self.flatten_tasks(chunk_meta)
        total_tasks = len(flat_tasks)
        cache_path = self.get_stage_cache_path(enhance_dir)
        enhance_cache = self.read_stage_cache_map(cache_path)
        if chunk_state["stages"]["enhance"] or len(enhance_cache) >= total_tasks:
            chunk_state["stages"]["enhance"] = True
            if total_tasks > 0:
                self.update_progress("enhance", detail="Reusing enhance cache", step_completed=total_tasks, step_total=total_tasks, step_unit="faces")
            return
        input_cache = self.read_stage_cache_map(self.get_stage_cache_path(chunk_dir / ("mask" if self.mask_name else "swap")))
        enhance_dir.mkdir(parents=True, exist_ok=True)
        enhance_mgr = ProcessMgr(None)
        enhance_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([self.enhancer_name]))
        processor = enhance_mgr.processors[0]
        processed_tasks = 0
        if getattr(processor, "supports_batch", False):
            for batch in chunked(flat_tasks, memory_plan["enhance_batch_size"]):
                pending = [(frame_meta, task_meta) for frame_meta, task_meta in batch if task_meta["cache_key"] not in enhance_cache]
                if not pending:
                    processed_tasks += len(batch)
                    self.update_progress("enhance", detail="Reusing packed enhance cache", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
                    continue
                current_frames = [input_cache[task_meta["cache_key"]] for _, task_meta in pending]
                target_faces = [enhance_mgr.deserialize_face(task_meta["target_face"]) for _, task_meta in pending]
                source_sets = [roop.globals.INPUT_FACESETS[task_meta["input_index"]] for _, task_meta in pending]
                enhanced_frames = processor.RunBatch(source_sets, target_faces, current_frames, memory_plan["enhance_batch_size"])
                for (_, task_meta), enhanced_frame in zip(pending, enhanced_frames):
                    enhance_cache[task_meta["cache_key"]] = normalize_cache_image(enhanced_frame)
                self.write_stage_cache_map(cache_path, enhance_cache)
                processed_tasks += len(batch)
                self.update_progress("enhance", detail="Running enhancement stage", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
        else:
            for _, task_meta in flat_tasks:
                if task_meta["cache_key"] in enhance_cache:
                    processed_tasks += 1
                    self.update_progress("enhance", detail="Reusing packed enhance cache", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
                    continue
                task = dict(task_meta)
                current_frame = input_cache[task_meta["cache_key"]]
                enhanced_frame, _ = enhance_mgr.run_enhance_task(task, current_frame, processor)
                enhance_cache[task_meta["cache_key"]] = normalize_cache_image(enhanced_frame)
                self.write_stage_cache_map(cache_path, enhance_cache)
                processed_tasks += 1
                self.update_progress("enhance", detail="Running enhancement stage", step_completed=processed_tasks, step_total=total_tasks, step_unit="faces")
        enhance_mgr.release_resources()
        chunk_state["stages"]["enhance"] = True


    def compose_image_from_cache(self, image, job_dir, frame_meta):
        if frame_meta["fallback"]:
            return self.get_fallback_mgr().process_frame(image)
        compose_mgr = ProcessMgr(None)
        compose_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([]))
        input_cache = self.read_stage_cache_map(self.get_stage_cache_path(job_dir / ("mask" if self.mask_name else "swap")))
        enhance_cache = self.read_stage_cache_map(self.get_stage_cache_path(job_dir / "enhance")) if self.enhancer_name is not None else {}
        result = image.copy()
        for task_meta in frame_meta["tasks"]:
            fake_frame = input_cache[task_meta["cache_key"]]
            enhanced_frame = enhance_cache.get(task_meta["cache_key"]) if self.enhancer_name is not None else None
            result = compose_mgr.compose_task(result, task_meta, fake_frame, enhanced_frame)
        compose_mgr.release_resources()
        return result


    def compose_chunk(self, entry, chunk_dir, chunk_meta, chunk_state, memory_plan, chunk_video):
        if chunk_state["stages"]["composite"] and chunk_video.exists():
            return
        compose_mgr = ProcessMgr(None)
        compose_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([]))
        frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in chunk_meta["frames"]}
        input_cache = self.read_stage_cache_map(self.get_stage_cache_path(chunk_dir / ("mask" if self.mask_name else "swap")))
        enhance_cache = self.read_stage_cache_map(self.get_stage_cache_path(chunk_dir / "enhance")) if self.enhancer_name is not None else {}
        cap = cv2.VideoCapture(entry.filename)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        output_to_file = self.output_method != "Virtual Camera"
        output_to_cam = self.output_method in ("Virtual Camera", "Both") and StreamWriter is not None
        writer = FFMPEG_VideoWriter(str(chunk_video), (width, height), entry.fps or util.detect_fps(entry.filename), codec=roop.globals.video_encoder, crf=roop.globals.video_quality) if output_to_file else None
        stream = StreamWriter((width, height), int(entry.fps or util.detect_fps(entry.filename))) if output_to_cam else None
        fallback_mgr = None

        try:
            processed_frames = 0
            total_frames = max(chunk_meta["end"] - chunk_meta["start"], 1)
            for frame_number, frame in iter_video_chunk(entry.filename, chunk_meta["start"], chunk_meta["end"], memory_plan["prefetch_frames"]):
                frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
                if frame_meta["fallback"]:
                    fallback_mgr = self.get_fallback_mgr()
                    result = fallback_mgr.process_frame(frame)
                else:
                    result = frame.copy()
                    for task_meta in frame_meta["tasks"]:
                        fake_frame = input_cache[task_meta["cache_key"]]
                        enhanced_frame = enhance_cache.get(task_meta["cache_key"]) if self.enhancer_name is not None else None
                        result = compose_mgr.compose_task(result, task_meta, fake_frame, enhanced_frame)
                    fallback_mgr = self.fallback_mgr
                    if fallback_mgr is not None and result is not None:
                        fallback_mgr.last_swapped_frame = result.copy()
                        fallback_mgr.num_frames_no_face = 0
                if result is not None:
                    if writer is not None:
                        writer.write_frame(result)
                    if stream is not None:
                        stream.WriteToStream(result)
                self.completed_units += 1
                processed_frames += 1
                self.update_progress("composite", detail="Compositing output frames", step_completed=processed_frames, step_total=total_frames, step_unit="frames")
        finally:
            compose_mgr.release_resources()
            if writer is not None:
                writer.close()
            if stream is not None:
                stream.Close()
        chunk_state["stages"]["composite"] = True


    def cleanup_job_dir(self, job_dir):
        if os.path.isdir(job_dir) and roop.globals.processing:
            shutil.rmtree(job_dir, ignore_errors=True)
