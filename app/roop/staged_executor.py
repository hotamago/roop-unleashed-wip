import hashlib
import json
import os
import queue
import shutil
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

try:
    from roop.StreamWriter import StreamWriter
except Exception:
    StreamWriter = None


def get_jobs_root():
    jobs_root = Path(os.environ.get("TEMP", os.getcwd())) / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    return jobs_root


def json_dumps(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(data), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    success, encoded = cv2.imencode(ext, image)
    if success:
        encoded.tofile(str(path))


def read_image(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


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


def get_entry_signature(entry, options):
    stat = os.stat(entry.filename)
    signature = {
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


    def update_progress(self, desc):
        if self.progress is not None:
            self.progress((self.completed_units, self.total_units), desc=desc, total=self.total_units, unit='frames')


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
        for index, entry in enumerate(files):
            if not roop.globals.processing:
                break
            if util.has_image_extension(entry.filename):
                self.process_image_entry(entry, index)
            else:
                self.process_video_entry(entry, index)


    def process_image_entry(self, entry, index):
        image = get_image_frame(entry.filename)
        if image is None:
            self.completed_units += 1
            self.update_progress(f"Skipping unreadable image {os.path.basename(entry.filename)}")
            return
        height, width = image.shape[:2]
        memory_plan = resolve_memory_plan(width, height)
        job_dir, manifest = self.prepare_job(entry, memory_plan)
        if manifest.get("status") == "completed" and os.path.isfile(entry.finalname):
            self.completed_units += 1
            self.update_progress(f"Skipping cached image {os.path.basename(entry.filename)}")
            return

        detect_dir = job_dir / "detect"
        task_meta = self.ensure_detect_cache(job_dir, detect_dir, image, 0)
        chunk_meta = {"frames": [task_meta]}
        chunk_state = {
            "stages": {
                "detect": True,
                "swap": False,
                "mask": self.mask_name is None,
                "enhance": self.enhancer_name is None,
                "composite": False,
            }
        }
        self.ensure_swap_stage(job_dir, chunk_meta, chunk_state, memory_plan)
        self.ensure_mask_stage(job_dir, chunk_meta, chunk_state, memory_plan)
        self.ensure_enhance_stage(job_dir, chunk_meta, chunk_state, memory_plan)
        result = self.compose_image_from_cache(image, job_dir, task_meta)
        if result is not None:
            write_image(Path(entry.finalname), result)
        manifest["status"] = "completed"
        write_json(job_dir / "manifest.json", manifest)
        self.completed_units += 1
        self.update_progress(f"Processed {os.path.basename(entry.filename)}")
        self.cleanup_job_dir(job_dir)


    def process_video_entry(self, entry, index):
        cap = cv2.VideoCapture(entry.filename)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        memory_plan = resolve_memory_plan(width, height)
        roop.globals.runtime_memory_status = describe_memory_plan(memory_plan)
        job_dir, manifest = self.prepare_job(entry, memory_plan)
        endframe = entry.endframe or get_video_frame_total(entry.filename)
        chunk_size = max(8, memory_plan["chunk_size"])
        chunks = [(start, min(start + chunk_size, endframe)) for start in range(entry.startframe, endframe, chunk_size)]

        chunk_outputs = []
        for chunk_index, (chunk_start, chunk_end) in enumerate(chunks):
            if not roop.globals.processing:
                break
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
                self.update_progress(f"Skipping cached chunk {chunk_key}")
                continue

            chunk_meta = self.ensure_chunk_detect(entry.filename, chunk_dir, chunk_start, chunk_end, memory_plan, chunk_state)
            if not roop.globals.processing:
                break
            self.ensure_swap_stage(chunk_dir, chunk_meta, chunk_state, memory_plan)
            if not roop.globals.processing:
                break
            self.ensure_mask_stage(chunk_dir, chunk_meta, chunk_state, memory_plan)
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


    def prepare_job(self, entry, memory_plan):
        job_hash = get_entry_signature(entry, self.options)
        job_dir = self.jobs_root / job_hash
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = job_dir / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
        else:
            manifest = {
                "job_hash": job_hash,
                "status": "running",
                "memory_plan": memory_plan,
                "chunks": {},
            }
            write_json(manifest_path, manifest)
        return job_dir, manifest


    def ensure_detect_cache(self, job_dir, detect_dir, image, frame_number):
        detect_dir.mkdir(parents=True, exist_ok=True)
        plan_path = detect_dir / "plan.json"
        if plan_path.exists():
            return read_json(plan_path)
        planner = ProcessMgr(None)
        planner.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([]))
        frame_plan = planner.build_frame_plan(image)
        frame_meta = {"frame_number": frame_number, "fallback": frame_plan["fallback"], "tasks": []}
        for task_index, task in enumerate(frame_plan["tasks"]):
            cache_key = f"f{frame_number:08d}_t{task_index:03d}"
            aligned_path = detect_dir / f"{cache_key}_aligned.png"
            write_image(aligned_path, task["aligned_frame"])
            task_meta = {key: value for key, value in task.items() if key != "aligned_frame"}
            task_meta["cache_key"] = cache_key
            task_meta["aligned_path"] = aligned_path.name
            frame_meta["tasks"].append(task_meta)
        write_json(plan_path, frame_meta)
        return frame_meta


    def ensure_chunk_detect(self, video_path, chunk_dir, chunk_start, chunk_end, memory_plan, chunk_state):
        detect_dir = chunk_dir / "detect"
        plan_path = chunk_dir / "plan.json"
        if plan_path.exists():
            chunk_state["stages"]["detect"] = True
            return read_json(plan_path)

        chunk_dir.mkdir(parents=True, exist_ok=True)
        detect_dir.mkdir(parents=True, exist_ok=True)
        planner = ProcessMgr(None)
        planner.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([]))
        chunk_meta = {"start": chunk_start, "end": chunk_end, "frames": []}
        for frame_number, frame in iter_video_chunk(video_path, chunk_start, chunk_end, memory_plan["prefetch_frames"]):
            frame_plan = planner.build_frame_plan(frame)
            frame_meta = {"frame_number": frame_number, "fallback": frame_plan["fallback"], "tasks": []}
            for task_index, task in enumerate(frame_plan["tasks"]):
                cache_key = f"f{frame_number:08d}_t{task_index:03d}"
                aligned_path = detect_dir / f"{cache_key}_aligned.png"
                write_image(aligned_path, task["aligned_frame"])
                task_meta = {key: value for key, value in task.items() if key != "aligned_frame"}
                task_meta["cache_key"] = cache_key
                task_meta["aligned_path"] = aligned_path.name
                frame_meta["tasks"].append(task_meta)
            chunk_meta["frames"].append(frame_meta)
        write_json(plan_path, chunk_meta)
        chunk_state["stages"]["detect"] = True
        return chunk_meta


    def flatten_tasks(self, chunk_meta):
        tasks = []
        for frame_meta in chunk_meta["frames"]:
            for task_meta in frame_meta["tasks"]:
                tasks.append((frame_meta, task_meta))
        return tasks


    def ensure_swap_stage(self, chunk_dir, chunk_meta, chunk_state, memory_plan):
        if not self.swap_enabled:
            chunk_state["stages"]["swap"] = True
            return
        swap_dir = chunk_dir / "swap"
        expected = [swap_dir / f"{task_meta['cache_key']}.png" for _, task_meta in self.flatten_tasks(chunk_meta)]
        if chunk_state["stages"]["swap"] or (expected and all(path.exists() for path in expected)):
            chunk_state["stages"]["swap"] = True
            return
        detect_dir = chunk_dir / "detect"
        swap_dir.mkdir(parents=True, exist_ok=True)
        swap_mgr = ProcessMgr(None)
        swap_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options(["faceswap"]))
        processor = swap_mgr.processors[0]
        for _, task_meta in self.flatten_tasks(chunk_meta):
            cache_path = swap_dir / f"{task_meta['cache_key']}.png"
            if cache_path.exists():
                continue
            task = dict(task_meta)
            task["aligned_frame"] = read_image(detect_dir / task_meta["aligned_path"])
            fake_frame = swap_mgr.run_swap_task(task, processor, memory_plan["swap_batch_size"])
            write_image(cache_path, fake_frame)
        swap_mgr.release_resources()
        chunk_state["stages"]["swap"] = True


    def ensure_mask_stage(self, chunk_dir, chunk_meta, chunk_state, memory_plan):
        if self.mask_name is None:
            chunk_state["stages"]["mask"] = True
            return
        mask_dir = chunk_dir / "mask"
        expected = [mask_dir / f"{task_meta['cache_key']}.png" for _, task_meta in self.flatten_tasks(chunk_meta)]
        if chunk_state["stages"]["mask"] or (expected and all(path.exists() for path in expected)):
            chunk_state["stages"]["mask"] = True
            return
        detect_dir = chunk_dir / "detect"
        swap_dir = chunk_dir / "swap"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_mgr = ProcessMgr(None)
        mask_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([self.mask_name]))
        processor = mask_mgr.processors[0]
        flat_tasks = self.flatten_tasks(chunk_meta)
        if getattr(processor, "supports_batch", False):
            for batch in chunked(flat_tasks, memory_plan["mask_batch_size"]):
                originals = [read_image(detect_dir / task_meta["aligned_path"]) for _, task_meta in batch]
                masks = processor.RunBatch(originals, self.options.masking_text, memory_plan["mask_batch_size"])
                for (_, task_meta), original, mask in zip(batch, originals, masks):
                    cache_path = mask_dir / f"{task_meta['cache_key']}.png"
                    if cache_path.exists():
                        continue
                    current_frame = read_image(swap_dir / f"{task_meta['cache_key']}.png")
                    mask = cv2.resize(mask, (current_frame.shape[1], current_frame.shape[0]))
                    mask = np.reshape(mask, [mask.shape[0], mask.shape[1], 1])
                    result = (1 - mask) * current_frame.astype(np.float32)
                    result += mask * original.astype(np.float32)
                    write_image(cache_path, np.uint8(result))
        else:
            for _, task_meta in flat_tasks:
                cache_path = mask_dir / f"{task_meta['cache_key']}.png"
                if cache_path.exists():
                    continue
                task = dict(task_meta)
                task["aligned_frame"] = read_image(detect_dir / task_meta["aligned_path"])
                current_frame = read_image(swap_dir / f"{task_meta['cache_key']}.png")
                result = mask_mgr.run_mask_task(task, current_frame, processor)
                write_image(cache_path, result)
        mask_mgr.release_resources()
        chunk_state["stages"]["mask"] = True


    def ensure_enhance_stage(self, chunk_dir, chunk_meta, chunk_state, memory_plan):
        if self.enhancer_name is None:
            chunk_state["stages"]["enhance"] = True
            return
        enhance_dir = chunk_dir / "enhance"
        expected = [enhance_dir / f"{task_meta['cache_key']}.png" for _, task_meta in self.flatten_tasks(chunk_meta)]
        if chunk_state["stages"]["enhance"] or (expected and all(path.exists() for path in expected)):
            chunk_state["stages"]["enhance"] = True
            return
        input_dir = chunk_dir / ("mask" if self.mask_name else "swap")
        enhance_dir.mkdir(parents=True, exist_ok=True)
        enhance_mgr = ProcessMgr(None)
        enhance_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([self.enhancer_name]))
        processor = enhance_mgr.processors[0]
        flat_tasks = self.flatten_tasks(chunk_meta)
        if getattr(processor, "supports_batch", False):
            for batch in chunked(flat_tasks, memory_plan["enhance_batch_size"]):
                current_frames = [read_image(input_dir / f"{task_meta['cache_key']}.png") for _, task_meta in batch]
                source_sets = [roop.globals.INPUT_FACESETS[task_meta["input_index"]] for _, task_meta in batch]
                target_faces = [enhance_mgr.deserialize_face(task_meta["target_face"]) for _, task_meta in batch]
                enhanced_frames = processor.RunBatch(source_sets, target_faces, current_frames, memory_plan["enhance_batch_size"])
                for (_, task_meta), enhanced_frame in zip(batch, enhanced_frames):
                    cache_path = enhance_dir / f"{task_meta['cache_key']}.png"
                    if cache_path.exists():
                        continue
                    write_image(cache_path, enhanced_frame)
        else:
            for _, task_meta in flat_tasks:
                cache_path = enhance_dir / f"{task_meta['cache_key']}.png"
                if cache_path.exists():
                    continue
                task = dict(task_meta)
                current_frame = read_image(input_dir / f"{task_meta['cache_key']}.png")
                enhanced_frame, _ = enhance_mgr.run_enhance_task(task, current_frame, processor)
                write_image(cache_path, enhanced_frame)
        enhance_mgr.release_resources()
        chunk_state["stages"]["enhance"] = True


    def compose_image_from_cache(self, image, job_dir, frame_meta):
        if frame_meta["fallback"]:
            return self.get_fallback_mgr().process_frame(image)
        compose_mgr = ProcessMgr(None)
        compose_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([]))
        result = image.copy()
        for task_meta in frame_meta["tasks"]:
            fake_path = job_dir / ("mask" if self.mask_name else "swap") / f"{task_meta['cache_key']}.png"
            fake_frame = read_image(fake_path)
            enhanced_frame = None
            if self.enhancer_name is not None:
                enhanced_frame = read_image(job_dir / "enhance" / f"{task_meta['cache_key']}.png")
            result = compose_mgr.compose_task(result, task_meta, fake_frame, enhanced_frame)
        compose_mgr.release_resources()
        return result


    def compose_chunk(self, entry, chunk_dir, chunk_meta, chunk_state, memory_plan, chunk_video):
        if chunk_state["stages"]["composite"] and chunk_video.exists():
            return
        compose_mgr = ProcessMgr(None)
        compose_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, self.build_stage_options([]))
        frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in chunk_meta["frames"]}
        input_dir = chunk_dir / ("mask" if self.mask_name else "swap")
        enhance_dir = chunk_dir / "enhance"
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
            for frame_number, frame in iter_video_chunk(entry.filename, chunk_meta["start"], chunk_meta["end"], memory_plan["prefetch_frames"]):
                frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
                if frame_meta["fallback"]:
                    fallback_mgr = self.get_fallback_mgr()
                    result = fallback_mgr.process_frame(frame)
                else:
                    result = frame.copy()
                    for task_meta in frame_meta["tasks"]:
                        fake_frame = read_image(input_dir / f"{task_meta['cache_key']}.png")
                        enhanced_frame = None
                        if self.enhancer_name is not None:
                            enhanced_frame = read_image(enhance_dir / f"{task_meta['cache_key']}.png")
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
                self.update_progress(f"Compositing {os.path.basename(entry.filename)}")
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
