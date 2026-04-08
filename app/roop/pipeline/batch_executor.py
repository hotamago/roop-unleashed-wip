import os
import cv2 
import numpy as np
import psutil
import time

from roop.pipeline.options import ProcessOptions

from roop.face import align_crop, get_first_face, get_all_faces, rotate_anticlockwise, rotate_clockwise, clamp_cut_values
from roop.utils import compute_cosine_distance, get_device, str_to_class
from roop.memory import resolve_single_batch_workers
import roop.utils.vr as vr

from typing import Any, List, Callable
from roop.config.types import Frame, Face
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock
from queue import Queue
from tqdm import tqdm
from roop.media.ffmpeg_writer import FFMPEG_VideoWriter
from roop.media.stream_writer import StreamWriter
from roop.media.video_io import open_video_capture, resolve_video_writer_config
import roop.config.globals
from roop.progress.status import format_duration, get_processing_status_line, publish_processing_progress



# Poor man's enum to be able to compare to int
class eNoFaceAction():
    USE_ORIGINAL_FRAME = 0
    RETRY_ROTATED = 1
    SKIP_FRAME = 2
    SKIP_FRAME_IF_DISSIMILAR = 3,
    USE_LAST_SWAPPED = 4



def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues



class ProcessMgr():
    plugins = {
        'faceswap'          : 'FaceSwapInsightFace',
        'mask_clip2seg'     : 'Mask_Clip2Seg',
        'mask_xseg'         : 'Mask_XSeg',
        'codeformer'        : 'Enhance_CodeFormer',
        'gfpgan'            : 'Enhance_GFPGAN',
        'dmdnet'            : 'Enhance_DMDNet',
        'gpen'              : 'Enhance_GPEN',
        'restoreformer++'   : 'Enhance_RestoreFormerPPlus',
        'colorizer'         : 'Frame_Colorizer',
        'filter_generic'    : 'Frame_Filter',
        'removebg'          : 'Frame_Masking',
        'upscale'           : 'Frame_Upscale'
    }

    def __init__(self, progress):
        # FIX: All mutable state as instance attributes (previously class-level,
        # which caused processor/model references to persist across ProcessMgr instances
        # and prevented VRAM from being released between runs).
        self.input_face_datas = []
        self.target_face_datas = []
        self.imagemask = None
        self.processors = []
        self.options = None
        self.num_threads = 1
        self.current_index = 0
        self.processing_threads = 1
        self.buffer_wait_time = 0.1
        self.lock = Lock()
        self.frames_queue = None
        self.processed_queue = None
        self.videowriter = None
        self.streamwriter = None
        self.progress_gradio = None
        self.total_frames = 0
        self.num_frames_no_face = 0
        self.last_swapped_frame = None
        self.output_to_file = None
        self.output_to_cam = None
        self.progress_stage = "processing"
        self.progress_target_name = None
        self.progress_file_index = None
        self.progress_file_total = None
        self.progress_unit = "frames"
        self.progress_started_at = None
        self.single_batch_worker_pools = {}

        if progress is not None:
            self.progress_gradio = progress

    def reuseOldProcessor(self, name:str):
        for p in self.processors:
            if p.processorname == name:
                return p
        return None


    def resolve_face_analysis_modules(self):
        modules = ["landmark_3d_68", "landmark_2d_106", "detection"]
        swap_mode = getattr(self.options, "swap_mode", None)
        if swap_mode == "selected":
            modules.append("recognition")
        if swap_mode == "all_female" or swap_mode == "all_male":
            modules.append("genderage")
        return modules


    def initialize(self, input_faces, target_faces, options):
        self.release_single_batch_worker_pools()
        self.input_face_datas = input_faces
        self.target_face_datas = target_faces
        self.num_frames_no_face = 0
        self.last_swapped_frame = None
        self.options = options
        devicename = get_device()

        roop.config.globals.g_desired_face_analysis = self.resolve_face_analysis_modules()

        for p in self.processors:
            newp = next((x for x in options.processors.keys() if x == p.processorname), None)
            if newp is None:
                p.Release()
                del p

        newprocessors = []
        for key, extoption in options.processors.items():
            p = self.reuseOldProcessor(key)
            if p is None:
                classname = self.plugins[key]
                module = 'roop.processors.' + classname
                p = str_to_class(module, classname)
            if p is not None:
                extoption.update({"devicename": devicename})
                p.Initialize(extoption)
                newprocessors.append(p)
            else:
                print(f"Not using {module}")
        self.processors = newprocessors

        if isinstance(self.options.imagemask, dict) and self.options.imagemask.get("layers") and len(self.options.imagemask["layers"]) > 0:
            self.options.imagemask = self.options.imagemask.get("layers")[0]
            # Get rid of alpha
            self.options.imagemask = cv2.cvtColor(self.options.imagemask, cv2.COLOR_RGBA2GRAY)
            if np.any(self.options.imagemask):
                mo = self.input_face_datas[0].faces[0].mask_offsets
                self.options.imagemask = self.blur_area(self.options.imagemask, mo[4])
                self.options.imagemask = self.options.imagemask.astype(np.float32) / 255
                self.options.imagemask = cv2.cvtColor(self.options.imagemask, cv2.COLOR_GRAY2RGB)
            else:
                self.options.imagemask = None

        self.options.frame_processing = False
        for p in self.processors:
            if p.type.startswith("frame_"):
                self.options.frame_processing = True


    def serialize_face(self, face: Face):
        data = {}
        for key in ("bbox", "kps", "landmark_2d_106"):
            value = getattr(face, key, None)
            if value is None:
                try:
                    value = face[key]
                except Exception:
                    value = None
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                data[key] = value.astype(np.float32, copy=False)
            else:
                data[key] = np.asarray(value, dtype=np.float32)
        matrix = getattr(face, "matrix", None)
        if matrix is not None:
            data["matrix"] = matrix.astype(np.float32, copy=False) if isinstance(matrix, np.ndarray) else np.asarray(matrix, dtype=np.float32)
        if hasattr(face, "sex"):
            data["sex"] = face.sex
        return data


    def deserialize_face(self, data):
        class FaceProxy(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

        face = FaceProxy()
        for key in ("bbox", "kps", "landmark_2d_106", "embedding", "matrix"):
            value = data.get(key)
            if value is None:
                continue
            face[key] = np.array(value, dtype=np.float32)
        if "sex" in data:
            face["sex"] = data["sex"]
        return face

    def _get_selected_input_index(self):
        if not self.input_face_datas:
            return None
        selected_index = self.options.selected_index
        if selected_index < 0:
            return 0
        max_index = len(self.input_face_datas) - 1
        return max(0, min(selected_index, max_index))


    def get_frame_face_targets(self, frame: Frame):
        tasks = []
        selected_input_index = self._get_selected_input_index()
        if self.options.swap_mode == "first":
            face = get_first_face(frame)
            if face is None:
                return tasks
            if selected_input_index is None:
                return tasks
            tasks.append((selected_input_index, face))
            return tasks

        faces = get_all_faces(frame)
        if faces is None:
            return tasks

        if self.options.swap_mode == "all":
            for face in faces:
                if selected_input_index is None:
                    break
                tasks.append((selected_input_index, face))
        elif self.options.swap_mode == "all_input":
            for i, face in enumerate(faces):
                if i >= len(self.input_face_datas):
                    break
                tasks.append((i, face))
        elif self.options.swap_mode == "selected":
            num_targetfaces = len(self.target_face_datas)
            use_index = num_targetfaces == 1
            if use_index and selected_input_index is None:
                return tasks
            for i, tf in enumerate(self.target_face_datas):
                for face in faces:
                    if compute_cosine_distance(tf.embedding, face.embedding) <= self.options.face_distance_threshold:
                        if i < len(self.input_face_datas):
                            input_index = selected_input_index if use_index else i
                            tasks.append((input_index, face))
                        if not roop.config.globals.vr_mode and len(tasks) == num_targetfaces:
                            break
        elif self.options.swap_mode == "all_female" or self.options.swap_mode == "all_male":
            gender = 'F' if self.options.swap_mode == "all_female" else 'M'
            for face in faces:
                if face.sex == gender:
                    if selected_input_index is None:
                        break
                    tasks.append((selected_input_index, face))

        if roop.config.globals.vr_mode and len(tasks) % 2 > 0:
            return []
        return tasks


    def prepare_face_task(self, face_index, target_face: Face, frame: Frame):
        inputface = self.input_face_datas[face_index].faces[0] if len(self.input_face_datas) > face_index else None
        working_frame = frame
        working_face = target_face
        rotation_action = None
        cutout_box = None
        if roop.config.globals.autorotate_faces:
            rotation_action = self.rotation_action(target_face, frame)
            if rotation_action is not None:
                (start_x, start_y, end_x, end_y) = target_face["bbox"].astype("int")
                width = end_x - start_x
                height = end_y - start_y
                offs = int(max(width, height) * 0.25)
                rotcutframe, start_x, start_y, end_x, end_y = self.cutout(frame, start_x - offs, start_y - offs, end_x + offs, end_y + offs)
                if rotation_action == "rotate_anticlockwise":
                    rotcutframe = rotate_anticlockwise(rotcutframe)
                elif rotation_action == "rotate_clockwise":
                    rotcutframe = rotate_clockwise(rotcutframe)
                rotface = get_first_face(rotcutframe)
                if rotface is None:
                    rotation_action = None
                else:
                    working_frame = rotcutframe
                    working_face = rotface
                    cutout_box = [start_x, start_y, end_x, end_y]

        aligned_img, matrix = align_crop(working_frame, working_face.kps, self.options.subsample_size)
        working_face.matrix = matrix
        mask_offsets = [0, 0, 0, 0, 20.0, 10.0, 1.0, 1.0, 1.0, 1.0]
        if inputface is not None and hasattr(inputface, "mask_offsets"):
            mask_offsets = list(inputface.mask_offsets)
        while len(mask_offsets) < 10:
            mask_offsets.append(1.0)
        return {
            "input_index": face_index,
            "aligned_frame": aligned_img,
            "target_face": self.serialize_face(working_face),
            "rotation_action": rotation_action,
            "cutout_box": cutout_box,
            "mask_offsets": mask_offsets,
        }


    def build_frame_plan(self, frame: Frame):
        selected_faces = self.get_frame_face_targets(frame)
        tasks = [self.prepare_face_task(face_index, face, frame) for face_index, face in selected_faces]
        fallback_required = len(tasks) == 0
        if roop.config.globals.no_face_action == eNoFaceAction.SKIP_FRAME_IF_DISSIMILAR and 0 < len(tasks) < len(self.input_face_datas):
            fallback_required = True
            tasks = []
        return {"tasks": tasks, "fallback": fallback_required}


    def rebuild_aligned_frame(self, frame: Frame, task):
        target_face = self.deserialize_face(task["target_face"])
        working_frame = frame
        rotation_action = task.get("rotation_action")
        cutout_box = task.get("cutout_box")
        if rotation_action is not None and cutout_box is not None:
            cutout_frame, _, _, _, _ = self.cutout(frame, cutout_box[0], cutout_box[1], cutout_box[2], cutout_box[3])
            if rotation_action == "rotate_anticlockwise":
                working_frame = rotate_anticlockwise(cutout_frame)
            elif rotation_action == "rotate_clockwise":
                working_frame = rotate_clockwise(cutout_frame)
            else:
                working_frame = cutout_frame
        aligned_img, _ = align_crop(working_frame, target_face.kps, self.options.subsample_size)
        return aligned_img


    def run_swap_task(self, task, processor, batch_size: int = 1):
        inputface = self.input_face_datas[task["input_index"]].faces[0] if len(self.input_face_datas) > task["input_index"] else None
        target_face = self.deserialize_face(task["target_face"])
        model_output_size = 128
        subsample_total = self.options.subsample_size // model_output_size
        current_frames = list(self.implode_pixel_boost(task["aligned_frame"], model_output_size, subsample_total))
        for _ in range(0, self.options.num_swap_steps):
            prepared_frames = [self.prepare_crop_frame(frame) for frame in current_frames]
            if getattr(processor, "supports_batch", False):
                raw_outputs = processor.RunBatch([inputface] * len(prepared_frames), [target_face] * len(prepared_frames), prepared_frames, max(1, batch_size))
                current_frames = [self.normalize_swap_frame(output) for output in raw_outputs]
            else:
                next_frames = []
                for prepared_frame in prepared_frames:
                    output = processor.Run(inputface, target_face, prepared_frame)
                    next_frames.append(self.normalize_swap_frame(output))
                current_frames = next_frames
        fake_frame = self.explode_pixel_boost(current_frames, model_output_size, subsample_total, self.options.subsample_size)
        return fake_frame.astype(np.uint8)


    def run_swap_single_outputs(self, processor, source_faces, target_faces, prepared_frames):
        return [
            self.normalize_swap_frame(processor.Run(source_face, target_face, prepared_frame))
            for source_face, target_face, prepared_frame in zip(source_faces, target_faces, prepared_frames)
        ]


    def disable_broken_swap_batch(self, processor):
        processor._batch_output_validation_state = "broken"
        processor.batch_size_limit = 1
        processor.supports_batch = False
        if hasattr(processor, "supports_parallel_single_batch"):
            processor.supports_parallel_single_batch = False


    def run_swap_tasks_batch(self, tasks, processor, batch_size: int = 1):
        if not tasks:
            return {}
        model_output_size = 128
        subsample_total = max(self.options.subsample_size // model_output_size, 1)
        prepared_tasks = []
        for task in tasks:
            inputface = self.input_face_datas[task["input_index"]].faces[0] if len(self.input_face_datas) > task["input_index"] else None
            target_face = self.deserialize_face(task["target_face"])
            current_frames = list(self.implode_pixel_boost(task["aligned_frame"], model_output_size, subsample_total))
            prepared_tasks.append({
                "cache_key": task["cache_key"],
                "input_face": inputface,
                "target_face": target_face,
                "current_frames": current_frames,
            })

        if self.should_parallelize_single_batch(processor):
            return self.run_swap_tasks_parallel_single_batch(prepared_tasks, processor)

        for _ in range(0, self.options.num_swap_steps):
            prepared_frames = []
            source_faces = []
            target_faces = []
            frame_map = []
            for task_index, prepared_task in enumerate(prepared_tasks):
                for slice_index, current_frame in enumerate(prepared_task["current_frames"]):
                    prepared_frames.append(self.prepare_crop_frame(current_frame))
                    source_faces.append(prepared_task["input_face"])
                    target_faces.append(prepared_task["target_face"])
                    frame_map.append((task_index, slice_index))

            if getattr(processor, "supports_batch", False):
                raw_outputs = processor.RunBatch(source_faces, target_faces, prepared_frames, max(1, batch_size))
                if len(raw_outputs) != len(prepared_frames):
                    self.disable_broken_swap_batch(processor)
                    normalized_outputs = self.run_swap_single_outputs(processor, source_faces, target_faces, prepared_frames)
                else:
                    normalized_outputs = [self.normalize_swap_frame(output) for output in raw_outputs]
                    normalized_outputs = self.validate_swap_batch_outputs(
                        processor,
                        source_faces,
                        target_faces,
                        prepared_frames,
                        normalized_outputs,
                    )
            else:
                normalized_outputs = self.run_swap_single_outputs(processor, source_faces, target_faces, prepared_frames)

            for (task_index, slice_index), normalized_output in zip(frame_map, normalized_outputs):
                prepared_tasks[task_index]["current_frames"][slice_index] = normalized_output

        outputs = {}
        for prepared_task in prepared_tasks:
            fake_frame = self.explode_pixel_boost(prepared_task["current_frames"], model_output_size, subsample_total, self.options.subsample_size)
            outputs[prepared_task["cache_key"]] = fake_frame.astype(np.uint8)
        return outputs


    def validate_swap_batch_outputs(self, processor, source_faces, target_faces, prepared_frames, normalized_outputs):
        if len(prepared_frames) <= 1:
            return normalized_outputs
        if getattr(processor, "_batch_output_validation_state", None) == "verified":
            return normalized_outputs
        if getattr(processor, "_batch_output_validation_state", None) == "broken":
            return self.run_swap_single_outputs(processor, source_faces, target_faces, prepared_frames)

        reference_output = self.normalize_swap_frame(processor.Run(source_faces[0], target_faces[0], prepared_frames[0]))
        batch_output = normalized_outputs[0]
        if np.allclose(batch_output.astype(np.float32), reference_output.astype(np.float32), atol=1.0):
            processor._batch_output_validation_state = "verified"
            return normalized_outputs

        self.disable_broken_swap_batch(processor)
        return [
            reference_output,
            *[
                self.normalize_swap_frame(processor.Run(source_face, target_face, prepared_frame))
                for source_face, target_face, prepared_frame in zip(source_faces[1:], target_faces[1:], prepared_frames[1:])
            ],
        ]


    def should_parallelize_single_batch(self, processor):
        batch_limit = getattr(processor, "batch_size_limit", None)
        supports_parallel_single_batch = getattr(processor, "supports_parallel_single_batch", False)
        return supports_parallel_single_batch and batch_limit == 1 and hasattr(processor, "CreateWorkerProcessor")


    def get_single_batch_worker_count(self, processor):
        if not self.should_parallelize_single_batch(processor):
            return 1
        active_plan = getattr(roop.config.globals, "active_memory_plan", None)
        if isinstance(active_plan, dict):
            explicit_workers = active_plan.get("single_batch_workers")
            if explicit_workers is not None:
                return max(1, int(explicit_workers))
        effective_workers, _, _ = resolve_single_batch_workers(getattr(roop.config.globals.CFG, "single_batch_workers", 1))
        return max(1, int(effective_workers))


    def get_single_batch_worker_processors(self, processor, worker_count):
        worker_count = max(1, int(worker_count))
        pool_key = id(processor)
        pool = self.single_batch_worker_pools.get(pool_key)
        if pool is None or pool.get("base") is not processor:
            pool = {
                "base": processor,
                "workers": [processor],
            }
            self.single_batch_worker_pools[pool_key] = pool

        workers = pool["workers"]
        while len(workers) < worker_count:
            workers.append(processor.CreateWorkerProcessor())
        while len(workers) > worker_count:
            extra_worker = workers.pop()
            if extra_worker is not processor:
                try:
                    extra_worker.Release()
                except Exception:
                    pass
        return list(workers)


    def release_single_batch_worker_pools(self):
        for pool in self.single_batch_worker_pools.values():
            workers = pool.get("workers", [])
            for worker_processor in workers[1:]:
                try:
                    worker_processor.Release()
                except Exception:
                    pass
        self.single_batch_worker_pools.clear()


    def run_tasks_parallel_single_batch(self, prepared_tasks, processor, run_task):
        if not prepared_tasks:
            return {}
        configured_worker_count = self.get_single_batch_worker_count(processor)
        if configured_worker_count <= 1 or len(prepared_tasks) <= 1:
            outputs = {}
            for prepared_task in prepared_tasks:
                outputs[prepared_task["cache_key"]] = run_task(prepared_task, processor)
            return outputs

        worker_processors = self.get_single_batch_worker_processors(processor, configured_worker_count)
        active_worker_processors = worker_processors[:min(len(prepared_tasks), configured_worker_count)]
        task_queue = Queue(maxsize=max(len(active_worker_processors) * 2, 2))
        outputs = {}
        output_lock = Lock()
        worker_error = {"exc": None}

        def worker_loop(worker_processor):
            while True:
                item = task_queue.get()
                try:
                    if item is None:
                        return
                    if worker_error["exc"] is not None:
                        continue
                    prepared_task = item
                    result = run_task(prepared_task, worker_processor)
                    with output_lock:
                        outputs[prepared_task["cache_key"]] = result
                except Exception as exc:
                    with output_lock:
                        if worker_error["exc"] is None:
                            worker_error["exc"] = exc
                finally:
                    task_queue.task_done()

        worker_threads = [Thread(target=worker_loop, args=(worker_processor,), daemon=True) for worker_processor in active_worker_processors]
        for worker_thread in worker_threads:
            worker_thread.start()
        for prepared_task in prepared_tasks:
            task_queue.put(prepared_task, block=True)
        for _ in worker_threads:
            task_queue.put(None, block=True)
        task_queue.join()
        for worker_thread in worker_threads:
            worker_thread.join()
        if worker_error["exc"] is not None:
            raise worker_error["exc"]
        return outputs


    def run_prepared_swap_task(self, prepared_task, processor):
        current_frames = [frame.copy() for frame in prepared_task["current_frames"]]
        for _ in range(0, self.options.num_swap_steps):
            next_frames = []
            for current_frame in current_frames:
                prepared_frame = self.prepare_crop_frame(current_frame)
                output = processor.Run(prepared_task["input_face"], prepared_task["target_face"], prepared_frame)
                next_frames.append(self.normalize_swap_frame(output))
            current_frames = next_frames
        model_output_size = 128
        subsample_total = max(self.options.subsample_size // model_output_size, 1)
        fake_frame = self.explode_pixel_boost(current_frames, model_output_size, subsample_total, self.options.subsample_size)
        return fake_frame.astype(np.uint8)


    def run_swap_tasks_parallel_single_batch(self, prepared_tasks, processor):
        return self.run_tasks_parallel_single_batch(prepared_tasks, processor, self.run_prepared_swap_task)


    def run_mask_task(self, task, current_frame, processor):
        return self.process_mask(processor, task["aligned_frame"], current_frame)


    def run_enhance_task(self, task, current_frame, processor):
        target_face = self.deserialize_face(task["target_face"])
        input_faceset = self.input_face_datas[task["input_index"]] if len(self.input_face_datas) > task["input_index"] else None
        return processor.Run(input_faceset, target_face, current_frame)


    def run_prepared_enhance_task(self, prepared_task, processor):
        enhanced_frame, _ = processor.Run(prepared_task["input_faceset"], prepared_task["target_face"], prepared_task["current_frame"])
        return enhanced_frame


    def run_enhance_tasks_batch(self, tasks, current_frames, processor, batch_size: int = 1):
        if not tasks:
            return {}
        prepared_tasks = []
        for task, current_frame in zip(tasks, current_frames):
            prepared_tasks.append({
                "cache_key": task["cache_key"],
                "input_faceset": self.input_face_datas[task["input_index"]] if len(self.input_face_datas) > task["input_index"] else None,
                "target_face": self.deserialize_face(task["target_face"]),
                "current_frame": current_frame,
            })

        if self.should_parallelize_single_batch(processor):
            return self.run_tasks_parallel_single_batch(prepared_tasks, processor, self.run_prepared_enhance_task)

        if getattr(processor, "supports_batch", False):
            source_sets = [prepared_task["input_faceset"] for prepared_task in prepared_tasks]
            target_faces = [prepared_task["target_face"] for prepared_task in prepared_tasks]
            batch_frames = [prepared_task["current_frame"] for prepared_task in prepared_tasks]
            enhanced_frames = processor.RunBatch(source_sets, target_faces, batch_frames, max(1, batch_size))
            return {
                prepared_task["cache_key"]: enhanced_frame
                for prepared_task, enhanced_frame in zip(prepared_tasks, enhanced_frames)
            }

        outputs = {}
        for prepared_task in prepared_tasks:
            outputs[prepared_task["cache_key"]] = self.run_prepared_enhance_task(prepared_task, processor)
        return outputs


    def compose_task(self, base_frame, task, fake_frame, enhanced_frame=None):
        target_face = self.deserialize_face(task["target_face"])
        target_img = base_frame
        rotation_action = task.get("rotation_action")
        cutout_box = task.get("cutout_box")
        if rotation_action is not None and cutout_box is not None:
            cutout_frame, _, _, _, _ = self.cutout(base_frame, cutout_box[0], cutout_box[1], cutout_box[2], cutout_box[3])
            if rotation_action == "rotate_anticlockwise":
                target_img = rotate_anticlockwise(cutout_frame)
            elif rotation_action == "rotate_clockwise":
                target_img = rotate_clockwise(cutout_frame)
            else:
                target_img = cutout_frame

        upscale = 512
        orig_width = fake_frame.shape[1]
        if orig_width != upscale:
            fake_frame = cv2.resize(fake_frame, (upscale, upscale), cv2.INTER_CUBIC)
        scale_factor = int(upscale / max(orig_width, 1))
        face_lm = target_face.landmark_2d_106 if hasattr(target_face, 'landmark_2d_106') and target_face.landmark_2d_106 is not None else None
        if enhanced_frame is None:
            result = self.paste_upscale(fake_frame, fake_frame, target_face.matrix, target_img, scale_factor, task["mask_offsets"], face_landmarks=face_lm)
        else:
            result = self.paste_upscale(fake_frame, enhanced_frame, target_face.matrix, target_img, scale_factor, task["mask_offsets"], face_landmarks=face_lm)

        if self.options.restore_original_mouth:
            mouth_cutout, mouth_bb, mouth_polygon = self.create_mouth_mask(target_face, target_img, task["mask_offsets"])
            result = self.apply_mouth_area(result, mouth_cutout, mouth_bb, mouth_polygon, task["mask_offsets"][5])

        if rotation_action is not None and cutout_box is not None:
            unrotated = self.auto_unrotate_frame(result, rotation_action)
            return self.paste_simple(unrotated, base_frame, cutout_box[0], cutout_box[1])
        return result


    def run_batch(self, source_files, target_files, threads:int = 1):
        progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        self.total_frames = len(source_files)
        self.num_threads = threads
        if self.progress_started_at is None:
            self.progress_started_at = time.time()
        with tqdm(total=self.total_frames, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = []
                queue = create_queue(source_files)
                queue_per_future = max(len(source_files) // threads, 1)
                while not queue.empty():
                    future = executor.submit(self.process_frames, source_files, target_files, pick_queue(queue, queue_per_future), lambda: self.update_progress(progress))
                    futures.append(future)
                for future in as_completed(futures):
                    future.result()


    def process_frames(self, source_files: List[str], target_files: List[str], current_files, update: Callable[[], None]) -> None:
        for f in current_files:
            if not roop.config.globals.processing:
                return
            temp_frame = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
            if temp_frame is not None:
                if self.options.frame_processing:
                    for p in self.processors:
                        frame = p.Run(temp_frame)
                    resimg = frame
                else:
                    resimg = self.process_frame(temp_frame)
                if resimg is not None:
                    i = source_files.index(f)
                    cv2.imwrite(target_files[i], resimg)
            if update:
                update()


    def read_frames_thread(self, cap, frame_start, frame_end, num_threads):
        num_frame = 0
        total_num = frame_end - frame_start
        if frame_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        while True and roop.config.globals.processing:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames_queue[num_frame % num_threads].put(frame, block=True)
            num_frame += 1
            if num_frame == total_num:
                break

        for i in range(num_threads):
            self.frames_queue[i].put(None)


    def process_videoframes(self, threadindex, progress) -> None:
        while True:
            frame = self.frames_queue[threadindex].get()
            if frame is None:
                self.processing_threads -= 1
                self.processed_queue[threadindex].put((False, None))
                return
            else:
                if self.options.frame_processing:
                    for p in self.processors:
                        frame = p.Run(frame)
                    resimg = frame
                else:
                    resimg = self.process_frame(frame)
                self.processed_queue[threadindex].put((True, resimg))
                del frame
                progress()


    def write_frames_thread(self):
        nextindex = 0
        num_producers = self.num_threads
        
        while True:
            process, frame = self.processed_queue[nextindex % self.num_threads].get()
            nextindex += 1
            if frame is not None:
                if self.output_to_file:
                    self.videowriter.write_frame(frame)
                if self.output_to_cam:
                    self.streamwriter.WriteToStream(frame)
                del frame
            elif process == False:
                num_producers -= 1
                if num_producers < 1:
                    return


    def run_batch_inmem(self, output_method, source_video, target_video, frame_start, frame_end, fps, threads:int = 1, skip_audio=False):
        cap = open_video_capture(source_video)
        frame_count = (frame_end - frame_start) + 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        processed_resolution = None
        for p in self.processors:
            if hasattr(p, 'getProcessedResolution'):
                processed_resolution = p.getProcessedResolution(width, height)
                print(f"Processed resolution: {processed_resolution}")
        if processed_resolution is not None:
            width = processed_resolution[0]
            height = processed_resolution[1]

        self.total_frames = frame_count
        self.num_threads = threads
        self.processing_threads = self.num_threads
        if self.progress_started_at is None:
            self.progress_started_at = time.time()
        self.frames_queue = []
        self.processed_queue = []
        for _ in range(threads):
            self.frames_queue.append(Queue(1))
            self.processed_queue.append(Queue(1))

        self.output_to_file = output_method != "Virtual Camera"
        self.output_to_cam = output_method == "Virtual Camera" or output_method == "Both"

        if self.output_to_file:
            writer_config = resolve_video_writer_config(roop.config.globals.video_encoder, roop.config.globals.video_quality)
            self.videowriter = FFMPEG_VideoWriter(
                target_video,
                (width, height),
                fps,
                codec=writer_config["codec"],
                crf=roop.config.globals.video_quality,
                audiofile=None,
                ffmpeg_params=writer_config["ffmpeg_params"],
                quality_args=writer_config["quality_args"],
            )
        if self.output_to_cam:
            self.streamwriter = StreamWriter((width, height), int(fps))

        readthread = Thread(target=self.read_frames_thread, args=(cap, frame_start, frame_end, threads))
        readthread.start()

        writethread = Thread(target=self.write_frames_thread)
        writethread.start()

        progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        with tqdm(total=self.total_frames, desc='Processing', unit='frames', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
            with ThreadPoolExecutor(thread_name_prefix='swap_proc', max_workers=self.num_threads) as executor:
                futures = []
                for threadindex in range(threads):
                    future = executor.submit(self.process_videoframes, threadindex, lambda: self.update_progress(progress))
                    futures.append(future)
                for future in as_completed(futures):
                    future.result()

        readthread.join()
        writethread.join()
        cap.release()
        if self.output_to_file:
            self.videowriter.close()
            self.videowriter = None  # FIX: null out so GC can collect
        if self.output_to_cam:
            self.streamwriter.Close()
            self.streamwriter = None  # FIX: null out so GC can collect

        self.frames_queue.clear()
        self.processed_queue.clear()


    def update_progress(self, progress: Any = None) -> None:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        progress.update(1)
        elapsed = progress.format_dict.get("elapsed")
        rate = progress.format_dict.get("rate")
        if (rate is None or rate <= 0) and elapsed and progress.n > 0:
            rate = progress.n / max(elapsed, 1e-9)
        eta = None
        if rate is not None and rate > 0 and self.total_frames > 0:
            eta = max(self.total_frames - progress.n, 0) / rate
        postfix = {
            'memory': '{:.2f}'.format(memory_usage).zfill(5) + 'GB',
            'workers': self.num_threads,
            'stage': self.progress_stage.replace('_', ' ')
        }
        if eta is not None:
            postfix['eta'] = format_duration(eta)
        if self.progress_target_name:
            postfix['file'] = os.path.basename(self.progress_target_name)
        progress.set_postfix(postfix)
        publish_processing_progress(
            stage=self.progress_stage,
            completed=progress.n,
            total=self.total_frames,
            unit=self.progress_unit,
            target_name=self.progress_target_name,
            file_index=self.progress_file_index,
            total_files=self.progress_file_total,
            rate=rate,
            elapsed=elapsed,
            eta=eta,
            detail=f"CPU workers: {self.num_threads}",
            memory_status=roop.config.globals.runtime_memory_status,
        )
        if self.progress_gradio is not None:
            self.progress_gradio((progress.n, self.total_frames), desc=get_processing_status_line(), total=self.total_frames, unit=self.progress_unit)


    def set_progress_context(self, stage, target_name=None, file_index=None, total_files=None, unit='frames'):
        self.progress_stage = stage
        self.progress_target_name = target_name
        self.progress_file_index = file_index
        self.progress_file_total = total_files
        self.progress_unit = unit
        self.progress_started_at = time.time()


    def process_frame(self, frame:Frame):
        if len(self.input_face_datas) < 1 and not self.options.show_face_masking:
            return frame
        temp_frame = frame.copy()
        num_swapped, temp_frame = self.swap_faces(frame, temp_frame)
        if num_swapped > 0:
            if roop.config.globals.no_face_action == eNoFaceAction.SKIP_FRAME_IF_DISSIMILAR:
                if len(self.input_face_datas) > num_swapped:
                    return None
            self.num_frames_no_face = 0
            self.last_swapped_frame = temp_frame.copy()
            return temp_frame
        if roop.config.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
            if self.last_swapped_frame is not None and self.num_frames_no_face < self.options.max_num_reuse_frame:
                self.num_frames_no_face += 1
                return self.last_swapped_frame.copy()
            return frame
        elif roop.config.globals.no_face_action == eNoFaceAction.USE_ORIGINAL_FRAME:
            return frame
        if roop.config.globals.no_face_action == eNoFaceAction.SKIP_FRAME:
            return None
        else:
            return self.retry_rotated(frame)

    def retry_rotated(self, frame):
        copyframe = frame.copy()
        copyframe = rotate_clockwise(copyframe)
        temp_frame = copyframe.copy()
        num_swapped, temp_frame = self.swap_faces(copyframe, temp_frame)
        if num_swapped > 0:
            return rotate_anticlockwise(temp_frame)
        
        copyframe = frame.copy()
        copyframe = rotate_anticlockwise(copyframe)
        temp_frame = copyframe.copy()
        num_swapped, temp_frame = self.swap_faces(copyframe, temp_frame)
        if num_swapped > 0:
            return rotate_clockwise(temp_frame)
        del copyframe
        return frame


    def swap_faces(self, frame, temp_frame):
        num_faces_found = 0

        if self.options.swap_mode == "first":
            face = get_first_face(frame)
            if face is None:
                return num_faces_found, frame
            num_faces_found += 1
            temp_frame = self.process_face(self.options.selected_index, face, temp_frame)
            del face

        else:
            faces = get_all_faces(frame)
            if faces is None:
                return num_faces_found, frame
            
            if self.options.swap_mode == "all":
                for face in faces:
                    num_faces_found += 1
                    temp_frame = self.process_face(self.options.selected_index, face, temp_frame)

            elif self.options.swap_mode == "all_input":
                for i, face in enumerate(faces):
                    num_faces_found += 1
                    if i < len(self.input_face_datas):
                        temp_frame = self.process_face(i, face, temp_frame)
                    else:
                        break
            
            elif self.options.swap_mode == "selected":
                num_targetfaces = len(self.target_face_datas)
                use_index = num_targetfaces == 1
                for i, tf in enumerate(self.target_face_datas):
                    for face in faces:
                        if compute_cosine_distance(tf.embedding, face.embedding) <= self.options.face_distance_threshold:
                            if i < len(self.input_face_datas):
                                if use_index:
                                    temp_frame = self.process_face(self.options.selected_index, face, temp_frame)
                                else:
                                    temp_frame = self.process_face(i, face, temp_frame)
                                num_faces_found += 1
                            if not roop.config.globals.vr_mode and num_faces_found == num_targetfaces:
                                break

            elif self.options.swap_mode == "all_female" or self.options.swap_mode == "all_male":
                gender = 'F' if self.options.swap_mode == "all_female" else 'M'
                for face in faces:
                    if face.sex == gender:
                        num_faces_found += 1
                        temp_frame = self.process_face(self.options.selected_index, face, temp_frame)
            
            for face in faces:
                del face
            faces.clear()

        if roop.config.globals.vr_mode and num_faces_found % 2 > 0:
            num_faces_found = 0
            return num_faces_found, frame
        if num_faces_found == 0:
            return num_faces_found, frame

        if self.options.imagemask is not None and self.options.imagemask.shape == frame.shape:
            temp_frame = self.simple_blend_with_mask(temp_frame, frame, self.options.imagemask)
        return num_faces_found, temp_frame


    def rotation_action(self, original_face:Face, frame:Frame):
        (height, width) = frame.shape[:2]

        bounding_box_width = original_face.bbox[2] - original_face.bbox[0]
        bounding_box_height = original_face.bbox[3] - original_face.bbox[1]
        horizontal_face = bounding_box_width > bounding_box_height

        center_x = width // 2.0
        start_x = original_face.bbox[0]
        end_x = original_face.bbox[2]
        bbox_center_x = start_x + (bounding_box_width // 2.0)

        forehead_x = original_face.landmark_2d_106[72][0]
        chin_x = original_face.landmark_2d_106[0][0]

        if horizontal_face:
            if chin_x < forehead_x:
                return "rotate_anticlockwise"
            elif forehead_x < chin_x:
                return "rotate_clockwise"
            if bbox_center_x >= center_x:
                return "rotate_anticlockwise"
            if bbox_center_x < center_x:
                return "rotate_clockwise"

        return None


    def auto_rotate_frame(self, original_face, frame:Frame):
        target_face = original_face
        original_frame = frame
        rotation_action = self.rotation_action(original_face, frame)
        if rotation_action == "rotate_anticlockwise":
            frame = rotate_anticlockwise(frame)
        elif rotation_action == "rotate_clockwise":
            frame = rotate_clockwise(frame)
        return target_face, frame, rotation_action


    def auto_unrotate_frame(self, frame:Frame, rotation_action):
        if rotation_action == "rotate_anticlockwise":
            return rotate_clockwise(frame)
        elif rotation_action == "rotate_clockwise":
            return rotate_anticlockwise(frame)
        return frame


    def process_face(self, face_index, target_face:Face, frame:Frame):
        enhanced_frame = None
        if len(self.input_face_datas) > 0:
            inputface = self.input_face_datas[face_index].faces[0]
        else:
            inputface = None

        rotation_action = None
        if roop.config.globals.autorotate_faces:
            rotation_action = self.rotation_action(target_face, frame)
            if rotation_action is not None:
                (startX, startY, endX, endY) = target_face["bbox"].astype("int")
                width = endX - startX
                height = endY - startY
                offs = int(max(width, height) * 0.25)
                rotcutframe, startX, startY, endX, endY = self.cutout(frame, startX - offs, startY - offs, endX + offs, endY + offs)
                if rotation_action == "rotate_anticlockwise":
                    rotcutframe = rotate_anticlockwise(rotcutframe)
                elif rotation_action == "rotate_clockwise":
                    rotcutframe = rotate_clockwise(rotcutframe)
                rotface = get_first_face(rotcutframe)
                if rotface is None:
                    rotation_action = None
                else:
                    saved_frame = frame.copy()
                    frame = rotcutframe
                    target_face = rotface

        model_output_size = 128
        subsample_size = self.options.subsample_size
        subsample_total = subsample_size // model_output_size
        aligned_img, M = align_crop(frame, target_face.kps, subsample_size)

        fake_frame = aligned_img
        target_face.matrix = M

        for p in self.processors:
            if p.type == 'swap':
                swap_result_frames = []
                subsample_frames = self.implode_pixel_boost(aligned_img, model_output_size, subsample_total)
                for sliced_frame in subsample_frames:
                    for _ in range(0, self.options.num_swap_steps):
                        sliced_frame = self.prepare_crop_frame(sliced_frame)
                        sliced_frame = p.Run(inputface, target_face, sliced_frame)
                        sliced_frame = self.normalize_swap_frame(sliced_frame)
                    swap_result_frames.append(sliced_frame)
                fake_frame = self.explode_pixel_boost(swap_result_frames, model_output_size, subsample_total, subsample_size)
                fake_frame = fake_frame.astype(np.uint8)
                scale_factor = 0.0
            elif p.type == 'mask':
                fake_frame = self.process_mask(p, aligned_img, fake_frame)
            else:
                enhanced_frame, scale_factor = p.Run(self.input_face_datas[face_index], target_face, fake_frame)

        upscale = 512
        orig_width = fake_frame.shape[1]
        if orig_width != upscale:
            fake_frame = cv2.resize(fake_frame, (upscale, upscale), cv2.INTER_CUBIC)
        mask_offsets = [0, 0, 0, 0, 20.0, 10.0] if inputface is None else inputface.mask_offsets

        face_lm = target_face.landmark_2d_106 if hasattr(target_face, 'landmark_2d_106') and target_face.landmark_2d_106 is not None else None
        if enhanced_frame is None:
            scale_factor = int(upscale / orig_width)
            result = self.paste_upscale(fake_frame, fake_frame, target_face.matrix, frame, scale_factor, mask_offsets, face_landmarks=face_lm)
        else:
            result = self.paste_upscale(fake_frame, enhanced_frame, target_face.matrix, frame, scale_factor, mask_offsets, face_landmarks=face_lm)

        if self.options.restore_original_mouth:
            mouth_cutout, mouth_bb, mouth_polygon = self.create_mouth_mask(target_face, frame, mask_offsets)
            result = self.apply_mouth_area(result, mouth_cutout, mouth_bb, mouth_polygon, mask_offsets[5])

        if rotation_action is not None:
            fake_frame = self.auto_unrotate_frame(result, rotation_action)
            result = self.paste_simple(fake_frame, saved_frame, startX, startY)
        
        return result


    def cutout(self, frame:Frame, start_x, start_y, end_x, end_y):
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        if end_x > frame.shape[1]:
            end_x = frame.shape[1]
        if end_y > frame.shape[0]:
            end_y = frame.shape[0]
        return frame[start_y:end_y, start_x:end_x], start_x, start_y, end_x, end_y

    def paste_simple(self, src:Frame, dest:Frame, start_x, start_y):
        end_x = start_x + src.shape[1]
        end_y = start_y + src.shape[0]
        start_x, end_x, start_y, end_y = clamp_cut_values(start_x, end_x, start_y, end_y, dest)
        dest[start_y:end_y, start_x:end_x] = src
        return dest

    def simple_blend_with_mask(self, image1, image2, mask):
        blended_image = image1.astype(np.float32) * (1.0 - mask) + image2.astype(np.float32) * mask
        return blended_image.astype(np.uint8)


    def paste_upscale(self, fake_face, upsk_face, M, target_img, scale_factor, mask_offsets, face_landmarks=None):
        M_scale = M * scale_factor
        IM = cv2.invertAffineTransform(M_scale)

        img_matte = np.zeros((upsk_face.shape[0], upsk_face.shape[1]), dtype=np.uint8)

        w = img_matte.shape[1]
        h = img_matte.shape[0]

        top = int(mask_offsets[0] * h)
        bottom = int(h - (mask_offsets[1] * h))
        left = int(mask_offsets[2] * w)
        right = int(w - (mask_offsets[3] * w))
        # Ellipse avoids rectangular corners that create visible box seams
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        ax = max(1, (right - left) // 2)
        ay = max(1, (bottom - top) // 2)
        cv2.ellipse(img_matte, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        img_matte = cv2.warpAffine(img_matte, IM, (target_img.shape[1], target_img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0.0)
        img_matte[:1, :] = img_matte[-1:, :] = img_matte[:, :1] = img_matte[:, -1:] = 0

        # Constrain mask to actual face outline using landmark convex hull.
        # For angled/profile faces this prevents the warped ellipse from covering
        # background regions where the swap model put grey fill pixels.
        if face_landmarks is not None:
            lm_mask = self.create_landmark_mask(face_landmarks, target_img.shape, mask_offsets[4])
            img_matte = np.minimum(img_matte, lm_mask)

        img_matte = self.blur_area(img_matte, mask_offsets[4])
        img_matte = img_matte.astype(np.float32) / 255

        # Save 2D mask before reshape so overlay can use gradient values
        mask_2d = img_matte if self.options.show_face_area_overlay else None

        img_matte = np.reshape(img_matte, [img_matte.shape[0], img_matte.shape[1], 1])
        paste_face = cv2.warpAffine(upsk_face, IM, (target_img.shape[1], target_img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        if upsk_face is not fake_face:
            fake_face = cv2.warpAffine(fake_face, IM, (target_img.shape[1], target_img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            paste_face = cv2.addWeighted(paste_face, self.options.blend_ratio, fake_face, 1.0 - self.options.blend_ratio, 0)

        paste_face = img_matte * paste_face
        paste_face = paste_face + (1 - img_matte) * target_img.astype(np.float32)

        if self.options.show_face_area_overlay:
            # Gradient overlay: green in the core (maskâ‰ˆ1), yellow/orange at the
            # edge blend zone (maskâ‰ˆ0.5), invisible outside (maskâ‰ˆ0).
            # G channel scales with mask strength; R channel peaks mid-transition.
            overlay = np.zeros_like(target_img, dtype=np.uint8)
            overlay[:, :, 1] = (mask_2d * 200).astype(np.uint8)
            overlay[:, :, 2] = np.clip((1.0 - mask_2d) * mask_2d * 4 * 255, 0, 255).astype(np.uint8)
            paste_face = cv2.addWeighted(paste_face.astype(np.uint8), 0.6, overlay, 0.4, 0)

        return paste_face.astype(np.uint8)


    def blur_area(self, img_matte, face_mask_blend):
        # Always apply minimal anti-aliasing after the affine warp
        img_matte = cv2.GaussianBlur(img_matte, (3, 3), 0)
        if face_mask_blend <= 0:
            return img_matte
        mask_h_inds, mask_w_inds = np.where(img_matte > 127)
        if len(mask_h_inds) == 0 or len(mask_w_inds) == 0:
            return img_matte
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        # blend_px controls ONLY edge softness â€” no erosion, mask coverage unchanged
        blend_px = max(1, int(mask_size * face_mask_blend / 200))
        blur_size = blend_px * 2 + 1
        return cv2.GaussianBlur(img_matte, (blur_size, blur_size), 0)


    def create_landmark_mask(self, landmarks_2d, frame_shape, blend_amount):
        """Build a binary mask from the convex hull of the 106-pt face landmarks.

        Works in target-frame space so the shape naturally matches the actual
        visible face area regardless of yaw/pitch â€” unlike the ellipse which is
        computed in canonical 512Ã—512 face-space and can bleed past the face
        edge on profile shots.

        A forehead extension is added because the 106-pt model only reaches
        the eyebrow line; we project upward by ~60 % of the brow-to-chin
        distance so the full forehead is covered on frontal faces without
        over-extending on profiles.
        """
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        pts = landmarks_2d.astype(np.int32)

        # Eyebrow region is roughly indices 33-52; find the topmost y there.
        brow_pts = pts[33:53]
        top_brow_y = int(np.min(brow_pts[:, 1]))
        chin_y    = int(np.max(pts[:, 1]))
        face_h    = max(1, chin_y - top_brow_y)

        # Extend upward to cover the forehead.
        forehead_y = max(0, top_brow_y - int(face_h * 0.6))

        # Horizontal extent of the top of the face (near brow line).
        top_zone = pts[pts[:, 1] < top_brow_y + int(face_h * 0.15)]
        if len(top_zone) >= 2:
            left_x  = int(np.min(top_zone[:, 0]))
            right_x = int(np.max(top_zone[:, 0]))
        else:
            left_x  = int(np.min(pts[:, 0]))
            right_x = int(np.max(pts[:, 0]))

        forehead_pts = np.array([
            [left_x,                    forehead_y],
            [(left_x + right_x) // 2,  forehead_y],
            [right_x,                   forehead_y],
        ], dtype=np.int32)

        all_pts = np.vstack([pts, forehead_pts])
        hull    = cv2.convexHull(all_pts)
        cv2.fillConvexPoly(mask, hull, 255)

        # Dilate slightly so the hull doesn't clip skin right at the landmark
        # boundary â€” especially at jaw/temple edges.
        if blend_amount > 0:
            face_w    = max(1, right_x - left_x)
            expand_px = max(1, int(np.sqrt(face_h * face_w) * blend_amount / 400))
            kernel    = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask


    def prepare_crop_frame(self, swap_frame):
        model_type = 'inswapper'
        model_mean = [0.0, 0.0, 0.0]
        model_standard_deviation = [1.0, 1.0, 1.0]

        if model_type == 'ghost':
            swap_frame = swap_frame[:, :, ::-1] / 127.5 - 1
        else:
            swap_frame = swap_frame[:, :, ::-1] / 255.0
        swap_frame = (swap_frame - model_mean) / model_standard_deviation
        swap_frame = swap_frame.transpose(2, 0, 1)
        swap_frame = np.expand_dims(swap_frame, axis=0).astype(np.float32)
        return swap_frame


    def normalize_swap_frame(self, swap_frame):
        model_type = 'inswapper'
        swap_frame = swap_frame.transpose(1, 2, 0)
        if model_type == 'ghost':
            swap_frame = (swap_frame * 127.5 + 127.5).round()
        else:
            swap_frame = (swap_frame * 255.0).round()
        swap_frame = swap_frame[:, :, ::-1]
        return swap_frame

    def implode_pixel_boost(self, aligned_face_frame, model_size, pixel_boost_total:int):
        subsample_frame = aligned_face_frame.reshape(model_size, pixel_boost_total, model_size, pixel_boost_total, 3)
        subsample_frame = subsample_frame.transpose(1, 3, 0, 2, 4).reshape(pixel_boost_total ** 2, model_size, model_size, 3)
        return subsample_frame

    def explode_pixel_boost(self, subsample_frame, model_size, pixel_boost_total, pixel_boost_size):
        final_frame = np.stack(subsample_frame, axis=0).reshape(pixel_boost_total, pixel_boost_total, model_size, model_size, 3)
        final_frame = final_frame.transpose(2, 0, 3, 1, 4).reshape(pixel_boost_size, pixel_boost_size, 3)
        return final_frame

    def process_mask(self, processor, frame:Frame, target:Frame):
        img_mask = processor.Run(frame, self.options.masking_text)
        img_mask = cv2.resize(img_mask, (target.shape[1], target.shape[0]))
        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])

        if self.options.show_face_masking:
            result = (1 - img_mask) * frame.astype(np.float32)
            return np.uint8(result)

        target = target.astype(np.float32)
        result = (1 - img_mask) * target
        result += img_mask * frame.astype(np.float32)
        return np.uint8(result)


    def create_mouth_mask(self, face:Face, frame:Frame, mask_offsets=None):
        mouth_cutout = None
        mouth_mask_points = None
        # Initialize so the return is always safe even when landmarks is absent
        min_x, min_y, max_x, max_y = 0, 0, 0, 0
        # Scale factors for each side of the mouth bounding box (indices 6-9).
        # 1.0 = default padding; 2.0 = double padding (larger mouth region).
        if mask_offsets is not None and len(mask_offsets) >= 10:
            s_top, s_bot, s_left, s_right = mask_offsets[6], mask_offsets[7], mask_offsets[8], mask_offsets[9]
        else:
            s_top = s_bot = s_left = s_right = 1.0
        landmarks = face.landmark_2d_106
        if landmarks is not None:
            mouth_points = landmarks[52:71].astype(np.int32)
            raw_min_x, raw_min_y = np.min(mouth_points, axis=0)
            raw_max_x, raw_max_y = np.max(mouth_points, axis=0)
            mouth_w = max(1, raw_max_x - raw_min_x)
            mouth_h = max(1, raw_max_y - raw_min_y)
            pad_top    = int(mouth_h * 0.35 * s_top)
            pad_bottom = int(mouth_h * 0.50 * s_bot)
            pad_left   = int(mouth_w * 0.40 * s_left)
            pad_right  = int(mouth_w * 0.40 * s_right)
            min_x = max(0, raw_min_x - pad_left)
            min_y = max(0, raw_min_y - pad_top)
            max_x = min(frame.shape[1], raw_max_x + pad_right)
            max_y = min(frame.shape[0], raw_max_y + pad_bottom)
            mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
            # Landmark points in cutout-local coordinates for polygon masking
            mouth_mask_points = mouth_points - np.array([min_x, min_y], dtype=np.int32)
        return mouth_cutout, (min_x, min_y, max_x, max_y), mouth_mask_points

    def create_feathered_mask(self, shape, feather_amount=30):
        mask = np.zeros(shape[:2], dtype=np.float32)
        center = (shape[1] // 2, shape[0] // 2)
        # Use full extent so lip-adjacent pixels are fully inside the ellipse.
        # Feathering then falls off only at the bounding-box edge, not into the lips.
        axes = (max(1, shape[1] // 2), max(1, shape[0] // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (feather_amount * 2 + 1, feather_amount * 2 + 1), 0)
        max_val = np.max(mask)
        return mask / max_val if max_val > 0 else mask

    def apply_mouth_area(self, frame:np.ndarray, mouth_cutout:np.ndarray, mouth_box:tuple, mouth_polygon=None, mouth_blend:float=10.0) -> np.ndarray:
        min_x, min_y, max_x, max_y = mouth_box
        box_width = max_x - min_x
        box_height = max_y - min_y
        if mouth_cutout is None or box_width <= 0 or box_height <= 0:
            return frame
        try:
            resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
            roi = frame[min_y:max_y, min_x:max_x]
            if roi.shape != resized_mouth_cutout.shape:
                resized_mouth_cutout = cv2.resize(resized_mouth_cutout, (roi.shape[1], roi.shape[0]))
            color_corrected_mouth = self.apply_color_transfer(resized_mouth_cutout, roi)

            if mouth_polygon is not None:
                # Scale polygon from original cutout coords to the resized box
                scale_x = box_width  / max(1, mouth_cutout.shape[1])
                scale_y = box_height / max(1, mouth_cutout.shape[0])
                scaled_pts = (mouth_polygon * [scale_x, scale_y]).astype(np.int32)
                hull = cv2.convexHull(scaled_pts)
                mask = np.zeros(resized_mouth_cutout.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                # mouth_blend (0-30) controls dilation and edge softness.
                # At 0: binary mask with only 3px anti-alias blur (hardest edge).
                # Higher values expand the mask outward and soften the transition.
                dilate_px = max(0, min(int(mouth_blend), box_width // 4))
                if dilate_px > 0:
                    dilate_kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (dilate_px * 2, dilate_px * 2))
                    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
                    blur_k = dilate_px * 2 + 1
                else:
                    blur_k = 3
                mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_k, blur_k), 0)
                mask /= 255.0
            else:
                feather_amount = max(1, min(30, box_width // 15, box_height // 15))
                mask = self.create_feathered_mask(resized_mouth_cutout.shape, feather_amount)

            mask = mask[:, :, np.newaxis]
            blended = (color_corrected_mouth * mask + roi * (1 - mask)).astype(np.uint8)
            frame[min_y:max_y, min_x:max_x] = blended

            if self.options.show_face_area_overlay:
                # Draw a red overlay on the mouth restore region so it's visible
                # alongside the green face-swap overlay
                red_overlay = np.zeros_like(frame[min_y:max_y, min_x:max_x])
                red_overlay[:, :, 2] = 255  # BGR red
                frame[min_y:max_y, min_x:max_x] = cv2.addWeighted(
                    frame[min_y:max_y, min_x:max_x], 0.5, red_overlay, 0.5, 0)
        except Exception as e:
            print(f'Error in apply_mouth_area: {e}')
        return frame

    def apply_color_transfer(self, source, target):
        # If source is effectively grayscale (B&W media), skip color transfer.
        # Chrominance std â‰ˆ 0 causes division explosion â†’ blue artifact.
        src_f = source.astype(np.float32)
        if (np.mean(np.abs(src_f[:, :, 0] - src_f[:, :, 1])) < 5 and
                np.mean(np.abs(src_f[:, :, 1] - src_f[:, :, 2])) < 5):
            return source
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
        source_mean, source_std = cv2.meanStdDev(source)
        target_mean, target_std = cv2.meanStdDev(target)
        source_mean = source_mean.reshape(1, 1, 3)
        source_std  = np.maximum(source_std.reshape(1, 1, 3), 1.0)  # guard near-zero
        target_mean = target_mean.reshape(1, 1, 3)
        target_std  = target_std.reshape(1, 1, 3)
        source = (source - source_mean) * (target_std / source_std) + target_mean
        return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)


    def unload_models():
        pass


    def release_resources(self):
        self.release_single_batch_worker_pools()
        for p in self.processors:
            p.Release()
        self.processors.clear()
        # FIX: Null out writer references after closing so GC can collect them
        if self.videowriter is not None:
            self.videowriter.close()
            self.videowriter = None
        if self.streamwriter is not None:
            self.streamwriter.Close()
            self.streamwriter = None
        # FIX: Clear face data and cached frame references so nothing holds VRAM-backed data
        self.input_face_datas = []
        self.target_face_datas = []
        self.last_swapped_frame = None

