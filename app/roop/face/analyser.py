import threading
from contextlib import contextmanager
from queue import Queue
from types import SimpleNamespace
from typing import Any

import insightface
import numpy as np
from insightface.app.common import Face
from insightface.model_zoo import get_model as get_insightface_model

import roop.config.globals
from roop.face.analytics_runtime import (
    create_face_detector,
    create_face_landmarker,
    get_face_analyser_providers,
    resolve_face_detector_size,
)
from roop.face_analytics_models import (
    ensure_face_detector_model_downloaded,
    ensure_face_landmarker_model_downloaded,
    get_face_detector_model_key,
    get_face_landmarker_model_key,
)
from roop.utils import resolve_relative_path


FACE_ANALYSER = None
FACE_ANALYSER_SIGNATURE = None
THREAD_LOCK_ANALYSER = threading.Lock()
DEFAULT_FACE_ANALYSIS_MODULES = []
INSIGHTFACE_COMPAT_MODEL_FILES = {
    "landmark_2d_106": "2d106det.onnx",
    "recognition": "w600k_r50.onnx",
    "genderage": "genderage.onnx",
}


class HybridFaceAnalyser:
    def __init__(self, compat_analyser, detector_model, landmarker_model, providers, det_size):
        self.compat_analyser = compat_analyser
        self.detector_model = get_face_detector_model_key(detector_model)
        self.landmarker_model = get_face_landmarker_model_key(landmarker_model)
        self.providers = list(providers or [])
        self.det_size = tuple(det_size)
        self.detector = create_face_detector(self.detector_model, self.providers, self.det_size)
        self.landmarker = create_face_landmarker(self.landmarker_model, self.providers)
        self.detector_worker_pools = {}

    def get(self, frame, max_num=0):
        if self.detector is None:
            faces = self.compat_analyser.get(frame, max_num=max_num)
            return self._enrich_faces(frame, faces)

        detections, kpss = self.detector.detect(frame, max_num=max_num)
        if detections is None or detections.shape[0] == 0:
            return []

        faces = []
        for index, detection in enumerate(detections):
            kps = None
            if kpss is not None and index < len(kpss):
                kps = np.asarray(kpss[index], dtype=np.float32)
            face = Face(
                bbox=np.asarray(detection[:4], dtype=np.float32),
                kps=kps,
                det_score=float(detection[4]),
            )
            for task_name, model in self.compat_analyser.models.items():
                if task_name == "detection":
                    continue
                model.get(frame, face)
            self._normalize_face_vectors(face)
            faces.append(face)
        return self._enrich_faces(frame, faces)

    def _create_faces_from_detection(self, frame, detections, kpss):
        if detections is None or detections.shape[0] == 0:
            return []
        faces = []
        for index, detection in enumerate(detections):
            kps = None
            if kpss is not None and index < len(kpss):
                kps = np.asarray(kpss[index], dtype=np.float32)
            face = Face(
                bbox=np.asarray(detection[:4], dtype=np.float32),
                kps=kps,
                det_score=float(detection[4]),
            )
            for task_name, model in self.compat_analyser.models.items():
                if task_name == "detection":
                    continue
                model.get(frame, face)
            self._normalize_face_vectors(face)
            faces.append(face)
        return faces

    def should_parallelize_single_batch_detector(self) -> bool:
        return bool(
            self.detector is not None
            and getattr(self.detector, "supports_parallel_single_batch", False)
            and getattr(self.detector, "batch_size_limit", None) == 1
            and hasattr(self.detector, "CreateWorkerDetector")
        )

    def get_detector_worker_count(self, worker_count=None) -> int:
        if not self.should_parallelize_single_batch_detector():
            return 1
        if worker_count is None:
            worker_count = getattr(getattr(roop.config.globals, "CFG", None), "detect_single_batch_workers", 1)
        try:
            return max(1, int(worker_count))
        except (TypeError, ValueError):
            return 1

    def get_detector_worker_instances(self, worker_count: int):
        worker_count = max(1, int(worker_count))
        pool_key = id(self.detector)
        pool = self.detector_worker_pools.get(pool_key)
        if pool is None or pool.get("base") is not self.detector:
            pool = {
                "base": self.detector,
                "workers": [self.detector],
            }
            self.detector_worker_pools[pool_key] = pool
        workers = pool["workers"]
        while len(workers) < worker_count:
            workers.append(self.detector.CreateWorkerDetector())
        while len(workers) > worker_count:
            extra_worker = workers.pop()
            if extra_worker is not self.detector:
                try:
                    extra_worker.Release()
                except Exception:
                    pass
        return list(workers)

    def release(self):
        for pool in self.detector_worker_pools.values():
            workers = pool.get("workers", [])
            for worker in workers[1:]:
                try:
                    worker.Release()
                except Exception:
                    pass
        self.detector_worker_pools.clear()

    def get_many(self, frames, max_num=0, batch_size=1, worker_count=1):
        frames = list(frames or [])
        if not frames:
            return []
        if self.detector is None:
            return [self.get(frame, max_num=max_num) for frame in frames]

        if getattr(self.detector, "supports_batch", False):
            detected_faces = []
            batch_outputs = self.detector.detect_batch(frames, max_num=max_num, batch_size=batch_size)
            for frame, (detections, kpss) in zip(frames, batch_outputs):
                detected_faces.append(self._enrich_faces(frame, self._create_faces_from_detection(frame, detections, kpss)))
            return detected_faces

        configured_worker_count = self.get_detector_worker_count(worker_count)
        if configured_worker_count <= 1 or len(frames) <= 1:
            return [self.get(frame, max_num=max_num) for frame in frames]

        workers = self.get_detector_worker_instances(configured_worker_count)
        active_workers = workers[: min(len(frames), configured_worker_count)]
        task_queue = Queue(maxsize=max(len(active_workers) * 2, 2))
        detection_outputs = [None] * len(frames)
        output_lock = threading.Lock()
        worker_error = {"exc": None}

        def worker_loop(worker_detector):
            while True:
                item = task_queue.get()
                try:
                    if item is None:
                        return
                    if worker_error["exc"] is not None:
                        continue
                    index, frame = item
                    detections, kpss = worker_detector.detect(frame, max_num=max_num)
                    with output_lock:
                        detection_outputs[index] = (detections, kpss)
                except Exception as exc:
                    with output_lock:
                        if worker_error["exc"] is None:
                            worker_error["exc"] = exc
                finally:
                    task_queue.task_done()

        worker_threads = [
            threading.Thread(target=worker_loop, args=(worker_detector,), daemon=True)
            for worker_detector in active_workers
        ]
        for worker_thread in worker_threads:
            worker_thread.start()
        for index, frame in enumerate(frames):
            task_queue.put((index, frame), block=True)
        for _ in worker_threads:
            task_queue.put(None, block=True)
        task_queue.join()
        for worker_thread in worker_threads:
            worker_thread.join()
        if worker_error["exc"] is not None:
            raise worker_error["exc"]
        outputs = []
        for frame, detection_output in zip(frames, detection_outputs):
            if detection_output is None:
                outputs.append([])
                continue
            detections, kpss = detection_output
            faces = self._create_faces_from_detection(frame, detections, kpss)
            outputs.append(self._enrich_faces(frame, faces))
        return outputs

    def _enrich_faces(self, frame, faces):
        if self.landmarker is None:
            for face in faces:
                self._normalize_face_vectors(face)
            return faces
        for face in faces:
            try:
                bbox = np.asarray(face.bbox, dtype=np.float32)
                kps = None if face.kps is None else np.asarray(face.kps, dtype=np.float32)
                landmark_2d_68, landmark_score = self.landmarker.detect(frame, bbox, kps)
            except Exception:
                continue
            if landmark_2d_68 is None:
                self._normalize_face_vectors(face)
                continue
            face["landmark_2d_68"] = np.asarray(landmark_2d_68, dtype=np.float32)
            face["landmark_2d_68_score"] = float(landmark_score)
            self._normalize_face_vectors(face)
        return faces

    @staticmethod
    def _normalize_face_vectors(face):
        for key in ("embedding", "normed_embedding"):
            value = getattr(face, key, None)
            if value is None:
                continue
            try:
                face[key] = np.asarray(value, dtype=np.float32).reshape(-1)
            except Exception:
                continue


def _get_allowed_modules() -> list[str]:
    desired = getattr(roop.config.globals, "g_desired_face_analysis", None)
    if desired:
        return list(desired)
    return list(DEFAULT_FACE_ANALYSIS_MODULES)


def build_face_analysis_modules(extra_modules=None) -> list[str]:
    modules = list(DEFAULT_FACE_ANALYSIS_MODULES)
    for module in extra_modules or []:
        if module and module not in modules:
            modules.append(module)
    return modules


def _build_face_analyser_signature() -> tuple:
    cfg = getattr(roop.config.globals, "CFG", None)
    detector_model = get_face_detector_model_key(getattr(cfg, "face_detector_model", None))
    landmarker_model = get_face_landmarker_model_key(getattr(cfg, "face_landmarker_model", None))
    providers = tuple(get_face_analyser_providers())
    return (
        tuple(_get_allowed_modules()),
        detector_model,
        landmarker_model,
        tuple(resolve_face_detector_size(detector_model)),
        bool(getattr(cfg, "force_cpu", False)),
        providers,
    )


def _create_face_analyser() -> Any:
    cfg = getattr(roop.config.globals, "CFG", None)
    detector_model = get_face_detector_model_key(getattr(cfg, "face_detector_model", None))
    landmarker_model = get_face_landmarker_model_key(getattr(cfg, "face_landmarker_model", None))
    providers = get_face_analyser_providers()
    det_size = resolve_face_detector_size(detector_model)
    requested_modules = _get_allowed_modules()

    if detector_model != "insightface":
        ensure_face_detector_model_downloaded(detector_model)
    if landmarker_model != "insightface_2d106":
        ensure_face_landmarker_model_downloaded(landmarker_model)

    compat_modules = [module for module in requested_modules if module in INSIGHTFACE_COMPAT_MODEL_FILES]

    if detector_model == "insightface":
        allowed_modules = ["detection", *compat_modules]
        model_path = resolve_relative_path("..")
        compat_analyser = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=model_path,
            providers=providers,
            allowed_modules=allowed_modules,
        )
        compat_analyser.prepare(
            ctx_id=-1 if providers == ["CPUExecutionProvider"] else 0,
            det_size=det_size,
        )
    else:
        compat_models = {}
        for module in compat_modules:
            model_filename = INSIGHTFACE_COMPAT_MODEL_FILES[module]
            model_path = resolve_relative_path(f"../models/buffalo_l/{model_filename}")
            compat_model = get_insightface_model(model_path, providers=providers)
            prepare = getattr(compat_model, "prepare", None)
            if callable(prepare):
                prepare(-1 if providers == ["CPUExecutionProvider"] else 0)
            compat_models[getattr(compat_model, "taskname", module)] = compat_model
        compat_analyser = SimpleNamespace(models=compat_models)
        allowed_modules = compat_modules
    print(
        "Face analytics pipeline:",
        f"detector={detector_model}",
        f"landmarker={landmarker_model}",
        f"compat={allowed_modules if allowed_modules else ['none']}",
    )
    roop.config.globals.g_current_face_analysis = requested_modules
    return HybridFaceAnalyser(compat_analyser, detector_model, landmarker_model, providers, det_size)


def release_face_analyser():
    global FACE_ANALYSER, FACE_ANALYSER_SIGNATURE
    with THREAD_LOCK_ANALYSER:
        if FACE_ANALYSER is not None:
            release = getattr(FACE_ANALYSER, "release", None)
            if callable(release):
                release()
            del FACE_ANALYSER
            FACE_ANALYSER = None
        FACE_ANALYSER_SIGNATURE = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER, FACE_ANALYSER_SIGNATURE

    with THREAD_LOCK_ANALYSER:
        signature = _build_face_analyser_signature()
        if FACE_ANALYSER is None or FACE_ANALYSER_SIGNATURE != signature:
            FACE_ANALYSER = _create_face_analyser()
            FACE_ANALYSER_SIGNATURE = signature
    return FACE_ANALYSER


@contextmanager
def use_face_analysis_modules(extra_modules=None):
    previous_modules = getattr(roop.config.globals, "g_desired_face_analysis", None)
    requested_modules = build_face_analysis_modules(extra_modules)
    previous_signature = tuple(previous_modules) if previous_modules else None
    requested_signature = tuple(requested_modules)

    if previous_signature != requested_signature:
        roop.config.globals.g_desired_face_analysis = list(requested_modules)
        release_face_analyser()
    try:
        yield list(requested_modules)
    finally:
        if previous_signature != requested_signature:
            roop.config.globals.g_desired_face_analysis = list(previous_modules) if previous_modules else None
            release_face_analyser()


__all__ = [
    "build_face_analysis_modules",
    "get_face_analyser",
    "release_face_analyser",
    "use_face_analysis_modules",
]
