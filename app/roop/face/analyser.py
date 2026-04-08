import threading
from contextlib import contextmanager
from typing import Any

import insightface
import numpy as np
from insightface.app.common import Face

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
DEFAULT_FACE_ANALYSIS_MODULES = ["landmark_3d_68", "landmark_2d_106", "detection"]


class HybridFaceAnalyser:
    def __init__(self, compat_analyser, detector_model, landmarker_model, providers, det_size):
        self.compat_analyser = compat_analyser
        self.detector_model = get_face_detector_model_key(detector_model)
        self.landmarker_model = get_face_landmarker_model_key(landmarker_model)
        self.providers = list(providers or [])
        self.det_size = tuple(det_size)
        self.detector = create_face_detector(self.detector_model, self.providers, self.det_size)
        self.landmarker = create_face_landmarker(self.landmarker_model, self.providers)

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
    allowed_modules = _get_allowed_modules()

    if detector_model != "insightface":
        ensure_face_detector_model_downloaded(detector_model)
    if landmarker_model != "insightface_2d106":
        ensure_face_landmarker_model_downloaded(landmarker_model)

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
    roop.config.globals.g_current_face_analysis = allowed_modules
    return HybridFaceAnalyser(compat_analyser, detector_model, landmarker_model, providers, det_size)


def release_face_analyser():
    global FACE_ANALYSER, FACE_ANALYSER_SIGNATURE
    with THREAD_LOCK_ANALYSER:
        if FACE_ANALYSER is not None:
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
