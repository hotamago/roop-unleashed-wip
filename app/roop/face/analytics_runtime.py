from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import cv2
import numpy as np
from insightface.model_zoo.retinaface import RetinaFace
from insightface.model_zoo.scrfd import SCRFD

import roop.config.globals
from roop.face_analytics_models import (
    get_face_detector_model_key,
    get_face_detector_model_paths,
    get_face_landmarker_model_key,
    get_face_landmarker_model_paths,
)
from roop.onnx.session import create_inference_session


WARP_TEMPLATE_FFHQ_512 = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    dtype=np.float32,
)


def get_face_analyser_providers(force_cpu: bool | None = None) -> list[str]:
    if force_cpu is None:
        force_cpu = bool(getattr(getattr(roop.config.globals, "CFG", None), "force_cpu", False))
    if force_cpu:
        return ["CPUExecutionProvider"]
    return list(getattr(roop.config.globals, "execution_providers", []) or ["CPUExecutionProvider"])


def resolve_face_detector_size(model_name=None) -> tuple[int, int]:
    model_key = get_face_detector_model_key(model_name)
    use_default_det_size = bool(getattr(roop.config.globals, "default_det_size", True))
    if model_key in {"yolo_face", "yunet"}:
        return (640, 640)
    return (640, 640) if use_default_det_size else (320, 320)


def restrict_frame(frame: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    restrict_width, restrict_height = resolution
    if height > restrict_height or width > restrict_width:
        scale = min(restrict_height / height, restrict_width / width)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return cv2.resize(frame, (new_width, new_height))
    return frame


def prepare_detect_frame(frame: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    width, height = resolution
    detect_frame = np.zeros((height, width, 3), dtype=np.float32)
    detect_frame[: frame.shape[0], : frame.shape[1], :] = frame
    detect_frame = np.expand_dims(detect_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
    return detect_frame


def prepare_detect_frames_batch(frames: Sequence[np.ndarray], resolution: tuple[int, int]) -> tuple[np.ndarray, list[np.ndarray], list[tuple[float, float]]]:
    width, height = resolution
    batch = np.zeros((len(frames), 3, height, width), dtype=np.float32)
    restricted_frames: list[np.ndarray] = []
    ratios: list[tuple[float, float]] = []
    for index, frame in enumerate(frames):
        temp_frame = restrict_frame(frame, resolution)
        ratio_height = frame.shape[0] / temp_frame.shape[0]
        ratio_width = frame.shape[1] / temp_frame.shape[1]
        batch[index, :, : temp_frame.shape[0], : temp_frame.shape[1]] = temp_frame.transpose(2, 0, 1)
        restricted_frames.append(temp_frame)
        ratios.append((ratio_width, ratio_height))
    return batch, restricted_frames, ratios


def normalize_detect_frame(detect_frame: np.ndarray, normalize_range: tuple[int, int]) -> np.ndarray:
    if normalize_range == (-1, 1):
        return (detect_frame - 127.5) / 128.0
    if normalize_range == (0, 1):
        return detect_frame / 255.0
    return detect_frame


def resolve_session_batch_size_limit(session) -> int | None:
    if session is None:
        return 1
    batch_size_limit = None
    for input_meta in session.get_inputs():
        shape = getattr(input_meta, "shape", None) or []
        if not shape:
            continue
        batch_dim = shape[0]
        if isinstance(batch_dim, int) and batch_dim > 0:
            batch_size_limit = batch_dim if batch_size_limit is None else min(batch_size_limit, batch_dim)
    return batch_size_limit


@lru_cache(maxsize=32)
def create_static_anchors(feature_stride: int, anchor_total: int, stride_height: int, stride_width: int) -> np.ndarray:
    x, y = np.mgrid[:stride_width, :stride_height]
    anchors = np.stack((y, x), axis=-1)
    anchors = (anchors * feature_stride).reshape((-1, 2))
    anchors = np.stack([anchors] * anchor_total, axis=1).reshape((-1, 2))
    return anchors.astype(np.float32)


def distance_to_bounding_box(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.column_stack([x1, y1, x2, y2]).astype(np.float32)


def distance_to_face_landmark_5(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x = points[:, 0::2] + distance[:, 0::2]
    y = points[:, 1::2] + distance[:, 1::2]
    return np.stack((x, y), axis=-1).astype(np.float32)


def create_rotation_matrix_and_size(angle: float, size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    rotation_matrix = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
    rotation_size = np.dot(np.abs(rotation_matrix[:, :2]), size)
    rotation_matrix[:, -1] += (rotation_size - size) * 0.5
    return rotation_matrix.astype(np.float32), (int(rotation_size[0]), int(rotation_size[1]))


def warp_face_by_translation(frame: np.ndarray, translation: np.ndarray, scale: float, crop_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    affine_matrix = np.array([[scale, 0, translation[0]], [0, scale, translation[1]]], dtype=np.float32)
    crop_frame = cv2.warpAffine(frame, affine_matrix, crop_size)
    return crop_frame, affine_matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = points.reshape(-1, 1, 2).astype(np.float32)
    points = cv2.transform(points, matrix)
    return points.reshape(-1, 2).astype(np.float32)


def estimate_matrix_by_face_landmark_5(face_landmark_5: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    warp_template_norm = WARP_TEMPLATE_FFHQ_512 * np.asarray(crop_size, dtype=np.float32)
    affine_matrix = cv2.estimateAffinePartial2D(
        face_landmark_5.astype(np.float32),
        warp_template_norm.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=100,
    )[0]
    return affine_matrix.astype(np.float32)


def conditional_optimize_contrast(rgb_frame: np.ndarray) -> np.ndarray:
    optimized = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2Lab)
    if float(np.mean(optimized[:, :, 0])) < 30.0:
        optimized[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(optimized[:, :, 0])
    optimized = cv2.cvtColor(optimized, cv2.COLOR_Lab2RGB)
    return optimized


def flatten_nms_indices(indices) -> list[int]:
    if indices is None:
        return []
    if isinstance(indices, np.ndarray):
        return [int(index) for index in indices.reshape(-1).tolist()]
    flat = []
    for index in indices:
        if isinstance(index, (list, tuple, np.ndarray)):
            flat.extend(flatten_nms_indices(index))
        else:
            flat.append(int(index))
    return flat


def apply_nms(bounding_boxes: list[np.ndarray], scores: list[float], score_threshold: float, nms_threshold: float) -> list[int]:
    if not bounding_boxes:
        return []
    bounding_boxes_norm = [
        [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
        for box in bounding_boxes
    ]
    keep_indices = cv2.dnn.NMSBoxes(
        bounding_boxes_norm,
        [float(score) for score in scores],
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )
    return flatten_nms_indices(keep_indices)


def limit_detections(
    bounding_boxes: list[np.ndarray],
    scores: list[float],
    landmarks_5: list[np.ndarray],
    image_shape: tuple[int, int, int],
    max_num: int = 0,
    nms_threshold: float = 0.4,
    score_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not bounding_boxes:
        return np.zeros((0, 5), dtype=np.float32), None

    keep_indices = apply_nms(bounding_boxes, scores, score_threshold, nms_threshold)
    if keep_indices:
        bounding_boxes = [bounding_boxes[index] for index in keep_indices]
        scores = [scores[index] for index in keep_indices]
        landmarks_5 = [landmarks_5[index] for index in keep_indices]

    detections = np.hstack(
        [
            np.asarray(bounding_boxes, dtype=np.float32),
            np.asarray(scores, dtype=np.float32).reshape(-1, 1),
        ]
    ).astype(np.float32)
    kpss = np.asarray(landmarks_5, dtype=np.float32) if landmarks_5 else None

    if max_num > 0 and detections.shape[0] > max_num:
        area = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        img_center = image_shape[0] // 2, image_shape[1] // 2
        offsets = np.vstack(
            [
                (detections[:, 0] + detections[:, 2]) / 2 - img_center[1],
                (detections[:, 1] + detections[:, 3]) / 2 - img_center[0],
            ]
        )
        offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)
        selected_indices = np.argsort(area - offset_dist_squared * 2.0)[::-1][:max_num]
        detections = detections[selected_indices]
        if kpss is not None:
            kpss = kpss[selected_indices]

    return detections.astype(np.float32), None if kpss is None else kpss.astype(np.float32)


@dataclass
class BaseFaceDetector:
    model_name: str
    providers: Iterable[str]
    det_size: tuple[int, int]
    det_thresh: float = 0.5
    supports_batch: bool = False
    batch_size_limit: int | None = 1
    supports_parallel_single_batch: bool = False

    def detect(self, frame: np.ndarray, max_num: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        max_num: int = 0,
        batch_size: int = 1,
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        del batch_size
        return [self.detect(frame, max_num=max_num) for frame in frames]

    def CreateWorkerDetector(self):
        return self.__class__(self.model_name, list(self.providers), self.det_size, self.det_thresh)

    def Release(self):
        for name in ("session", "detector"):
            if hasattr(self, name):
                setattr(self, name, None)


@dataclass
class RetinaFaceDetector(BaseFaceDetector):
    def __post_init__(self):
        model_path = get_face_detector_model_paths(self.model_name)[0]
        session = create_inference_session(model_path, providers=list(self.providers))
        self.detector = RetinaFace(model_file=model_path, session=session)
        self.detector.prepare(-1 if "CPUExecutionProvider" in self.providers else 0, input_size=self.det_size, det_thresh=self.det_thresh)
        self.supports_parallel_single_batch = True

    def detect(self, frame: np.ndarray, max_num: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
        return self.detector.detect(frame, input_size=self.det_size, max_num=max_num, metric="default")


@dataclass
class ScrfdDetector(BaseFaceDetector):
    def __post_init__(self):
        model_path = get_face_detector_model_paths(self.model_name)[0]
        session = create_inference_session(model_path, providers=list(self.providers))
        self.detector = SCRFD(model_file=model_path, session=session)
        self.detector.prepare(-1 if "CPUExecutionProvider" in self.providers else 0, input_size=self.det_size, det_thresh=self.det_thresh)
        self.supports_parallel_single_batch = True

    def detect(self, frame: np.ndarray, max_num: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
        return self.detector.detect(frame, input_size=self.det_size, max_num=max_num, metric="default")


@dataclass
class YoloFaceDetector(BaseFaceDetector):
    def __post_init__(self):
        model_path = get_face_detector_model_paths(self.model_name)[0]
        self.session = create_inference_session(model_path, providers=list(self.providers))
        self.input_name = self.session.get_inputs()[0].name
        self.batch_size_limit = resolve_session_batch_size_limit(self.session)
        self.supports_batch = self.batch_size_limit is None or self.batch_size_limit > 1
        self.supports_parallel_single_batch = True

    @staticmethod
    def _normalize_detection_output(detection: np.ndarray) -> np.ndarray:
        detection = np.asarray(detection)
        if detection.ndim != 2:
            return np.zeros((0, 0), dtype=np.float32)
        if detection.shape[0] < detection.shape[1]:
            detection = detection.T
        return detection.astype(np.float32, copy=False)

    def _parse_detection(
        self,
        frame: np.ndarray,
        detection: np.ndarray,
        ratio_width: float,
        ratio_height: float,
        max_num: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        detection = self._normalize_detection_output(detection)
        if detection.ndim != 2 or detection.shape[0] == 0:
            return np.zeros((0, 5), dtype=np.float32), None

        bounding_boxes_raw, face_scores_raw, face_landmarks_5_raw = np.split(detection, [4, 5], axis=1)
        keep_indices = np.where(face_scores_raw.reshape(-1) > self.det_thresh)[0]
        if not np.any(keep_indices):
            return np.zeros((0, 5), dtype=np.float32), None

        bounding_boxes = []
        scores = []
        landmarks_5 = []
        for index in keep_indices:
            bounding_box_raw = bounding_boxes_raw[index]
            bounding_boxes.append(
                np.array(
                    [
                        (bounding_box_raw[0] - bounding_box_raw[2] / 2) * ratio_width,
                        (bounding_box_raw[1] - bounding_box_raw[3] / 2) * ratio_height,
                        (bounding_box_raw[0] + bounding_box_raw[2] / 2) * ratio_width,
                        (bounding_box_raw[1] + bounding_box_raw[3] / 2) * ratio_height,
                    ],
                    dtype=np.float32,
                )
            )
            scores.append(float(face_scores_raw[index][0]))
            landmarks = face_landmarks_5_raw[index].reshape(-1, 3)[:, :2].astype(np.float32)
            landmarks[:, 0] *= ratio_width
            landmarks[:, 1] *= ratio_height
            landmarks_5.append(landmarks)

        return limit_detections(
            bounding_boxes,
            scores,
            landmarks_5,
            frame.shape,
            max_num=max_num,
            nms_threshold=0.4,
            score_threshold=self.det_thresh,
        )

    def detect(self, frame: np.ndarray, max_num: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
        detect_frame, _restricted_frames, ratios = prepare_detect_frames_batch([frame], self.det_size)
        detect_frame = normalize_detect_frame(detect_frame, (0, 1))
        detection = self.session.run(None, {self.input_name: detect_frame})[0]
        detection = detection[0] if np.asarray(detection).ndim >= 3 else detection
        ratio_width, ratio_height = ratios[0]
        return self._parse_detection(frame, detection, ratio_width, ratio_height, max_num=max_num)

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        max_num: int = 0,
        batch_size: int = 1,
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        if not frames:
            return []
        effective_batch_size = max(1, int(batch_size))
        if self.batch_size_limit is not None:
            effective_batch_size = min(effective_batch_size, max(1, int(self.batch_size_limit)))
        if effective_batch_size <= 1:
            return [self.detect(frame, max_num=max_num) for frame in frames]

        outputs: list[tuple[np.ndarray, np.ndarray | None]] = []
        for start in range(0, len(frames), effective_batch_size):
            frame_batch = list(frames[start : start + effective_batch_size])
            detect_frame, _restricted_frames, ratios = prepare_detect_frames_batch(frame_batch, self.det_size)
            detect_frame = normalize_detect_frame(detect_frame, (0, 1))
            detections = self.session.run(None, {self.input_name: detect_frame})[0]
            detections = np.asarray(detections)
            for batch_index, frame in enumerate(frame_batch):
                batch_detection = detections[batch_index] if detections.ndim >= 3 else detections
                ratio_width, ratio_height = ratios[batch_index]
                outputs.append(self._parse_detection(frame, batch_detection, ratio_width, ratio_height, max_num=max_num))
        return outputs


@dataclass
class YuNetDetector(BaseFaceDetector):
    def __post_init__(self):
        model_path = get_face_detector_model_paths(self.model_name)[0]
        self.session = create_inference_session(model_path, providers=list(self.providers))
        self.input_name = self.session.get_inputs()[0].name
        self.feature_strides = [8, 16, 32]
        self.feature_map_channel = 3
        self.anchor_total = 1
        self.batch_size_limit = resolve_session_batch_size_limit(self.session)
        self.supports_batch = self.batch_size_limit is None or self.batch_size_limit > 1
        self.supports_parallel_single_batch = True

    def _parse_detection(
        self,
        frame: np.ndarray,
        detection: list[np.ndarray],
        ratio_width: float,
        ratio_height: float,
        max_num: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None]:

        bounding_boxes = []
        scores = []
        landmarks_5 = []
        face_detector_width, face_detector_height = self.det_size

        for index, feature_stride in enumerate(self.feature_strides):
            face_scores_raw = (detection[index] * detection[index + self.feature_map_channel]).reshape(-1)
            keep_indices = np.where(face_scores_raw >= self.det_thresh)[0]
            if not np.any(keep_indices):
                continue

            stride_height = face_detector_height // feature_stride
            stride_width = face_detector_width // feature_stride
            anchors = create_static_anchors(feature_stride, self.anchor_total, stride_height, stride_width)
            bounding_boxes_center = detection[index + self.feature_map_channel * 2].squeeze(0)[:, :2] * feature_stride + anchors
            bounding_boxes_size = np.exp(detection[index + self.feature_map_channel * 2].squeeze(0)[:, 2:4]) * feature_stride
            face_landmarks_raw = detection[index + self.feature_map_channel * 3].squeeze(0)
            bounding_boxes_raw = np.stack(
                [
                    bounding_boxes_center[:, 0] - bounding_boxes_size[:, 0] / 2,
                    bounding_boxes_center[:, 1] - bounding_boxes_size[:, 1] / 2,
                    bounding_boxes_center[:, 0] + bounding_boxes_size[:, 0] / 2,
                    bounding_boxes_center[:, 1] + bounding_boxes_size[:, 1] / 2,
                ],
                axis=-1,
            )
            face_landmarks_raw = np.concatenate(
                [
                    face_landmarks_raw[:, [0, 1]] * feature_stride + anchors,
                    face_landmarks_raw[:, [2, 3]] * feature_stride + anchors,
                    face_landmarks_raw[:, [4, 5]] * feature_stride + anchors,
                    face_landmarks_raw[:, [6, 7]] * feature_stride + anchors,
                    face_landmarks_raw[:, [8, 9]] * feature_stride + anchors,
                ],
                axis=-1,
            ).reshape(-1, 5, 2)

            for keep_index in keep_indices:
                bounding_box_raw = bounding_boxes_raw[keep_index]
                bounding_boxes.append(
                    np.array(
                        [
                            bounding_box_raw[0] * ratio_width,
                            bounding_box_raw[1] * ratio_height,
                            bounding_box_raw[2] * ratio_width,
                            bounding_box_raw[3] * ratio_height,
                        ],
                        dtype=np.float32,
                    )
                )
                scores.append(float(face_scores_raw[keep_index]))
                landmarks = face_landmarks_raw[keep_index].astype(np.float32)
                landmarks[:, 0] *= ratio_width
                landmarks[:, 1] *= ratio_height
                landmarks_5.append(landmarks)

        return limit_detections(
            bounding_boxes,
            scores,
            landmarks_5,
            frame.shape,
            max_num=max_num,
            nms_threshold=0.4,
            score_threshold=self.det_thresh,
        )

    def detect(self, frame: np.ndarray, max_num: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
        detect_frame, _restricted_frames, ratios = prepare_detect_frames_batch([frame], self.det_size)
        detection = self.session.run(None, {self.input_name: detect_frame})
        ratio_width, ratio_height = ratios[0]
        sample_detection = [np.asarray(output)[0] if np.asarray(output).ndim >= 3 else np.asarray(output) for output in detection]
        return self._parse_detection(frame, sample_detection, ratio_width, ratio_height, max_num=max_num)

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        max_num: int = 0,
        batch_size: int = 1,
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        if not frames:
            return []
        effective_batch_size = max(1, int(batch_size))
        if self.batch_size_limit is not None:
            effective_batch_size = min(effective_batch_size, max(1, int(self.batch_size_limit)))
        if effective_batch_size <= 1:
            return [self.detect(frame, max_num=max_num) for frame in frames]

        outputs: list[tuple[np.ndarray, np.ndarray | None]] = []
        for start in range(0, len(frames), effective_batch_size):
            frame_batch = list(frames[start : start + effective_batch_size])
            detect_frame, _restricted_frames, ratios = prepare_detect_frames_batch(frame_batch, self.det_size)
            detection_outputs = self.session.run(None, {self.input_name: detect_frame})
            detection_outputs = [np.asarray(output) for output in detection_outputs]
            for batch_index, frame in enumerate(frame_batch):
                sample_detection = [
                    output[batch_index] if output.ndim >= 3 else output
                    for output in detection_outputs
                ]
                ratio_width, ratio_height = ratios[batch_index]
                outputs.append(self._parse_detection(frame, sample_detection, ratio_width, ratio_height, max_num=max_num))
        return outputs


@dataclass
class BaseFaceLandmarker:
    model_name: str
    providers: Iterable[str]

    def detect(self, frame: np.ndarray, bbox: np.ndarray, kps: np.ndarray | None) -> tuple[np.ndarray | None, float]:
        raise NotImplementedError


@dataclass
class TwoDFAN4Landmarker(BaseFaceLandmarker):
    def __post_init__(self):
        model_path = get_face_landmarker_model_paths(self.model_name)[0]
        self.session = create_inference_session(model_path, providers=list(self.providers))
        self.input_name = self.session.get_inputs()[0].name
        self.model_size = 256

    def detect(self, frame: np.ndarray, bbox: np.ndarray, kps: np.ndarray | None) -> tuple[np.ndarray | None, float]:
        del kps
        scale = 195.0 / max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]), 1.0)
        translation = (self.model_size - np.add(bbox[2:], bbox[:2]) * scale) * 0.5
        rotation_matrix, rotation_size = create_rotation_matrix_and_size(0, (self.model_size, self.model_size))
        crop_frame, affine_matrix = warp_face_by_translation(frame, translation, scale, (self.model_size, self.model_size))
        crop_frame = cv2.warpAffine(crop_frame, rotation_matrix, rotation_size)
        crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
        crop_frame = conditional_optimize_contrast(crop_frame)
        crop_frame = crop_frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        prediction, heatmap = self.session.run(None, {self.input_name: np.expand_dims(crop_frame, axis=0)})
        landmarks_68 = prediction[:, :, :2][0] / 64 * 256
        landmarks_68 = transform_points(landmarks_68, cv2.invertAffineTransform(rotation_matrix))
        landmarks_68 = transform_points(landmarks_68, cv2.invertAffineTransform(affine_matrix))
        landmark_score = float(np.mean(np.amax(heatmap, axis=(2, 3))))
        landmark_score = float(np.interp(landmark_score, [0.0, 0.9], [0.0, 1.0]))
        return landmarks_68.astype(np.float32), landmark_score


@dataclass
class PeppaWutzLandmarker(BaseFaceLandmarker):
    def __post_init__(self):
        model_path = get_face_landmarker_model_paths(self.model_name)[0]
        self.session = create_inference_session(model_path, providers=list(self.providers))
        self.input_name = self.session.get_inputs()[0].name
        self.model_size = 256

    def detect(self, frame: np.ndarray, bbox: np.ndarray, kps: np.ndarray | None) -> tuple[np.ndarray | None, float]:
        del kps
        scale = 195.0 / max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]), 1.0)
        translation = (self.model_size - np.add(bbox[2:], bbox[:2]) * scale) * 0.5
        rotation_matrix, rotation_size = create_rotation_matrix_and_size(0, (self.model_size, self.model_size))
        crop_frame, affine_matrix = warp_face_by_translation(frame, translation, scale, (self.model_size, self.model_size))
        crop_frame = cv2.warpAffine(crop_frame, rotation_matrix, rotation_size)
        crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
        crop_frame = conditional_optimize_contrast(crop_frame)
        crop_frame = crop_frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        crop_frame = np.expand_dims(crop_frame, axis=0)
        prediction = self.session.run(None, {self.input_name: crop_frame})[0]
        landmarks_68 = prediction.reshape(-1, 3)[:, :2] / 64 * self.model_size
        landmarks_68 = transform_points(landmarks_68, cv2.invertAffineTransform(rotation_matrix))
        landmarks_68 = transform_points(landmarks_68, cv2.invertAffineTransform(affine_matrix))
        landmark_score = float(prediction.reshape(-1, 3)[:, 2].mean())
        landmark_score = float(np.interp(landmark_score, [0.0, 0.95], [0.0, 1.0]))
        return landmarks_68.astype(np.float32), landmark_score


@dataclass
class Fan68From5Landmarker(BaseFaceLandmarker):
    def __post_init__(self):
        model_path = get_face_landmarker_model_paths(self.model_name)[0]
        self.session = create_inference_session(model_path, providers=list(self.providers))
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame: np.ndarray, bbox: np.ndarray, kps: np.ndarray | None) -> tuple[np.ndarray | None, float]:
        del frame, bbox
        if kps is None or np.asarray(kps).shape != (5, 2):
            return None, 0.0
        # FaceFusion feeds fan_68_5 normalized FFHQ points in the [0, 1] range.
        affine_matrix = estimate_matrix_by_face_landmark_5(np.asarray(kps, dtype=np.float32), (1, 1))
        face_landmark_5 = cv2.transform(np.asarray(kps, dtype=np.float32).reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
        face_landmark_68 = self.session.run(None, {self.input_name: np.expand_dims(face_landmark_5, axis=0)})[0][0]
        face_landmark_68 = cv2.transform(
            face_landmark_68.reshape(1, -1, 2).astype(np.float32),
            cv2.invertAffineTransform(affine_matrix),
        ).reshape(-1, 2)
        return face_landmark_68.astype(np.float32), 1.0


def create_face_detector(model_name=None, providers: Iterable[str] | None = None, det_size: tuple[int, int] | None = None):
    model_key = get_face_detector_model_key(model_name)
    if model_key == "insightface":
        return None
    providers = list(providers or get_face_analyser_providers())
    det_size = det_size or resolve_face_detector_size(model_key)
    detector_map = {
        "retinaface": RetinaFaceDetector,
        "scrfd": ScrfdDetector,
        "yolo_face": YoloFaceDetector,
        "yunet": YuNetDetector,
    }
    detector_class = detector_map.get(model_key)
    if detector_class is None:
        return None
    return detector_class(model_key, providers, det_size)


def create_face_landmarker(model_name=None, providers: Iterable[str] | None = None):
    model_key = get_face_landmarker_model_key(model_name)
    if model_key == "insightface_2d106":
        return None
    providers = list(providers or get_face_analyser_providers())
    landmarker_map = {
        "2dfan4": TwoDFAN4Landmarker,
        "peppa_wutz": PeppaWutzLandmarker,
        "fan_68_5": Fan68From5Landmarker,
    }
    landmarker_class = landmarker_map.get(model_key)
    if landmarker_class is None:
        return None
    return landmarker_class(model_key, providers)


__all__ = [
    "create_face_detector",
    "create_face_landmarker",
    "get_face_analyser_providers",
    "limit_detections",
    "resolve_face_detector_size",
]
