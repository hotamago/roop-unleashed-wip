from types import SimpleNamespace

import numpy as np

import roop.config.globals
from roop.face.analyser import HybridFaceAnalyser
from roop.face.analytics_runtime import Fan68From5Landmarker, YoloFaceDetector
from roop.pipeline.batch_executor import ProcessMgr


def test_process_mgr_limits_face_analysis_modules_for_non_matching_modes():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="all")
    roop.config.globals.CFG = SimpleNamespace(face_landmarker_model="fan_68_5")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == []


def test_process_mgr_adds_recognition_only_for_selected_matching():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="selected")
    roop.config.globals.CFG = SimpleNamespace(face_landmarker_model="fan_68_5")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["recognition"]


def test_process_mgr_adds_genderage_for_gender_filtering():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="all_female")
    roop.config.globals.CFG = SimpleNamespace(face_landmarker_model="fan_68_5")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["genderage"]


def test_process_mgr_keeps_106_only_when_feature_explicitly_needs_it():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="all", restore_original_mouth=True)
    roop.config.globals.CFG = SimpleNamespace(face_landmarker_model="fan_68_5")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["landmark_2d_106"]


def test_hybrid_face_analyser_custom_detector_preserves_compatibility_landmarks(monkeypatch):
    class FakeCompatModel:
        def get(self, _frame, face):
            face["landmark_2d_106"] = np.zeros((106, 2), dtype=np.float32)

    fake_detector = SimpleNamespace(
        detect=lambda _frame, max_num=0: (
            np.array([[1.0, 2.0, 11.0, 12.0, 0.9]], dtype=np.float32),
            np.array([[[2.0, 3.0], [5.0, 3.0], [3.5, 5.0], [2.5, 8.0], [5.5, 8.0]]], dtype=np.float32),
        )
    )
    fake_landmarker = SimpleNamespace(
        detect=lambda _frame, _bbox, _kps: (np.zeros((68, 2), dtype=np.float32), 0.88)
    )
    compat_analyser = SimpleNamespace(models={"detection": object(), "landmark_2d_106": FakeCompatModel()})

    monkeypatch.setattr("roop.face.analyser.create_face_detector", lambda *args, **kwargs: fake_detector)
    monkeypatch.setattr("roop.face.analyser.create_face_landmarker", lambda *args, **kwargs: fake_landmarker)

    analyser = HybridFaceAnalyser(
        compat_analyser,
        "scrfd",
        "fan_68_5",
        ["CPUExecutionProvider"],
        (640, 640),
    )
    faces = analyser.get(np.zeros((16, 16, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert faces[0].landmark_2d_106.shape == (106, 2)
    assert faces[0].landmark_2d_68.shape == (68, 2)
    assert faces[0].kps.shape == (5, 2)


def test_hybrid_face_analyser_uses_detector_batch_when_available(monkeypatch):
    calls = []

    class FakeBatchDetector:
        supports_batch = True
        batch_size_limit = None
        supports_parallel_single_batch = False

        def detect(self, _frame, max_num=0):
            raise AssertionError("single detect should not be used when batch is available")

        def detect_batch(self, frames, max_num=0, batch_size=1):
            calls.append((len(frames), max_num, batch_size))
            return [
                (
                    np.array([[1.0, 2.0, 11.0, 12.0, 0.9]], dtype=np.float32),
                    np.array([[[2.0, 3.0], [5.0, 3.0], [3.5, 5.0], [2.5, 8.0], [5.5, 8.0]]], dtype=np.float32),
                )
                for _frame in frames
            ]

    compat_analyser = SimpleNamespace(models={})

    monkeypatch.setattr("roop.face.analyser.create_face_detector", lambda *args, **kwargs: FakeBatchDetector())
    monkeypatch.setattr("roop.face.analyser.create_face_landmarker", lambda *args, **kwargs: None)

    analyser = HybridFaceAnalyser(
        compat_analyser,
        "yolo_face",
        "insightface_2d106",
        ["CPUExecutionProvider"],
        (640, 640),
    )

    faces_per_frame = analyser.get_many(
        [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
        ],
        batch_size=4,
    )

    assert calls == [(2, 0, 4)]
    assert len(faces_per_frame) == 2
    assert all(len(faces) == 1 for faces in faces_per_frame)


def test_hybrid_face_analyser_parallelizes_single_batch_detector_workers(monkeypatch):
    calls = []

    class FakeSingleBatchDetector:
        supports_batch = False
        batch_size_limit = 1
        supports_parallel_single_batch = True

        def __init__(self, worker_id=0):
            self.worker_id = worker_id

        def detect(self, _frame, max_num=0):
            calls.append((self.worker_id, max_num))
            return (
                np.array([[1.0, 2.0, 11.0, 12.0, 0.9]], dtype=np.float32),
                np.array([[[2.0, 3.0], [5.0, 3.0], [3.5, 5.0], [2.5, 8.0], [5.5, 8.0]]], dtype=np.float32),
            )

        def CreateWorkerDetector(self):
            return FakeSingleBatchDetector(worker_id=self.worker_id + 1)

        def Release(self):
            return None

    compat_analyser = SimpleNamespace(models={})

    monkeypatch.setattr("roop.face.analyser.create_face_detector", lambda *args, **kwargs: FakeSingleBatchDetector())
    monkeypatch.setattr("roop.face.analyser.create_face_landmarker", lambda *args, **kwargs: None)

    analyser = HybridFaceAnalyser(
        compat_analyser,
        "yolo_face",
        "insightface_2d106",
        ["CPUExecutionProvider"],
        (640, 640),
    )

    faces_per_frame = analyser.get_many(
        [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
        ],
        worker_count=2,
    )

    assert len(faces_per_frame) == 3
    assert len(calls) == 3
    assert len(analyser.get_detector_worker_instances(2)) == 2


def test_create_face_analyser_skips_faceanalysis_wrapper_for_custom_detector(monkeypatch):
    captured = {}
    roop.config.globals.CFG = SimpleNamespace(
        face_detector_model="yolo_face",
        face_landmarker_model="fan_68_5",
        force_cpu=True,
    )
    roop.config.globals.g_desired_face_analysis = ["recognition"]

    monkeypatch.setattr("roop.face.analyser.ensure_face_detector_model_downloaded", lambda *_args, **_kwargs: ["det.onnx"])
    monkeypatch.setattr("roop.face.analyser.ensure_face_landmarker_model_downloaded", lambda *_args, **_kwargs: ["lmk.onnx"])
    monkeypatch.setattr("roop.face.analyser.create_face_detector", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr("roop.face.analyser.create_face_landmarker", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr("roop.face.analyser.get_face_analyser_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr("roop.face.analyser.resolve_face_detector_size", lambda *_args, **_kwargs: (640, 640))
    monkeypatch.setattr("roop.face.analyser.resolve_relative_path", lambda path: f"ROOT::{path}")
    def fake_get_insightface_model(model_path, providers=None):
        captured.setdefault("paths", []).append((model_path, tuple(providers or [])))
        return SimpleNamespace(taskname="recognition", prepare=lambda *_args, **_kwargs: None)

    monkeypatch.setattr("roop.face.analyser.get_insightface_model", fake_get_insightface_model)

    def fail_faceanalysis(*_args, **_kwargs):
        raise AssertionError("FaceAnalysis wrapper should not be created for custom detectors")

    monkeypatch.setattr("roop.face.analyser.insightface.app.FaceAnalysis", fail_faceanalysis)

    from roop.face.analyser import _create_face_analyser

    analyser = _create_face_analyser()

    assert analyser.compat_analyser.models.keys() == {"recognition"}
    assert captured["paths"] == [("ROOT::../models/buffalo_l/w600k_r50.onnx", ("CPUExecutionProvider",))]


def test_yolo_face_detector_keeps_bgr_channel_order(monkeypatch):
    captured = {}

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, _outputs, inputs):
            captured["input"] = inputs["input"]
            return [np.zeros((1, 20, 8400), dtype=np.float32)]

    monkeypatch.setattr(
        "roop.face.analytics_runtime.create_inference_session",
        lambda *_args, **_kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        "roop.face.analytics_runtime.get_face_detector_model_paths",
        lambda *_args, **_kwargs: ["fake_yolo.onnx"],
    )

    detector = YoloFaceDetector("yolo_face", ["CPUExecutionProvider"], (640, 640))
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = np.array([10, 20, 30], dtype=np.uint8)

    detector.detect(frame)

    observed = captured["input"][0, :, 0, 0]
    assert np.allclose(observed, np.array([10.0, 20.0, 30.0], dtype=np.float32) / 255.0)


def test_yolo_face_detector_detect_batch_preserves_batch_dimension(monkeypatch):
    captured = {}

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[None, 3, 640, 640])]

        def run(self, _outputs, inputs):
            captured["input_shape"] = inputs["input"].shape
            return [np.zeros((2, 20, 8400), dtype=np.float32)]

    monkeypatch.setattr(
        "roop.face.analytics_runtime.create_inference_session",
        lambda *_args, **_kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        "roop.face.analytics_runtime.get_face_detector_model_paths",
        lambda *_args, **_kwargs: ["fake_yolo.onnx"],
    )

    detector = YoloFaceDetector("yolo_face", ["CPUExecutionProvider"], (640, 640))
    outputs = detector.detect_batch(
        [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),
        ],
        batch_size=4,
    )

    assert captured["input_shape"] == (2, 3, 640, 640)
    assert len(outputs) == 2


def test_fan_68_5_landmarker_normalizes_points_before_inference(monkeypatch):
    captured = {}

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, _outputs, inputs):
            captured["input"] = inputs["input"]
            return [np.zeros((1, 68, 2), dtype=np.float32)]

    monkeypatch.setattr(
        "roop.face.analytics_runtime.create_inference_session",
        lambda *_args, **_kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        "roop.face.analytics_runtime.get_face_landmarker_model_paths",
        lambda *_args, **_kwargs: ["fake_fan_68_5.onnx"],
    )

    landmarker = Fan68From5Landmarker("fan_68_5", ["CPUExecutionProvider"])
    kps = np.array(
        [[10.0, 10.0], [20.0, 10.0], [15.0, 15.0], [11.0, 20.0], [19.0, 20.0]],
        dtype=np.float32,
    )

    landmarker.detect(np.zeros((32, 32, 3), dtype=np.uint8), np.array([8.0, 8.0, 22.0, 22.0], dtype=np.float32), kps)

    normalized_points = captured["input"][0]
    assert normalized_points.shape == (5, 2)
    assert np.all(normalized_points >= 0.0)
    assert np.all(normalized_points <= 1.0)


def test_process_mgr_prefers_refined_landmarks_for_swap_alignment():
    mgr = ProcessMgr(None)
    face = SimpleNamespace(
        landmark_2d_68=np.zeros((68, 2), dtype=np.float32),
        kps=np.array([[1.0, 1.0], [2.0, 1.0], [1.5, 1.5], [1.2, 2.0], [1.8, 2.0]], dtype=np.float32),
    )
    face.landmark_2d_68[36:42] = np.array([[10.0, 11.0]] * 6, dtype=np.float32)
    face.landmark_2d_68[42:48] = np.array([[20.0, 21.0]] * 6, dtype=np.float32)
    face.landmark_2d_68[30] = np.array([15.0, 16.0], dtype=np.float32)
    face.landmark_2d_68[48] = np.array([12.0, 25.0], dtype=np.float32)
    face.landmark_2d_68[54] = np.array([18.0, 25.0], dtype=np.float32)

    landmarks = mgr.get_face_alignment_landmarks(face)

    assert np.allclose(
        landmarks,
        np.array(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [15.0, 16.0],
                [12.0, 25.0],
                [18.0, 25.0],
            ],
            dtype=np.float32,
        ),
    )


def test_process_mgr_rotation_action_falls_back_to_68_landmarks():
    mgr = ProcessMgr(None)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    face = SimpleNamespace(
        bbox=np.array([10.0, 20.0, 90.0, 50.0], dtype=np.float32),
        landmark_2d_68=np.zeros((68, 2), dtype=np.float32),
        landmark_2d_106=None,
    )
    face.landmark_2d_68[17:27] = np.array([[20.0, 24.0]] * 10, dtype=np.float32)
    face.landmark_2d_68[8] = np.array([45.0, 48.0], dtype=np.float32)

    rotation = mgr.rotation_action(face, frame)

    assert rotation == "rotate_clockwise"


def test_process_mgr_create_mouth_mask_falls_back_to_68_landmarks():
    mgr = ProcessMgr(None)
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    face = SimpleNamespace(
        landmark_2d_106=None,
        landmark_2d_68=np.zeros((68, 2), dtype=np.float32),
    )
    mouth_points = np.array(
        [
            [48, 60], [52, 58], [56, 57], [60, 58], [64, 60],
            [60, 62], [56, 63], [52, 62], [50, 60], [53, 59],
            [56, 59], [59, 59], [62, 60], [59, 61], [56, 61],
            [53, 61], [52, 60], [56, 60], [60, 60], [56, 62],
        ],
        dtype=np.float32,
    )
    face.landmark_2d_68[48:68] = mouth_points

    mouth_cutout, mouth_box, mouth_polygon = mgr.create_mouth_mask(face, frame)

    assert mouth_cutout is not None
    assert mouth_cutout.size > 0
    assert mouth_box[2] > mouth_box[0]
    assert mouth_box[3] > mouth_box[1]
    assert mouth_polygon.shape == (20, 2)
