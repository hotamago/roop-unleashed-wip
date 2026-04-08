from types import SimpleNamespace

import numpy as np

from roop.face.analyser import HybridFaceAnalyser
from roop.face.analytics_runtime import Fan68From5Landmarker, YoloFaceDetector
from roop.pipeline.batch_executor import ProcessMgr


def test_process_mgr_limits_face_analysis_modules_for_non_matching_modes():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="all")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["landmark_3d_68", "landmark_2d_106", "detection"]


def test_process_mgr_adds_recognition_only_for_selected_matching():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="selected")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["landmark_3d_68", "landmark_2d_106", "detection", "recognition"]


def test_process_mgr_adds_genderage_for_gender_filtering():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="all_female")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["landmark_3d_68", "landmark_2d_106", "detection", "genderage"]


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
