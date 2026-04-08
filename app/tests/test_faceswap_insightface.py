from types import SimpleNamespace

import numpy as np
import onnx

from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace


class FakeInputMeta:
    def __init__(self, shape, name="input"):
        self.shape = shape
        self.name = name


class FakeOutputMeta:
    def __init__(self, name="output"):
        self.name = name


class FakeSession:
    def __init__(self, fail_on_batch_over_one=False):
        self.fail_on_batch_over_one = fail_on_batch_over_one
        self.calls = []

    def get_inputs(self):
        return [
            FakeInputMeta([1, 3, 128, 128], name="target"),
            FakeInputMeta([1, 512], name="source"),
        ]

    def get_outputs(self):
        return [FakeOutputMeta(name="output")]

    def run(self, _, inputs):
        target = inputs["target"]
        self.calls.append(target.shape[0])
        if self.fail_on_batch_over_one and target.shape[0] > 1:
            raise RuntimeError(
                "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: target "
                "for the following indices index: 0 Got: 80 Expected: 1"
            )
        return [target.copy()]


class FakeDynamicSession(FakeSession):
    def get_inputs(self):
        return [
            FakeInputMeta(["batch", 3, 128, 128], name="target"),
            FakeInputMeta(["batch", 512], name="source"),
        ]


def make_source_face():
    return SimpleNamespace(normed_embedding=np.ones((512,), dtype=np.float32))


def test_faceswap_insightface_detects_single_batch_model():
    processor = FaceSwapInsightFace()
    processor.model_swap_insightface = FakeSession()
    processor.batch_size_limit = processor._resolve_batch_size_limit()

    assert processor._resolve_batch_size_limit() == 1
    assert processor._effective_batch_size(64) == 1


def test_faceswap_insightface_falls_back_to_single_batch_when_ort_rejects_batch():
    processor = FaceSwapInsightFace()
    processor.model_swap_insightface = FakeSession(fail_on_batch_over_one=True)
    processor.emap = np.eye(512, dtype=np.float32)
    processor.batch_size_limit = None
    processor.supports_batch = True

    source_faces = [make_source_face(), make_source_face()]
    target_faces = [None, None]
    temp_frames = [
        np.ones((1, 3, 128, 128), dtype=np.float32),
        np.ones((1, 3, 128, 128), dtype=np.float32) * 2,
    ]

    outputs = processor.RunBatch(source_faces, target_faces, temp_frames, batch_size=8)

    assert len(outputs) == 2
    assert processor.batch_size_limit == 1
    assert processor.supports_batch is False
    assert processor.model_swap_insightface.calls == [2, 1, 1]


def test_faceswap_insightface_initialize_uses_native_batch_model(monkeypatch):
    captured = {}
    fake_graph = SimpleNamespace(
        initializer=[onnx.numpy_helper.from_array(np.eye(512, dtype=np.float32), name="emap")]
    )

    monkeypatch.setattr("roop.processors.FaceSwapInsightFace.resolve_relative_path", lambda _path: "original.onnx")
    monkeypatch.setattr("roop.processors.FaceSwapInsightFace.onnx.load", lambda _path: SimpleNamespace(graph=fake_graph))

    try:
        monkeypatch.setattr(
            "roop.processors.FaceSwapInsightFace.resolve_model_path_for_processor",
            lambda model_path, _processor_name: "patched.onnx",
        )
    except AttributeError as exc:
        raise AssertionError("FaceSwapInsightFace should import native batch helper") from exc

    def fake_inference_session(model_path, *_args, **_kwargs):
        captured["model_path"] = model_path
        return FakeDynamicSession()

    monkeypatch.setattr("roop.processors.FaceSwapInsightFace.create_inference_session", fake_inference_session)

    processor = FaceSwapInsightFace()
    processor.Initialize({"devicename": "cuda"})

    assert captured["model_path"] == "patched.onnx"
    assert processor.batch_size_limit is None
    assert processor.supports_batch is True
