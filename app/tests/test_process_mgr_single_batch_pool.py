from types import SimpleNamespace

import numpy as np

import roop.config.globals
import roop.pipeline.batch_executor as batch_executor
from roop.pipeline.batch_executor import ProcessMgr


class FakeSingleBatchProcessor:
    supports_parallel_single_batch = True
    batch_size_limit = 1

    def __init__(self, worker_id=0):
        self.worker_id = worker_id
        self.released = False
        self.create_calls = 0

    def CreateWorkerProcessor(self):
        self.create_calls += 1
        return FakeSingleBatchProcessor(worker_id=self.worker_id + self.create_calls)

    def Release(self):
        self.released = True

    def Run(self, source_face, target_face, frame):
        return np.full_like(frame, self.worker_id + 1), 1


class FakeSingleBatchMaskProcessor:
    supports_parallel_single_batch = True
    batch_size_limit = 1

    def __init__(self, worker_id=0):
        self.worker_id = worker_id
        self.released = False
        self.create_calls = 0

    def CreateWorkerProcessor(self):
        self.create_calls += 1
        return FakeSingleBatchMaskProcessor(worker_id=self.worker_id + self.create_calls)

    def Release(self):
        self.released = True

    def Run(self, image, _keywords):
        return np.zeros(image.shape[:2] + (1,), dtype=np.float32)


class FakeBrokenBatchSwapProcessor:
    supports_batch = True
    batch_size_limit = None
    supports_parallel_single_batch = True

    def Run(self, _source_face, _target_face, frame):
        return np.ones_like(frame[0], dtype=np.float32) * 0.5

    def RunBatch(self, _source_faces, _target_faces, frames, _batch_size):
        return [frame[0].copy() for frame in frames]


class FakeShortBatchSwapProcessor:
    supports_batch = True
    batch_size_limit = None
    supports_parallel_single_batch = True

    def Run(self, _source_face, _target_face, frame):
        return np.ones_like(frame[0], dtype=np.float32) * 0.5

    def RunBatch(self, _source_faces, _target_faces, frames, _batch_size):
        return [np.ones_like(frames[0][0], dtype=np.float32) * 0.5]


class FakeInitCaptureProcessor:
    processorname = "faceswap"
    type = "swap"

    def __init__(self):
        self.initialized_with = None

    def Initialize(self, options):
        self.initialized_with = dict(options)

    def Release(self):
        return None


def test_process_mgr_parallelizes_single_batch_processors(monkeypatch):
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(num_swap_steps=1, subsample_size=256)
    monkeypatch.setattr(mgr, "get_single_batch_worker_count", lambda processor: 2)
    monkeypatch.setattr(
        mgr,
        "run_prepared_swap_task",
        lambda prepared_task, processor: np.full((1, 1, 3), processor.worker_id + len(prepared_task["cache_key"]), dtype=np.uint8),
    )

    prepared_tasks = [
        {"cache_key": "task_a", "current_frames": [], "input_face": None, "target_face": None},
        {"cache_key": "task_b", "current_frames": [], "input_face": None, "target_face": None},
        {"cache_key": "task_c", "current_frames": [], "input_face": None, "target_face": None},
    ]
    processor = FakeSingleBatchProcessor()

    outputs = mgr.run_swap_tasks_parallel_single_batch(prepared_tasks, processor)

    assert set(outputs.keys()) == {"task_a", "task_b", "task_c"}
    assert all(isinstance(value, np.ndarray) for value in outputs.values())
    assert processor.released is False


def test_process_mgr_passes_face_swap_model_from_options_into_processor(monkeypatch):
    mgr = ProcessMgr(None)
    processor = FakeInitCaptureProcessor()
    mgr.processors = [processor]
    mgr.input_face_datas = []
    mgr.target_face_datas = []
    options = SimpleNamespace(
        processors={"faceswap": {}},
        face_swap_model="hyperswap_1b_256",
        swap_mode="all",
        imagemask=None,
    )

    monkeypatch.setattr("roop.pipeline.batch_executor.get_device", lambda: "cuda")

    mgr.initialize([], [], options)

    assert processor.initialized_with["face_swap_model"] == "hyperswap_1b_256"
    assert processor.initialized_with["devicename"] == "cuda"


def test_process_mgr_aligns_faces_with_model_template_and_refined_landmarks(monkeypatch):
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(face_swap_model="hyperswap_1b_256", subsample_size=256)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    captured = {}
    target_face = SimpleNamespace(
        landmark_2d_68=np.zeros((68, 2), dtype=np.float32),
        kps=np.array([[1.0, 1.0], [2.0, 1.0], [1.5, 1.5], [1.2, 2.0], [1.8, 2.0]], dtype=np.float32),
    )
    target_face.landmark_2d_68[36:42] = np.array([[10.0, 11.0]] * 6, dtype=np.float32)
    target_face.landmark_2d_68[42:48] = np.array([[20.0, 21.0]] * 6, dtype=np.float32)
    target_face.landmark_2d_68[30] = np.array([15.0, 16.0], dtype=np.float32)
    target_face.landmark_2d_68[48] = np.array([12.0, 25.0], dtype=np.float32)
    target_face.landmark_2d_68[54] = np.array([18.0, 25.0], dtype=np.float32)

    def fake_align_crop(image, landmark, image_size, mode):
        captured["image_shape"] = image.shape
        captured["landmark"] = landmark
        captured["image_size"] = image_size
        captured["mode"] = mode
        return np.zeros((image_size, image_size, 3), dtype=np.uint8), np.eye(2, 3, dtype=np.float32)

    monkeypatch.setattr(batch_executor, "align_crop", fake_align_crop)

    mgr.align_face_for_swap(frame, target_face)

    assert captured["mode"] == "arcface_128"
    assert captured["image_size"] == 256
    assert np.allclose(
        captured["landmark"],
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


def test_process_mgr_disables_broken_swap_batch_outputs():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(num_swap_steps=1, subsample_size=128)
    mgr.input_face_datas = [SimpleNamespace(faces=[SimpleNamespace()])]
    mgr.deserialize_face = lambda payload: payload
    processor = FakeBrokenBatchSwapProcessor()
    tasks = [
        {
            "cache_key": "task_a",
            "input_index": 0,
            "target_face": {"id": "a"},
            "aligned_frame": np.zeros((128, 128, 3), dtype=np.uint8),
        },
        {
            "cache_key": "task_b",
            "input_index": 0,
            "target_face": {"id": "b"},
            "aligned_frame": np.zeros((128, 128, 3), dtype=np.uint8),
        },
    ]

    outputs = mgr.run_swap_tasks_batch(tasks, processor, batch_size=8)

    assert processor.supports_batch is False
    assert processor.batch_size_limit == 1
    assert outputs["task_a"].mean() > 0
    assert outputs["task_b"].mean() > 0


def test_process_mgr_disables_swap_batch_when_output_count_mismatches():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(num_swap_steps=1, subsample_size=128)
    mgr.input_face_datas = [SimpleNamespace(faces=[SimpleNamespace()])]
    mgr.deserialize_face = lambda payload: payload
    processor = FakeShortBatchSwapProcessor()
    tasks = [
        {
            "cache_key": "task_a",
            "input_index": 0,
            "target_face": {"id": "a"},
            "aligned_frame": np.zeros((128, 128, 3), dtype=np.uint8),
        },
        {
            "cache_key": "task_b",
            "input_index": 0,
            "target_face": {"id": "b"},
            "aligned_frame": np.zeros((128, 128, 3), dtype=np.uint8),
        },
    ]

    outputs = mgr.run_swap_tasks_batch(tasks, processor, batch_size=8)

    assert processor.supports_batch is False
    assert processor.batch_size_limit == 1
    assert outputs["task_a"].mean() > 0
    assert outputs["task_b"].mean() > 0


def test_process_mgr_reuses_single_batch_worker_sessions_across_calls(monkeypatch):
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(num_swap_steps=1, subsample_size=256)
    monkeypatch.setattr(mgr, "get_single_batch_worker_count", lambda processor: 3)
    monkeypatch.setattr(
        mgr,
        "run_prepared_swap_task",
        lambda prepared_task, processor: np.full((1, 1, 3), processor.worker_id + 1, dtype=np.uint8),
    )

    prepared_tasks = [
        {"cache_key": "task_a", "current_frames": [], "input_face": None, "target_face": None},
        {"cache_key": "task_b", "current_frames": [], "input_face": None, "target_face": None},
    ]
    processor = FakeSingleBatchProcessor()

    mgr.run_swap_tasks_parallel_single_batch(prepared_tasks, processor)
    mgr.run_swap_tasks_parallel_single_batch(prepared_tasks, processor)

    assert processor.create_calls == 2
    assert len(mgr.single_batch_worker_pools[id(processor)]["workers"]) == 3

    mgr.processors = [processor]
    mgr.release_resources()

    assert processor.released is True


def test_process_mgr_single_batch_worker_count_uses_memory_plan(monkeypatch):
    mgr = ProcessMgr(None)
    processor = FakeSingleBatchProcessor()
    monkeypatch.setattr(roop.config.globals, "active_memory_plan", {"single_batch_workers": 3}, raising=False)

    assert mgr.get_single_batch_worker_count(processor) == 3


def test_process_mgr_single_batch_worker_count_uses_requested_workers_without_memory_plan(monkeypatch):
    mgr = ProcessMgr(None)
    processor = FakeSingleBatchProcessor()
    monkeypatch.setattr(roop.config.globals, "active_memory_plan", None, raising=False)
    monkeypatch.setattr(roop.config.globals, "CFG", SimpleNamespace(single_batch_workers=4), raising=False)
    monkeypatch.setattr("roop.pipeline.batch_executor.resolve_single_batch_workers", lambda configured_workers: (configured_workers, configured_workers, None))

    assert mgr.get_single_batch_worker_count(processor) == 4


def test_process_mgr_parallelizes_single_batch_enhancers(monkeypatch):
    mgr = ProcessMgr(None)
    mgr.input_face_datas = [SimpleNamespace(faces=[])]
    mgr.deserialize_face = lambda payload: payload
    monkeypatch.setattr(mgr, "get_single_batch_worker_count", lambda processor: 2)

    tasks = [
        {"cache_key": "task_a", "input_index": 0, "target_face": {"id": "a"}},
        {"cache_key": "task_b", "input_index": 0, "target_face": {"id": "b"}},
    ]
    current_frames = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
    ]

    outputs = mgr.run_enhance_tasks_batch(tasks, current_frames, FakeSingleBatchProcessor(), batch_size=8)

    assert set(outputs.keys()) == {"task_a", "task_b"}
    assert all(isinstance(value, np.ndarray) for value in outputs.values())


def test_process_mgr_parallelizes_single_batch_masks(monkeypatch):
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(masking_text="", show_face_masking=False)
    monkeypatch.setattr(mgr, "get_single_batch_worker_count", lambda processor: 2)

    tasks = [
        {"cache_key": "task_a", "aligned_frame": np.zeros((2, 2, 3), dtype=np.uint8)},
        {"cache_key": "task_b", "aligned_frame": np.zeros((2, 2, 3), dtype=np.uint8)},
    ]
    current_frames = [
        np.full((2, 2, 3), 255, dtype=np.uint8),
        np.full((2, 2, 3), 255, dtype=np.uint8),
    ]

    outputs = mgr.run_mask_tasks_batch(tasks, current_frames, FakeSingleBatchMaskProcessor(), batch_size=8)

    assert set(outputs.keys()) == {"task_a", "task_b"}
    assert all(isinstance(value, np.ndarray) for value in outputs.values())


def test_paste_upscale_uses_roi_warp_sizes(monkeypatch):
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(show_face_area_overlay=False, blend_ratio=0.5)
    target = np.zeros((512, 512, 3), dtype=np.uint8)
    fake = np.full((512, 512, 3), 255, dtype=np.uint8)
    matrix = np.array([[2.0, 0.0, 64.0], [0.0, 2.0, 64.0]], dtype=np.float32)
    calls = []
    real_warp_affine = batch_executor.cv2.warpAffine

    def tracking_warp_affine(image, matrix_arg, dsize, *args, **kwargs):
        calls.append(tuple(int(v) for v in dsize))
        return real_warp_affine(image, matrix_arg, dsize, *args, **kwargs)

    monkeypatch.setattr(batch_executor.cv2, "warpAffine", tracking_warp_affine)

    result = mgr.paste_upscale(fake, fake, matrix, target, 1, [0, 0, 0, 0, 20.0])

    assert result is target
    assert calls
    assert all(width < target.shape[1] and height < target.shape[0] for width, height in calls)


def test_compose_task_mutates_base_frame_in_place(monkeypatch):
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(show_face_area_overlay=False, blend_ratio=0.5, restore_original_mouth=False)
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    fake = np.full((64, 64, 3), 123, dtype=np.uint8)
    target_face = batch_executor.deserialize_face_payload(
        {
            "bbox": np.array([0, 0, 10, 10], dtype=np.float32),
            "kps": np.array([[1, 1], [2, 1], [1.5, 1.5], [1.2, 2], [1.8, 2]], dtype=np.float32),
            "matrix": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        }
    )
    monkeypatch.setattr(mgr, "deserialize_face", lambda payload: target_face)

    task = {
        "target_face": {},
        "mask_offsets": [0, 0, 0, 0, 20.0],
        "rotation_action": None,
        "cutout_box": None,
    }

    result = mgr.compose_task(base, task, fake)

    assert result is base

