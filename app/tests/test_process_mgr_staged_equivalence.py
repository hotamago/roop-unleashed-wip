from types import SimpleNamespace

import numpy as np

import roop.config.globals
from roop.pipeline.batch_executor import ProcessMgr


class FakeSwapProcessor:
    type = "swap"
    supports_batch = True

    def Run(self, _source_face, _target_face, prepared_frame):
        return np.ones_like(prepared_frame[0], dtype=np.float32) * 0.5

    def RunBatch(self, source_faces, target_faces, prepared_frames, _batch_size):
        return [np.ones_like(frame[0], dtype=np.float32) * 0.5 for frame in prepared_frames]


def test_process_mgr_staged_plan_matches_legacy_process_frame(monkeypatch):
    mgr = ProcessMgr(None)
    mgr.input_face_datas = [SimpleNamespace(faces=[SimpleNamespace(mask_offsets=[0] * 10)])]
    mgr.target_face_datas = [SimpleNamespace(embedding=np.array([1.0], dtype=np.float32))]
    mgr.options = SimpleNamespace(
        swap_mode="selected",
        selected_index=0,
        face_distance_threshold=1.0,
        subsample_size=128,
        num_swap_steps=1,
        imagemask=None,
        show_face_masking=False,
        restore_original_mouth=False,
        blend_ratio=1.0,
    )
    mgr.processors = [FakeSwapProcessor()]

    detected_face = SimpleNamespace(
        bbox=np.array([0, 0, 128, 128], dtype=np.float32),
        kps=np.zeros((5, 2), dtype=np.float32),
        embedding=np.array([1.0], dtype=np.float32),
        landmark_2d_106=np.zeros((106, 2), dtype=np.float32),
    )

    monkeypatch.setattr("roop.pipeline.batch_executor.get_all_faces", lambda _frame: [detected_face])
    monkeypatch.setattr("roop.pipeline.batch_executor.compute_cosine_distance", lambda _a, _b: 0.0)
    monkeypatch.setattr("roop.pipeline.batch_executor.align_crop", lambda frame, _kps, _size, _mode=None: (frame.copy(), np.eye(2, 3, dtype=np.float32)))
    monkeypatch.setattr(ProcessMgr, "paste_upscale", lambda self, fake_face, _upsk_face, _matrix, _target_img, _scale_factor, _mask_offsets, face_landmarks=None: fake_face)
    monkeypatch.setattr(roop.config.globals, "autorotate_faces", False, raising=False)
    monkeypatch.setattr(roop.config.globals, "vr_mode", False, raising=False)
    monkeypatch.setattr(roop.config.globals, "no_face_action", 0, raising=False)

    frame = np.full((128, 128, 3), 7, dtype=np.uint8)

    legacy_result = mgr.process_frame(frame.copy())

    frame_plan = mgr.build_frame_plan(frame.copy())
    assert frame_plan["fallback"] is False
    staged_tasks = []
    for task_index, task_meta in enumerate(frame_plan["tasks"]):
        task = dict(task_meta)
        task["cache_key"] = f"task_{task_index}"
        task["aligned_frame"] = mgr.rebuild_aligned_frame(frame.copy(), task_meta)
        staged_tasks.append(task)

    swap_outputs = mgr.run_swap_tasks_batch(staged_tasks, mgr.processors[0], batch_size=8)

    staged_result = frame.copy()
    for task_index, task_meta in enumerate(frame_plan["tasks"]):
        task_with_cache_key = dict(task_meta)
        task_with_cache_key["cache_key"] = f"task_{task_index}"
        staged_result = mgr.compose_task(
            staged_result,
            task_with_cache_key,
            swap_outputs[task_with_cache_key["cache_key"]],
            None,
        )

    assert np.array_equal(staged_result, legacy_result)

