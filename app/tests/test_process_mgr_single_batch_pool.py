from types import SimpleNamespace

import numpy as np

import roop.globals
from roop.ProcessMgr import ProcessMgr


class FakeSingleBatchProcessor:
    supports_parallel_single_batch = True
    batch_size_limit = 1

    def __init__(self, worker_id=0):
        self.worker_id = worker_id
        self.released = False

    def CreateWorkerProcessor(self):
        return FakeSingleBatchProcessor(worker_id=self.worker_id + 1)

    def Release(self):
        self.released = True


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


def test_process_mgr_single_batch_worker_count_uses_memory_plan(monkeypatch):
    mgr = ProcessMgr(None)
    processor = FakeSingleBatchProcessor()
    monkeypatch.setattr(roop.globals, "active_memory_plan", {"swap_single_batch_workers": 3}, raising=False)

    assert mgr.get_single_batch_worker_count(processor) == 3
