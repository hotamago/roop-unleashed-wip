import numpy as np


class BaseProcessor:
    plugin_options = None
    batch_size_limit = None
    supports_batch = False
    supports_parallel_single_batch = False

    def _get_active_session(self):
        for name in dir(self):
            if not name.startswith("model_"):
                continue
            value = getattr(self, name, None)
            if hasattr(value, "get_inputs") and callable(value.get_inputs):
                return value
        return None

    def _resolve_batch_size_limit(self):
        session = self._get_active_session()
        if session is None:
            return 1
        batch_size_limit = None
        for input_meta in session.get_inputs():
            shape = getattr(input_meta, "shape", None) or []
            if len(shape) < 1:
                continue
            batch_dim = shape[0]
            if isinstance(batch_dim, int) and batch_dim > 0:
                batch_size_limit = batch_dim if batch_size_limit is None else min(batch_size_limit, batch_dim)
        return batch_size_limit

    def _effective_batch_size(self, requested_batch_size):
        effective_batch_size = max(1, int(requested_batch_size))
        if self.batch_size_limit is not None:
            effective_batch_size = min(effective_batch_size, max(1, int(self.batch_size_limit)))
        return effective_batch_size

    def CreateWorkerProcessor(self):
        worker = self.__class__()
        if hasattr(worker, "Initialize"):
            worker.Initialize(dict(self.plugin_options or {}))
        return worker

    def _get_batch_buffer(self, name: str, shape, dtype=np.float32):
        shape = tuple(int(dim) for dim in shape)
        dtype = np.dtype(dtype)
        batch_buffers = getattr(self, "_batch_buffers", None)
        if batch_buffers is None:
            batch_buffers = {}
            self._batch_buffers = batch_buffers
        buffer_key = (name, shape, dtype.str)
        batch_buffer = batch_buffers.get(buffer_key)
        if batch_buffer is None:
            batch_buffer = np.empty(shape, dtype=dtype)
            batch_buffers[buffer_key] = batch_buffer
        return batch_buffer

    def _clear_batch_buffers(self):
        batch_buffers = getattr(self, "_batch_buffers", None)
        if isinstance(batch_buffers, dict):
            batch_buffers.clear()
