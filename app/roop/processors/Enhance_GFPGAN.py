from typing import Any, List, Callable
import cv2 
import numpy as np
import roop.config.globals

from roop.onnx.runtime import resolve_model_path_for_processor
from roop.onnx.session import create_inference_session
from roop.processors.base import BaseProcessor
from roop.config.types import Face, Frame, FaceSet
from roop.utils import resolve_relative_path


# THREAD_LOCK = threading.Lock()


class Enhance_GFPGAN(BaseProcessor):
    plugin_options:dict = None

    model_gfpgan = None
    name = None
    devicename = None

    processorname = 'gfpgan'
    type = 'enhance'
    supports_batch = True
    batch_size_limit = None
    supports_parallel_single_batch = True


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_gfpgan is None:
            model_path = resolve_model_path_for_processor(resolve_relative_path('../models/GFPGANv1.4.onnx'), self.processorname)
            self.model_gfpgan = create_inference_session(model_path, self.processorname)
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            self.model_inputs = self.model_gfpgan.get_inputs()
            self.model_outputs = self.model_gfpgan.get_outputs()
            self.input_name = self.model_inputs[0].name
            self.output_name = self.model_outputs[0].name

        self.name = getattr(self, "input_name", self.model_gfpgan.get_inputs()[0].name)
        self.batch_size_limit = self._resolve_batch_size_limit()
        self.supports_batch = self.batch_size_limit is None or self.batch_size_limit > 1


    def _resolve_batch_size_limit(self):
        if self.model_gfpgan is None:
            return 1
        batch_size_limit = None
        for input_meta in self.model_gfpgan.get_inputs():
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
        worker = Enhance_GFPGAN()
        worker.Initialize(dict(self.plugin_options or {}))
        return worker


    def _preprocess_frame(self, temp_frame):
        temp_frame = cv2.resize(temp_frame, (512, 512), cv2.INTER_CUBIC)
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        return temp_frame


    def _postprocess_frame(self, result):
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result.astype(np.uint8)

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        # preprocess
        input_size = temp_frame.shape[1]
        batch_input = self._get_batch_buffer("gfpgan_input_single", (1, 3, 512, 512), np.float32)
        batch_input[0] = self._preprocess_frame(temp_frame).transpose(2, 0, 1)
        temp_frame = np.ascontiguousarray(batch_input[:1])

        io_binding = self.model_gfpgan.io_binding()           
        io_binding.bind_cpu_input(getattr(self, "input_name", "input"), temp_frame)
        io_binding.bind_output(getattr(self, "output_name", "1288"), self.devicename)
        self.model_gfpgan.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]

        # post-process
        result = self._postprocess_frame(result)
        scale_factor = int(result.shape[1] / input_size)       
        return result, scale_factor


    def RunBatch(self, source_facesets, target_faces, temp_frames, batch_size=1):
        outputs = []
        effective_batch_size = self._effective_batch_size(batch_size)
        for batch_start in range(0, len(temp_frames), effective_batch_size):
            current_frames = temp_frames[batch_start:batch_start + effective_batch_size]
            if not current_frames:
                continue
            batch_input = self._get_batch_buffer("gfpgan_input", (effective_batch_size, 3, 512, 512), np.float32)
            for index, temp_frame in enumerate(current_frames):
                batch_input[index] = self._preprocess_frame(temp_frame).transpose(2, 0, 1)
            active_batch = np.ascontiguousarray(batch_input[: len(current_frames)])
            try:
                batch_outputs = self.model_gfpgan.run(None, {getattr(self, "input_name", "input"): active_batch})[0]
            except Exception as exc:
                should_disable_batch = len(current_frames) > 1 and "Got invalid dimensions for input" in str(exc)
                if not should_disable_batch:
                    raise
                self.batch_size_limit = 1
                self.supports_batch = False
                return self.RunBatch(source_facesets, target_faces, temp_frames, batch_size=1)
            for result in batch_outputs:
                outputs.append(self._postprocess_frame(result))
        return outputs


    def Release(self):
        del self.model_gfpgan
        self.model_gfpgan = None
        self._clear_batch_buffers()












