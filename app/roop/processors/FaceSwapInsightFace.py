import roop.globals
import cv2
import numpy as np
import onnx
import onnxruntime

from roop.onnx_runtime import get_execution_providers_for_processor, resolve_model_path_for_processor
from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path



class FaceSwapInsightFace():
    plugin_options:dict = None
    model_swap_insightface = None
    source_latent_cache = None

    processorname = 'faceswap'
    type = 'swap'
    supports_batch = True
    batch_size_limit = None
    supports_parallel_single_batch = True


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_swap_insightface is None:
            original_model_path = resolve_relative_path('../models/inswapper_128.onnx')
            graph = onnx.load(original_model_path).graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
            model_path = resolve_model_path_for_processor(original_model_path, self.processorname)
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            self.input_mean = 0.0
            self.input_std = 255.0
            #cuda_options = {"arena_extend_strategy": "kSameAsRequested", 'cudnn_conv_algo_search': 'DEFAULT'}            
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False
            self.model_swap_insightface = onnxruntime.InferenceSession(
                model_path,
                sess_options,
                providers=get_execution_providers_for_processor(self.processorname),
            )
            self.batch_size_limit = self._resolve_batch_size_limit()
            self.supports_batch = self.batch_size_limit is None or self.batch_size_limit > 1
        self.source_latent_cache = {}


    def _resolve_batch_size_limit(self):
        if self.model_swap_insightface is None:
            return 1
        batch_size_limit = None
        for input_meta in self.model_swap_insightface.get_inputs():
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
        worker = FaceSwapInsightFace()
        worker.Initialize(dict(self.plugin_options or {}))
        return worker


    def _project_source_latent(self, source_face: Face):
        if self.source_latent_cache is None:
            self.source_latent_cache = {}
        cache_key = id(source_face)
        cached_latent = self.source_latent_cache.get(cache_key)
        if cached_latent is not None:
            return cached_latent
        latent = np.asarray(source_face.normed_embedding, dtype=np.float32).reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        latent = latent.astype(np.float32, copy=False)
        self.source_latent_cache[cache_key] = latent
        return latent


    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        latent = self._project_source_latent(source_face)
        io_binding = self.model_swap_insightface.io_binding()           
        io_binding.bind_cpu_input("target", temp_frame)
        io_binding.bind_cpu_input("source", latent)
        io_binding.bind_output("output", self.devicename)
        self.model_swap_insightface.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()[0]
        return ort_outs[0]


    def RunBatch(self, source_faces, target_faces, temp_frames, batch_size=1):
        outputs = []
        effective_batch_size = self._effective_batch_size(batch_size)
        for batch_start in range(0, len(temp_frames), effective_batch_size):
            batch_end = batch_start + effective_batch_size
            batch_frames = np.concatenate(temp_frames[batch_start:batch_end], axis=0).astype(np.float32)
            latents = []
            for source_face in source_faces[batch_start:batch_end]:
                latents.append(self._project_source_latent(source_face))
            batch_latents = np.concatenate(latents, axis=0)
            try:
                batch_outputs = self.model_swap_insightface.run(None, {"target": batch_frames, "source": batch_latents})[0]
            except Exception as exc:
                should_disable_batch = batch_frames.shape[0] > 1 and "Got invalid dimensions for input: target" in str(exc)
                if not should_disable_batch:
                    raise
                self.batch_size_limit = 1
                self.supports_batch = False
                return self.RunBatch(source_faces, target_faces, temp_frames, batch_size=1)
            outputs.extend([output for output in batch_outputs])
        return outputs


    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None
        self.source_latent_cache = None


                



