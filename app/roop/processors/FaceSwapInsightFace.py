import cv2
import numpy as np
import onnx

import roop.config.globals
from roop.config.types import Face, Frame
from roop.face_swap_models import (
    ensure_face_swap_model_downloaded,
    get_face_swap_model_key,
    get_face_swap_model_tile_size,
    get_face_swap_model_type,
)
from roop.onnx.runtime import resolve_model_path_for_processor
from roop.onnx.session import create_inference_session
from roop.processors.base import BaseProcessor


class FaceSwapInsightFace(BaseProcessor):
    plugin_options: dict = None
    model_swap_insightface = None
    source_latent_cache = None

    processorname = "faceswap"
    type = "swap"
    supports_batch = True
    batch_size_limit = None
    supports_parallel_single_batch = True

    def _resolve_projection_matrix(self, graph, model_type: str):
        if model_type != "inswapper":
            return None

        preferred_names = {"buff2fs", "initializer"}
        matching_initializers = []
        for initializer in graph.initializer:
            dims = tuple(int(dim) for dim in initializer.dims)
            if dims != (512, 512):
                continue
            matching_initializers.append(initializer)
            if initializer.name in preferred_names:
                return onnx.numpy_helper.to_array(initializer)

        if graph.initializer:
            last_initializer = graph.initializer[-1]
            last_dims = tuple(int(dim) for dim in last_initializer.dims)
            if last_dims == (512, 512):
                return onnx.numpy_helper.to_array(last_initializer)

        if len(matching_initializers) == 1:
            return onnx.numpy_helper.to_array(matching_initializers[0])
        return None

    def _resolve_selected_model_key(self, plugin_options: dict | None = None) -> str:
        selected_model = None
        if isinstance(plugin_options, dict):
            selected_model = plugin_options.get("face_swap_model")
        if not selected_model:
            selected_model = getattr(roop.config.globals.CFG, "face_swap_model", None)
        return get_face_swap_model_key(selected_model)

    def _resolve_model_input_size(self):
        for input_meta in getattr(self, "model_inputs", []) or []:
            shape = getattr(input_meta, "shape", None) or []
            if len(shape) < 4:
                continue
            height = shape[2]
            width = shape[3]
            if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0 and height == width:
                return int(height)
        return get_face_swap_model_tile_size(getattr(self, "active_model_key", None))

    def _get_active_model_type(self):
        return get_face_swap_model_type(getattr(self, "active_model_key", None))

    def Initialize(self, plugin_options: dict):
        plugin_options = dict(plugin_options or {})
        selected_model = self._resolve_selected_model_key(plugin_options)
        plugin_options["face_swap_model"] = selected_model

        if self.plugin_options is not None:
            previous_model = self.plugin_options.get("face_swap_model")
            if self.plugin_options["devicename"] != plugin_options["devicename"] or previous_model != selected_model:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_swap_insightface is None:
            original_model_path = ensure_face_swap_model_downloaded(selected_model)
            selected_model_type = get_face_swap_model_type(selected_model)
            graph = onnx.load(original_model_path).graph
            self.emap = self._resolve_projection_matrix(graph, selected_model_type)
            if selected_model_type == "inswapper":
                model_path = resolve_model_path_for_processor(original_model_path, self.processorname)
            else:
                model_path = original_model_path
            self.devicename = self.plugin_options["devicename"].replace("mps", "cpu")
            self.active_model_key = selected_model
            self.model_swap_insightface = create_inference_session(model_path, self.processorname)
            self.model_inputs = self.model_swap_insightface.get_inputs()
            self.model_outputs = self.model_swap_insightface.get_outputs()
            self.target_input_name = next(
                (meta.name for meta in self.model_inputs if meta.name.lower() == "target"),
                self.model_inputs[0].name,
            )
            self.source_input_name = next(
                (meta.name for meta in self.model_inputs if meta.name.lower() == "source"),
                self.model_inputs[-1].name,
            )
            self.output_name = next(
                (meta.name for meta in self.model_outputs if meta.name.lower() == "output"),
                self.model_outputs[0].name,
            )
            self.model_input_size = self._resolve_model_input_size()
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

        if self._get_active_model_type() == "inswapper":
            source_embedding = getattr(source_face, "embedding", None)
            if source_embedding is None:
                source_embedding = getattr(source_face, "normed_embedding", None)
        else:
            source_embedding = getattr(source_face, "normed_embedding", None)
            if source_embedding is None:
                source_embedding = getattr(source_face, "embedding", None)

        latent = np.asarray(source_embedding, dtype=np.float32).reshape((1, -1))
        if self.emap is not None:
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)
        latent = latent.astype(np.float32, copy=False)
        self.source_latent_cache[cache_key] = latent
        return latent

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        del target_face
        latent = np.ascontiguousarray(self._project_source_latent(source_face), dtype=np.float32)
        temp_frame = np.ascontiguousarray(temp_frame, dtype=np.float32)
        io_binding = self.model_swap_insightface.io_binding()
        io_binding.bind_cpu_input(getattr(self, "target_input_name", "target"), temp_frame)
        io_binding.bind_cpu_input(getattr(self, "source_input_name", "source"), latent)
        io_binding.bind_output(getattr(self, "output_name", "output"), self.devicename)
        self.model_swap_insightface.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()[0]
        return ort_outs[0]

    def RunBatch(self, source_faces, target_faces, temp_frames, batch_size=1):
        outputs = []
        effective_batch_size = self._effective_batch_size(batch_size)
        for batch_start in range(0, len(temp_frames), effective_batch_size):
            batch_end = batch_start + effective_batch_size
            current_frames = temp_frames[batch_start:batch_end]
            current_faces = source_faces[batch_start:batch_end]
            if not current_frames:
                continue
            batch_count = len(current_frames)
            frame_shape = tuple(current_frames[0].shape[1:])
            batch_frames = self._get_batch_buffer("swap_target", (effective_batch_size, *frame_shape), np.float32)
            latent_width = int(self._project_source_latent(current_faces[0]).shape[-1])
            batch_latents = self._get_batch_buffer("swap_source", (effective_batch_size, latent_width), np.float32)
            for index, (temp_frame, source_face) in enumerate(zip(current_frames, current_faces)):
                batch_frames[index] = np.ascontiguousarray(temp_frame[0], dtype=np.float32)
                batch_latents[index] = self._project_source_latent(source_face)[0]
            try:
                batch_outputs = self.model_swap_insightface.run(
                    None,
                    {
                        getattr(self, "target_input_name", "target"): np.ascontiguousarray(batch_frames[:batch_count]),
                        getattr(self, "source_input_name", "source"): np.ascontiguousarray(batch_latents[:batch_count]),
                    },
                )[0]
            except Exception as exc:
                error_text = str(exc)
                should_disable_batch = batch_count > 1 and (
                    "Got invalid dimensions for input: target" in error_text
                    or "Input channels C is not equal to kernel channels * group" in error_text
                )
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
        self.active_model_key = None
        self.model_input_size = None
        self._clear_batch_buffers()
