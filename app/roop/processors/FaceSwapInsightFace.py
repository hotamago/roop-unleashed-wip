import roop.globals
import cv2
import numpy as np
import onnx
import onnxruntime

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path



class FaceSwapInsightFace():
    plugin_options:dict = None
    model_swap_insightface = None

    processorname = 'faceswap'
    type = 'swap'
    supports_batch = True


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_swap_insightface is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            graph = onnx.load(model_path).graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            self.input_mean = 0.0
            self.input_std = 255.0
            #cuda_options = {"arena_extend_strategy": "kSameAsRequested", 'cudnn_conv_algo_search': 'DEFAULT'}            
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False
            self.model_swap_insightface = onnxruntime.InferenceSession(model_path, sess_options, providers=roop.globals.execution_providers)



    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        latent = source_face.normed_embedding.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        io_binding = self.model_swap_insightface.io_binding()           
        io_binding.bind_cpu_input("target", temp_frame)
        io_binding.bind_cpu_input("source", latent)
        io_binding.bind_output("output", self.devicename)
        self.model_swap_insightface.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()[0]
        return ort_outs[0]


    def RunBatch(self, source_faces, target_faces, temp_frames, batch_size=1):
        outputs = []
        for batch_start in range(0, len(temp_frames), max(1, batch_size)):
            batch_end = batch_start + max(1, batch_size)
            batch_frames = np.concatenate(temp_frames[batch_start:batch_end], axis=0).astype(np.float32)
            latents = []
            for source_face in source_faces[batch_start:batch_end]:
                latent = source_face.normed_embedding.reshape((1, -1))
                latent = np.dot(latent, self.emap)
                latent /= np.linalg.norm(latent)
                latents.append(latent.astype(np.float32))
            batch_latents = np.concatenate(latents, axis=0)
            batch_outputs = self.model_swap_insightface.run(None, {"target": batch_frames, "source": batch_latents})[0]
            outputs.extend([output for output in batch_outputs])
        return outputs


    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None


                



